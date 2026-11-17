"""Casos de construção de catálogo sem rede, com decisões semânticas controladas."""

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path

from determinism.modules.semantic_catalog.adapters.json_artifacts import JsonResponseCache
from determinism.modules.semantic_catalog.application import CatalogStage
from determinism.modules.semantic_catalog.domain import CatalogError, simulate_removal
from determinism.modules.semantic_catalog.ports.provider import CatalogResponse


APPEARANCE = "Harry e Meghan apareceram em público com o filho."
FIRST_APPEARANCE = "Foi a primeira aparição pública de Harry e Meghan com o filho."
NAME = "O filho de Harry e Meghan se chama Archie."


def dataset_for(contents):
    """Um item por fonte; um item aninhado permite vários segmentos na fonte."""
    return [
        {"name": f"Fonte {source_index}", "metadata": {"date": "2020-01-01"},
         "segments": [{"content": text, "start": index * 2.0}
                      for index, text in enumerate([items] if isinstance(items, str) else items)]}
        for source_index, items in enumerate(contents)
    ]


def candidate(statement, quote=None, indices=None):
    return {"statement": statement, "evidence_quotes": [quote or statement],
            "information_indices": [] if indices is None else list(indices)}


class ControlledProvider:
    """Provedor com semântica explicitamente fornecida pelo caso de teste."""

    def __init__(self, extracted=None, *, relations=None, supports=None, mutations=None, fail_once=None):
        self.extracted = extracted or {}
        self.relations = relations or {}
        self.supports = supports or {}
        self.mutations = mutations or {}
        self.fail_once = set(fail_once or ())
        self.calls = []
        self.lock = threading.Lock()

    def _result(self, operation, arguments, payload):
        with self.lock:
            self.calls.append((operation, copy.deepcopy(arguments)))
            should_fail = operation in self.fail_once
            self.fail_once.discard(operation)
        if should_fail:
            raise CatalogError("Falha simulada durante a construção.")
        mutation = self.mutations.get(operation)
        if mutation is not None:
            payload = mutation(payload)
        return CatalogResponse(payload, "controlled-model-v1", "fp_controlled")

    def extract(self, segment):
        items = self.extracted.get(segment["id"], [candidate(segment["content"])])
        return self._result("extract", segment, {"facts": copy.deepcopy(items)})

    def embed(self, texts):
        # Escores iguais mantêm estes cenários focados nas decisões factuais.
        return self._result("embed", texts, {"embeddings": [[1.0, 0.0] for text in texts]})

    def compare(self, pairs):
        items = []
        for pair in pairs:
            key = frozenset((pair["left"]["id"], pair["right"]["id"]))
            default = "equivalent" if pair["left"]["statement"] == pair["right"]["statement"] else "different"
            items.append({"pair_id": pair["pair_id"], "relation": self.relations.get(key, default),
                          "reason": "Decisão fornecida pelo experimento controlado."})
        return self._result("compare", pairs, {"comparisons": items})

    def assess(self, segment, facts):
        original_statements = {
            item["statement"] for item in self.extracted.get(segment["id"], [candidate(segment["content"])])
        }
        items = []
        for fact in facts:
            default = "supported" if fact["statement"] in original_statements else "not_supported"
            status = self.supports.get((segment["id"], fact["statement"]), default)
            items.append({"fact_id": fact["id"], "status": status,
                          "evidence_quotes": [segment["content"]] if status == "supported" else [],
                          "reason": "Suporte configurado para este segmento e esta informação."})
        return self._result("assess", (segment, facts), {"assessments": items})

    def count(self, operation):
        return sum(name == operation for name, _ in self.calls)


class NeverProvider:
    def embed(self, texts):
        raise AssertionError("O cache deveria impedir novos embeddings.")

    def extract(self, segment):
        raise AssertionError("O cache deveria impedir uma nova extração.")

    def compare(self, pairs):
        raise AssertionError("O cache deveria impedir uma nova comparação.")

    def assess(self, segment, facts):
        raise AssertionError("O cache deveria impedir uma nova avaliação.")


class CatalogBuildingTests(unittest.TestCase):
    def test_extraction_triples_reach_provider_and_are_preserved_in_catalog_snapshot(self):
        dataset = dataset_for(["Ana dormiu."])
        information = [{
            "info": "Ana dormiu.", "tokens": [0, 1], "context": "O que Ana fez?",
            "syntagm": {"subject": "Ana", "relation": "dormiu", "object": ""},
        }]
        dataset[0]["segments"][0]["information"] = information
        original = copy.deepcopy(dataset)
        provider = ControlledProvider({"s0.seg0": [candidate("Ana dormiu.", indices=[0])]})
        catalog = CatalogStage(provider)(dataset)
        self.assertEqual(dataset, original)
        self.assertEqual(catalog["segments"][0]["information"], information)
        extracted_segment = next(arguments for operation, arguments in provider.calls if operation == "extract")
        self.assertEqual(extracted_segment["information"], information)
        self.assertEqual(catalog["candidates"][0]["information_indices"], [0])
        information[0]["syntagm"]["subject"] = "Outra pessoa"
        self.assertEqual(catalog["segments"][0]["information"], original[0]["segments"][0]["information"])

    def test_dataset_is_unchanged_and_grouping_does_not_erase_specific_detail(self):
        original_text = "Harry e Meghan apareceram em público com o filho pela primeira vez."
        dataset = dataset_for([original_text, APPEARANCE])
        first = dataset[0]["segments"][0]
        first["map"] = {"0": {"token": 0, "word": "Harry", "count": 99}}
        first["information"] = [{"info": APPEARANCE}, {"info": FIRST_APPEARANCE}]
        first["set_different_information"] = [[
            {"info": APPEARANCE, "context": "aparição", "avg_similarity": 0.9},
            {"info": FIRST_APPEARANCE, "context": "primeira", "avg_similarity": 0.9},
        ]]
        before = copy.deepcopy(dataset)
        provider = ControlledProvider({
            "s0.seg0": [candidate(APPEARANCE, original_text, [0]),
                        candidate(FIRST_APPEARANCE, original_text, [1])],
            "s1.seg0": [candidate(APPEARANCE)],
        })

        catalog = CatalogStage(provider)(dataset)

        self.assertEqual(dataset, before)
        self.assertEqual(len(catalog["facts"]), 2)
        self.assertEqual({fact["statement"] for fact in catalog["facts"]}, {APPEARANCE, FIRST_APPEARANCE})
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual([item["statement"] for item in report["preserved_information"]], [APPEARANCE])
        self.assertEqual([item["statement"] for item in report["lost_information"]], [FIRST_APPEARANCE])
        self.assertEqual(catalog["segments"][0]["information"], first["information"])
        self.assertEqual(catalog["segments"][0]["set_different_information"], first["set_different_information"])
        self.assertEqual(catalog["candidates"][0]["information_indices"], [0])

    def test_every_candidate_pair_and_every_segment_fact_are_assessed(self):
        dataset = dataset_for([[APPEARANCE, NAME], FIRST_APPEARANCE])
        provider = ControlledProvider()
        catalog = CatalogStage(provider, comparison_batch_size=2, assessment_batch_size=2)(dataset)

        pairs = [pair for operation, batch in provider.calls if operation == "compare" for pair in batch]
        self.assertEqual({frozenset((pair["left"]["id"], pair["right"]["id"])) for pair in pairs},
                         {frozenset(("c0", "c1")), frozenset(("c0", "c2")), frozenset(("c1", "c2"))})
        self.assertTrue(any(pair["left"]["segment_id"].startswith("s0.")
                            and pair["right"]["segment_id"].startswith("s0.") for pair in pairs))
        checked = [(segment["id"], fact["id"])
                   for operation, args in provider.calls if operation == "assess"
                   for segment, facts in [args] for fact in facts]
        expected = {(segment["id"], fact["id"]) for segment in catalog["segments"] for fact in catalog["facts"]}
        self.assertEqual(set(checked), expected)
        self.assertEqual(len(checked), len(expected))

    def test_complete_link_prevents_transitive_chain_from_becoming_one_fact(self):
        dataset = dataset_for(["Formulação A.", "Formulação B.", "Formulação C."])
        provider = ControlledProvider(relations={
            frozenset(("c0", "c1")): "equivalent",
            frozenset(("c1", "c2")): "equivalent",
            frozenset(("c0", "c2")): "different",
        })
        catalog = CatalogStage(provider)(dataset)
        self.assertEqual([fact["candidate_ids"] for fact in catalog["facts"]], [["c0", "c1"], ["c2"]])
        self.assertEqual(catalog["facts"][0]["statement"], "Formulação A.")

    def test_proposed_equivalence_cannot_import_a_detail_from_shared_content(self):
        general = "A ponte foi inaugurada."
        dated = "A inauguração da ponte ocorreu hoje."
        text = "A ponte foi inaugurada hoje."
        for verification_result in ("different", "uncertain"):
            with self.subTest(verification=verification_result):
                class Reviewer(ControlledProvider):
                    def compare(self, pairs):
                        isolated = pairs[0].get("mode") == "statement_only_verification"
                        if isolated:
                            for pair in pairs:
                                for side in ("left", "right"):
                                    self.assert_isolated(pair[side])
                        return CatalogResponse({"comparisons": [
                            {"pair_id": pair["pair_id"], "relation": verification_result if isolated else "equivalent",
                             "reason": "A formulação geral não contém o detalhe temporal."}
                            for pair in pairs
                        ]})

                    def assert_isolated(self, claim):
                        assert claim["content"] == claim["statement"]
                        assert claim["source_metadata"] == claim["segment_metadata"] == {}
                        assert "evidence" not in claim

                provider = Reviewer({"s0.seg0": [candidate(general, text), candidate(dated, text)]})
                catalog = CatalogStage(provider)(dataset_for([text]))
                self.assertEqual(len(catalog["facts"]), 2)
                self.assertEqual(catalog["comparisons"][0]["relation"], verification_result)

    def test_uncertain_equivalence_stays_separate_and_uncertain_support_is_reported(self):
        dataset = dataset_for([APPEARANCE, "Eles apareceram em público."])
        provider = ControlledProvider(
            relations={frozenset(("c0", "c1")): "uncertain"},
            supports={("s1.seg0", APPEARANCE): "uncertain"},
        )
        catalog = CatalogStage(provider)(dataset)
        self.assertEqual(len(catalog["facts"]), 2)
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual([item["statement"] for item in report["uncertain_information"]], [APPEARANCE])
        self.assertEqual(report["lost_information"], [])

    def test_rejected_extraction_is_not_counted_as_information_loss(self):
        dataset = dataset_for([APPEARANCE])
        provider = ControlledProvider(supports={("s0.seg0", APPEARANCE): "not_supported"})
        catalog = CatalogStage(provider)(dataset)
        self.assertEqual(catalog["facts"], [])
        self.assertEqual(len(catalog["rejected_facts"]), 1)
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual(report["lost_information"], [])
        self.assertEqual(report["summary"]["rejected"], 1)

    def test_original_content_can_add_information_missing_from_previous_extraction(self):
        dataset = dataset_for([NAME])
        dataset[0]["segments"][0].update(information=[], set_different_information=[])
        catalog = CatalogStage(ControlledProvider())(dataset)
        self.assertEqual(catalog["facts"][0]["statement"], NAME)
        self.assertEqual(catalog["candidates"][0]["information_indices"], [])

    def test_duplicate_within_segment_merges_provenance_and_literal_repetitions(self):
        text = "A ponte abriu. A ponte abriu."
        dataset = dataset_for([text])
        dataset[0]["segments"][0]["information"] = [{"info": "A ponte abriu."}, {"info": "A ponte abriu."}]
        provider = ControlledProvider({"s0.seg0": [candidate("A ponte abriu.", indices=[0]),
                                                  candidate("A ponte abriu.", indices=[1])]})
        catalog = CatalogStage(provider)(dataset)
        self.assertEqual(len(catalog["candidates"]), 1)
        self.assertEqual(catalog["candidates"][0]["information_indices"], [0, 1])
        self.assertEqual([(item["start"], item["end"]) for item in catalog["candidates"][0]["evidence"]],
                         [(0, 14), (15, 29)])
        self.assertEqual(provider.count("compare"), 0)

    def test_identical_text_across_segments_still_requires_equivalence_assessment(self):
        dataset = dataset_for(["Ele nasceu ontem.", "Ele nasceu ontem."])
        provider = ControlledProvider(relations={frozenset(("c0", "c1")): "uncertain"})
        catalog = CatalogStage(provider)(dataset)
        self.assertEqual(provider.count("compare"), 1)
        self.assertEqual(len(catalog["candidates"]), 2)
        self.assertEqual(len(catalog["facts"]), 2)

    def test_shuffled_response_items_do_not_change_catalog_order_or_digest(self):
        dataset = dataset_for([APPEARANCE, FIRST_APPEARANCE, NAME])
        normal = CatalogStage(ControlledProvider())(dataset)
        shuffled = ControlledProvider(mutations={
            "compare": lambda value: {"comparisons": list(reversed(value["comparisons"]))},
            "assess": lambda value: {"assessments": list(reversed(value["assessments"]))},
        })
        self.assertEqual(CatalogStage(shuffled)(dataset), normal)

    def test_missing_or_duplicate_comparison_cannot_be_interpreted_as_different(self):
        dataset = dataset_for([APPEARANCE, NAME])
        mutations = [
            lambda value: {"comparisons": []},
            lambda value: {"comparisons": value["comparisons"] * 2},
            lambda value: {"comparisons": [{**value["comparisons"][0], "pair_id": "unknown"}]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                provider = ControlledProvider(mutations={"compare": mutation})
                with self.assertRaises(CatalogError):
                    CatalogStage(provider, max_concurrency=1)(dataset)
                self.assertEqual(provider.count("assess"), 0)

    def test_missing_duplicate_or_unknown_assessment_is_rejected(self):
        dataset = dataset_for([APPEARANCE])
        mutations = [
            lambda value: {"assessments": []},
            lambda value: {"assessments": value["assessments"] * 2},
            lambda value: {"assessments": [{**value["assessments"][0], "fact_id": "unknown"}]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(CatalogError):
                    CatalogStage(ControlledProvider(mutations={"assess": mutation}))(dataset)

    def test_nonliteral_evidence_is_rejected_in_extraction_and_support(self):
        dataset = dataset_for([APPEARANCE])
        for operation, root in (("extract", "facts"), ("assess", "assessments")):
            with self.subTest(operation=operation):
                def invalid(value, root=root):
                    value[root][0]["evidence_quotes"] = ["Esta citação não está na transcrição."]
                    return value
                with self.assertRaisesRegex(CatalogError, "literalmente"):
                    CatalogStage(ControlledProvider(mutations={operation: invalid}))(dataset)

    def test_supported_assessment_requires_literal_evidence(self):
        def no_evidence(payload):
            payload["assessments"][0]["evidence_quotes"] = []
            return payload

        with self.assertRaises(CatalogError):
            CatalogStage(ControlledProvider(mutations={"assess": no_evidence}))(dataset_for([APPEARANCE]))

    def test_invalid_information_reference_or_boolean_index_is_rejected(self):
        dataset = dataset_for([APPEARANCE])
        dataset[0]["segments"][0]["information"] = [{"info": APPEARANCE}]
        for indices in ([1], [True], [0, 0], [-1]):
            with self.subTest(indices=indices):
                provider = ControlledProvider({"s0.seg0": [candidate(APPEARANCE, indices=indices)]})
                with self.assertRaises(CatalogError):
                    CatalogStage(provider)(dataset)

    def test_validates_entire_dataset_before_first_provider_call(self):
        invalid = [None, {}, "wrong"]
        for information in invalid:
            with self.subTest(information=information):
                dataset = dataset_for([APPEARANCE, NAME])
                dataset[1]["segments"][0]["information"] = information
                provider = ControlledProvider()
                with self.assertRaises(CatalogError):
                    CatalogStage(provider)(dataset)
                self.assertEqual(provider.calls, [])

    def test_empty_datasets_and_empty_segments_do_not_call_provider(self):
        for dataset in ([], [{"segments": []}], dataset_for(["", "   "])):
            with self.subTest(dataset=dataset):
                provider = ControlledProvider()
                catalog = CatalogStage(provider)(dataset)
                self.assertEqual(catalog["facts"], [])
                self.assertEqual(catalog["candidates"], [])
                self.assertEqual(catalog["comparisons"], [])
                self.assertEqual(provider.calls, [])
                self.assertEqual(simulate_removal(catalog)["summary"]["total_information"], 0)

    def test_disk_cache_replays_byte_identical_catalog_without_inference(self):
        dataset = dataset_for([APPEARANCE, NAME])
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonResponseCache(Path(directory))
            first = CatalogStage(ControlledProvider(), cache=cache, provenance={"revision": "v1"})(dataset)
            second = CatalogStage(NeverProvider(), cache=cache, provenance={"revision": "v1"})(dataset)
            self.assertEqual(json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False))
            self.assertTrue(list(Path(directory).glob("*.json")))

    def test_cache_requires_explicit_evaluator_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CatalogError, "provenance"):
                CatalogStage(ControlledProvider(), cache=JsonResponseCache(Path(directory)))

    def test_cache_can_resume_after_failure_and_provenance_change_invalidates_it(self):
        dataset = dataset_for([APPEARANCE, NAME])
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonResponseCache(Path(directory))
            interrupted = ControlledProvider(fail_once={"compare"})
            with self.assertRaisesRegex(CatalogError, "Falha simulada"):
                CatalogStage(interrupted, max_concurrency=1, cache=cache, provenance={"revision": "v1"})(dataset)
            resumed = ControlledProvider()
            catalog = CatalogStage(resumed, max_concurrency=1, cache=cache, provenance={"revision": "v1"})(dataset)
            self.assertEqual(resumed.count("extract"), 0)
            self.assertGreater(resumed.count("compare"), 0)
            self.assertEqual(len(catalog["facts"]), 2)
            changed = ControlledProvider()
            CatalogStage(changed, cache=cache, provenance={"revision": "v2"})(dataset)
            self.assertEqual(changed.count("extract"), 2)
            self.assertGreater(changed.count("compare"), 0)
            self.assertGreater(changed.count("assess"), 0)

    def test_invalid_response_is_not_persisted_in_cache(self):
        dataset = dataset_for([APPEARANCE])
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonResponseCache(Path(directory))
            bad = ControlledProvider(mutations={"extract": lambda value: {"facts": [{**value["facts"][0], "extra": True}]}})
            with self.assertRaises(CatalogError):
                CatalogStage(bad, cache=cache, provenance={"revision": "test-v1"})(dataset)
            self.assertEqual(list(Path(directory).glob("*.json")), [])
            good = ControlledProvider()
            CatalogStage(good, cache=cache, provenance={"revision": "test-v1"})(dataset)
            self.assertEqual(good.count("extract"), 1)


    def test_malformed_unicode_statement_fails_before_cache_or_other_calls(self):
        dataset = dataset_for([APPEARANCE])
        with tempfile.TemporaryDirectory() as directory:
            provider = ControlledProvider({"s0.seg0": [candidate("Fato " + chr(0xD800), APPEARANCE)]})
            with self.assertRaisesRegex(CatalogError, "UTF-8"):
                CatalogStage(provider, cache=JsonResponseCache(Path(directory)),
                             provenance={"revision": "unicode-test"})(dataset)
            self.assertEqual(provider.count("compare"), 0)
            self.assertEqual(provider.count("assess"), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_malformed_unicode_reason_is_rejected_in_both_semantic_operations(self):
        for operation, key in (("compare", "comparisons"), ("assess", "assessments")):
            with self.subTest(operation=operation):
                def invalid_reason(payload, key=key):
                    payload[key][0]["reason"] = "Motivo " + chr(0xD800)
                    return payload
                provider = ControlledProvider(mutations={operation: invalid_reason})
                with self.assertRaisesRegex(CatalogError, "UTF-8"):
                    CatalogStage(provider)(dataset_for([APPEARANCE, NAME]))

    def test_malformed_unicode_provider_metadata_is_not_cached(self):
        for metadata_field in ("model", "system_fingerprint"):
            with self.subTest(metadata_field=metadata_field), tempfile.TemporaryDirectory() as directory:
                class InvalidMetadataProvider(ControlledProvider):
                    def _result(self, operation, arguments, payload):
                        response = super()._result(operation, arguments, payload)
                        values = {"model": response.model, "system_fingerprint": response.system_fingerprint}
                        values[metadata_field] = "bad-" + chr(0xD800)
                        return CatalogResponse(response.payload, **values)

                with self.assertRaisesRegex(CatalogError, "UTF-8"):
                    CatalogStage(InvalidMetadataProvider(), cache=JsonResponseCache(Path(directory)),
                                 provenance={"revision": "unicode-test"})(dataset_for([APPEARANCE]))
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
