"""Resultados observáveis da canonicalização global, sem depender de API externa."""

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from determinism.experiments import run_catalog
from determinism.modules.semantic_catalog.adapters import CatalogConfig, ConfigurationError
from determinism.modules.semantic_catalog.adapters.json_artifacts import JsonResponseCache
from determinism.modules.semantic_catalog.application import CatalogStage
from determinism.modules.semantic_catalog.domain import CatalogError, catalog_digest, simulate_removal, validate_catalog
from determinism.modules.semantic_catalog.domain.clustering import complete_linkage, cosine_matrix, validate_embeddings
from determinism.modules.semantic_catalog.ports import CatalogResponse


def dataset_for(statements):
    return [{"name": f"Fonte {index}", "segments": [{"content": text}]} for index, text in enumerate(statements)]


def angle(degrees):
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


class Claims:
    def __init__(self, *, context="equivalent", isolated="equivalent"):
        self.context, self.isolated = context, isolated
        self.calls = []

    def extract(self, segment):
        self.calls.append(("extract", copy.deepcopy(segment)))
        return CatalogResponse({"facts": [{"statement": segment["content"],
                                          "evidence_quotes": [segment["content"]], "information_indices": []}]}, "claims-v1")

    def compare(self, pairs):
        self.calls.append(("compare", copy.deepcopy(pairs)))
        relation = self.isolated if pairs[0].get("mode") == "statement_only_verification" else self.context
        return CatalogResponse({"comparisons": [
            {"pair_id": pair["pair_id"], "relation": relation, "reason": "Decisão controlada para testar o contrato factual."}
            for pair in pairs]}, "claims-v1")

    def assess(self, segment, facts):
        self.calls.append(("assess", copy.deepcopy(facts)))
        return CatalogResponse({"assessments": [
            {"fact_id": fact["id"], "status": "supported" if any(
                origin["segment_id"] == segment["id"] for origin in fact["origin_contexts"]) else "not_supported",
             "evidence_quotes": [segment["content"]] if any(
                 origin["segment_id"] == segment["id"] for origin in fact["origin_contexts"]) else [],
             "reason": "Suporte presente apenas nos segmentos dos membros deste cenário."} for fact in facts]}, "claims-v1")


class Vectors:
    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    def embed(self, texts):
        self.calls.append(list(texts))
        return CatalogResponse({"embeddings": [self.mapping[text] for text in texts]}, "vectors-v1")


class CanonicalizationTests(unittest.TestCase):
    def build(self, statements, vectors, **kwargs):
        claims = Claims()
        embeddings = Vectors(dict(zip(statements, vectors)))
        catalog = CatalogStage(claims, embedding_provider=embeddings, **kwargs)(dataset_for(statements))
        return catalog, claims, embeddings

    def test_variants_from_all_sources_share_one_real_medoid_and_keep_evidence(self):
        statements = ["A ponte abriu ao público.", "A ponte foi aberta ao público.", "O público já pode usar a ponte."]
        catalog, _, _ = self.build(statements, [angle(0), angle(10), angle(20)])
        self.assertEqual(catalog["schema_version"], 2)
        fact = catalog["facts"][0]
        self.assertEqual(len(catalog["facts"]), 1)
        self.assertEqual(fact["candidate_ids"], ["c0", "c1", "c2"])
        self.assertEqual(fact["canonical_candidate_id"], "c1")
        self.assertEqual(fact["statement"], statements[1])
        self.assertEqual({candidate["evidence"][0]["segment_id"] for candidate in catalog["candidates"]},
                         {"s0.seg0", "s1.seg0", "s2.seg0"})
        self.assertEqual(simulate_removal(catalog, removed_source_ids=["s0"])["summary"]["preserved"], 1)
        self.assertEqual(simulate_removal(catalog, removed_source_ids=["s0", "s1", "s2"])["summary"]["lost"], 1)

    def test_strongest_numeric_link_wins_over_first_boolean_pair_and_chain_does_not_merge(self):
        catalog, claims, _ = self.build(["Variante A.", "Variante B.", "Variante C."],
                                        [angle(0), angle(40), angle(50)], similarity_threshold=0.7)
        self.assertEqual([fact["candidate_ids"] for fact in catalog["facts"]], [["c0"], ["c1", "c2"]])
        self.assertEqual(catalog["clustering"]["merges"][0]["candidate_ids"], ["c1", "c2"])
        skipped = next(pair for pair in catalog["comparisons"] if (pair["left_id"], pair["right_id"]) == ("c0", "c2"))
        self.assertEqual(skipped["relation"], "below_threshold")
        self.assertIsNone(skipped["context_decision"])
        self.assertIsNone(skipped["statement_decision"])
        self.assertEqual(sum(len(batch) for operation, batch in claims.calls if operation == "compare"), 4)

    def test_complete_linkage_blocks_low_pair_even_when_average_passes(self):
        catalog, _, _ = self.build(["Variante A.", "Variante B.", "Variante C."],
                                   [angle(0), angle(10), angle(50)], similarity_threshold=0.7)
        self.assertEqual([fact["candidate_ids"] for fact in catalog["facts"]], [["c0", "c1"], ["c2"]])
        self.assertLess(catalog["comparisons"][1]["similarity"], 0.7)

    def test_threshold_is_inclusive_and_below_threshold_is_not_a_factual_judgment(self):
        for threshold, expected in ((0.8, 1), (0.8000000001, 2)):
            with self.subTest(threshold=threshold):
                catalog, claims, _ = self.build(["Primeira formulação.", "Segunda formulação."],
                                                [[1, 0], [0.8, 0.6]], similarity_threshold=threshold)
                self.assertEqual(len(catalog["facts"]), expected)
                self.assertEqual(catalog["comparisons"][0]["similarity"], 0.8)
                if expected == 2:
                    self.assertEqual(catalog["comparisons"][0]["relation"], "below_threshold")
                    self.assertFalse(any(operation == "compare" for operation, _ in claims.calls))

    def test_equal_scores_and_medoid_use_snapshot_indices_including_c10(self):
        ids = [f"c{index}" for index in range(12)]
        pairs = [{"left_id": left, "right_id": right, "similarity": 1.0,
                  "relation": "equivalent" if {left, right} in ({"c2", "c3"}, {"c2", "c10"}) else "different"}
                 for first, left in enumerate(ids) for right in ids[first + 1:]]
        groups, merges, representatives = complete_linkage(ids, pairs, 0.85)
        self.assertEqual(merges[0]["candidate_ids"], ["c2", "c3"])
        self.assertIn(["c10"], groups)
        self.assertEqual(representatives[groups.index(["c2", "c3"])], "c2")

    def test_high_similarity_does_not_override_negation_date_quantity_attribution_or_detail(self):
        statements = ["A ponte abriu.", "A ponte não abriu.", "A ponte abriu em 2019.",
                      "A ponte abriu em 2020.", "A ponte tem duas faixas.", "A ponte tem três faixas.",
                      "Ana disse que a ponte abriu.", "A ponte pode abrir."]
        # O provedor contextual propõe uma fusão incorreta: a confirmação isolada a rejeita.
        claims = Claims(isolated="different")
        catalog = CatalogStage(claims, embedding_provider=Vectors({text: [1, 0] for text in statements}))(dataset_for(statements))
        self.assertEqual(len(catalog["facts"]), len(statements))
        self.assertEqual({fact["statement"] for fact in catalog["facts"]}, set(statements))
        for pair in catalog["comparisons"]:
            self.assertEqual(pair["similarity"], 1.0)
            self.assertEqual(pair["context_decision"]["relation"], "equivalent")
            self.assertEqual(pair["statement_decision"]["relation"], "different")
            self.assertEqual(pair["relation"], "different")

    def test_contextual_uncertainty_cannot_be_promoted_by_isolated_comparison(self):
        statements = ["Ele nasceu ontem.", "Ele nasceu ontem."]
        claims = Claims(context="uncertain")
        vectors = Vectors({statements[0]: [1, 0]})
        catalog = CatalogStage(claims, embedding_provider=vectors)(dataset_for(statements))
        self.assertEqual(len(catalog["facts"]), 2)
        self.assertEqual(vectors.calls, [[statements[0]]])
        pair = catalog["comparisons"][0]
        self.assertEqual(pair["context_decision"]["relation"], "uncertain")
        self.assertIsNone(pair["statement_decision"])
        self.assertEqual(sum(operation == "compare" for operation, _ in claims.calls), 1)

    def test_conflicting_locations_reach_contextual_reviewer_and_block_same_named_claim(self):
        class ContextualClaims(Claims):
            def compare(self, pairs):
                for pair in pairs:
                    assert pair["left"]["source_metadata"]["city"] == "Vale Claro"
                    assert pair["right"]["source_metadata"]["city"] == "Serra Azul"
                    assert pair["left"]["content"] == pair["right"]["content"]
                return super().compare(pairs)

        dataset = dataset_for(["A Ponte Aurora tem duas faixas."] * 2)
        dataset[0]["city"], dataset[1]["city"] = "Vale Claro", "Serra Azul"
        claims = ContextualClaims(context="uncertain")
        catalog = CatalogStage(claims, embedding_provider=Vectors({dataset[0]["segments"][0]["content"]: [1, 0]}))(dataset)
        self.assertEqual(len(catalog["facts"]), 2)
        self.assertEqual(catalog["comparisons"][0]["similarity"], 1.0)
        self.assertEqual(catalog["comparisons"][0]["relation"], "uncertain")
        self.assertIsNone(catalog["comparisons"][0]["statement_decision"])

    def test_validator_rejects_resealed_hierarchy_representative_and_decision_tampering(self):
        catalog, _, _ = self.build(["Variante A.", "Variante B.", "Variante C."], [angle(0), angle(10), angle(20)])
        mutations = [
            lambda value: value["clustering"]["merges"][0].update(similarity=0.86),
            lambda value: value["clustering"].update(merges=[]),
            lambda value: value["facts"][0].update(canonical_candidate_id="c0", statement="Variante A."),
            lambda value: value["comparisons"][0].update(statement_decision=None),
            lambda value: value["comparisons"][0].update(context_decision={"relation": "uncertain", "reason": "Ambíguo."}),
            lambda value: value["clustering"].update(embedding_inputs=[]),
            lambda value: value["clustering"].update(similarity_threshold=True),
            lambda value: value["provenance"].update(configuration={"similarity_threshold": 0.1}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(catalog)
                mutation(altered)
                altered["catalog_id"] = catalog_digest(altered)
                with self.assertRaises(CatalogError):
                    validate_catalog(altered)

    def test_invalid_embeddings_are_rejected_before_comparison_and_not_cached(self):
        for vector in ([], [0, 0], [True, 1], [float("nan"), 1], [float("inf"), 1], [1, "x"]):
            with self.subTest(vector=vector), tempfile.TemporaryDirectory() as directory:
                claims = Claims()
                cache = JsonResponseCache(Path(directory))
                with self.assertRaises(CatalogError):
                    CatalogStage(claims, embedding_provider=Vectors({"Texto.": vector}), cache=cache,
                                 provenance={"revision": "vector-fixture-v1"})(dataset_for(["Texto."]))
                self.assertFalse(any(operation in ("compare", "assess") for operation, _ in claims.calls))
                self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)  # Só extração válida.
        with self.assertRaises(CatalogError):
            validate_embeddings({"embeddings": [[1, 0], [1, 0, 0]]}, 2)

    def test_dimension_mismatch_across_batches_and_explicit_dimensions_fail(self):
        claims = Claims()
        embeddings = Vectors({"A.": [1, 0], "B.": [1, 0, 0]})
        for options in ({"embedding_batch_size": 1}, {"embedding_dimensions": 3}):
            with self.subTest(options=options), self.assertRaisesRegex(CatalogError, "[Dd]imens"):
                CatalogStage(claims, embedding_provider=embeddings, **options)(dataset_for(["A.", "B."]))

    def test_real_cosines_are_finite_for_large_or_small_valid_vectors(self):
        for scale in (1e300, 1e-300):
            vectors = validate_embeddings({"embeddings": [[scale, scale], [scale, -scale]]}, 2)
            self.assertEqual(cosine_matrix(vectors)[0][1], 0.0)

    def test_cache_warm_replay_has_no_provider_calls_and_revision_changes_invalidate(self):
        statements = ["Variante A.", "Variante B."]
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonResponseCache(Path(directory))
            provenance = {"claims": "v1", "embeddings": "v1"}
            claims, vectors = Claims(), Vectors(dict.fromkeys(statements, [1, 0]))
            first = CatalogStage(claims, embedding_provider=vectors, cache=cache, provenance=provenance)(dataset_for(statements))
            with patch.object(claims, "extract", side_effect=AssertionError("extração nova")), patch.object(
                    claims, "compare", side_effect=AssertionError("comparação nova")), patch.object(
                    claims, "assess", side_effect=AssertionError("avaliação nova")), patch.object(
                    vectors, "embed", side_effect=AssertionError("embedding novo")):
                self.assertEqual(CatalogStage(claims, embedding_provider=vectors, cache=cache,
                                             provenance=provenance)(dataset_for(statements)), first)
            changed = Vectors(dict.fromkeys(statements, [1, 0]))
            CatalogStage(Claims(), embedding_provider=changed, cache=cache,
                         provenance={**provenance, "embeddings": "v2"})(dataset_for(statements))
            self.assertEqual(len(changed.calls), 1)
            changed_threshold = Vectors(dict.fromkeys(statements, [1, 0]))
            CatalogStage(Claims(), embedding_provider=changed_threshold, cache=cache, provenance=provenance,
                         similarity_threshold=0.9)(dataset_for(statements))
            self.assertEqual(len(changed_threshold.calls), 1)

    def test_configuration_and_injected_embedding_cache_identity_are_explicit(self):
        for changes in ({"similarity_threshold": True}, {"similarity_threshold": float("nan")},
                        {"similarity_threshold": 1.1}, {"embedding_dimensions": 0},
                        {"embedding_batch_size": True}, {"embedding_base_url": "https://u:p@host/v1"},
                        {"embedding_provider": "unknown"}, {"embedding_cache_revision": "bad\nrevision"}):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                CatalogConfig(**changes)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "output.json"
            source.write_text('[{"segments":[{"content":"Texto."}]}]', encoding="utf-8")
            claims = Claims()
            with self.assertRaisesRegex(CatalogError, "embedding_provider_identity"):
                run_catalog(source, output, provider=claims, provider_identity="claims-v1",
                            embedding_provider=Vectors({"Texto.": [1, 0]}), cache_dir=Path(directory) / "cache")
            self.assertEqual(claims.calls, [])

    def test_missing_embedding_port_has_an_explicit_v2_migration_error(self):
        with self.assertRaisesRegex(CatalogError, "catálogo v2 exige embedding_provider"):
            CatalogStage(Claims())(dataset_for(["Texto."]))

    def test_embedding_failure_preserves_existing_artifact_and_original_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "output.json"
            original = b'[{"segments":[{"content":"Texto."}]}]'
            source.write_bytes(original)
            output.write_bytes(b"previous catalog")
            with self.assertRaises(CatalogError):
                run_catalog(source, output, provider=Claims(), embedding_provider=Vectors({"Texto.": [0, 0]}))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), b"previous catalog")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
