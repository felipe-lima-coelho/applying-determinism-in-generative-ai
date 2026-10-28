"""Requisitos ancorados, integridade v2 e preservação parcial de afirmações."""

import copy
import unittest

from determinism.modules.semantic_catalog.domain import (
    CatalogError, RemovalSimulator, candidate_identifier, catalog_digest,
    compile_requirements, dataset_digest, decision_digest, normalize_statement,
    reduce_requirement_status, simulate_removal, statement_tokens, validate_catalog,
)


def seal_requirements(catalog):
    catalog["decision_id"] = decision_digest(catalog)
    catalog["catalog_id"] = catalog_digest(catalog)
    return catalog


def sample_requirement_catalog():
    statement = "A ponte abriu na terça-feira."
    dataset = [
        {"name": "Band", "segments": [{"content": statement, "information": [{"info": statement}]}]},
        {"name": "Record", "segments": [{"content": "A ponte abriu.", "information": []}]},
    ]
    segments = [
        {"id": f"s{i}.seg0", "source_id": f"s{i}", "source_index": i, "segment_index": 0,
         "source_name": source["name"], "content": source["segments"][0]["content"],
         "information": copy.deepcopy(source["segments"][0]["information"])}
        for i, source in enumerate(dataset)
    ]

    def evidence(segment):
        return [{"segment_id": segment["id"], "quote": segment["content"], "start": 0, "end": len(segment["content"])}]

    requirements = compile_requirements({"requirements": [
        {"kind": "predicate", "token_ids": [0, 1, 2]},
        {"kind": "time", "token_ids": [3, 4, 5]},
    ]}, statement)
    identifier = candidate_identifier(segments[0]["id"], statement, statement)
    candidate = {"id": identifier, "segment_id": segments[0]["id"], "statement": statement,
                 "evidence": evidence(segments[0]), "information_indices": [0], "requirements": requirements}
    assessments = []
    for index, segment in enumerate(segments):
        assessments.append({
            "segment_ids": [segment["id"]], "status": "supported" if index == 0 else "not_supported",
            "evidence": evidence(segment), "reason": "Redução dos requisitos.", "reference_status": "compatible",
            "requirement_results": [
                {"requirement_id": requirements[0]["id"], "status": "entailed", "evidence": evidence(segment)},
                {"requirement_id": requirements[1]["id"], "status": "entailed" if index == 0 else "absent",
                 "evidence": evidence(segment) if index == 0 else []},
            ],
        })
    catalog = {
        "schema_version": 2, "dataset_fingerprint": dataset_digest(dataset),
        "policy": {"support": "independent_segments", "method": "requirements"}, "provenance": {"model": "fixture"},
        "segments": segments, "candidates": [candidate], "comparisons": [],
        "facts": [{"id": "f_full", "statement": statement, "candidate_ids": [identifier], "assessments": assessments}],
        "rejected_facts": [],
    }
    return dataset, seal_requirements(catalog)


class RequirementCompilationTests(unittest.TestCase):
    def test_unicode_tokens_have_literal_offsets_and_punctuation_is_not_a_token(self):
        statement = '“Meghan”: 500 metros, terça-feira.'
        tokens = statement_tokens(statement)
        self.assertEqual([item["text"] for item in tokens], ["Meghan", "500", "metros", "terça", "feira"])
        for item in tokens:
            self.assertEqual(statement[item["start"]:item["end"]], item["text"])

    def test_whitespace_unicode_normalization_preserves_semantic_qualifiers(self):
        self.assertEqual(normalize_statement("  A  aparic\u0327a\u0303o\nrápida. "), "A aparição rápida.")
        self.assertNotEqual(normalize_statement("A aparição rápida."), normalize_statement("A aparição."))

    def test_compiler_preserves_contiguous_punctuation_and_marks_discontinuous_spans(self):
        text = "João, Pedro e Ana chegaram ontem."
        requirements = compile_requirements({"requirements": [
            {"kind": "participant", "token_ids": [0, 3]},
            {"kind": "predicate", "token_ids": [5, 4, 3, 2, 1, 0]},
        ]}, text)
        by_kind = {item["kind"]: item for item in requirements}
        self.assertEqual(by_kind["participant"]["text"], "João … Ana")
        self.assertEqual(by_kind["predicate"]["text"], "João, Pedro e Ana chegaram ontem")

    def test_order_of_model_groups_and_tokens_does_not_change_compilation(self):
        a = {"kind": "predicate", "token_ids": [0, 1, 2]}
        b = {"kind": "time", "token_ids": [3]}
        first = compile_requirements({"requirements": [a, b]}, "A ponte abriu ontem.")
        second = compile_requirements({"requirements": [b, {**a, "token_ids": [2, 0, 1]}]}, "A ponte abriu ontem.")
        self.assertEqual(first, second)
        self.assertTrue(all(item["id"].startswith("r_") for item in first))

    def test_omitted_qualifier_cannot_be_silently_accepted(self):
        with self.assertRaisesRegex(CatalogError, "omitidas"):
            compile_requirements({"requirements": [{"kind": "predicate", "token_ids": [0, 1, 2]}]}, "A ponte abriu ontem.")

    def test_bad_identifiers_duplicates_unknown_kinds_and_free_text_are_rejected(self):
        invalid = [
            {"requirements": []},
            {"requirements": [{"kind": "predicate", "token_ids": [True, 1, 2]}]},
            {"requirements": [{"kind": "predicate", "token_ids": [-1, 0, 1, 2]}]},
            {"requirements": [{"kind": "predicate", "token_ids": [0, 1, 2, 3]}]},
            {"requirements": [{"kind": "predicate", "token_ids": [0, 0, 1, 2]}]},
            {"requirements": [{"kind": "invented", "token_ids": [0, 1, 2]}]},
            {"requirements": [{"kind": "predicate", "token_ids": [0, 1, 2], "text": "Another fact"}]},
            {"requirements": [{"kind": "predicate", "token_ids": [0, 1, 2]}] * 2},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(CatalogError):
                compile_requirements(payload, "A ponte abriu.")

    def test_token_keyed_contract_compiles_to_same_persisted_requirements(self):
        text = "A ponte abriu ontem."
        old = {"requirements": [
            {"kind": "predicate", "token_ids": [0, 1, 2]},
            {"kind": "time", "token_ids": [3]},
        ]}
        keyed = {"requirements_by_token": {
            "t3": [{"kind": "time", "group": 0}],
            "t1": [{"kind": "predicate", "group": 3}],
            "t2": [{"kind": "predicate", "group": 3}],
            "t0": [{"kind": "predicate", "group": 3}],
        }}
        self.assertEqual(compile_requirements(keyed, text), compile_requirements(old, text))

    def test_token_assignment_order_and_group_numbers_are_not_semantic_identity(self):
        text = "A ponte abriu."
        first = {"requirements_by_token": {
            "t0": [{"kind": "participant", "group": 0}, {"kind": "predicate", "group": 1}],
            "t1": [{"kind": "participant", "group": 0}, {"kind": "predicate", "group": 1}],
            "t2": [{"kind": "predicate", "group": 1}],
        }}
        second = {"requirements_by_token": {
            key: [{**item, "group": 2 if item["group"] == 0 else 0} for item in reversed(items)]
            for key, items in reversed(list(first["requirements_by_token"].items()))
        }}
        self.assertEqual(compile_requirements(first, text), compile_requirements(second, text))

    def test_token_keyed_contract_rejects_missing_unknown_or_empty_token(self):
        valid = {f"t{i}": [{"kind": "predicate", "group": 0}] for i in range(3)}
        for mutation in (lambda mapping: mapping.pop("t1"),
                         lambda mapping: mapping.update(t3=[{"kind": "predicate", "group": 0}]),
                         lambda mapping: mapping.update(t1=[])):
            bad = copy.deepcopy(valid)
            mutation(bad)
            with self.subTest(mapping=bad), self.assertRaises(CatalogError):
                compile_requirements({"requirements_by_token": bad}, "A ponte abriu.")

    def test_token_keyed_contract_validates_group_and_kind_strictly(self):
        invalid_assignments = [
            {"kind": "predicate", "group": True},
            {"kind": "predicate", "group": -1},
            {"kind": "predicate", "group": 3},
            {"kind": "predicate", "group": "0"},
            {"kind": "unknown", "group": 0},
            {"kind": ["predicate"], "group": 0},
            {"kind": "predicate", "group": 0, "text": "free text"},
        ]
        for assignment in invalid_assignments:
            values = {f"t{i}": [{"kind": "predicate", "group": 0}] for i in range(3)}
            values["t1"] = [assignment]
            with self.subTest(assignment=assignment), self.assertRaises(CatalogError):
                compile_requirements({"requirements_by_token": values}, "A ponte abriu.")

    def test_same_group_number_in_different_kinds_keeps_separate_requirements(self):
        values = {
            "t0": [{"kind": "participant", "group": 0}, {"kind": "predicate", "group": 0}],
            "t1": [{"kind": "participant", "group": 0}, {"kind": "predicate", "group": 0}],
            "t2": [{"kind": "predicate", "group": 0}],
        }
        compiled = compile_requirements({"requirements_by_token": values}, "A ponte abriu.")
        by_kind = {item["kind"]: item for item in compiled}
        self.assertEqual(len(compiled), 2)
        self.assertEqual(by_kind["participant"]["token_ids"], [0, 1])
        self.assertEqual(by_kind["predicate"]["token_ids"], [0, 1, 2])

    def test_duplicate_kind_group_assignment_for_same_token_is_normalized(self):
        values = {f"t{i}": [{"kind": "predicate", "group": 0}] for i in range(3)}
        expected = compile_requirements({"requirements_by_token": values}, "A ponte abriu.")
        values["t1"] *= 2
        self.assertEqual(compile_requirements({"requirements_by_token": values}, "A ponte abriu."), expected)

    def test_different_group_numbers_for_same_requirement_are_normalized(self):
        values = {f"t{i}": [{"kind": "predicate", "group": 0}, {"kind": "predicate", "group": 1}]
                  for i in range(3)}
        expected = compile_requirements({"requirements": [{"kind": "predicate", "token_ids": [0, 1, 2]}]}, "A ponte abriu.")
        self.assertEqual(compile_requirements({"requirements_by_token": values}, "A ponte abriu."), expected)

    def test_invalid_duplicate_assignment_is_not_hidden_by_normalization(self):
        values = {f"t{i}": [{"kind": "predicate", "group": 0}] for i in range(3)}
        values["t1"].append({"kind": "predicate", "group": False})
        with self.assertRaises(CatalogError):
            compile_requirements({"requirements_by_token": values}, "A ponte abriu.")

    def test_reduction_prioritizes_missing_or_contradicted_over_ambiguity(self):
        self.assertEqual(reduce_requirement_status("ambiguous", ["entailed", "absent"]), "not_supported")
        self.assertEqual(reduce_requirement_status("compatible", ["ambiguous", "contradicted"]), "not_supported")
        self.assertEqual(reduce_requirement_status("different", ["entailed"]), "not_supported")
        self.assertEqual(reduce_requirement_status("ambiguous", ["entailed"]), "uncertain")
        self.assertEqual(reduce_requirement_status("compatible", ["entailed", "ambiguous"]), "uncertain")
        self.assertEqual(reduce_requirement_status("compatible", ["entailed"]), "supported")

    def test_reduction_rejects_missing_or_invalid_states(self):
        for reference, states in (("invented", ["entailed"]), ("compatible", []), ("compatible", "entailed"),
                                  ("compatible", [True]), ("compatible", ["supported"]), ("compatible", None)):
            with self.subTest(reference=reference, states=states), self.assertRaises(CatalogError):
                reduce_requirement_status(reference, states)


class RequirementCatalogContractTests(unittest.TestCase):
    def test_fixture_is_valid_and_frozen_simulation_keeps_partial_information(self):
        dataset, catalog = sample_requirement_catalog()
        self.assertIs(validate_catalog(catalog, dataset), catalog)
        report = simulate_removal(catalog, ["s0.seg0"], dataset=dataset)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["decision_id"], catalog["decision_id"])
        self.assertEqual(report["summary"]["lost"], 1)
        self.assertEqual(report["summary"]["total_information"], 1)
        components = {item["kind"]: item for item in report["lost_information"][0]["requirement_coverage"]}
        self.assertEqual(components["predicate"]["classification"], "preserved")
        self.assertEqual(components["predicate"]["remaining_support_segment_ids"], ["s1.seg0"])
        self.assertEqual(components["time"]["classification"], "lost")
        self.assertEqual(components["time"]["removed_support_segment_ids"], ["s0.seg0"])

    def test_ambiguous_reference_makes_partial_component_uncertain(self):
        _, catalog = sample_requirement_catalog()
        catalog["facts"][0]["assessments"][1]["reference_status"] = "ambiguous"
        seal_requirements(catalog)
        report = simulate_removal(catalog, ["s0.seg0"])
        components = {item["kind"]: item for item in report["lost_information"][0]["requirement_coverage"]}
        self.assertEqual(components["predicate"]["classification"], "uncertain")
        self.assertEqual(components["predicate"]["possible_segment_ids"], ["s1.seg0"])

    def test_immutable_compilation_does_not_share_requirement_records(self):
        _, catalog = sample_requirement_catalog()
        simulator = RemovalSimulator(catalog)
        expected = simulator.simulate(["s0.seg0"])
        catalog["candidates"][0]["requirements"][0]["text"] = "Modified"
        changed = simulator.simulate(["s0.seg0"])
        changed["lost_information"][0]["requirement_coverage"][0]["remaining_support_segment_ids"].clear()
        self.assertEqual(simulator.simulate(["s0.seg0"]), expected)

    def test_decision_digest_ignores_audit_and_response_order(self):
        _, original = sample_requirement_catalog()
        changed = copy.deepcopy(original)
        changed["provenance"] = {"model": "different-response-metadata"}
        changed["facts"][0]["assessments"].reverse()
        for assessment in changed["facts"][0]["assessments"]:
            assessment["reason"] = "Outra redação."
            assessment["requirement_results"].reverse()
        self.assertEqual(decision_digest(original), decision_digest(changed))
        self.assertNotEqual(catalog_digest(original), catalog_digest(changed))
        seal_requirements(changed)
        validate_catalog(changed)

    def test_decision_digest_changes_with_requirement_decision_and_rejects_stale_hash(self):
        _, catalog = sample_requirement_catalog()
        before = catalog["decision_id"]
        catalog["facts"][0]["assessments"][1]["reference_status"] = "ambiguous"
        self.assertNotEqual(before, decision_digest(catalog))
        catalog["catalog_id"] = catalog_digest(catalog)
        with self.assertRaisesRegex(CatalogError, "decision_id"):
            validate_catalog(catalog)

    def test_status_cannot_override_missing_requirement(self):
        _, catalog = sample_requirement_catalog()
        catalog["facts"][0]["assessments"][1]["status"] = "supported"
        seal_requirements(catalog)
        with self.assertRaisesRegex(CatalogError, "redução"):
            validate_catalog(catalog)

    def test_unknown_missing_and_duplicate_requirement_results_are_rejected(self):
        for mutation in (lambda items: items.pop(), lambda items: items.append(copy.deepcopy(items[0])),
                         lambda items: items[0].update(requirement_id="unknown")):
            _, catalog = sample_requirement_catalog()
            mutation(catalog["facts"][0]["assessments"][0]["requirement_results"])
            seal_requirements(catalog)
            with self.assertRaises(CatalogError):
                validate_catalog(catalog)

    def test_missing_evidence_for_entailed_ambiguous_or_contradicted_is_rejected(self):
        for status in ("entailed", "ambiguous", "contradicted"):
            _, catalog = sample_requirement_catalog()
            result = catalog["facts"][0]["assessments"][0]["requirement_results"][0]
            result.update(status=status, evidence=[])
            seal_requirements(catalog)
            with self.subTest(status=status), self.assertRaisesRegex(CatalogError, "evidência literal"):
                validate_catalog(catalog)

    def test_absent_requirement_cannot_have_evidence_and_aggregate_must_be_union(self):
        _, catalog = sample_requirement_catalog()
        assessment = catalog["facts"][0]["assessments"][1]
        assessment["requirement_results"][1]["evidence"] = copy.deepcopy(assessment["evidence"])
        seal_requirements(catalog)
        with self.assertRaises(CatalogError):
            validate_catalog(catalog)
        _, catalog = sample_requirement_catalog()
        catalog["facts"][0]["assessments"][1]["evidence"] = []
        seal_requirements(catalog)
        with self.assertRaisesRegex(CatalogError, "união"):
            validate_catalog(catalog)

    def test_fixed_inventory_rejects_rewritten_claim_or_missing_occurrence(self):
        _, catalog = sample_requirement_catalog()
        catalog["segments"][0]["information"][0]["info"] = "A ponte abriu."
        seal_requirements(catalog)
        with self.assertRaisesRegex(CatalogError, "sem reescrita"):
            validate_catalog(catalog)
        _, catalog = sample_requirement_catalog()
        catalog["segments"][0]["information"].append({"info": "Nova informação."})
        seal_requirements(catalog)
        with self.assertRaisesRegex(CatalogError, "ocorrências"):
            validate_catalog(catalog)

    def test_compiled_text_and_ids_cannot_be_forged(self):
        for key, value in (("text", "A ponte abriu hoje"), ("id", "r_bad")):
            _, catalog = sample_requirement_catalog()
            catalog["candidates"][0]["requirements"][0][key] = value
            seal_requirements(catalog)
            with self.subTest(key=key), self.assertRaisesRegex(CatalogError, "compilação"):
                validate_catalog(catalog)


if __name__ == "__main__":
    unittest.main()
