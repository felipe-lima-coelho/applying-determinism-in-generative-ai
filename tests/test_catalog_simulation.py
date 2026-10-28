import copy
import itertools
import math
import unittest
from unittest.mock import patch

from determinism.modules.semantic_catalog.domain import (
    CatalogError, RemovalSimulator, catalog_digest, dataset_digest, simulate_removal, validate_catalog,
)


def seal(catalog):
    catalog["catalog_id"] = catalog_digest(catalog)
    return catalog


def sample_catalog(specifications=None):
    dataset = [
        {"name": "Band", "segments": [{"content": "A ponte foi inaugurada na segunda-feira."}, {"content": "A ponte tem duas faixas."}]},
        {"name": "Record", "segments": [{"content": "A ponte foi inaugurada. Ela tem duas faixas."}]},
    ]
    specifications = specifications if specifications is not None else [
        ("A ponte foi inaugurada.", ["supported", "not_supported", "supported"]),
    ]
    segments = [
        {"id": f"s{si}.seg{gi}", "source_id": f"s{si}", "source_index": si, "segment_index": gi,
         "source_name": source["name"], "content": segment["content"], "information": []}
        for si, source in enumerate(dataset) for gi, segment in enumerate(source["segments"])
    ]

    def evidence(segment):
        return [{"segment_id": segment["id"], "quote": segment["content"], "start": 0, "end": len(segment["content"])}]

    candidates, facts, rejected = [], [], []
    for index, (statement, statuses) in enumerate(specifications):
        origin = next((position for position, status in enumerate(statuses) if status != "not_supported"), 0)
        candidates.append({"id": f"c{index}", "segment_id": segments[origin]["id"], "statement": statement,
                           "evidence": evidence(segments[origin]), "information_indices": []})
        fact = {"id": f"f{index}", "statement": statement, "candidate_ids": [f"c{index}"], "assessments": [
            {"segment_ids": [segment["id"]], "status": status,
             "evidence": evidence(segment) if status == "supported" else [], "reason": "Decisão de teste."}
            for segment, status in zip(segments, statuses)
        ]}
        (facts if any(status != "not_supported" for status in statuses) else rejected).append(fact)
    comparisons = [
        {"pair_id": f"p{index}", "left_id": left["id"], "right_id": right["id"], "relation": "different", "reason": "Conteúdos distintos."}
        for index, (left, right) in enumerate(itertools.combinations(candidates, 2))
    ]
    return dataset, seal({
        "schema_version": 1, "dataset_fingerprint": dataset_digest(dataset),
        "policy": {"support": "independent_segments"}, "provenance": {"model": "fixture"},
        "segments": segments, "candidates": candidates, "comparisons": comparisons,
        "facts": facts, "rejected_facts": rejected,
    })


class RemovalSimulationTests(unittest.TestCase):
    def test_joint_removal_cannot_be_computed_by_adding_individual_losses(self):
        _, catalog = sample_catalog()
        self.assertEqual(simulate_removal(catalog, ["s0.seg0"])["summary"]["lost"], 0)
        self.assertEqual(simulate_removal(catalog, ["s1.seg0"])["summary"]["lost"], 0)
        joint = simulate_removal(catalog, ["s0.seg0", "s1.seg0"])
        self.assertEqual(joint["summary"]["lost"], 1)
        self.assertEqual(joint["lost_information"][0]["fact_id"], "f0")

    def test_general_information_survives_but_specific_detail_is_lost(self):
        _, catalog = sample_catalog([
            ("A ponte foi inaugurada.", ["supported", "not_supported", "supported"]),
            ("A inauguração da ponte ocorreu na segunda-feira.", ["supported", "not_supported", "not_supported"]),
        ])
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual([item["fact_id"] for item in report["preserved_information"]], ["f0"])
        self.assertEqual([item["fact_id"] for item in report["lost_information"]], ["f1"])

    def test_uncertain_remaining_support_does_not_become_preserved_or_lost(self):
        _, catalog = sample_catalog([("A ponte foi inaugurada.", ["supported", "not_supported", "uncertain"])])
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual(report["summary"]["uncertain"], 1)
        entry = report["uncertain_information"][0]
        self.assertEqual(entry["baseline_status"], "supported")
        self.assertEqual(entry["remaining_status"], "uncertain")
        self.assertEqual(entry["possible_supports"][0]["segment_ids"], ["s1.seg0"])

    def test_uncertain_baseline_is_explicit_even_when_all_possible_support_is_removed(self):
        _, catalog = sample_catalog([("A ponte foi inaugurada ontem.", ["uncertain", "not_supported", "not_supported"])])
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual(report["summary"]["lost"], 0)
        self.assertEqual(report["uncertain_information"][0]["baseline_status"], "uncertain")
        self.assertEqual(report["uncertain_information"][0]["remaining_status"], "not_supported")

    def test_definite_support_takes_priority_over_uncertain_alternative(self):
        _, catalog = sample_catalog([("A ponte foi inaugurada.", ["supported", "uncertain", "supported"])])
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual(report["summary"]["preserved"], 1)
        entry = report["preserved_information"][0]
        self.assertEqual(entry["remaining_source_ids"], ["s1"])
        self.assertEqual(len(entry["possible_supports"]), 1)

    def test_same_source_other_segment_also_preserves_information(self):
        _, catalog = sample_catalog([("A ponte tem duas faixas.", ["supported", "supported", "not_supported"])])
        report = simulate_removal(catalog, ["s0.seg0"])
        self.assertEqual(report["summary"]["preserved"], 1)
        self.assertEqual(report["preserved_information"][0]["remaining_source_ids"], ["s0"])

    def test_multi_segment_dependency_requires_all_of_its_evidence(self):
        _, catalog = sample_catalog([("A ponte tem duas faixas.", ["not_supported"] * 3)])
        fact = catalog["rejected_facts"].pop()
        fact["assessments"].append({
            "segment_ids": ["s0.seg0", "s0.seg1"], "status": "supported", "reason": "Identificação e descrição conjuntas.",
            "evidence": [
                {"segment_id": segment["id"], "quote": segment["content"], "start": 0, "end": len(segment["content"])}
                for segment in catalog["segments"][:2]
            ],
        })
        catalog["facts"].append(fact)
        seal(catalog)
        self.assertEqual(simulate_removal(catalog)["summary"]["preserved"], 1)
        for segment_id in ("s0.seg0", "s0.seg1"):
            report = simulate_removal(catalog, [segment_id])
            self.assertEqual(report["summary"]["lost"], 1)
            self.assertEqual(report["lost_information"][0]["removed_supports"][0]["missing_segment_ids"], [segment_id])

    def test_source_selection_and_repeated_ids_are_sets(self):
        _, catalog = sample_catalog()
        first = simulate_removal(catalog, ["s0.seg0", "s0.seg0"], ["s0", "s0"])
        second = simulate_removal(catalog, ["s0.seg1", "s0.seg0"])
        self.assertEqual(first, second)
        self.assertEqual(first["removed_segment_ids"], ["s0.seg0", "s0.seg1"])

    def test_unknown_or_malformed_selection_fails(self):
        _, catalog = sample_catalog()
        for ids in (["unknown"], [True], "s0.seg0", None):
            with self.subTest(ids=ids), self.assertRaises(CatalogError):
                simulate_removal(catalog, ids)
        with self.assertRaises(CatalogError):
            simulate_removal(catalog, removed_source_ids=["Band"])

    def test_empty_catalog_and_rejected_facts_are_not_losses(self):
        _, catalog = sample_catalog([("Texto não sustentado.", ["not_supported"] * 3)])
        report = simulate_removal(catalog, removed_source_ids=["s0", "s1"])
        self.assertEqual(report["summary"]["total_information"], 0)
        self.assertEqual(report["summary"]["rejected"], 1)
        empty = seal({"schema_version": 1, "dataset_fingerprint": dataset_digest([]), "policy": {}, "provenance": {},
                      "segments": [], "candidates": [], "comparisons": [], "facts": [], "rejected_facts": []})
        self.assertEqual(simulate_removal(empty)["summary"]["remaining_segments"], 0)

    def test_deterministic_pure_output_does_not_share_mutable_evidence(self):
        dataset, catalog = sample_catalog()
        before = copy.deepcopy(catalog)
        report = simulate_removal(catalog, dataset=dataset)
        self.assertEqual(report, simulate_removal(catalog, dataset=dataset))
        self.assertEqual(catalog, before)
        report["preserved_information"][0]["remaining_supports"][0]["evidence"][0]["quote"] = "changed"
        self.assertEqual(catalog, before)

    def test_compiled_simulator_reuses_validation_and_matches_all_removal_scenarios(self):
        dataset, catalog = sample_catalog([
            ("A ponte foi inaugurada.", ["supported", "not_supported", "supported"]),
            ("A ponte tem duas faixas.", ["not_supported", "supported", "uncertain"]),
        ])
        scenarios = [list(ids) for size in range(4) for ids in itertools.combinations(["s0.seg0", "s0.seg1", "s1.seg0"], size)]
        expected = [simulate_removal(catalog, ids, dataset=dataset) for ids in scenarios]
        with patch("determinism.modules.semantic_catalog.domain.simulation.validate_catalog", wraps=validate_catalog) as validator:
            simulator = RemovalSimulator(catalog, dataset)
            for ids, report in zip(scenarios, expected):
                self.assertEqual(simulator.simulate(ids), report)
            self.assertEqual(simulator.simulate(removed_source_ids=["s0"]), expected[4])
            self.assertEqual(validator.call_count, 1)

    def test_compiled_simulator_is_isolated_from_catalog_and_report_mutations(self):
        dataset, catalog = sample_catalog()
        expected = simulate_removal(catalog)
        simulator = RemovalSimulator(catalog, dataset)
        catalog["facts"][0]["statement"] = "Modified caller catalog"
        catalog["facts"][0]["assessments"][0]["evidence"][0]["quote"] = "Modified evidence"
        catalog["segments"].clear()
        dataset.clear()
        report = simulator.simulate()
        self.assertEqual(report, expected)
        report["preserved_information"][0]["remaining_supports"][0]["evidence"][0]["quote"] = "Modified output"
        report["preserved_information"][0]["remaining_supports"][0]["segment_ids"].clear()
        report["retained_segment_ids"].clear()
        self.assertEqual(simulator.simulate(), expected)


class CatalogValidationTests(unittest.TestCase):
    def test_catalog_hash_and_full_dataset_snapshot_are_checked(self):
        dataset, catalog = sample_catalog()
        self.assertIs(validate_catalog(catalog, dataset), catalog)
        edited = copy.deepcopy(catalog)
        edited["facts"][0]["statement"] += " changed"
        with self.assertRaisesRegex(CatalogError, "catalog_id"):
            validate_catalog(edited)
        changed_dataset = copy.deepcopy(dataset)
        changed_dataset[0]["metadata"] = "even metadata changes invalidate the snapshot"
        with self.assertRaisesRegex(CatalogError, "snapshot"):
            validate_catalog(catalog, changed_dataset)

    def test_equivalence_chain_cannot_merge_a_non_equivalent_pair(self):
        _, catalog = sample_catalog([(f"Informação {i}.", ["supported"] * 3) for i in range(3)])
        for comparison in catalog["comparisons"]:
            comparison["relation"] = "equivalent"
        catalog["comparisons"][1]["relation"] = "uncertain"
        catalog["facts"][0]["candidate_ids"] = ["c0", "c1", "c2"]
        catalog["facts"] = catalog["facts"][:1]
        with self.assertRaisesRegex(CatalogError, "todos os pares"):
            validate_catalog(seal(catalog))

    def test_missing_assessment_is_not_treated_as_negative_evidence(self):
        _, catalog = sample_catalog()
        catalog["facts"][0]["assessments"].pop()
        with self.assertRaisesRegex(CatalogError, "cada segmento"):
            validate_catalog(seal(catalog))

    def test_invalid_references_evidence_partition_and_types_are_rejected(self):
        edits = [
            lambda c: c["facts"][0]["candidate_ids"].append("absent"),
            lambda c: c["facts"].append(copy.deepcopy(c["facts"][0])),
            lambda c: c["candidates"][0].update(segment_id="absent"),
            lambda c: c["candidates"][0].update(information_indices=[True]),
            lambda c: c["candidates"][0]["evidence"][0].update(start=True),
            lambda c: c["candidates"][0]["evidence"][0].update(quote="Invented evidence"),
            lambda c: c["facts"][0]["assessments"][0].update(evidence=[]),
            lambda c: c["facts"][0]["assessments"][0].update(segment_ids=["unknown"]),
            lambda c: c["facts"][0]["assessments"].append(copy.deepcopy(c["facts"][0]["assessments"][0])),
            lambda c: c["segments"][0].update(source_index=True),
            lambda c: c.update(schema_version=True),
            lambda c: c["facts"][0].update(statement="Untraceable canonical formulation"),
            lambda c: c.update(facts=[]),
        ]
        for edit in edits:
            _, catalog = sample_catalog()
            edit(catalog)
            with self.subTest(edit=edit), self.assertRaises(CatalogError):
                validate_catalog(seal(catalog))

    def test_incomplete_pair_matrix_is_rejected(self):
        _, catalog = sample_catalog([(f"Informação {i}.", ["supported"] * 3) for i in range(2)])
        catalog["comparisons"] = []
        with self.assertRaisesRegex(CatalogError, "todos os pares"):
            validate_catalog(seal(catalog))

    def test_non_json_and_nonfinite_values_do_not_produce_hashes(self):
        for value in (math.nan, math.inf, (1, 2), {1: "numeric key"}, "\ud800"):
            with self.subTest(value=repr(value)), self.assertRaises(CatalogError):
                catalog_digest({"value": value})


if __name__ == "__main__":
    unittest.main()
