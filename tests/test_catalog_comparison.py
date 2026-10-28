"""Comparação offline de decisões v2 e controle de promoção por código de saída."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from determinism.entrypoints.cli import main
from determinism.experiments.catalog_comparison import compare_catalogs, run_catalog_comparison
from determinism.modules.semantic_catalog.adapters.json_artifacts import write_json
from determinism.modules.semantic_catalog.application.requirement_stage import RequirementCatalogStage
from determinism.modules.semantic_catalog.domain import CatalogError, catalog_digest, compile_requirements, dataset_digest
from determinism.modules.semantic_catalog.ports import CatalogResponse
from test_catalog_requirements_domain import sample_requirement_catalog, seal_requirements


class PartitionProvider:
    def __init__(self, merge):
        self.merge = merge

    def requirements(self, candidate):
        return CatalogResponse({"requirements": [{"kind": "predicate", "token_ids": [item["id"] for item in candidate["statement_tokens"]]}]})

    def compare(self, pairs):
        return CatalogResponse({"comparisons": [
            {"pair_id": pair["pair_id"], "relation": "equivalent" if self.merge else "different", "reason": "Cenário controlado."}
            for pair in pairs
        ]})

    def assess_requirements(self, segment, facts):
        return CatalogResponse({"assessments": [
            {"fact_id": fact["id"], "reference_status": "compatible", "requirements": [
                {"requirement_id": requirement["id"], "status": "entailed", "evidence_quotes": [segment["content"]]}
                for requirement in fact["requirements"]
            ]}
            for fact in facts
        ]})


def partition_catalog(merge):
    text = "A ponte abriu."
    dataset = [{"segments": [{"content": text, "information": [{"info": text}]}]} for _ in range(2)]
    return RequirementCatalogStage(PartitionProvider(merge))(dataset)


def uncertain_variant(catalog):
    result = copy.deepcopy(catalog)
    support = result["facts"][0]["assessments"][1]
    support["status"] = "uncertain"
    support["requirement_results"][1]["status"] = "ambiguous"
    support["requirement_results"][1]["evidence"] = copy.deepcopy(support["evidence"])
    return seal_requirements(result)


class CatalogComparisonTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.catalog = sample_requirement_catalog()

    def test_same_decisions_ignore_reason_metadata_and_response_order(self):
        changed = copy.deepcopy(self.catalog)
        changed["provenance"] = {"model": "another-returned-name", "system_fingerprint": "another-fingerprint"}
        changed["facts"][0]["assessments"].reverse()
        for support in changed["facts"][0]["assessments"]:
            support["reason"] = "Outra justificativa textual para a mesma decisão."
            support["requirement_results"].reverse()
        seal_requirements(changed)
        report = compare_catalogs(self.catalog, changed)
        self.assertTrue(report["same_decisions"])
        self.assertFalse(report["same_catalog_content"])
        self.assertEqual(report["difference_kind"], "audit_only")
        self.assertFalse(any(report["summary"].values()))
        self.assertEqual(report["affected_candidates"], [])

    def test_uncertainty_change_is_inference_instability_with_both_decisions(self):
        changed = uncertain_variant(self.catalog)
        before = copy.deepcopy((self.catalog, changed))
        report = compare_catalogs(self.catalog, changed)
        self.assertFalse(report["same_decisions"])
        self.assertEqual(report["difference_kind"], "inference_instability")
        self.assertEqual(report["summary"]["support_changes"], 1)
        difference = report["differences"]["support"][0]
        self.assertEqual(difference["segment_ids"], ["s1.seg0"])
        self.assertEqual(difference["first"]["status"], "not_supported")
        self.assertEqual(difference["second"]["status"], "uncertain")
        report["affected_candidates"][0]["information_indices"].clear()
        report["differences"]["support"][0]["first"]["requirements"].clear()
        self.assertEqual((self.catalog, changed), before)

    def test_changed_requirement_contract_is_explicit_even_if_final_statuses_match(self):
        changed = copy.deepcopy(self.catalog)
        requirements = compile_requirements({"requirements": [{"kind": "predicate", "token_ids": list(range(6))}]}, changed["candidates"][0]["statement"])
        changed["candidates"][0]["requirements"] = requirements
        for support in changed["facts"][0]["assessments"]:
            evidence = copy.deepcopy(support["evidence"]) if support["status"] == "supported" else []
            support["evidence"] = evidence
            support["requirement_results"] = [{"requirement_id": requirements[0]["id"], "status": "entailed" if support["status"] == "supported" else "absent", "evidence": copy.deepcopy(evidence)}]
        seal_requirements(changed)
        report = compare_catalogs(self.catalog, changed)
        self.assertFalse(report["same_decisions"])
        self.assertEqual(report["summary"]["inventory_changes"], 0)
        self.assertEqual(report["summary"]["requirement_changes"], 1)
        self.assertEqual(report["summary"]["support_changes"], 2)

    def test_partition_changes_are_not_hidden_by_identical_statement_counts(self):
        first, second = partition_catalog(False), partition_catalog(True)
        report = compare_catalogs(first, second)
        self.assertEqual(report["summary"]["inventory_changes"], 0)
        self.assertEqual(report["summary"]["relation_changes"], 1)
        self.assertEqual(report["summary"]["removed_groups"], 2)
        self.assertEqual(report["summary"]["added_groups"], 1)
        self.assertEqual(len(report["affected_candidates"]), 2)

    def test_canonical_member_change_is_reported_but_fact_identifier_is_not_aligned(self):
        first = partition_catalog(True)
        renamed = copy.deepcopy(first)
        renamed["facts"][0]["id"] = "different-derived-fact-id"
        seal_requirements(renamed)
        self.assertTrue(compare_catalogs(first, renamed)["same_decisions"])
        changed = copy.deepcopy(first)
        changed["facts"][0]["candidate_ids"].reverse()
        seal_requirements(changed)
        report = compare_catalogs(first, changed)
        self.assertFalse(report["same_decisions"])
        self.assertEqual(report["summary"]["representative_changes"], 1)

    def test_policy_change_is_reported(self):
        changed = copy.deepcopy(self.catalog)
        changed["policy"]["revision"] = "another-rule"
        seal_requirements(changed)
        report = compare_catalogs(self.catalog, changed)
        self.assertTrue(report["summary"]["policy_changed"])
        self.assertFalse(report["same_decisions"])

    def test_different_dataset_is_not_called_model_instability(self):
        changed = copy.deepcopy(self.catalog)
        changed["dataset_fingerprint"] = dataset_digest([])
        seal_requirements(changed)
        with self.assertRaisesRegex(CatalogError, "datasets diferentes"):
            compare_catalogs(self.catalog, changed)

    def test_legacy_positional_identifiers_are_rejected_explicitly(self):
        legacy = {"schema_version": 1, "dataset_fingerprint": dataset_digest([]), "policy": {}, "provenance": {},
                  "segments": [], "candidates": [], "comparisons": [], "facts": [], "rejected_facts": []}
        legacy["catalog_id"] = catalog_digest(legacy)
        with self.assertRaisesRegex(CatalogError, "v2"):
            compare_catalogs(legacy, legacy)

    def test_invalid_artifact_cannot_be_compared_by_hash_only(self):
        changed = copy.deepcopy(self.catalog)
        changed["decision_id"] = "0" * 64
        changed["catalog_id"] = catalog_digest(changed)
        with self.assertRaises(CatalogError):
            compare_catalogs(self.catalog, changed)


class CatalogComparisonFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = self.root / "first.json"
        self.second = self.root / "second.json"
        self.output = self.root / "comparison.json"
        _, self.catalog = sample_requirement_catalog()
        write_json(self.first, self.catalog)
        write_json(self.second, self.catalog)

    def test_formatting_is_separate_from_decisions_and_artifact_content(self):
        self.second.write_text(json.dumps(self.catalog), encoding="utf-8")
        report = run_catalog_comparison(self.first, self.second, self.output)
        self.assertTrue(report["same_decisions"])
        self.assertTrue(report["same_catalog_content"])
        self.assertFalse(report["same_file_bytes"])
        self.assertEqual(report["difference_kind"], "formatting_only")
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)

    def test_output_cannot_replace_inputs_including_hardlink(self):
        original = self.first.read_bytes()
        for destination in (self.first, self.second):
            with self.subTest(destination=destination), self.assertRaises(CatalogError):
                run_catalog_comparison(self.first, self.second, destination)
        link = self.root / "link.json"
        os.link(self.first, link)
        with self.assertRaises(CatalogError):
            run_catalog_comparison(self.first, self.second, link)
        self.assertEqual(self.first.read_bytes(), original)

    def test_invalid_second_catalog_preserves_existing_report(self):
        self.second.write_text("{}", encoding="utf-8")
        self.output.write_bytes(b"previous")
        with self.assertRaises(CatalogError):
            run_catalog_comparison(self.first, self.second, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous")

    def test_cli_gate_returns_two_for_decisions_and_zero_for_audit_only(self):
        changed = uncertain_variant(self.catalog)
        write_json(self.second, changed)
        originals = (self.first.read_bytes(), self.second.read_bytes())
        with redirect_stdout(io.StringIO()) as captured:
            result = main(["compare-catalogs", str(self.first), str(self.second), "--output", str(self.output), "--fail-on-difference"])
        self.assertEqual(result, 2)
        self.assertIn("Instabilidade", captured.getvalue())
        self.assertEqual((self.first.read_bytes(), self.second.read_bytes()), originals)
        self.assertTrue(self.output.is_file())
        changed = copy.deepcopy(self.catalog)
        changed["provenance"]["note"] = "Another run"
        write_json(self.second, seal_requirements(changed))
        with redirect_stdout(io.StringIO()):
            result = main(["compare-catalogs", str(self.first), str(self.second), "--output", str(self.output), "--fail-on-difference"])
        self.assertEqual(result, 0)

    def test_cli_difference_is_diagnostic_without_gate_option(self):
        write_json(self.second, uncertain_variant(self.catalog))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["compare-catalogs", str(self.first), str(self.second), "--output", str(self.output)]), 0)

    def test_cli_invalid_input_returns_one(self):
        self.second.write_text("{}", encoding="utf-8")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["compare-catalogs", str(self.first), str(self.second), "--output", str(self.output)])
        self.assertEqual(raised.exception.code, 1)

    def test_simulation_cli_shows_preserved_component_of_lost_compound_claim(self):
        with redirect_stdout(io.StringIO()) as captured:
            result = main(["simulate", str(self.first), "--remove", "s0.seg0", "--output", str(self.output)])
        self.assertEqual(result, 0)
        self.assertIn("Preservado: A ponte abriu", captured.getvalue())
        self.assertIn("Perdido: na terça-feira", captured.getvalue())
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["total_information"], 1)


if __name__ == "__main__":
    unittest.main()
