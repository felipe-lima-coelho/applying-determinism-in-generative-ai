"""Integração do inventário fixo, contrato automático, redução e persistência v2."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from determinism.entrypoints.cli import main
from determinism.experiments.semantic_catalog import build_catalog, run_catalog, run_simulation
from determinism.modules.semantic_catalog.adapters import CatalogConfig
from determinism.modules.semantic_catalog.adapters.json_artifacts import JsonResponseCache
from determinism.modules.semantic_catalog.application import RequirementCatalogStage
from determinism.modules.semantic_catalog.domain import CatalogError, RemovalSimulator, statement_tokens
from determinism.modules.semantic_catalog.ports import CatalogResponse


OPEN = "A ponte abriu."
DETAIL = "A ponte tem duas faixas."


def dataset():
    return [
        {"name": "Band", "segments": [{"content": f"{OPEN} {DETAIL}",
          "information": [{"info": OPEN, "tokens": [0, 1, 2]}, {"info": DETAIL}, {"info": OPEN}],
          "set_different_information": [[{"info": OPEN}, {"info": DETAIL}]]}]},
        {"name": "Record", "segments": [{"content": OPEN, "information": [{"info": OPEN}]}]},
    ]


class FixedProvider:
    def __init__(self, *, reverse=False, reason="Teste controlado.", mutations=None):
        self.calls = []
        self.reverse, self.reason, self.mutations = reverse, reason, mutations or {}

    def extract(self, segment):
        raise AssertionError("Catálogo v2 não pode reextrair ou reformular.")

    def assess(self, segment, facts):
        raise AssertionError("Catálogo v2 não usa decisão global livre.")

    def emit(self, operation, value):
        return CatalogResponse(self.mutations.get(operation, lambda x: x)(copy.deepcopy(value)), "fixture-v2")

    def requirements(self, candidate):
        self.calls.append(("requirements", candidate))
        ids = [item["id"] for item in candidate["statement_tokens"]]
        return self.emit("requirements", {"requirements": [{"kind": "predicate", "token_ids": list(reversed(ids)) if self.reverse else ids}]})

    def compare(self, pairs):
        self.calls.append(("compare", pairs))
        items = [{"pair_id": pair["pair_id"], "relation": "equivalent" if pair["left"]["statement"] == pair["right"]["statement"] else "different", "reason": self.reason} for pair in pairs]
        return self.emit("compare", {"comparisons": list(reversed(items)) if self.reverse else items})

    def assess_requirements(self, segment, facts):
        self.calls.append(("assess_requirements", (segment, facts)))
        items = []
        for fact in facts:
            present = fact["statement"] in segment["content"]
            checks = [{"requirement_id": req["id"], "status": "entailed" if present else "absent",
                       "evidence_quotes": [fact["statement"]] if present else []} for req in fact["requirements"]]
            items.append({"fact_id": fact["id"], "reference_status": "compatible", "requirements": checks})
        return self.emit("assess_requirements", {"assessments": list(reversed(items)) if self.reverse else items})


class NeverProvider:
    def __getattr__(self, operation):
        raise AssertionError(f"Cache deveria evitar {operation}.")


class RequirementStageTests(unittest.TestCase):
    def test_fixed_inventory_preserves_source_and_groups_do_not_erase_details(self):
        data = dataset()
        before = copy.deepcopy(data)
        provider = FixedProvider()
        catalog = build_catalog(data, provider=provider)
        self.assertEqual(data, before)
        self.assertEqual(catalog["schema_version"], 2)
        self.assertEqual(len(catalog["candidates"]), 3)
        self.assertEqual(len(catalog["facts"]), 2)
        self.assertEqual({item["statement"] for item in catalog["candidates"]}, {OPEN, DETAIL})
        candidate = next(item for item in catalog["candidates"] if item["segment_id"] == "s0.seg0" and item["statement"] == OPEN)
        self.assertEqual(candidate["information_indices"], [0, 2])
        self.assertEqual(sum(op == "requirements" for op, _ in provider.calls), 3)
        report = RemovalSimulator(catalog).simulate(["s0.seg0"])
        self.assertEqual([item["statement"] for item in report["lost_information"]], [DETAIL])
        self.assertEqual([item["statement"] for item in report["preserved_information"]], [OPEN])
        self.assertEqual(report["decision_id"], catalog["decision_id"])

    def test_missing_information_is_rejected_before_any_inference(self):
        provider = FixedProvider()
        with self.assertRaisesRegex(CatalogError, "information"):
            build_catalog([{"segments": [{"content": OPEN}]}], provider=provider)
        self.assertEqual(provider.calls, [])

    def test_explicit_empty_inventory_never_invents_units_from_content(self):
        provider = FixedProvider()
        result = build_catalog([{"segments": [{"content": OPEN, "information": []}]}], provider=provider)
        self.assertEqual(result["facts"], [])
        self.assertEqual(provider.calls, [])

    def test_source_extraction_can_be_rejected_instead_of_assumed_true(self):
        data = dataset()
        data[0]["segments"][0]["information"].append({"info": "A ponte tem cem faixas."})
        result = build_catalog(data, provider=FixedProvider())
        self.assertEqual([f["statement"] for f in result["rejected_facts"]], ["A ponte tem cem faixas."])

    def test_requirement_omission_cannot_silently_remove_a_qualifier(self):
        provider = FixedProvider(mutations={"requirements": lambda _: {"requirements": [{"kind": "predicate", "token_ids": [0]}]}})
        with self.assertRaisesRegex(CatalogError, "cobrir todos"):
            build_catalog(dataset(), provider=provider, max_concurrency=1)
        self.assertTrue(all(op == "requirements" for op, _ in provider.calls))

    def test_assessment_requires_every_requirement_and_literal_evidence(self):
        for mode in ("omitted", "nonliteral", "unexpected"):
            def mutate(payload, mode=mode):
                item = payload["assessments"][0]
                if mode == "omitted":
                    item["requirements"] = []
                elif mode == "nonliteral":
                    item["requirements"][0].update(status="entailed", evidence_quotes=["Texto inventado."])
                else:
                    item["status"] = "supported"
                return payload
            with self.subTest(mode=mode), self.assertRaises(CatalogError):
                build_catalog(dataset(), provider=FixedProvider(mutations={"assess_requirements": mutate}), max_concurrency=1)

    def test_order_and_free_comparison_reason_do_not_change_decisions(self):
        first = build_catalog(dataset(), provider=FixedProvider(), assessment_batch_size=2)
        second = build_catalog(dataset(), provider=FixedProvider(reverse=True, reason="Outra explicação equivalente."), assessment_batch_size=2)
        self.assertNotEqual(first["catalog_id"], second["catalog_id"])
        self.assertEqual(first["decision_id"], second["decision_id"])

    def test_cache_replays_requirements_and_decisions_without_provider(self):
        with tempfile.TemporaryDirectory() as path:
            cache = JsonResponseCache(Path(path))
            first = RequirementCatalogStage(FixedProvider(), cache=cache, provenance={"fixture": "v2"})(dataset())
            second = RequirementCatalogStage(NeverProvider(), cache=cache, provenance={"fixture": "v2"})(dataset())
            self.assertEqual(first, second)

    def test_default_public_composition_persists_v2_and_simulates_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, report = (root / name for name in ("data.json", "catalog.json", "report.json"))
            source.write_text(json.dumps(dataset()), encoding="utf-8")
            before = source.read_bytes()
            catalog = run_catalog(source, output, provider=FixedProvider(), config=CatalogConfig())
            self.assertEqual(catalog["schema_version"], 2)
            result = run_simulation(output, report, removed_segment_ids=["s0.seg0"], dataset_path=source)
            self.assertEqual(result["summary"]["lost"], 1)
            self.assertEqual(source.read_bytes(), before)

    def test_catalog_cli_uses_requirements_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "data.json", root / "catalog.json"
            source.write_text(json.dumps(dataset()), encoding="utf-8")
            provider = FixedProvider()
            with patch("determinism.experiments.semantic_catalog._ConfiguredProvider", return_value=provider), redirect_stdout(io.StringIO()):
                status = main(["catalog", str(source), "--output", str(output), "--no-cache"])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], 2)
            self.assertTrue(any(op == "requirements" for op, _ in provider.calls))
            self.assertTrue(any(op == "assess_requirements" for op, _ in provider.calls))


if __name__ == "__main__":
    unittest.main()
