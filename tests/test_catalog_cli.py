"""Contratos públicos, persistência atômica e CLI do catálogo e da simulação."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from determinism.entrypoints.cli import main
from determinism.experiments import run_catalog, run_simulation
from determinism.modules.semantic_catalog.adapters import CatalogConfig
from determinism.modules.semantic_catalog.adapters.json_artifacts import write_json
from determinism.modules.semantic_catalog.domain import CatalogError
from determinism.modules.semantic_catalog.ports.provider import CatalogInferenceError, CatalogResponse


SHARED = "A ponte foi inaugurada."
EXCLUSIVE = "A ponte tem duas faixas."
UNCERTAIN = "O projeto custou dez milhões."
MAYBE = "O custo pode ter sido dez milhões."


class ScenarioProvider:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    def extract(self, segment):
        self.calls.append("extract")
        if self.failure is not None:
            raise self.failure
        statements = (SHARED, EXCLUSIVE, UNCERTAIN) if segment["id"] == "s0.seg0" else (SHARED,)
        return CatalogResponse({"facts": [
            {"statement": statement, "evidence_quotes": [statement], "information_indices": []}
            for statement in statements
        ]}, "fixture-v1")

    def compare(self, pairs):
        self.calls.append("compare")
        return CatalogResponse({"comparisons": [
            {"pair_id": pair["pair_id"],
             "relation": "equivalent" if pair["left"]["statement"] == pair["right"]["statement"] else "different",
             "reason": "Equivalência controlada do cenário."}
            for pair in pairs
        ]}, "fixture-v1")

    def assess(self, segment, facts):
        self.calls.append("assess")
        items = []
        for fact in facts:
            statement = fact["statement"]
            status = ("supported" if statement in segment["content"] else
                      "uncertain" if statement == UNCERTAIN else "not_supported")
            items.append({"fact_id": fact["id"], "status": status,
                          "evidence_quotes": [statement] if status == "supported" else [],
                          "reason": "Suporte controlado para testar o cenário de remoção."})
        return CatalogResponse({"assessments": items}, "fixture-v1")


class CatalogCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "dataset original.json"
        self.catalog_path = self.root / "catálogo.json"
        self.report_path = self.root / "simulação.json"
        self.dataset = [
            {"name": "Band", "segments": [{"content": " ".join((SHARED, EXCLUSIVE, UNCERTAIN)), "start": 1.5}]},
            {"name": "Record", "segments": [{"content": " ".join((SHARED, MAYBE)), "start": 2.5}]},
        ]
        self.input.write_text(json.dumps(self.dataset, ensure_ascii=False), encoding="utf-8")
        self.original_bytes = self.input.read_bytes()
        self.config = CatalogConfig(max_concurrency=1)

    def create_catalog(self, provider=None):
        return run_catalog(self.input, self.catalog_path, provider=provider or ScenarioProvider(), config=self.config)

    def assert_cli_error(self, arguments, expected):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(arguments)
        self.assertEqual(raised.exception.code, 1)
        self.assertIn(expected, stderr.getvalue())
        self.assertNotIn("salv", stdout.getvalue().lower())

    def test_run_catalog_persists_exact_result_without_changing_dataset(self):
        catalog = self.create_catalog()
        self.assertEqual(json.loads(self.catalog_path.read_text(encoding="utf-8")), catalog)
        self.assertEqual(self.input.read_bytes(), self.original_bytes)
        self.assertEqual(len(catalog["facts"]), 3)
        self.assertEqual([segment["id"] for segment in catalog["segments"]], ["s0.seg0", "s1.seg0"])

    def test_run_simulation_persists_all_categories_and_keeps_inputs(self):
        self.create_catalog()
        original_catalog = self.catalog_path.read_bytes()
        report = run_simulation(self.catalog_path, self.report_path, removed_segment_ids=["s0.seg0"],
                                dataset_path=self.input)
        self.assertEqual(json.loads(self.report_path.read_text(encoding="utf-8")), report)
        self.assertEqual([item["statement"] for item in report["lost_information"]], [EXCLUSIVE])
        self.assertEqual([item["statement"] for item in report["preserved_information"]], [SHARED])
        self.assertEqual([item["statement"] for item in report["uncertain_information"]], [UNCERTAIN])
        self.assertEqual(self.catalog_path.read_bytes(), original_catalog)
        self.assertEqual(self.input.read_bytes(), self.original_bytes)

    def test_failed_inference_keeps_existing_output_and_dataset_untouched(self):
        sentinel = b'{"previous": "valid output"}\n'
        self.catalog_path.write_bytes(sentinel)
        provider = ScenarioProvider(failure=CatalogInferenceError("Falha de API simulada."))
        with self.assertRaisesRegex(CatalogInferenceError, "API"):
            self.create_catalog(provider)
        self.assertEqual(self.catalog_path.read_bytes(), sentinel)
        self.assertEqual(self.input.read_bytes(), self.original_bytes)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_invalid_dataset_fails_before_inference_and_preserves_output(self):
        self.dataset[1]["segments"][0]["information"] = "inválido"
        self.input.write_text(json.dumps(self.dataset), encoding="utf-8")
        bad_input = self.input.read_bytes()
        self.catalog_path.write_bytes(b"previous")
        provider = ScenarioProvider()
        with self.assertRaises(CatalogError):
            self.create_catalog(provider)
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.catalog_path.read_bytes(), b"previous")
        self.assertEqual(self.input.read_bytes(), bad_input)

    def test_atomic_replace_failure_keeps_previous_output_and_cleans_temporary_file(self):
        self.catalog_path.write_bytes(b"previous")
        with patch("determinism.modules.semantic_catalog.adapters.json_artifacts.os.replace",
                   side_effect=OSError("Disco simulado indisponível.")):
            with self.assertRaises(OSError):
                self.create_catalog()
        self.assertEqual(self.catalog_path.read_bytes(), b"previous")
        self.assertEqual(self.input.read_bytes(), self.original_bytes)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_output_cannot_overwrite_dataset_or_catalog(self):
        provider = ScenarioProvider()
        with self.assertRaisesRegex(CatalogError, "diferente"):
            run_catalog(self.input, self.input, provider=provider, config=self.config)
        self.assertEqual(provider.calls, [])
        self.create_catalog()
        saved = self.catalog_path.read_bytes()
        for destination in (self.catalog_path, self.input):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(CatalogError, "diferente"):
                    run_simulation(self.catalog_path, destination, dataset_path=self.input)
        self.assertEqual(self.catalog_path.read_bytes(), saved)
        self.assertEqual(self.input.read_bytes(), self.original_bytes)

    def test_hardlinks_cannot_bypass_input_protection(self):
        alias = self.root / "alias.json"
        try:
            os.link(self.input, alias)
        except (OSError, NotImplementedError):
            self.skipTest("O filesystem não oferece hardlinks neste ambiente.")
        provider = ScenarioProvider()
        with self.assertRaisesRegex(CatalogError, "diferente"):
            run_catalog(self.input, alias, provider=provider, config=self.config)
        self.assertEqual(provider.calls, [])
        self.create_catalog()
        catalog_alias = self.root / "catalog-alias.json"
        os.link(self.catalog_path, catalog_alias)
        saved = self.catalog_path.read_bytes()
        with self.assertRaisesRegex(CatalogError, "diferente"):
            run_simulation(self.catalog_path, catalog_alias)
        self.assertEqual(self.input.read_bytes(), self.original_bytes)
        self.assertEqual(self.catalog_path.read_bytes(), saved)

    def test_missing_output_parent_is_rejected_before_inference(self):
        provider = ScenarioProvider()
        destination = self.root / "missing" / "catalog.json"
        with self.assertRaisesRegex(CatalogError, "diretório"):
            run_catalog(self.input, destination, provider=provider, config=self.config)
        self.assertEqual(provider.calls, [])
        self.assertFalse(destination.parent.exists())

    def test_injected_provider_cache_requires_explicit_version_identity(self):
        provider = ScenarioProvider()
        self.catalog_path.write_bytes(b"previous")
        with self.assertRaisesRegex(CatalogError, "provider_identity"):
            run_catalog(self.input, self.catalog_path, provider=provider, config=self.config,
                        cache_dir=self.root / "cache")
        self.assertEqual(provider.calls, [])
        self.assertEqual(self.catalog_path.read_bytes(), b"previous")

    def test_run_catalog_can_resume_injected_cache_without_credentials(self):
        cache_dir = self.root / "cache"
        first = run_catalog(self.input, self.catalog_path, provider=ScenarioProvider(), config=self.config,
                            cache_dir=cache_dir, provider_identity="fixture-revision-1")
        provider = ScenarioProvider(failure=AssertionError("Inferência não deveria ocorrer."))
        with patch.dict(os.environ, {}, clear=True), patch(
                "determinism.experiments.semantic_catalog.read_api_key", side_effect=AssertionError("Credencial não deveria ser lida.")):
            second = run_catalog(self.input, self.catalog_path, provider=provider, config=self.config,
                                 cache_dir=cache_dir, provider_identity="fixture-revision-1", env_file=None)
        self.assertEqual(first, second)
        self.assertEqual(provider.calls, [])

    def test_simulation_cli_is_offline_and_prints_all_three_categories(self):
        self.create_catalog()
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch(
                "determinism.experiments.semantic_catalog.read_api_key", side_effect=AssertionError("Simulação deve ser offline.")), patch(
                "determinism.experiments.semantic_catalog.OpenAICompatibleCatalogProvider", side_effect=AssertionError("Sem provedor na simulação.")), redirect_stdout(stdout):
            status = main(["simulate", str(self.catalog_path), "--output", str(self.report_path),
                           "--remove", "s0.seg0", "--dataset", str(self.input)])
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        for expected in ("Perdidas: 1", "Preservadas: 1", "Incertas: 1", SHARED, EXCLUSIVE, UNCERTAIN):
            self.assertIn(expected, text)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["removed_segment_ids"], ["s0.seg0"])

    def test_catalog_cli_builds_output_and_lists_removable_segment_ids(self):
        provider = ScenarioProvider()
        stdout = io.StringIO()
        with patch("determinism.experiments.semantic_catalog._ConfiguredProvider", return_value=provider), patch(
                "determinism.experiments.semantic_catalog.JsonResponseCache", side_effect=AssertionError("--no-cache foi ignorado.")), redirect_stdout(stdout):
            status = main(["catalog", str(self.input), "--output", str(self.catalog_path), "--no-cache"])
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        for expected in ("Catálogo salvo", "informações: 3", "s0.seg0 | Band", "s1.seg0 | Record"):
            self.assertIn(expected, text)
        self.assertEqual(self.input.read_bytes(), self.original_bytes)
        self.assertEqual(len(json.loads(self.catalog_path.read_text(encoding="utf-8"))["facts"]), 3)

    def test_simulation_cli_rejects_unknown_segment_and_preserves_old_report(self):
        self.create_catalog()
        self.report_path.write_bytes(b"previous")
        self.assert_cli_error(["simulate", str(self.catalog_path), "--output", str(self.report_path),
                               "--remove", "s99.seg99"], "desconhecido")
        self.assertEqual(self.report_path.read_bytes(), b"previous")

    def test_simulation_cli_rejects_stale_dataset_and_preserves_old_report(self):
        self.create_catalog()
        self.dataset[0]["segments"][0]["content"] += " Conteúdo atualizado."
        self.input.write_text(json.dumps(self.dataset), encoding="utf-8")
        self.report_path.write_bytes(b"previous")
        self.assert_cli_error(["simulate", str(self.catalog_path), "--output", str(self.report_path),
                               "--dataset", str(self.input)], "snapshot")
        self.assertEqual(self.report_path.read_bytes(), b"previous")

    def test_simulation_cli_combines_source_and_segment_removals(self):
        self.create_catalog()
        with redirect_stdout(io.StringIO()):
            status = main(["simulate", str(self.catalog_path), "--output", str(self.report_path),
                           "--remove-source", "s0", "--remove", "s1.seg0", "--remove", "s1.seg0"])
        self.assertEqual(status, 0)
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["removed_segment_ids"], ["s0.seg0", "s1.seg0"])
        self.assertEqual(report["summary"]["lost"], 3)

    def test_existing_commands_keep_their_original_dispatch_and_messages(self):
        payload = [{"name": "Original", "segments": [{"content": "Texto.", "map": {"0": {}},
                                                         "information": [{"info": "Texto."}],
                                                         "set_different_information": [[{"info": "Texto."}]]}]}]
        for command, function, expected in (("map", "run_mapping", "Mapeamento salvo"),
                                             ("extract", "run_extraction", "Extração salva"),
                                             ("group", "run_grouping", "Agrupamento salvo")):
            with self.subTest(command=command):
                stdout = io.StringIO()
                with patch(f"determinism.entrypoints.cli.{function}", return_value=payload) as original, patch(
                        "determinism.entrypoints.cli.run_catalog", side_effect=AssertionError("Etapa antiga desviada para catálogo.")), patch(
                        "determinism.entrypoints.cli.run_simulation", side_effect=AssertionError("Etapa antiga desviada para simulação.")), redirect_stdout(stdout):
                    self.assertEqual(main([command, str(self.input)]), 0)
                original.assert_called_once()
                self.assertEqual(original.call_args.args[:2], (self.input, None))
                self.assertIn(expected, stdout.getvalue())

    def test_invalid_catalog_json_does_not_replace_existing_simulation(self):
        self.catalog_path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
        self.report_path.write_bytes(b"previous")
        with self.assertRaises(CatalogError):
            run_simulation(self.catalog_path, self.report_path)
        self.assertEqual(self.report_path.read_bytes(), b"previous")


    def test_malformed_unicode_artifact_is_rejected_before_opening_temporary_file(self):
        self.report_path.write_bytes(b"previous")
        with patch("determinism.modules.semantic_catalog.adapters.json_artifacts.NamedTemporaryFile",
                   side_effect=AssertionError("Não deve abrir arquivo para conteúdo não serializável.")):
            with self.assertRaises(CatalogError):
                write_json(self.report_path, {"statement": "Fato " + chr(0xD800)})
        self.assertEqual(self.report_path.read_bytes(), b"previous")

    def test_catalog_cli_handles_malformed_unicode_without_replacing_artifacts(self):
        class InvalidUnicodeProvider(ScenarioProvider):
            def extract(self, segment):
                response = super().extract(segment)
                response.payload["facts"][0]["statement"] += chr(0xD800)
                return response

        self.catalog_path.write_bytes(b"previous")
        cache_dir = self.root / "cache"
        with patch("determinism.experiments.semantic_catalog._ConfiguredProvider", return_value=InvalidUnicodeProvider()):
            self.assert_cli_error(["catalog", str(self.input), "--output", str(self.catalog_path),
                                   "--cache-dir", str(cache_dir)], "UTF-8")
        self.assertEqual(self.catalog_path.read_bytes(), b"previous")
        self.assertEqual(self.input.read_bytes(), self.original_bytes)
        self.assertFalse(cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
