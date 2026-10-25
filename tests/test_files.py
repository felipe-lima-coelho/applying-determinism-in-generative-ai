import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from determinism.experiments import run_mapping
from determinism.modules.text_mapping.adapters import DatasetFileError, JsonDatasetRepository
from determinism.modules.text_mapping.domain import DatasetFormatError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FileMappingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "entrada com espaço.json"

    def write_dataset(self, dataset):
        self.source.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    def test_in_place_mapping_persists_unicode_and_is_repeatable(self):
        self.write_dataset([
            {"name": "fonte", "segments": [{"content": "É a casa!", "start": 1.25}], "extra": [True, None]}
        ])

        mapped = run_mapping(self.source)

        self.assertEqual(json.loads(self.source.read_text(encoding="utf-8")), mapped)
        text = self.source.read_text(encoding="utf-8")
        self.assertIn("É", text)
        self.assertNotIn("\\u00c9", text)
        self.assertEqual(mapped[0]["extra"], [True, None])
        self.assertEqual(mapped[0]["segments"][0]["start"], 1.25)
        first_bytes = self.source.read_bytes()
        run_mapping(self.source)
        self.assertEqual(self.source.read_bytes(), first_bytes)

    def test_utf8_bom_is_accepted(self):
        self.source.write_text('[{"segments": [{"content": "É."}]}]', encoding="utf-8-sig")

        mapped = run_mapping(self.source)

        self.assertEqual(mapped[0]["segments"][0]["map"]["0"]["word"], "É")
        self.assertFalse(self.source.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_invalid_files_leave_source_and_existing_destination_untouched(self):
        invalid_files = [
            b"{ invalid json",
            b"\xff\xfe",
            b"{}",
            b'[{"segments": [{"content": "valid"}]}, {"segments": [{"content": null}]}]',
            b'[{"segments": [], "name": "a", "name": "b"}]',
            b'[{"segments": [], "extra": NaN}]',
            b'[{"segments": [], "extra": Infinity}]',
            b'[{"segments": [], "extra": -Infinity}]',
        ]
        destination = self.root / "saida.json"
        for raw in invalid_files:
            with self.subTest(raw=raw):
                self.source.write_bytes(raw)
                destination.write_bytes(b"destino anterior")
                for output in (None, destination):
                    with self.assertRaises((DatasetFileError, DatasetFormatError)):
                        run_mapping(self.source, output)
                    self.assertEqual(self.source.read_bytes(), raw)
                    self.assertEqual(destination.read_bytes(), b"destino anterior")
                    self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_write_or_replace_failure_preserves_file_and_cleans_temporary(self):
        self.write_dataset([{"segments": [{"content": "não perder"}]}])
        original = self.source.read_bytes()
        for operation in ("os.fsync", "os.replace"):
            with self.subTest(operation=operation):
                with patch(
                    f"determinism.modules.text_mapping.adapters.json_repository.{operation}",
                    side_effect=OSError("falha simulada"),
                ):
                    with self.assertRaisesRegex(OSError, "falha simulada"):
                        run_mapping(self.source)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_serialization_failure_happens_before_opening_a_temporary_file(self):
        self.source.write_bytes(b"conteudo anterior")
        repository = JsonDatasetRepository()
        for metadata in ({1, 2}, float("nan")):
            with self.subTest(metadata=metadata):
                with self.assertRaises(DatasetFileError):
                    repository.save(self.source, [{"segments": [], "metadata": metadata}])
                self.assertEqual(self.source.read_bytes(), b"conteudo anterior")
                self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_tokenization_failure_leaves_source_and_destination_untouched(self):
        class FailingTokenizer:
            def tokenize(self, text):
                if text == "falha":
                    raise RuntimeError("tokenização falhou")
                return [text]

        self.write_dataset([{"segments": [{"content": "ok"}, {"content": "falha"}]}])
        destination = self.root / "saida.json"
        destination.write_bytes(b"resultado anterior")
        original = self.source.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "tokenização falhou"):
            run_mapping(self.source, destination, tokenizer=FailingTokenizer())

        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(destination.read_bytes(), b"resultado anterior")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_cli_separate_output_preserves_source_and_reports_totals(self):
        self.write_dataset([
            {"name": "a", "segments": [{"content": "A casa é verde, porém não tem porta."}, {"content": "Olá!"}]},
            {"name": "b", "segments": [{"content": ""}]},
        ])
        original = self.source.read_bytes()
        output = self.root / "resultado com espaço.json"

        completed = subprocess.run(
            [sys.executable, "-m", "determinism", "map", str(self.source), "--output", str(output)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=self.utf8_environment(),
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.source.read_bytes(), original)
        mapped = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(mapped[0]["segments"][0]["map"]["7"], {"token": 7, "word": "porta", "count": 0})
        self.assertEqual(mapped[1]["segments"][0]["map"], {})
        self.assertIn("fontes: 2, segmentos: 3, palavras: 9", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_cli_invalid_input_reports_clear_error_without_replacing_file(self):
        self.write_dataset([{"segments": [{"content": "válido"}]}, {"segments": [{}]}])
        original = self.source.read_bytes()

        completed = subprocess.run(
            [sys.executable, "-m", "determinism", "map", str(self.source)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=self.utf8_environment(),
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("$[1].segments[0].content", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(self.source.read_bytes(), original)

    @staticmethod
    def utf8_environment():
        # O processo filho precisa emitir UTF-8 também quando o Windows usa outra code page.
        import os

        return {**os.environ, "PYTHONIOENCODING": "utf-8"}


if __name__ == "__main__":
    unittest.main()
