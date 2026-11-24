import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import chdir
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
        self.source = self.root / "data" / "transcriptions" / "entrada com espaço.json"
        self.destination = self.root / "data" / "processed" / self.source.name
        self.source.parent.mkdir(parents=True)
        self.destination.parent.mkdir(parents=True)

    def write_dataset(self, dataset):
        self.source.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    def test_default_mapping_preserves_transcription_and_is_repeatable(self):
        self.write_dataset([
            {"name": "fonte", "segments": [{"content": "É a casa!", "start": 1.25}], "extra": [True, None]}
        ])
        original = self.source.read_bytes()

        mapped = run_mapping(self.source)

        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(json.loads(self.destination.read_text(encoding="utf-8")), mapped)
        text = self.destination.read_text(encoding="utf-8")
        self.assertIn("É", text)
        self.assertNotIn("\\u00c9", text)
        self.assertEqual(mapped[0]["extra"], [True, None])
        self.assertEqual(mapped[0]["segments"][0]["start"], 1.25)
        first_bytes = self.destination.read_bytes()
        run_mapping(self.source)
        self.assertEqual(self.destination.read_bytes(), first_bytes)
        self.assertEqual(self.source.read_bytes(), original)

    def test_utf8_bom_is_accepted(self):
        self.source.write_text('[{"segments": [{"content": "É."}]}]', encoding="utf-8-sig")
        original = self.source.read_bytes()

        mapped = run_mapping(self.source)

        self.assertEqual(mapped[0]["segments"][0]["map"]["0"]["word"], "É")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.destination.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_relative_transcription_path_writes_to_processed_with_the_same_name(self):
        self.write_dataset([{"segments": [{"content": "Uma transcrição."}]}])
        original = self.source.read_bytes()

        with chdir(self.root):
            mapped = run_mapping(self.source.relative_to(self.root))

        self.assertEqual(json.loads(self.destination.read_text(encoding="utf-8")), mapped)
        self.assertEqual(self.source.read_bytes(), original)

    def test_default_output_cannot_overwrite_an_input_already_in_processed(self):
        self.destination.write_text('[{"segments": [{"content": "Preservar."}]}]', encoding="utf-8")
        original = self.destination.read_bytes()

        with chdir(self.root):
            with self.assertRaisesRegex(DatasetFileError, "preservar a fonte"):
                run_mapping(self.destination.relative_to(self.root))

        self.assertEqual(self.destination.read_bytes(), original)

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
                self.destination.write_bytes(b"mapeamento anterior")
                for output in (None, destination):
                    with self.assertRaises((DatasetFileError, DatasetFormatError)):
                        run_mapping(self.source, output)
                    self.assertEqual(self.source.read_bytes(), raw)
                    self.assertEqual(destination.read_bytes(), b"destino anterior")
                    self.assertEqual(self.destination.read_bytes(), b"mapeamento anterior")
                    self.assertEqual(list(self.root.rglob(".*.tmp")), [])

    def test_write_or_replace_failure_preserves_file_and_cleans_temporary(self):
        self.write_dataset([{"segments": [{"content": "não perder"}]}])
        original = self.source.read_bytes()
        self.destination.write_bytes(b"mapeamento anterior")
        for operation in ("os.fsync", "os.replace"):
            with self.subTest(operation=operation):
                with patch(
                    f"determinism.modules.text_mapping.adapters.json_repository.{operation}",
                    side_effect=OSError("falha simulada"),
                ):
                    with self.assertRaisesRegex(OSError, "falha simulada"):
                        run_mapping(self.source)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertEqual(self.destination.read_bytes(), b"mapeamento anterior")
                self.assertEqual(list(self.root.rglob(".*.tmp")), [])

    def test_serialization_failure_happens_before_opening_a_temporary_file(self):
        self.source.write_bytes(b"conteudo anterior")
        repository = JsonDatasetRepository()
        for metadata in ({1, 2}, float("nan")):
            with self.subTest(metadata=metadata):
                with self.assertRaises(DatasetFileError):
                    repository.save(self.source, [{"segments": [], "metadata": metadata}])
                self.assertEqual(self.source.read_bytes(), b"conteudo anterior")
                self.assertEqual(list(self.root.rglob(".*.tmp")), [])

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
        self.assertEqual(list(self.root.rglob(".*.tmp")), [])

    def test_output_cannot_replace_the_source_even_through_an_alias(self):
        self.write_dataset([{"segments": [{"content": "Preservar a fonte."}]}])
        original = self.source.read_bytes()
        alias = self.root / "alias.json"
        alias.hardlink_to(self.source)
        for output in (self.source, self.source.parent / ".." / "transcriptions" / self.source.name, alias):
            with self.subTest(output=output):
                with self.assertRaisesRegex(DatasetFileError, "preservar a fonte"):
                    run_mapping(self.source, output)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertEqual(alias.read_bytes(), original)
        self.assertFalse(self.destination.exists())

    def test_cli_default_output_preserves_source_and_reports_processed_path(self):
        self.write_dataset([{"name": "fonte", "segments": [{"content": "Olá, mundo!"}]}])
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

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.source.read_bytes(), original)
        mapped = json.loads(self.destination.read_text(encoding="utf-8"))
        self.assertEqual(mapped[0]["segments"][0]["content"], "Olá, mundo!")
        self.assertEqual(mapped[0]["segments"][0]["map"]["1"], {"token": 1, "word": "mundo", "count": 0})
        self.assertIn(f"Mapeamento salvo em {self.destination}", completed.stdout)
        self.assertIn("fontes: 1, segmentos: 1, palavras: 2", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_cli_rejects_output_equal_to_source_without_a_traceback(self):
        self.write_dataset([{"segments": [{"content": "Preservar a fonte."}]}])
        original = self.source.read_bytes()

        completed = subprocess.run(
            [sys.executable, "-m", "determinism", "map", str(self.source), "--output", str(self.source)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=self.utf8_environment(),
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("preservar a fonte", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(self.source.read_bytes(), original)

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
