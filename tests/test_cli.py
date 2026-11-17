"""Interface pública dos experimentos independentes de map, extract e group."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch, sentinel

from determinism.entrypoints.cli import main


class CliTests(unittest.TestCase):
    def test_help_lists_only_the_three_pipeline_commands(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("{map,extract,group}", stdout.getvalue())

    def test_removed_global_commands_are_rejected_before_processing(self):
        for command in ("catalog", "simulate"):
            with self.subTest(command=command):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    main([command, "dataset.json"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("invalid choice", stderr.getvalue())

    def test_pipeline_commands_preserve_default_dispatch_and_messages(self):
        payload = [{"name": "Original", "segments": [{
            "content": "Texto.", "map": {"0": {}},
            "information": [{"info": "Texto."}],
            "set_different_information": [[{"info": "Texto."}]],
        }]}]
        cases = (
            ("map", "run_mapping", Path("data/transcriptions/bebe_real_sources.json"),
             f"Mapeamento salvo em {Path('data/processed/bebe_real_sources.json')}", "palavras: 1"),
            ("extract", "run_extraction", Path("data/processed/bebe_real_sources.json"),
             "Extração salva", "informações acumuladas: 1"),
            ("group", "run_grouping", Path("data/processed/bebe_real_sources.json"),
             "Agrupamento salvo", "grupos: 1"),
        )
        for command, function, source, expected, total in cases:
            with self.subTest(command=command):
                stdout = io.StringIO()
                with patch(f"determinism.entrypoints.cli.{function}", return_value=payload) as run, redirect_stdout(stdout):
                    self.assertEqual(main([command, str(source)]), 0)
                run.assert_called_once()
                self.assertEqual(run.call_args.args, (source, None))
                if command != "map":
                    self.assertEqual(run.call_args.kwargs["env_file"], Path(".env"))
                self.assertIn(expected, stdout.getvalue())
                self.assertIn("fontes: 1, segmentos: 1", stdout.getvalue())
                self.assertIn(total, stdout.getvalue())

    def test_inference_commands_forward_explicit_output_config_and_environment(self):
        payload = [{"segments": [{"content": "", "information": [], "set_different_information": []}]}]
        source, output = Path("input.json"), Path("output.json")
        config, environment = Path("custom.toml"), Path("custom.env")
        for command, run, loader in (
            ("extract", "run_extraction", "load_extraction_config"),
            ("group", "run_grouping", "load_grouping_config"),
        ):
            with self.subTest(command=command):
                with patch(f"determinism.entrypoints.cli.{loader}", return_value=sentinel.config) as load, patch(
                    f"determinism.entrypoints.cli.{run}", return_value=payload,
                ) as execute, redirect_stdout(io.StringIO()):
                    self.assertEqual(main([command, str(source), "--output", str(output),
                                           "--config", str(config), "--env-file", str(environment)]), 0)
                load.assert_called_once_with(config)
                execute.assert_called_once_with(source, output, config=sentinel.config, env_file=environment)


if __name__ == "__main__":
    unittest.main()
