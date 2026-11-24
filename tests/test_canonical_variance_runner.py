"""Exercise real pipeline composition through a local HTTP server, without APIs."""

import json
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from determinism.experiments import canonical_variance_runner as runner
from determinism.modules.semantic_extraction.adapters import load_extraction_config
from determinism.modules.semantic_extraction.domain import CoverageFeedback, SegmentInput, TokenReference
from determinism.modules.semantic_grouping.adapters import load_grouping_config
from determinism.modules.semantic_grouping.adapters.postgres_vectors import canonical_float32


class LocalAPI:
    def __init__(self, *, fail_chat_calls=0, cached_tokens=0):
        self.calls = []
        self.fail_chat_calls, self.cached_tokens = fail_chat_calls, cached_tokens
        self.lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with api.lock:
                    api.calls.append((self.path, body))
                    failed = self.path.endswith("/chat/completions") and api.fail_chat_calls > 0
                    if failed:
                        api.fail_chat_calls -= 1
                if failed:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"private error omitted"}}')
                    return
                if self.path.endswith("/chat/completions"):
                    information = {"info": "Ana dormiu.", "tokens": [0, 1], "context": "O que Ana fez?",
                                   "syntagm": {"subject": "Ana", "relation": "dormiu", "object": ""}}
                    value = {"choices": [{"finish_reason": "stop", "message": {
                        "content": json.dumps({"informacoes": [information, information]})}}],
                        "model": body["model"], "system_fingerprint": "fp_mock",
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
                                  "prompt_tokens_details": {"cached_tokens": api.cached_tokens, "cache_write_tokens": 0}}}
                else:
                    value = {"data": [{"index": index, "embedding": [0.123456789, 0.987654321]}
                                      for index in reversed(range(len(body["input"])))],
                             "model": body["model"], "usage": {"prompt_tokens": len(body["input"]), "total_tokens": len(body["input"])}}
                raw = json.dumps(value).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("x-request-id", "request_mock")
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"


class CanonicalVarianceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw = self.directory / "source.json"
        self.raw.write_text(json.dumps([{"name": "source", "segments": [{"content": "Ana dormiu."}]}]), encoding="utf-8")
        self.api = LocalAPI()
        self.addCleanup(self.api.close)
        self.extraction = replace(load_extraction_config(), provider="openai-compatible", base_url=self.api.base_url, api_key_env="TEST_MISSING_KEY")
        self.grouping = replace(load_grouping_config(), provider="openai-compatible", base_url=self.api.base_url, api_key_env="TEST_MISSING_KEY")

    def prepare(self):
        root = self.directory / "experiment"
        manifest = runner.prepare_experiment(self.raw, root, 2, self.extraction, self.grouping)
        return root, manifest

    def test_every_repeat_reexecutes_production_stages_and_both_embedding_requests(self):
        root, manifest = self.prepare()
        observed = []
        originals = {name: getattr(runner, name) for name in ("run_mapping", "extract_dataset", "group_dataset", "build_global_catalog")}

        def call(name):
            def wrapped(*args, **kwargs):
                observed.append(name)
                return originals[name](*args, **kwargs)
            return wrapped

        with patch.object(runner, "run_mapping", side_effect=call("run_mapping")), \
             patch.object(runner, "extract_dataset", side_effect=call("extract_dataset")), \
             patch.object(runner, "group_dataset", side_effect=call("group_dataset")), \
             patch.object(runner, "build_global_catalog", side_effect=call("build_global_catalog")):
            runs = [runner.execute_run(index, root, self.extraction, self.grouping, None, None,
                                       manifest["specification"]["input_sha256"]) for index in (1, 2)]
        self.assertEqual(observed, list(originals) * 2)
        self.assertEqual([run["status"] for run in runs], ["completed", "completed"])
        self.assertEqual([run["counts"]["global_canonical_information"] for run in runs], [1, 1])
        self.assertEqual([path for path, _ in self.api.calls], ["/v1/chat/completions", "/v1/embeddings", "/v1/embeddings"] * 2)
        for run in runs:
            self.assertEqual(run["api"]["attempted_calls"], 3)
            self.assertEqual(run["api"]["calls_by_stage"], {"extraction": 1, "local_grouping": 1, "global_grouping": 1})
            self.assertEqual(run["api"]["prompt_cache"]["reported_cached_tokens"], 0)
            self.assertTrue(run["api"]["prompt_cache"]["zero_writes_verified_for_all_calls"])
            self.assertEqual(run["local_vector_handoff"][0]["returned"], 0)
            self.assertEqual(run["threshold_local"], .85)
            self.assertEqual(run["threshold_global"], .85)
            directory = root / run["directory"]
            self.assertEqual(set(run["artifact_sha256"]), {"raw.json", "mapped.json", "extracted.json", "grouped.json", "global.json"})
            mapped = json.loads((directory / "mapped.json").read_text(encoding="utf-8"))
            self.assertNotIn("information", mapped[0]["segments"][0])
            self.assertTrue(all(entry["count"] == 0 for entry in mapped[0]["segments"][0]["map"].values()))
            self.assertNotIn("Authorization", (directory / "api" / "calls.jsonl").read_text(encoding="utf-8"))
        segment = SegmentInput(0, 0, "Ana dormiu.", (TokenReference(0, "Ana"), TokenReference(1, "dormiu")), CoverageFeedback(1, ((0, 0), (1, 0))))
        expected = runner.OpenAICompatibleProvider(self.extraction).build_request_body(segment)
        for _, request in (self.api.calls[0], self.api.calls[3]):
            self.assertEqual(request["prompt_cache_options"], {"mode": "explicit"})
            self.assertEqual({key: value for key, value in request.items() if key != "prompt_cache_options"}, expected)

    def test_local_handoff_is_float32_consumed_once_and_global_fetches_are_fresh(self):
        profile = self.grouping.profile
        store = runner.LocalVectorHandoff()
        self.assertEqual(store.fetch(profile, ("text",)), {})
        vector = (0.123456789, 0.987654321)
        store.put(profile, {"text": vector})
        self.assertEqual(store.fetch(profile, ("text",))["text"], canonical_float32(vector))
        with self.assertRaises(runner.ExperimentError):
            store.fetch(profile, ("text",))
        provider = runner.AuditedEmbeddingProvider(self.grouping, None, runner.AuditJournal(self.directory / "audit"), "global_grouping")
        global_store = runner.FreshGlobalVectors(provider)
        self.assertEqual(global_store.fetch(profile, ("text",)), global_store.fetch(profile, ("text",)))
        self.assertEqual(len(self.api.calls), 2)

    def test_provider_cache_violation_is_retained_as_failure_without_request_retry(self):
        root, manifest = self.prepare()
        self.api.cached_tokens = 12
        result = runner.execute_run(1, root, self.extraction, self.grouping, None, None, manifest["specification"]["input_sha256"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "ExperimentError")
        self.assertEqual(result["api"]["attempted_calls"], 1)
        self.assertEqual(result["api"]["prompt_cache"]["reported_cached_tokens"], 12)
        self.assertIn("extraction", result["stages_seconds"])
        self.assertFalse((root / result["directory"] / "global.json").exists())

    def _write_config(self, path, config):
        from dataclasses import asdict
        path.write_text("\n".join(f"{name} = {json.dumps(value)}" for name, value in asdict(config).items() if value is not None), encoding="utf-8")

    def test_failed_attempts_retained_full_pipeline_retry_and_pilot_resume(self):
        extraction_path, grouping_path = self.directory / "extract.toml", self.directory / "group.toml"
        self._write_config(extraction_path, self.extraction)
        self._write_config(grouping_path, self.grouping)
        root = self.directory / "experiment"
        argv = [str(self.raw), "--output-root", str(root), "--repeats", "2", "--extraction-config", str(extraction_path),
                "--grouping-config", str(grouping_path), "--env-file", str(self.directory / "missing.env")]
        self.api.fail_chat_calls = 1
        self.assertEqual(runner.main(argv + ["--max-new-runs", "1"]), 0)
        manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual([run["status"] for run in manifest["attempts"]], ["failed", "completed"])
        self.assertEqual([(run["run"], run["attempt"]) for run in manifest["runs"]], [(1, 2)])
        failed_path = root / manifest["attempts"][0]["directory"] / "mapped.json"
        failed_bytes = failed_path.read_bytes()
        self.assertEqual(runner.main(argv + ["--resume"]), 0)
        manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["attempts"]), 3)
        self.assertEqual(len(manifest["runs"]), 2)
        self.assertEqual(failed_path.read_bytes(), failed_bytes)
        self.assertEqual(len(self.api.calls), 7)
        failed_path.write_bytes(failed_bytes + b" ")
        self.assertEqual(runner.main(argv + ["--resume"]), 1)
        self.assertEqual(len(self.api.calls), 7)

    def test_resume_rejects_changed_input_or_configuration_before_new_calls(self):
        root, _ = self.prepare()
        self.raw.write_text("[]", encoding="utf-8")
        with self.assertRaises(runner.ExperimentError):
            runner.prepare_experiment(self.raw, root, 2, self.extraction, self.grouping, resume=True)
        self.assertEqual(self.api.calls, [])

    def test_interrupted_checkpoint_is_retained_and_next_attempt_starts_from_raw(self):
        root, manifest = self.prepare()
        manifest["attempts"].append({"run": 1, "attempt": 1, "directory": "run_001/attempt_01",
                                      "status": "started", "artifact_sha256": {}})
        runner._save_json(root / "experiment.json", manifest)
        resumed = runner.prepare_experiment(self.raw, root, 2, self.extraction, self.grouping, resume=True)
        self.assertEqual(resumed["attempts"][0]["status"], "interrupted")
        self.assertEqual(resumed["runs"], [])
        fresh = runner.execute_run(1, root, self.extraction, self.grouping, None, None,
                                   resumed["specification"]["input_sha256"], attempt=2)
        self.assertEqual(fresh["status"], "completed")
        self.assertEqual(fresh["attempt"], 2)
        self.assertEqual(fresh["api"]["attempted_calls"], 3)
        self.assertTrue((root / "run_001/attempt_01/run.json").exists())


if __name__ == "__main__":
    unittest.main()
