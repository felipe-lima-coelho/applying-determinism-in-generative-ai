import copy
import io
import json
import tempfile
import threading
import unittest
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from determinism.modules.semantic_catalog.adapters import (
    CatalogConfig, ConfigurationError, OpenAICompatibleCatalogProvider,
    contract_fingerprint, load_catalog_config, read_api_key,
)
from determinism.modules.semantic_catalog.adapters.prompts import OPERATIONS, structured_output
from determinism.modules.semantic_catalog.ports import CatalogInferenceError, CatalogResponse


def envelope(payload):
    return {"model": "local-test", "system_fingerprint": "fp_test", "choices": [
        {"finish_reason": "stop", "message": {"content": json.dumps(payload)}}
    ]}


@contextmanager
def local_server():
    requests = []
    outputs = {
        "catalog_extract_v1": {"facts": [{"statement": "A ponte abriu.", "evidence_quotes": ["A ponte abriu."], "information_indices": []}]},
        "catalog_compare_v1": {"comparisons": [{"pair_id": "p0", "relation": "equivalent", "reason": "Mesma abertura."}]},
        "catalog_assess_v1": {"assessments": [{"fact_id": "f0", "status": "supported", "evidence_quotes": ["A ponte abriu."], "reason": "Conteúdo explícito."}]},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            result = envelope(outputs[body["response_format"]["json_schema"]["name"]])
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests, outputs
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class CatalogAdapterTests(unittest.TestCase):
    def test_three_operations_use_http_schemas_and_keep_data_out_of_system(self):
        segment = {"id": "s0.seg0", "content": "A ponte abriu.", "information": [], "metadata": "UNTRUSTED-INSTRUCTION"}
        pairs = [{"pair_id": "p0", "left": {"statement": "A ponte abriu."}, "right": {"statement": "A ponte foi aberta."}}]
        facts = [{"id": "f0", "statement": "A ponte abriu."}]
        original = copy.deepcopy((segment, pairs, facts))
        with local_server() as (url, requests, outputs):
            provider = OpenAICompatibleCatalogProvider(CatalogConfig(provider="openai-compatible", base_url=url), "local-secret")
            responses = [provider.extract(segment), provider.compare(pairs), provider.assess(segment, facts)]
        self.assertEqual((segment, pairs, facts), original)
        self.assertEqual(len(requests), 3)
        for (path, auth, body), operation, response in zip(requests, OPERATIONS, responses):
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(auth, "Bearer local-secret")
            self.assertEqual(body["seed"], 42)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertFalse(body["store"])
            self.assertEqual([item["role"] for item in body["messages"]], ["system", "user"])
            self.assertNotIn("UNTRUSTED-INSTRUCTION", body["messages"][0]["content"])
            contract = body["response_format"]["json_schema"]
            self.assertTrue(contract["strict"])
            self.assertFalse(contract["schema"]["additionalProperties"])
            self.assertEqual(response, CatalogResponse(outputs[f"catalog_{operation}_v1"], "local-test", "fp_test"))
        self.assertEqual(json.loads(requests[0][2]["messages"][1]["content"]), {"segment": segment})
        self.assertEqual(json.loads(requests[1][2]["messages"][1]["content"]), {"pairs": pairs})
        self.assertEqual(json.loads(requests[2][2]["messages"][1]["content"]), {"segment": segment, "facts": facts})
        self.assertNotIn("local-secret", repr(provider))

    def test_schemas_require_every_property_and_forbid_extra_properties(self):
        def verify(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                for value in node.values():
                    verify(value)
            elif isinstance(node, list):
                for value in node:
                    verify(value)
        for operation in OPERATIONS:
            verify(structured_output(operation)["schema"])
        self.assertEqual(len(contract_fingerprint()), 64)
        self.assertEqual(contract_fingerprint(), contract_fingerprint())

    def test_compatible_server_can_explicitly_omit_reasoning_without_other_fallbacks(self):
        config = CatalogConfig(provider="openai-compatible", base_url="http://127.0.0.1:1234/v1", reasoning_effort="omit")
        body = OpenAICompatibleCatalogProvider(config).build_request_body("extract", {"segment": {"content": "texto"}})
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["model"], config.model)
        self.assertIn("temperature", body)
        self.assertIn("seed", body)
        with self.assertRaises(ValueError):
            OpenAICompatibleCatalogProvider(config).build_request_body("invented", {})

    def test_invalid_envelopes_refusal_truncation_and_json_are_rejected(self):
        invalid = [None, {}, {"choices": []}, {"choices": [{"message": None}]},
            {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"refusal": "SECRET", "content": None}}]},
        ]
        for content in (None, "", " ", "not-json", '{"facts":[],"facts":[]}', '{"facts":NaN}'):
            invalid.append({"choices": [{"finish_reason": "stop", "message": {"content": content}}]})
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(CatalogInferenceError) as caught:
                OpenAICompatibleCatalogProvider._parse_response(response)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_http_failures_are_sanitized_and_never_retried(self):
        for status in (400, 401, 429, 503):
            raw = json.dumps({"error": {"code": "invalid_request", "message": "PRIVATE-KEY"}}).encode()
            failure = urllib.error.HTTPError("https://secret.invalid/PRIVATE-KEY", status, "PRIVATE-KEY", {}, io.BytesIO(raw))
            opener = Mock()
            opener.open.side_effect = failure
            with patch("determinism.modules.semantic_catalog.adapters.openai_compatible.urllib.request.build_opener", return_value=opener):
                with self.assertRaises(CatalogInferenceError) as caught:
                    OpenAICompatibleCatalogProvider(CatalogConfig(), "PRIVATE-KEY").extract({"content": "texto"})
            self.assertIn(f"HTTP {status}", str(caught.exception))
            self.assertNotIn("PRIVATE-KEY", str(caught.exception))
            self.assertEqual(opener.open.call_count, 1)

    def test_transport_invalid_json_and_response_limit(self):
        for outcome in (OSError("PRIVATE-KEY"), io.BytesIO(b"not-json"), io.BytesIO(b'{"choices":[],"choices":[]}')):
            opener = Mock()
            if isinstance(outcome, Exception):
                opener.open.side_effect = outcome
            else:
                opener.open.return_value = outcome
            with patch("determinism.modules.semantic_catalog.adapters.openai_compatible.urllib.request.build_opener", return_value=opener):
                with self.assertRaises(CatalogInferenceError) as caught:
                    OpenAICompatibleCatalogProvider(CatalogConfig(), "PRIVATE-KEY").extract({"content": "texto"})
            self.assertNotIn("PRIVATE-KEY", str(caught.exception))
        opener = Mock()
        opener.open.return_value = io.BytesIO(b"123456789")
        with patch("determinism.modules.semantic_catalog.adapters.openai_compatible.urllib.request.build_opener", return_value=opener), patch("determinism.modules.semantic_catalog.adapters.openai_compatible.MAX_RESPONSE_BYTES", 4):
            with self.assertRaisesRegex(CatalogInferenceError, "excedeu"):
                OpenAICompatibleCatalogProvider(CatalogConfig(), "safe-key").extract({"content": "texto"})


class CatalogConfigurationTests(unittest.TestCase):
    def test_defaults_local_server_and_environment_precedence(self):
        self.assertEqual(load_catalog_config(), CatalogConfig())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('provider="openai-compatible"\nmodel="offline"\nbase_url="http://localhost:1234/v1"\nreasoning_effort="omit"\ncomparison_batch_size=3\nassessment_batch_size=2\ncache_revision="weights-v2"', encoding="utf-8")
            config = load_catalog_config(path)
            self.assertEqual((config.comparison_batch_size, config.assessment_batch_size), (3, 2))
            self.assertEqual(config.cache_revision, "weights-v2")
            self.assertIsNone(read_api_key(config, None, environ={}))
            path.write_text('OPENAI_API_KEY="file-key"', encoding="utf-8")
            self.assertEqual(read_api_key(config, path, environ={"OPENAI_API_KEY": "environment-key"}), "environment-key")
            self.assertEqual(read_api_key(config, path, environ={}), "file-key")

    def test_invalid_config_and_unknown_secret_field_are_rejected(self):
        for changes in (
            {"comparison_batch_size": 0}, {"comparison_batch_size": True}, {"assessment_batch_size": 257},
            {"cache_revision": ""}, {"cache_revision": "bad\nrevision"}, {"cache_revision": 1},
            {"model": "invalid model"}, {"base_url": "https://user:password@api.openai.com/v1"},
            {"base_url": "https://api.openai.com:8443/v1"}, {"timeout_seconds": float("nan")},
            {"max_concurrency": 0}, {"seed": True}, {"reasoning_effort": "omit"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                CatalogConfig(**changes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text('api_key="PRIVATE-KEY"', encoding="utf-8")
            with self.assertRaises(ConfigurationError) as caught:
                load_catalog_config(path)
            self.assertNotIn("PRIVATE-KEY", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
