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
from determinism.modules.semantic_catalog.adapters.prompts import ALL_OPERATIONS, OPERATIONS, structured_output
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
    def test_id_maps_are_required_by_schema_and_normalized_back_to_port_lists(self):
        pairs = [{"pair_id": "p_a", "left": {}, "right": {}}, {"pair_id": "p_b", "left": {}, "right": {}}]
        segment = {"id": "s0.seg0", "content": "A ponte abriu hoje."}
        facts = [
            {"id": "f_a", "statement": "A ponte abriu.", "requirements": [{"id": "r_a"}, {"id": "r_b"}]},
            {"id": "f_b", "statement": "A ponte abriu hoje.", "requirements": [{"id": "r_c"}]},
        ]
        original = copy.deepcopy((pairs, segment, facts))
        with local_server() as (url, requests, outputs):
            outputs["catalog_compare_v1"] = {"comparisons": {
                "p_a": {"relation": "equivalent", "reason": "Mesma informação."},
                "p_b": {"relation": "different", "reason": "Detalhe adicional."},
            }}
            outputs["catalog_assess_requirements_v1"] = {"assessments": {
                "f_a": {"reference_status": "compatible", "requirements": {
                    "r_a": {"status": "entailed", "evidence_quotes": ["A ponte"]},
                    "r_b": {"status": "entailed", "evidence_quotes": ["abriu"]},
                }},
                "f_b": {"reference_status": "ambiguous", "requirements": {
                    "r_c": {"status": "absent", "evidence_quotes": []},
                }},
            }}
            provider = OpenAICompatibleCatalogProvider(CatalogConfig(provider="openai-compatible", base_url=url), "local-secret")
            comparison = provider.compare(pairs)
            assessment = provider.assess_requirements(segment, facts)
        self.assertEqual((pairs, segment, facts), original)
        self.assertEqual(comparison, CatalogResponse({"comparisons": [
            {"pair_id": "p_a", "relation": "equivalent", "reason": "Mesma informação."},
            {"pair_id": "p_b", "relation": "different", "reason": "Detalhe adicional."},
        ]}, "local-test", "fp_test"))
        self.assertEqual(assessment.payload, {"assessments": [
            {"fact_id": "f_a", "reference_status": "compatible", "requirements": [
                {"requirement_id": "r_a", "status": "entailed", "evidence_quotes": ["A ponte"]},
                {"requirement_id": "r_b", "status": "entailed", "evidence_quotes": ["abriu"]},
            ]},
            {"fact_id": "f_b", "reference_status": "ambiguous", "requirements": [
                {"requirement_id": "r_c", "status": "absent", "evidence_quotes": []},
            ]},
        ]})
        pair_schema = requests[0][2]["response_format"]["json_schema"]["schema"]
        mapping = pair_schema["properties"]["comparisons"]
        self.assertEqual(mapping["required"], ["p_a", "p_b"])
        self.assertEqual(set(mapping["properties"]), {"p_a", "p_b"})
        self.assertFalse(mapping["additionalProperties"])
        self.assertEqual(pair_schema["$defs"]["comparison"]["required"], ["relation", "reason"])
        facts_schema = requests[1][2]["response_format"]["json_schema"]["schema"]["properties"]["assessments"]
        self.assertEqual(facts_schema["required"], ["f_a", "f_b"])
        self.assertFalse(facts_schema["additionalProperties"])
        for fact_id, expected in (("f_a", ["r_a", "r_b"]), ("f_b", ["r_c"])):
            required = facts_schema["properties"][fact_id]["properties"]["requirements"]
            self.assertEqual(required["required"], expected)
            self.assertEqual(set(required["properties"]), set(expected))
            self.assertFalse(required["additionalProperties"])

    def test_dynamic_id_maps_do_not_mutate_shared_template_or_other_requests(self):
        from determinism.modules.semantic_catalog.adapters import openai_compatible

        provider = OpenAICompatibleCatalogProvider(CatalogConfig(), "local-secret")
        for operation, data, changed_data, key in (
            ("compare", {"pairs": [{"pair_id": "p_a"}]}, {"pairs": [{"pair_id": "p_b"}]}, "comparisons"),
            ("assess_requirements", {"facts": [{"id": "f_a", "requirements": [{"id": "r_a"}]}]},
             {"facts": [{"id": "f_b", "requirements": [{"id": "r_b"}]}]}, "assessments"),
        ):
            with self.subTest(operation=operation):
                template = structured_output(operation)
                saved = copy.deepcopy(template)
                with patch.object(openai_compatible, "structured_output", return_value=template):
                    first = provider.build_request_body(operation, data)["response_format"]["json_schema"]["schema"]
                    second = provider.build_request_body(operation, changed_data)["response_format"]["json_schema"]["schema"]
                self.assertEqual(template, saved)
                first["properties"][key]["required"].append("mutated")
                self.assertNotIn("mutated", second["properties"][key]["required"])
                self.assertEqual(template, saved)

    def test_dynamic_id_maps_reject_missing_or_duplicate_input_ids_before_http(self):
        provider = OpenAICompatibleCatalogProvider(CatalogConfig(), "local-secret")
        cases = [
            ("compare", []), ("compare", [{"pair_id": "p"}, {"pair_id": "p"}]),
            ("compare", [{}]), ("assess_requirements", []),
            ("assess_requirements", [{"id": "f", "requirements": []}]),
            ("assess_requirements", [{"id": "f", "requirements": [{"id": "r"}, {"id": "r"}]}]),
        ]
        for operation, items in cases:
            with self.subTest(operation=operation, items=items), patch("determinism.modules.semantic_catalog.adapters.openai_compatible.urllib.request.build_opener") as opener:
                with self.assertRaises(CatalogInferenceError):
                    provider.compare(items) if operation == "compare" else provider.assess_requirements({}, items)
                opener.assert_not_called()

    def test_map_normalization_does_not_drop_unexpected_fields_or_repair_omissions(self):
        from determinism.modules.semantic_catalog.adapters.openai_compatible import _normalize_wire_response

        for operation, payload in (
            ("compare", {"comparisons": {"p": {"relation": "different", "reason": "Fato diferente", "pair_id": "spoofed"}}}),
            ("assess_requirements", {"assessments": {"f": {"reference_status": "compatible", "requirements": {}, "extra": True}}}),
            ("assess_requirements", {"assessments": {"f": {"reference_status": "compatible", "requirements": {"r": {"status": "entailed", "evidence_quotes": [], "extra": True}}}}}),
        ):
            with self.subTest(operation=operation), self.assertRaises(CatalogInferenceError):
                _normalize_wire_response(operation, CatalogResponse(payload))
        empty = _normalize_wire_response("compare", CatalogResponse({"comparisons": {}}))
        self.assertEqual(empty.payload, {"comparisons": []})
        legacy = CatalogResponse({"comparisons": [{"pair_id": "p", "relation": "different", "reason": "Fato diferente"}]})
        self.assertIs(_normalize_wire_response("compare", legacy), legacy)

    def test_requirements_operations_use_distinct_schemas_without_rewriting_input(self):
        candidate = {
            "id": "candidate", "statement": "A ponte abriu hoje.",
            "statement_tokens": [{"id": 0, "text": "A", "start": 0, "end": 1},
                                 {"id": 1, "text": "ponte", "start": 2, "end": 7},
                                 {"id": 2, "text": "abriu", "start": 8, "end": 13},
                                 {"id": 3, "text": "hoje", "start": 14, "end": 18}],
        }
        segment = {"id": "s0.seg0", "content": "A ponte abriu hoje.", "metadata": "UNTRUSTED-INSTRUCTION"}
        facts = [{"id": "fact", "statement": candidate["statement"], "requirements": [
            {"id": "requirement", "kind": "time", "token_ids": [2, 3], "text": "abriu hoje"}
        ]}]
        original = copy.deepcopy((candidate, segment, facts))
        with local_server() as (url, requests, outputs):
            outputs["catalog_requirements_v1"] = {"requirements_by_token": {
                "t0": [{"kind": "predicate", "group": 0}],
                "t1": [{"kind": "predicate", "group": 0}],
                "t2": [{"kind": "predicate", "group": 0}, {"kind": "time", "group": 1}],
                "t3": [{"kind": "time", "group": 1}],
            }}
            outputs["catalog_assess_requirements_v1"] = {"assessments": [
                {"fact_id": "fact", "reference_status": "compatible", "requirements": [
                    {"requirement_id": "requirement", "status": "entailed", "evidence_quotes": ["abriu hoje"]}
                ]}
            ]}
            provider = OpenAICompatibleCatalogProvider(CatalogConfig(provider="openai-compatible", base_url=url), "local-secret")
            replies = [provider.requirements(candidate), provider.assess_requirements(segment, facts)]
        self.assertEqual((candidate, segment, facts), original)
        self.assertEqual(len(requests), 2)
        for (_, auth, body), operation, reply in zip(requests, ("requirements", "assess_requirements"), replies):
            self.assertEqual(auth, "Bearer local-secret")
            self.assertEqual(body["response_format"]["json_schema"]["name"], f"catalog_{operation}_v1")
            self.assertEqual(reply.payload, outputs[f"catalog_{operation}_v1"])
            self.assertNotIn("UNTRUSTED-INSTRUCTION", body["messages"][0]["content"])
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["seed"], 42)
            self.assertNotIn("method", body)
        self.assertEqual(json.loads(requests[0][2]["messages"][1]["content"]), {"candidate": candidate})
        self.assertEqual(json.loads(requests[1][2]["messages"][1]["content"]), {"segment": segment, "facts": facts})

    def test_requirement_wire_schema_requires_every_token_and_reuses_assignment_definition(self):
        from determinism.modules.semantic_catalog.adapters import openai_compatible
        from determinism.modules.semantic_catalog.domain.requirements import statement_tokens, REQUIREMENT_KINDS

        provider = OpenAICompatibleCatalogProvider(CatalogConfig(), "local-secret")
        template = structured_output("requirements")
        snapshot = copy.deepcopy(template)
        schemas = []
        with patch.object(openai_compatible, "structured_output", return_value=template):
            for statement in ("Ponte.", "A ponte abriu hoje."):
                tokens = statement_tokens(statement)
                schema = provider.build_request_body("requirements", {"candidate": {"statement": statement, "statement_tokens": tokens}})["response_format"]["json_schema"]["schema"]
                schemas.append(schema)
                self.assertEqual(set(schema["properties"]), {"requirements_by_token"})
                mapping = schema["properties"]["requirements_by_token"]
                keys = [f"t{i}" for i in range(len(tokens))]
                self.assertEqual(list(mapping["properties"]), keys)
                self.assertEqual(mapping["required"], keys)
                self.assertFalse(mapping["additionalProperties"])
                definition = schema["$defs"]["requirement_assignment"]
                self.assertEqual(set(definition["properties"]["kind"]["enum"]), REQUIREMENT_KINDS)
                self.assertEqual(definition["properties"]["group"], {"type": "integer", "minimum": 0, "maximum": len(tokens) - 1})
                for token_schema in mapping["properties"].values():
                    self.assertEqual(token_schema["minItems"], 1)
                    self.assertEqual(token_schema["items"], {"$ref": "#/$defs/requirement_assignment"})
                    self.assertNotIn("enum", json.dumps(token_schema))
        self.assertEqual(template, snapshot)
        self.assertEqual(schemas[0]["$defs"]["requirement_assignment"]["properties"]["group"]["maximum"], 0)
        schemas[1]["properties"]["requirements_by_token"]["properties"]["t0"]["minItems"] = 99
        self.assertEqual(schemas[1]["properties"]["requirements_by_token"]["properties"]["t1"]["minItems"], 1)
        self.assertEqual(template, snapshot)

    def test_requirement_schema_rejects_empty_or_malformed_token_inventory_before_http(self):
        provider = OpenAICompatibleCatalogProvider(CatalogConfig(), "local-secret")
        for tokens in (None, [], {}, [{"id": 1}], [{"id": True}], [{"id": 0}, {"id": 0}], [None]):
            with self.subTest(tokens=tokens), patch("determinism.modules.semantic_catalog.adapters.openai_compatible.urllib.request.build_opener") as opener:
                with self.assertRaisesRegex(CatalogInferenceError, "statement_tokens"):
                    provider.requirements({"statement": "A ponte abriu.", "statement_tokens": tokens})
                opener.assert_not_called()

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
        for operation in ALL_OPERATIONS:
            verify(structured_output(operation)["schema"])
        self.assertEqual(len(contract_fingerprint()), 64)
        self.assertEqual(contract_fingerprint(), contract_fingerprint())

    def test_new_prompt_or_schema_changes_invalidate_contract_fingerprint(self):
        from determinism.modules.semantic_catalog.adapters import prompts

        original = prompts._asset
        baseline = contract_fingerprint()
        for operation in ("requirements", "assess_requirements"):
            for suffix in ("system", "schema"):
                with self.subTest(operation=operation, suffix=suffix):
                    def changed(requested_operation, requested_suffix):
                        text = original(requested_operation, requested_suffix)
                        return text + "\n " if (requested_operation, requested_suffix) == (operation, suffix) else text
                    with patch.object(prompts, "_asset", side_effect=changed):
                        self.assertNotEqual(contract_fingerprint(), baseline)
        with patch.object(prompts, "DYNAMIC_CONTRACT_REVISION", "different-dynamic-schema"):
            self.assertNotEqual(contract_fingerprint(), baseline)

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
        self.assertEqual(CatalogConfig().method, "requirements")
        self.assertEqual(CatalogConfig().assessment_batch_size, 1)
        self.assertEqual(CatalogConfig(method="legacy").method, "legacy")
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
            {"method": "automatic"}, {"method": ""}, {"method": None}, {"method": []},
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
