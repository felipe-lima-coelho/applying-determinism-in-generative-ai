import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiment_results.comparacao_luna_terra.compare_models import (
    EXPECTED_ANALYSIS_STAGE,
    _validate_cohort_configuration,
    _validate_run_binding,
    _validate_shared_manifest,
    _verify_embedding_trace,
)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


class CompareModelsIntegrityTests(unittest.TestCase):
    def test_run_binding_rejects_stale_global_catalog_hash(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run_001" / "attempt_01"
            run_dir.mkdir(parents=True)
            raw = b'{"source":"raw"}'
            catalog = b'{"canonical_information":[]}'
            (run_dir / "raw.json").write_bytes(raw)
            (run_dir / "global.json").write_bytes(catalog)
            run = {
                "run": 1,
                "attempt": 1,
                "directory": "run_001/attempt_01",
                "artifact_sha256": {"raw.json": sha256(raw), "global.json": "0" * 64},
            }
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
            manifest = {
                "specification": {"input_sha256": sha256(raw)},
                "attempts": [run],
            }

            with self.assertRaisesRegex(ValueError, "artifact hash mismatch: global.json"):
                _validate_run_binding(root, manifest, run)

    def test_configuration_guard_rejects_wrong_model_and_nonmodel_settings(self):
        config = {
            "extraction": {
                "model": "gpt-6.2-sol",
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "max_concurrency": 4,
                "temperature": 0.0,
                "seed": 42,
                "reasoning_effort": "none",
                "timeout_seconds": 60.0,
                "max_completion_tokens": 4096,
                "api_key_env": "OPENAI_API_KEY",
                "max_generations_per_segment": 20,
            },
            "grouping": {"model": "text-embedding-3-small", "threshold": 0.85},
        }
        cohort = {
            "configuration": config,
            "manifest": {"protocol": {}},
            "runs": [],
        }

        with self.assertRaisesRegex(ValueError, "Expected extraction model"):
            _validate_cohort_configuration(cohort, "gpt-6-luna")

        config["extraction"]["model"] = "gpt-6-luna"
        config["extraction"]["temperature"] = 0.7
        with self.assertRaisesRegex(ValueError, "temperature"):
            _validate_cohort_configuration(cohort, "gpt-6-luna")

    def test_shared_embedding_trace_rejects_artifact_vector_not_in_hashed_response(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            api_dir = root / "api"
            api_dir.mkdir()
            profile = {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "base_url": "https://api.openai.com/v1",
                "dimensions": None,
                "encoding_format": "float",
                "cache_revision": "audit-test",
            }
            context = {"stage": EXPECTED_ANALYSIS_STAGE, "batch_index": 1, "input_count": 1, "unique_input_count": 1}
            request = {"model": profile["model"], "input": ["fact"], "encoding_format": "float"}
            response = {"model": profile["model"], "data": [{"index": 0, "embedding": [1.0, 0.0]}]}
            request_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8")
            response_bytes = json.dumps(response, ensure_ascii=False).encode("utf-8")
            (api_dir / "call_0001.request.json").write_bytes(request_bytes)
            (api_dir / "call_0001.response.json").write_bytes(response_bytes)
            events = [
                {
                    "event": "start", "call_id": 1, "url": profile["base_url"] + "/embeddings",
                    "request_sha256": sha256(request_bytes), "context": context, "retries": 0,
                },
                {
                    "event": "finish", "call_id": 1, "context": context, "status": "received",
                    "response_sha256": sha256(response_bytes), "model": profile["model"], "retries": 0,
                },
            ]
            (api_dir / "calls.jsonl").write_text(
                "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n", encoding="utf-8",
            )
            embeddings_path = root / "embeddings.json"
            embeddings_path.write_text(json.dumps({
                "texts": ["fact"], "vectors": [[0.0, 1.0]], "profile": profile,
            }), encoding="utf-8")
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "attempted_calls": 1,
                "finished_calls": 1,
                "failed_http_or_transport_calls": 0,
                "calls_by_stage": {EXPECTED_ANALYSIS_STAGE: 1},
                "retries": 0,
                "effective_models": [profile["model"]],
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match its hashed raw API response"):
                _verify_embedding_trace(embeddings_path, audit_path, {"fact"}, profile)

    def test_shared_manifest_rejects_stale_embedding_artifact_hash(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            profile = {
                "provider": "openai", "model": "text-embedding-3-small",
                "base_url": "https://api.openai.com/v1", "dimensions": None,
                "encoding_format": "float", "cache_revision": "audit-test",
            }
            cohorts = {}
            entries = {}
            for key in ("luna", "terra"):
                cohort_root = root / key
                cohort_root.mkdir()
                config_path = cohort_root / "configuration.json"
                config_path.write_text("{}", encoding="utf-8")
                global_path = cohort_root / "global.json"
                global_path.write_text("{}", encoding="utf-8")
                run = {
                    "run": 1, "attempt": 1, "directory": ".",
                    "artifact_sha256": {"global.json": sha256(global_path.read_bytes())},
                }
                manifest = {
                    "specification": {
                        "configuration_sha256": "config-spec-hash",
                        "input_sha256": "input-hash",
                        "code_sha256": "code-hash",
                    },
                }
                manifest_path = cohort_root / "experiment.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                cohorts[key] = {
                    "root": cohort_root,
                    "manifest_path": manifest_path,
                    "config_path": config_path,
                    "manifest": manifest,
                    "runs": [run],
                }
                entries[key] = {
                    "root": str(cohort_root.resolve()),
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": sha256(manifest_path.read_bytes()),
                    "configuration_path": str(config_path.resolve()),
                    "configuration_sha256": sha256(config_path.read_bytes()),
                    "specification_configuration_sha256": "config-spec-hash",
                    "input_sha256": "input-hash",
                    "code_sha256": "code-hash",
                    "run_count": 1,
                    "catalogs": [{
                        "run": 1, "attempt": 1, "path": str(global_path.resolve()),
                        "sha256": sha256(global_path.read_bytes()),
                    }],
                }

            shared_root = root / "shared"
            shared_root.mkdir()
            embeddings_path = shared_root / "embeddings.json"
            embeddings_path.write_text(json.dumps({"texts": ["fact"], "vectors": [[1, 0]], "profile": profile}), encoding="utf-8")
            audit_path = shared_root / "audit.json"
            audit_path.write_text("{}", encoding="utf-8")
            api_dir = shared_root / "api"
            api_dir.mkdir()
            trace_path = api_dir / "calls.jsonl"
            trace_path.write_text("{}\n", encoding="utf-8")
            shared_manifest_path = shared_root / "manifest.json"
            shared_manifest = {
                "schema_version": 1,
                "cohorts": entries,
                "shared": {
                    "embeddings_path": str(embeddings_path.resolve()),
                    "embeddings_sha256": sha256(embeddings_path.read_bytes()),
                    "audit_path": str(audit_path.resolve()),
                    "audit_sha256": sha256(audit_path.read_bytes()),
                    "trace_path": str(trace_path.resolve()),
                    "trace_sha256": sha256(trace_path.read_bytes()),
                    "stage": EXPECTED_ANALYSIS_STAGE,
                    "text_count": 1,
                    "profile": profile,
                },
            }
            shared_manifest_path.write_text(json.dumps(shared_manifest), encoding="utf-8")

            embeddings_path.write_text(json.dumps({"texts": ["fact"], "vectors": [[0, 1]], "profile": profile}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hashes/profile/count"):
                _validate_shared_manifest(
                    shared_manifest_path, cohorts["luna"], cohorts["terra"],
                    embeddings_path, audit_path, 1, profile,
                )


if __name__ == "__main__":
    unittest.main()
