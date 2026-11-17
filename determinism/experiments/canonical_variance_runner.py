"""Audited, uncached repetitions of map -> extract -> local/global grouping.

Run as a module. Artifacts must live outside the repository. A bounded first
invocation (``--max-new-runs 1``) permits checking the pilot before ``--resume``.
Failures remain in distinct attempt directories. Bounded retries repeat the
entire pipeline; individual API requests have no retry or inference fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .semantic_extraction import extract_dataset
from .semantic_grouping import build_global_catalog, group_dataset
from .word_mapping import run_mapping
from ..modules.semantic_extraction.adapters import OpenAICompatibleProvider, load_extraction_config, read_api_key
from ..modules.semantic_extraction.adapters.openai_compatible import _decode_json
from ..modules.semantic_extraction.domain import IncompleteCoverageError, SegmentInput
from ..modules.semantic_grouping.adapters import OpenAIEmbeddingProvider, load_grouping_config, read_embedding_api_key
from ..modules.semantic_grouping.adapters.postgres_vectors import canonical_float32
from ..modules.semantic_grouping.domain import EmbeddingProfile, Vector, validate_vector
from ..modules.text_mapping.adapters import JsonDatasetRepository


class ExperimentError(ValueError):
    """An experimental invariant or an explicitly audited API request failed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


@contextmanager
def _time_stage(result: dict[str, Any], stage: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        result["stages_seconds"][stage] = time.perf_counter() - start


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _named_usage_values(value: Any, name: str) -> list[int]:
    if not isinstance(value, dict):
        return []
    result = []
    for key, item in value.items():
        if key == name and type(item) is int:
            result.append(item)
        elif isinstance(item, dict):
            result.extend(_named_usage_values(item, name))
    return result


def _flatten_usage(value: Any, prefix: str = "") -> dict[str, int]:
    result = {}
    if isinstance(value, dict):
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else name
            if type(item) is int:
                result[key] = item
            elif isinstance(item, dict):
                result.update(_flatten_usage(item, key))
    return result


class AuditJournal:
    """Thread-safe request/response artifacts, without authorization headers.

    Request bodies contain experimental inputs, never the API key. Error bodies
    and arbitrary transport exception messages are deliberately not persisted.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        saved = directory / "calls.jsonl"
        self._events = [json.loads(line) for line in saved.read_text(encoding="utf-8").splitlines()] if saved.exists() else []
        self._next = max((event["call_id"] for event in self._events), default=0)

    def _event(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(value)
            with (self.directory / "calls.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def post(self, *, url: str, body: dict[str, Any], api_key: str | None,
             timeout: float, max_bytes: int, context: dict[str, Any]) -> Any:
        raw_request = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        with self._lock:
            self._next += 1
            call_id = self._next
        stem = f"call_{call_id:04d}"
        (self.directory / f"{stem}.request.json").write_bytes(raw_request)
        self._event({"event": "start", "call_id": call_id, "started_at": _now(),
                     "url": url, "request_sha256": _hash(raw_request), "context": context,
                     "retries": 0})
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=raw_request, headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
                raw = response.read(max_bytes + 1)
                status = response.status
                request_id = response.headers.get("x-request-id")
            if len(raw) > max_bytes:
                raise ExperimentError("API response exceeded the configured size limit.")
            (self.directory / f"{stem}.response.json").write_bytes(raw)
            envelope = _decode_json(raw)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            self._event({"event": "finish", "call_id": call_id, "context": context,
                         "finished_at": _now(), "duration_seconds": time.perf_counter() - start,
                         "status": "http_error", "http_status": status, "retries": 0})
            raise ExperimentError(f"API request failed: HTTP {status}; no retry or fallback.") from None
        except (OSError, ValueError) as error:
            self._event({"event": "finish", "call_id": call_id, "context": context,
                         "finished_at": _now(), "duration_seconds": time.perf_counter() - start,
                         "status": "transport_or_decode_error", "error_type": type(error).__name__, "retries": 0})
            raise ExperimentError("API transport/JSON failure; no retry or fallback. See the sanitized audit.") from None
        self._event({"event": "finish", "call_id": call_id, "context": context,
                     "finished_at": _now(), "duration_seconds": time.perf_counter() - start,
                     "status": "received", "http_status": status, "request_id": request_id,
                     "response_sha256": _hash(raw), "response_bytes": len(raw),
                     "model": envelope.get("model") if isinstance(envelope, dict) else None,
                     "system_fingerprint": envelope.get("system_fingerprint") if isinstance(envelope, dict) else None,
                     "usage": envelope.get("usage") if isinstance(envelope, dict) else None, "retries": 0})
        return envelope

    def summary(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        starts = [event for event in events if event["event"] == "start"]
        finished = [event for event in events if event["event"] == "finish"]
        totals: Counter[str] = Counter()
        stages: Counter[str] = Counter()
        cache_read, cache_write = [], []
        extraction_calls = 0
        for event in starts:
            stages[event["context"]["stage"]] += 1
        for event in finished:
            totals.update(_flatten_usage(event.get("usage")))
            if event["context"]["stage"] == "extraction":
                extraction_calls += 1
                cache_read.extend(_named_usage_values(event.get("usage"), "cached_tokens"))
                cache_write.extend(_named_usage_values(event.get("usage"), "cache_write_tokens"))
        return {"attempted_calls": len(starts), "finished_calls": len(finished),
                "failed_http_or_transport_calls": sum(event["status"] != "received" for event in finished),
                "calls_by_stage": dict(stages), "usage_totals": dict(totals), "retries": 0,
                "effective_models": sorted({event["model"] for event in finished if isinstance(event.get("model"), str)}),
                "system_fingerprints": sorted({event["system_fingerprint"] for event in finished if isinstance(event.get("system_fingerprint"), str)}),
                "prompt_cache": {"requested_mode": "explicit_without_breakpoints", "extraction_calls": extraction_calls,
                                 "read_fields_reported": len(cache_read), "write_fields_reported": len(cache_write),
                                 "reported_cached_tokens": sum(cache_read) if cache_read else None,
                                 "reported_cache_write_tokens": sum(cache_write) if cache_write else None,
                                 "zero_reads_verified_for_all_calls": len(cache_read) == extraction_calls and bool(cache_read) and not any(cache_read),
                                 "zero_writes_verified_for_all_calls": len(cache_write) == extraction_calls and bool(cache_write) and not any(cache_write)}}


class AuditedExtractionProvider:
    def __init__(self, config, api_key: str | None, journal: AuditJournal):
        self.config, self.api_key, self.journal = config, api_key, journal
        self.adapter = OpenAICompatibleProvider(config, api_key)

    def extract(self, segment: SegmentInput):
        body = self.adapter.build_request_body(segment)
        # Supported by GPT-5.6+ Chat Completions. No prompt content or inference
        # parameter changes, and no breakpoints are supplied anywhere.
        body["prompt_cache_options"] = {"mode": "explicit"}
        envelope = self.journal.post(
            url=self.config.base_url.rstrip("/") + "/chat/completions", body=body,
            api_key=self.api_key, timeout=self.config.timeout_seconds, max_bytes=8 * 1024 * 1024,
            context={"stage": "extraction", "source_index": segment.source_index,
                     "segment_index": segment.segment_index,
                     "generation": segment.coverage.generation if segment.coverage else None},
        )
        usage = envelope.get("usage") if isinstance(envelope, dict) else None
        if any(value != 0 for name in ("cached_tokens", "cache_write_tokens") for value in _named_usage_values(usage, name)):
            raise ExperimentError("Provider reported cache reads/writes despite explicit mode without breakpoints.")
        return self.adapter._parse_response(envelope)


class AuditedEmbeddingProvider:
    def __init__(self, config, api_key: str | None, journal: AuditJournal, stage: str):
        self.config, self.api_key, self.journal, self.stage = config, api_key, journal, stage
        self.adapter = OpenAIEmbeddingProvider(config, api_key)

    def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        result = []
        dimension = self.config.dimensions
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start:start + self.config.batch_size])
            body: dict[str, Any] = {"model": self.config.model, "input": batch, "encoding_format": "float"}
            if self.config.dimensions is not None:
                body["dimensions"] = self.config.dimensions
            envelope = self.journal.post(
                url=self.config.base_url.rstrip("/") + "/embeddings", body=body,
                api_key=self.api_key, timeout=self.config.timeout_seconds, max_bytes=64 * 1024 * 1024,
                context={"stage": self.stage, "batch_index": start // self.config.batch_size + 1,
                         "input_count": len(batch), "unique_input_count": len(set(batch))},
            )
            vectors = self.adapter._parse_response(envelope, len(batch), dimension)
            dimension = dimension or len(vectors[0])
            result.extend(vectors)
        return tuple(result)


class LocalVectorHandoff:
    """Empty lookup then a single float32 handoff of this local-stage response.

    This satisfies GroupingStage's write/read contract without PostgreSQL or a
    persistent cache. Vectors are consumed once and never exposed to global
    grouping, another repeat, or the final statistical comparison.
    """

    def __init__(self):
        self._state = "empty"
        self._profile_key: str | None = None
        self._vectors: dict[str, Vector] = {}
        self.events: list[dict[str, Any]] = []

    def fetch(self, profile: EmbeddingProfile, texts: Sequence[str]) -> dict[str, Vector]:
        if self._state == "empty":
            self._state = "awaiting_fresh_vectors"
            self._profile_key = profile.key
            self.events.append({"action": "empty_lookup", "requested": len(texts), "returned": 0})
            return {}
        if self._state != "ready" or profile.key != self._profile_key or set(texts) != set(self._vectors):
            raise ExperimentError("Local vector handoff was reused or did not contain exactly this stage's fresh vectors.")
        result = self._vectors
        self._vectors = {}
        self._state = "consumed"
        self.events.append({"action": "consume_fresh_float32_vectors_once", "requested": len(texts), "returned": len(result)})
        return result

    def put(self, profile: EmbeddingProfile, vectors: Mapping[str, Vector]) -> None:
        if self._state != "awaiting_fresh_vectors" or profile.key != self._profile_key:
            raise ExperimentError("Unexpected local vector handoff write.")
        dimension = profile.dimensions
        for text, vector in vectors.items():
            checked = canonical_float32(validate_vector(vector, dimension))
            dimension = dimension or len(checked)
            self._vectors[text] = checked
        self._state = "ready"
        self.events.append({"action": "receive_fresh_api_vectors", "count": len(vectors)})


class FreshGlobalVectors:
    """Every fetch embeds its inputs anew; there is no stored vector state."""

    def __init__(self, provider: AuditedEmbeddingProvider):
        self.provider = provider
        self.events: list[dict[str, Any]] = []

    def fetch(self, profile: EmbeddingProfile, texts: Sequence[str]) -> dict[str, Vector]:
        if profile != self.provider.config.profile:
            raise ExperimentError("Global embedding profile differs from the fixed configuration.")
        generated = self.provider.embed(texts)
        if len(generated) != len(texts):
            raise ExperimentError("Fresh global embeddings did not cover all requested texts.")
        self.events.append({"action": "fresh_global_api_embeddings", "count": len(texts)})
        return {text: canonical_float32(validate_vector(vector, profile.dimensions)) for text, vector in zip(texts, generated)}

    def put(self, profile, vectors):
        raise ExperimentError("Global embeddings cannot be cached.")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {path.name: _hash(path.read_bytes()) for path in sorted(directory.glob("*.json"))}


def execute_run(index: int, root: Path, extraction_config, grouping_config,
                extraction_key: str | None, embedding_key: str | None, expected_input_hash: str,
                *, attempt: int = 1) -> dict[str, Any]:
    directory = root / f"run_{index:03d}" / f"attempt_{attempt:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    journal = AuditJournal(directory / "api")
    result: dict[str, Any] = {"run": index, "attempt": attempt, "directory": directory.relative_to(root).as_posix(),
                              "full_pipeline_retries_for_sample": attempt - 1,
                              "status": "running", "started_at": _now(), "stages_seconds": {}}
    start = time.perf_counter()
    try:
        raw = (root / "raw.json").read_bytes()
        if _hash(raw) != expected_input_hash:
            raise ExperimentError("The fixed raw input snapshot changed.")
        (directory / "raw.json").write_bytes(raw)
        with _time_stage(result, "mapping"):
            mapped = run_mapping(directory / "raw.json", directory / "mapped.json")
        with _time_stage(result, "extraction"):
            extracted = extract_dataset(
                mapped, provider=AuditedExtractionProvider(extraction_config, extraction_key, journal),
                max_concurrency=extraction_config.max_concurrency,
                max_generations_per_segment=extraction_config.max_generations_per_segment,
            )
            JsonDatasetRepository().save(directory / "extracted.json", extracted)
        local_vectors = LocalVectorHandoff()
        with _time_stage(result, "local_grouping"):
            grouped = group_dataset(
                extracted, provider=AuditedEmbeddingProvider(grouping_config, embedding_key, journal, "local_grouping"),
                vector_store=local_vectors, profile=grouping_config.profile, threshold=grouping_config.threshold,
            )
            JsonDatasetRepository().save(directory / "grouped.json", grouped)
        global_vectors = FreshGlobalVectors(AuditedEmbeddingProvider(grouping_config, embedding_key, journal, "global_grouping"))
        with _time_stage(result, "global_grouping"):
            catalog = build_global_catalog(grouped, vector_store=global_vectors, profile=grouping_config.profile,
                                           threshold=grouping_config.threshold)
            catalog["metadata"]["dataset_file"] = "grouped.json"
            _save_json(directory / "global.json", catalog)
        result.update(status="completed", counts=catalog["metadata"]["counts"],
                      lexical_tokens=sum(len(segment["map"]) for source in mapped for segment in source["segments"]),
                      local_vector_handoff=local_vectors.events, global_vector_requests=global_vectors.events,
                      threshold_local=grouping_config.threshold, threshold_global=catalog["metadata"]["threshold"])
    except Exception as error:
        if isinstance(error, IncompleteCoverageError) and error.partial_dataset is not None:
            JsonDatasetRepository().save(directory / "extracted.partial.json", error.partial_dataset)
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
    result.update(finished_at=_now(), total_seconds=time.perf_counter() - start, api=journal.summary(),
                  artifact_sha256=_artifact_hashes(directory))
    _save_json(directory / "run.json", result)
    return result


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {path.relative_to(package).as_posix(): _hash(path.read_bytes())
            for path in sorted(package.rglob("*")) if path.is_file() and path.suffix in (".py", ".txt", ".json", ".toml")}


def prepare_experiment(input_path: Path, root: Path, repeats: int, extraction_config, grouping_config,
                       *, resume: bool = False, max_attempts: int = 3) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    if root.resolve().is_relative_to(repository):
        raise ExperimentError("Experiment artifacts must be outside the repository.")
    if repeats < 2:
        raise ExperimentError("At least two repetitions are required for sample variance.")
    if max_attempts < 1:
        raise ExperimentError("The full-pipeline attempt limit must be positive.")
    if grouping_config.threshold != 0.85:
        raise ExperimentError("This protocol requires threshold 0.85 at both grouping stages.")
    raw = input_path.read_bytes()
    dataset = JsonDatasetRepository().load(input_path)
    if any(set(segment) & {"map", "information", "canonical_information", "set_different_information"}
           for source in dataset for segment in source["segments"]):
        raise ExperimentError("Start from raw transcripts without derived stage fields.")
    configs = {"extraction": asdict(extraction_config), "grouping": asdict(grouping_config)}
    spec = {"input_sha256": _hash(raw), "configuration_sha256": _hash(_json_bytes(configs)),
            "repeats": repeats, "max_attempts_per_sample": max_attempts,
            "code_sha256": _hash(_json_bytes(_code_hashes()))}
    if resume:
        manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        if manifest["specification"] != spec or _hash((root / "raw.json").read_bytes()) != spec["input_sha256"]:
            raise ExperimentError("Input, configuration, code, or repeat count changed since the pilot.")
        for position, attempt in enumerate(manifest["attempts"]):
            if attempt["status"] != "started":
                continue
            directory = root / attempt["directory"]
            completed_result = directory / "run.json"
            if completed_result.exists():
                recovered = json.loads(completed_result.read_text(encoding="utf-8"))
            else:
                directory.mkdir(parents=True, exist_ok=True)
                recovered = {**attempt, "status": "interrupted", "finished_at": _now(),
                             "error_type": "InterruptedAttempt", "error": "Process ended before attempt completion; next attempt repeats all stages.",
                             "api": AuditJournal(directory / "api").summary(), "artifact_sha256": _artifact_hashes(directory)}
                _save_json(completed_result, recovered)
            manifest["attempts"][position] = recovered
            if recovered["status"] == "completed":
                if recovered["run"] != len(manifest["runs"]) + 1:
                    raise ExperimentError("Recovered successful attempts are not in sequential sample order.")
                manifest["runs"].append(recovered)
        # Do not trust a continuation if an earlier result was edited afterwards.
        for run in manifest["attempts"]:
            directory = root / run["directory"]
            for name, expected_hash in run["artifact_sha256"].items():
                if _hash((directory / name).read_bytes()) != expected_hash:
                    raise ExperimentError("An earlier run artifact changed since the pilot.")
        _save_json(root / "experiment.json", manifest)
        return manifest
    root.mkdir(parents=True, exist_ok=False)
    (root / "raw.json").write_bytes(raw)
    _save_json(root / "configuration.json", configs)
    _save_json(root / "code_hashes.json", _code_hashes())
    manifest = {"schema_version": 1, "created_at": _now(), "status": "prepared",
                "input_file": str(input_path.resolve()), "specification": spec,
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "unicode_version": unicodedata.unidata_version, "git": _git_state()},
                "protocol": {"order": ["mapping", "extraction", "local_grouping", "global_grouping"],
                             "sequential_repeats": True, "persisted_cache_reads": 0,
                             "cross_run_vector_reuse": False, "local_to_global_vector_reuse": False,
                             "embedding_float_format": "float32, matching production pgvector",
                             "same_call_exact_text_deduplication": True,
                             "prompt_cache_options": {"mode": "explicit"}, "prompt_cache_breakpoints": [],
                             "embedding_provider_internal_caching": "not controllable or observable through this API",
                             "per_request_retries": 0, "max_full_pipeline_attempts_per_sample": max_attempts,
                             "failed_attempts_retained": True, "analysis_selects": "first_successful_attempt_of_each_sample"},
                "attempts": [], "runs": []}
    _save_json(root / "experiment.json", manifest)
    return manifest


def analyze_experiment(root: Path, manifest: dict[str, Any], grouping_config, embedding_key: str | None) -> dict[str, Any]:
    from .canonical_variance_analysis import analyze_catalogs, write_analysis_artifacts

    if len(manifest["runs"]) != manifest["specification"]["repeats"] or any(run["status"] != "completed" for run in manifest["runs"]):
        raise ExperimentError("All planned repetitions must succeed before analysis.")
    directory = root / "analysis"
    directory.mkdir(exist_ok=False)
    journal = AuditJournal(directory / "api")
    catalogs = [json.loads((root / run["directory"] / "global.json").read_text(encoding="utf-8"))
                for run in manifest["runs"]]
    texts = list(dict.fromkeys(item["info"] for catalog in catalogs for item in catalog["canonical_information"]))
    provider = AuditedEmbeddingProvider(grouping_config, embedding_key, journal, "cross_run_analysis")
    vectors = [canonical_float32(vector) for vector in provider.embed(texts)]
    _save_json(directory / "embeddings.json", {"texts": texts, "vectors": vectors, "profile": grouping_config.profile.metadata})
    summary = analyze_catalogs(catalogs, dict(zip(texts, vectors)), threshold=grouping_config.threshold)
    summary["experiment"] = {"specification": manifest["specification"], "run_api": [run["api"] for run in manifest["runs"]],
                             "analysis_api": journal.summary(), "protocol": manifest["protocol"],
                             "failed_attempts": [run for run in manifest["attempts"] if run["status"] != "completed"]}
    write_analysis_artifacts(summary, directory)
    _save_json(directory / "audit.json", journal.summary())
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--max-attempts", type=int, default=3, help="Full fresh-pipeline attempts per sample; all failures retained.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="Fresh final comparison embeddings and statistical artifacts.")
    parser.add_argument("--extraction-config", type=Path)
    parser.add_argument("--grouping-config", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    try:
        extraction_config = load_extraction_config(args.extraction_config)
        grouping_config = load_grouping_config(args.grouping_config)
        extraction_key = read_api_key(extraction_config, args.env_file)
        embedding_key = read_embedding_api_key(grouping_config, args.env_file)
        manifest = prepare_experiment(args.input, args.output_root, args.repeats, extraction_config, grouping_config,
                                      resume=args.resume, max_attempts=args.max_attempts)
        completed = len(manifest["runs"])
        end = min(args.repeats, completed + (args.max_new_runs or args.repeats))
        for index in range(completed + 1, end + 1):
            previous = [run for run in manifest["attempts"] if run["run"] == index]
            succeeded = False
            for attempt in range(len(previous) + 1, args.max_attempts + 1):
                manifest["attempts"].append({"run": index, "attempt": attempt,
                    "directory": f"run_{index:03d}/attempt_{attempt:02d}", "status": "started",
                    "started_at": _now(), "artifact_sha256": {}})
                _save_json(args.output_root / "experiment.json", manifest)
                run = execute_run(index, args.output_root, extraction_config, grouping_config, extraction_key, embedding_key,
                                  manifest["specification"]["input_sha256"], attempt=attempt)
                manifest["attempts"][-1] = run
                if run["status"] == "completed":
                    manifest["runs"].append(run)
                    succeeded = True
                manifest["status"] = "running" if not succeeded else "completed" if index == args.repeats else "pilot_or_batch_complete"
                manifest["updated_at"] = _now()
                _save_json(args.output_root / "experiment.json", manifest)
                print(json.dumps({"run": index, "attempt": attempt, "status": run["status"], "counts": run.get("counts"),
                                  "api_calls": run["api"]["attempted_calls"], "seconds": round(run["total_seconds"], 3)}, ensure_ascii=False), flush=True)
                if succeeded:
                    break
            if not succeeded:
                manifest["status"] = "failed_attempt_limit"
                _save_json(args.output_root / "experiment.json", manifest)
                print(f"Full-pipeline attempt limit reached; all attempts retained at {args.output_root}", file=sys.stderr)
                return 1
        if args.analyze:
            analyze_experiment(args.output_root, manifest, grouping_config, embedding_key)
            manifest["analysis_completed_at"] = _now()
            _save_json(args.output_root / "experiment.json", manifest)
        print(f"Artifacts: {args.output_root.resolve()}", flush=True)
    except (OSError, ValueError) as error:
        print(f"Experiment error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
