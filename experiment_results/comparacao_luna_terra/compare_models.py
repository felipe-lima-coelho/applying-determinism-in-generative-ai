"""Offline Luna/Terra comparison for two completed 100-run cohorts.

This script reads immutable runner artifacts and a separate fresh shared-union
embedding audit. It never calls model APIs or reads a vector store.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from determinism.experiments.canonical_variance_analysis import (
    _catalog_texts,
    _cosine_matrix,
    _exact_match_count,
    _match_metrics,
    _pairwise_summary,
    _semantic_match_count,
    _validated_vectors,
)
from determinism.modules.semantic_grouping.adapters.postgres_vectors import canonical_float32


EXPECTED_MODELS = {"gpt-6-luna", "gpt-6.1-terra"}
ALLOWED_CODE_HASH_CHANGES = {
    "experiments/configs/semantic_extraction.toml",
    "modules/semantic_extraction/adapters/configuration.py",
}
EXPECTED_EXTRACTION_SETTINGS = {
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
}
EXPECTED_ANALYSIS_STAGE = "luna_terra_shared_analysis"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_run_binding(root: Path, manifest: Mapping[str, Any], run: Mapping[str, Any]) -> Path:
    run_dir = (root / run["directory"]).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("Accepted run directory escapes its experiment root.")
    run_json_path = run_dir / "run.json"
    if _read_json(run_json_path) != run:
        raise ValueError(f"Accepted run {run.get('run')} differs from its retained run.json audit record.")
    attempt = next((
        item for item in manifest.get("attempts", [])
        if item.get("run") == run.get("run") and item.get("attempt") == run.get("attempt")
    ), None)
    if attempt is None or attempt != run:
        raise ValueError(f"Accepted run {run.get('run')} is not bound to its matching manifest attempt.")

    expected_input_hash = manifest["specification"]["input_sha256"]
    run_hashes = run.get("artifact_sha256", {})
    required_artifacts = {"raw.json", "global.json"}
    if not required_artifacts <= set(run_hashes):
        raise ValueError(f"Accepted run {run.get('run')} lacks raw/global artifact hashes.")
    for filename, expected_hash in run_hashes.items():
        artifact = (run_dir / filename).resolve()
        if (not artifact.is_relative_to(run_dir) or not artifact.is_file()
                or _sha256(artifact) != expected_hash):
            raise ValueError(f"Accepted run {run.get('run')} artifact hash mismatch: {filename}.")
    if run_hashes["raw.json"] != expected_input_hash:
        raise ValueError(f"Accepted run {run.get('run')} raw snapshot hash differs from its manifest input.")
    return run_dir


def _validate_cohort_configuration(cohort: Mapping[str, Any], expected_model: str) -> None:
    if expected_model not in EXPECTED_MODELS:
        raise ValueError(f"Unsupported comparison model {expected_model!r}.")
    config = cohort["configuration"]
    extraction = config.get("extraction", {})
    grouping = config.get("grouping", {})
    if extraction.get("model") != expected_model:
        raise ValueError(f"Expected extraction model {expected_model}, got {extraction.get('model')!r}.")
    for key, expected in EXPECTED_EXTRACTION_SETTINGS.items():
        if extraction.get(key) != expected:
            raise ValueError(f"{expected_model} extraction setting {key} must be {expected!r}.")
    if grouping.get("threshold") != 0.85:
        raise ValueError(f"{expected_model} grouping threshold must be 0.85.")
    protocol = cohort["manifest"].get("protocol", {})
    if (protocol.get("persisted_cache_reads") != 0 or protocol.get("cross_run_vector_reuse") is not False
            or protocol.get("local_to_global_vector_reuse") is not False
            or protocol.get("sequential_repeats") is not True
            or protocol.get("order") != ["mapping", "extraction", "local_grouping", "global_grouping"]):
        raise ValueError(f"{expected_model} manifest does not certify fresh, uncached pipeline embeddings.")
    expected_effective = {expected_model, grouping.get("model")}
    for run in cohort["runs"]:
        api = run.get("api", {})
        if api.get("attempted_calls") != api.get("finished_calls") or api.get("failed_http_or_transport_calls") != 0 or api.get("retries") != 0:
            raise ValueError(f"{expected_model} run {run['run']} has incomplete/failed/retried API calls.")
        if set(api.get("effective_models", [])) != expected_effective:
            raise ValueError(f"{expected_model} run {run['run']} has unexpected effective model names.")
        prompt_cache = api.get("prompt_cache", {})
        if prompt_cache.get("zero_reads_verified_for_all_calls") is not True or prompt_cache.get("zero_writes_verified_for_all_calls") is not True:
            raise ValueError(f"{expected_model} run {run['run']} does not verify zero prompt-cache reads/writes.")
        local_events = [event.get("action") for event in run.get("local_vector_handoff", [])]
        if local_events != ["empty_lookup", "receive_fresh_api_vectors", "consume_fresh_float32_vectors_once"]:
            raise ValueError(f"{expected_model} run {run['run']} local embeddings were not freshly handed off once.")
        global_events = run.get("global_vector_requests", [])
        if not global_events or any(event.get("action") != "fresh_global_api_embeddings" for event in global_events):
            raise ValueError(f"{expected_model} run {run['run']} global embeddings were not freshly requested.")


def _verify_run_api_trace(run_dir: Path, run: Mapping[str, Any], configuration: Mapping[str, Any]) -> None:
    trace_dir = run_dir / "api"
    calls_path = trace_dir / "calls.jsonl"
    if not calls_path.is_file():
        raise ValueError(f"Run {run.get('run')} raw API trace calls.jsonl is missing.")
    events = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = {event["call_id"]: event for event in events if event.get("event") == "start"}
    finishes = {event["call_id"]: event for event in events if event.get("event") == "finish"}
    api_summary = run.get("api", {})
    if (not starts or set(starts) != set(finishes)
            or len(starts) != api_summary.get("attempted_calls")
            or len(finishes) != api_summary.get("finished_calls")):
        raise ValueError(f"Run {run.get('run')} API summary is not bound to a complete call trace.")
    if api_summary.get("failed_http_or_transport_calls") != 0 or api_summary.get("retries") != 0:
        raise ValueError(f"Run {run.get('run')} API summary records a failed or retried call.")

    extraction = configuration["extraction"]
    grouping = configuration["grouping"]
    stages = Counter()
    observed_models = set()
    for call_id, start in starts.items():
        finish = finishes[call_id]
        if finish.get("status") != "received" or finish.get("retries") != 0 or finish.get("context") != start.get("context"):
            raise ValueError(f"Run {run.get('run')} API call {call_id} has an invalid response audit.")
        stage = start.get("context", {}).get("stage")
        stages[stage] += 1
        stem = f"call_{call_id:04d}"
        request_path = trace_dir / f"{stem}.request.json"
        response_path = trace_dir / f"{stem}.response.json"
        if (not request_path.is_file() or not response_path.is_file()
                or _sha256(request_path) != start.get("request_sha256")
                or _sha256(response_path) != finish.get("response_sha256")):
            raise ValueError(f"Run {run.get('run')} API call {call_id} request/response hash mismatch.")
        body = json.loads(request_path.read_text(encoding="utf-8"))
        if stage == "extraction":
            expected_url = extraction["base_url"].rstrip("/") + "/chat/completions"
            expected_values = {
                "model": extraction["model"],
                "temperature": 0.0,
                "seed": 42,
                "reasoning_effort": "none",
                "max_completion_tokens": extraction["max_completion_tokens"],
                "store": False,
                "prompt_cache_options": {"mode": "explicit"},
            }
        elif stage in ("local_grouping", "global_grouping"):
            expected_url = grouping["base_url"].rstrip("/") + "/embeddings"
            expected_values = {"model": grouping["model"], "encoding_format": "float"}
            if grouping.get("dimensions") is None:
                if "dimensions" in body:
                    raise ValueError(f"Run {run.get('run')} embedding call {call_id} changed dimensions.")
            elif body.get("dimensions") != grouping["dimensions"]:
                raise ValueError(f"Run {run.get('run')} embedding call {call_id} changed dimensions.")
            if not isinstance(body.get("input"), list) or not body["input"]:
                raise ValueError(f"Run {run.get('run')} embedding call {call_id} has no input texts.")
        else:
            raise ValueError(f"Run {run.get('run')} API call {call_id} has unexpected stage {stage!r}.")
        if start.get("url") != expected_url or any(body.get(key) != value for key, value in expected_values.items()):
            raise ValueError(f"Run {run.get('run')} API call {call_id} used unexpected model parameters.")
        if finish.get("model") != body.get("model"):
            raise ValueError(f"Run {run.get('run')} API call {call_id} returned a different model than requested.")
        observed_models.add(finish.get("model"))

    if dict(stages) != api_summary.get("calls_by_stage"):
        raise ValueError(f"Run {run.get('run')} API stage counts do not match its call trace.")
    expected_models = {extraction["model"], grouping["model"]}
    if observed_models != expected_models or set(api_summary.get("effective_models", [])) != expected_models:
        raise ValueError(f"Run {run.get('run')} observed API models differ from configured models.")


def _load_cohort(label: str, root: Path, expected_model: str, expected_runs: int) -> dict[str, Any]:
    manifest_path = root / "experiment.json"
    config_path = root / "configuration.json"
    hashes_path = root / "code_hashes.json"
    manifest, config, code_hashes = map(_read_json, (manifest_path, config_path, hashes_path))
    if manifest.get("status") != "completed":
        raise ValueError(f"{label} experiment is not completed.")
    specification = manifest.get("specification", {})
    if specification.get("repeats") != expected_runs:
        raise ValueError(f"{label} manifest does not specify exactly {expected_runs} repetitions.")
    if specification.get("configuration_sha256") != _json_hash(config):
        raise ValueError(f"{label} configuration.json does not match its manifest configuration hash.")
    if specification.get("code_sha256") != _json_hash(code_hashes):
        raise ValueError(f"{label} code_hashes.json does not match its manifest code hash.")
    raw_path = root / "raw.json"
    if not raw_path.is_file() or _sha256(raw_path) != specification.get("input_sha256"):
        raise ValueError(f"{label} root raw.json does not match its manifest input hash.")
    accepted = manifest.get("runs")
    if not isinstance(accepted, list) or len(accepted) != expected_runs:
        raise ValueError(f"{label} must have exactly {expected_runs} accepted runs.")
    accepted = sorted(accepted, key=lambda row: row.get("run", -1))
    run_ids = [row.get("run") for row in accepted]
    if run_ids != list(range(1, expected_runs + 1)):
        raise ValueError(f"{label} accepted runs must have unique IDs 1..{expected_runs}.")
    _validate_cohort_configuration({"configuration": config, "manifest": manifest, "runs": accepted}, expected_model)

    catalogs: list[dict[str, Any]] = []
    texts_by_run: list[list[str]] = []
    for run in accepted:
        if run.get("status") != "completed":
            raise ValueError(f"{label} run {run['run']} is not completed.")
        if run.get("threshold_local") != 0.85 or run.get("threshold_global") != 0.85:
            raise ValueError(f"{label} run {run['run']} does not use threshold 0.85 at both grouping stages.")
        directory = _validate_run_binding(root, manifest, run)
        _verify_run_api_trace(directory, run, config)
        catalog_path = directory / "global.json"
        catalog = _read_json(catalog_path)
        canonical = catalog.get("canonical_information")
        metadata = catalog.get("metadata", {})
        count_in_sidecar = metadata.get("counts", {}).get("global_canonical_information")
        if not isinstance(canonical, list) or count_in_sidecar != len(canonical):
            raise ValueError(f"{label} run {run['run']} catalog count does not match its canonical list.")
        run_count = run.get("counts", {}).get("global_canonical_information")
        if run_count != len(canonical):
            raise ValueError(f"{label} run {run['run']} manifest count does not match global.json.")
        if metadata.get("threshold") != 0.85:
            raise ValueError(f"{label} run {run['run']} does not use grouping threshold 0.85.")
        catalogs.append(catalog)
        texts_by_run.append(_catalog_texts(catalog, run["run"]))

    return {
        "label": label,
        "expected_model": expected_model,
        "root": root,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "code_hashes_path": hashes_path,
        "manifest": manifest,
        "configuration": config,
        "code_hashes": code_hashes,
        "runs": accepted,
        "catalogs": catalogs,
        "texts_by_run": texts_by_run,
    }


def _count_summary(texts_by_run: Sequence[Sequence[str]]) -> dict[str, int | float | None]:
    counts = [len(texts) for texts in texts_by_run]
    mean = math.fsum(counts) / len(counts)
    variance = (
        math.fsum((count - mean) ** 2 for count in counts) / (len(counts) - 1)
        if len(counts) > 1 else None
    )
    return {
        "n": len(counts),
        "mean": mean,
        "sample_variance_ddof1": variance,
        "sample_standard_deviation": math.sqrt(variance) if variance is not None else None,
        "min": min(counts),
        "max": max(counts),
        "range": max(counts) - min(counts),
    }


def _exact_frequency(texts_by_run: Sequence[Sequence[str]]) -> dict[str, Any]:
    counts_by_run = [Counter(texts) for texts in texts_by_run]
    all_texts = sorted({text for counts in counts_by_run for text in counts})
    distribution = Counter()
    items = []
    multiset_intersection = 0
    multiset_union = 0
    intersection_distinct = 0
    for text in all_texts:
        occurrences = [counts.get(text, 0) for counts in counts_by_run]
        runs_present = sum(count > 0 for count in occurrences)
        distribution[runs_present] += 1
        multiset_intersection += min(occurrences)
        multiset_union += max(occurrences)
        intersection_distinct += min(occurrences) > 0
        items.append({"info": text, "runs_present": runs_present, "total_occurrences": sum(occurrences)})
    return {
        "comparison": "literal raw string equality; occurrences are a multiset",
        "multiset_intersection_count": multiset_intersection,
        "multiset_union_count": multiset_union,
        "intersection_distinct_text_count": intersection_distinct,
        "union_distinct_text_count": len(all_texts),
        "run_frequency_distribution": [
            {"runs_present": frequency, "distinct_texts": distribution[frequency]}
            for frequency in range(1, len(texts_by_run) + 1)
        ],
        "items": items,
    }


def _pair_row(
    label_a: str,
    run_a: int,
    texts_a: list[str],
    label_b: str,
    run_b: int,
    texts_b: list[str],
    text_index: Mapping[str, int],
    similarities: list[list[float]],
    threshold: float,
) -> dict[str, Any]:
    exact = _match_metrics(_exact_match_count(texts_a, texts_b), len(texts_a), len(texts_b))
    semantic = _match_metrics(
        _semantic_match_count(texts_a, texts_b, text_index, similarities, threshold),
        len(texts_a), len(texts_b),
    )
    return {
        "cohort_a": label_a,
        "run_a": run_a,
        "count_a": len(texts_a),
        "cohort_b": label_b,
        "run_b": run_b,
        "count_b": len(texts_b),
        "exact": exact,
        "semantic": semantic,
    }


def _pair_rows(
    cohort_a: dict[str, Any],
    cohort_b: dict[str, Any],
    text_index: Mapping[str, int],
    similarities: list[list[float]],
    threshold: float,
    *,
    cartesian: bool,
) -> list[dict[str, Any]]:
    rows = []
    for index_a, texts_a in enumerate(cohort_a["texts_by_run"]):
        start = 0 if cartesian else index_a + 1
        for index_b in range(start, len(cohort_b["texts_by_run"])):
            rows.append(_pair_row(
                cohort_a["label"], index_a + 1, texts_a,
                cohort_b["label"], index_b + 1, cohort_b["texts_by_run"][index_b],
                text_index, similarities, threshold,
            ))
    return rows


def _summarize_pairs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for criterion in ("exact", "semantic"):
        base = _pairwise_summary(rows, criterion)
        matched_fractions = []
        unmatched_fractions = []
        for row in rows:
            metric = row[criterion]
            for count, matched, unmatched in (
                (row["count_a"], metric["matched"], metric["unmatched_a"]),
                (row["count_b"], metric["matched"], metric["unmatched_b"]),
            ):
                if count:
                    matched_fractions.append(matched / count)
                    unmatched_fractions.append(unmatched / count)
        base["mean_matched_fraction_per_catalog"] = (
            math.fsum(matched_fractions) / len(matched_fractions) if matched_fractions else None
        )
        base["mean_unmatched_fraction_per_catalog"] = (
            math.fsum(unmatched_fractions) / len(unmatched_fractions) if unmatched_fractions else None
        )
        base["complete_match_rate"] = (
            base["identical_pair_count"] / base["pair_count"] if base["pair_count"] else None
        )
        summaries[criterion] = base
    return {
        "pair_count": len(rows),
        **summaries,
        "total_unmatched_definition": "unmatched_a + unmatched_b = n_a + n_b - 2 * matched",
        "fraction_definition": "Mean of per-catalog matched/count and unmatched/count across the two catalogs in each pair; empty catalogs are omitted from these fractions.",
    }


def _jackknife_pair_mean(rows: Sequence[dict[str, Any]], run_count: int, criterion: str = "semantic") -> dict[str, float | int]:
    if run_count < 3:
        raise ValueError("Jackknife requires at least three runs.")
    values = [row[criterion]["jaccard"] for row in rows]
    estimate = math.fsum(values) / len(values)
    leave_one_out = []
    for run in range(1, run_count + 1):
        remaining = [
            row[criterion]["jaccard"]
            for row in rows
            if row["run_a"] != run and row["run_b"] != run
        ]
        expected = (run_count - 1) * (run_count - 2) // 2
        if len(remaining) != expected:
            raise ValueError("Within-cohort pair rows are incomplete for jackknife.")
        leave_one_out.append(math.fsum(remaining) / len(remaining))
    loo_mean = math.fsum(leave_one_out) / len(leave_one_out)
    variance = (run_count - 1) / run_count * math.fsum((value - loo_mean) ** 2 for value in leave_one_out)
    return {"estimate": estimate, "standard_error": math.sqrt(variance), "n_runs": run_count}


def _primary_outcome(
    luna_rows: Sequence[dict[str, Any]],
    terra_rows: Sequence[dict[str, Any]],
    luna_count: int,
    terra_count: int,
    luna_counts: Mapping[str, Any],
    terra_counts: Mapping[str, Any],
) -> dict[str, Any]:
    luna_estimate = _jackknife_pair_mean(luna_rows, luna_count)
    terra_estimate = _jackknife_pair_mean(terra_rows, terra_count)
    difference = terra_estimate["estimate"] - luna_estimate["estimate"]
    difference_se = math.hypot(luna_estimate["standard_error"], terra_estimate["standard_error"])
    ci = [difference - 1.96 * difference_se, difference + 1.96 * difference_se]
    return {
        "outcome": "mean within-model pairwise semantic Jaccard",
        "difference_direction": "Terra minus Luna; positive favors higher observed semantic repeatability for Terra under this proxy.",
        "luna": luna_estimate,
        "terra": terra_estimate,
        "difference_terra_minus_luna": difference,
        "difference_percentage_points": 100 * difference,
        "approximate_95_percent_jackknife_interval_difference": ci,
        "approximate_95_percent_jackknife_interval_percentage_points": [100 * ci[0], 100 * ci[1]],
        "difference_standard_error": difference_se,
        "jackknife_method": "Delete one run at a time within each cohort; recompute the mean of all remaining unordered run-pair semantic Jaccards. For the difference of independently collected cohorts, combine cohort jackknife variances as sqrt(SE_Terra^2 + SE_Luna^2).",
        "assumptions_and_limitations": "Approximate interval assumes exchangeable independent runs within each cohort and independent cohorts. Runs were sequential, so service/time dependence can violate this assumption; treat the interval as exploratory. Pairwise values are not treated as independent observations.",
        "count_context": {
            "luna_mean_global_canonical_count": luna_counts["mean"],
            "terra_mean_global_canonical_count": terra_counts["mean"],
            "terra_minus_luna_mean_count": terra_counts["mean"] - luna_counts["mean"],
            "note": "Jaccard reflects both item identity and catalog size; count mean, sample SD, and range are reported alongside it to distinguish output volume from repeatability.",
        },
        "accuracy_note": "No ground-truth labeling was used; this analysis does not measure factual accuracy.",
    }


def _effective_model_audit(cohort: dict[str, Any]) -> dict[str, Any]:
    runs = cohort["runs"]
    effective_models = Counter(
        model for run in runs for model in run.get("api", {}).get("effective_models", [])
    )
    fingerprints = Counter(
        fingerprint for run in runs for fingerprint in run.get("api", {}).get("system_fingerprints", [])
    )
    attempts = cohort["manifest"].get("attempts", [])
    return {
        "accepted_runs": len(runs),
        "effective_model_appearances": dict(sorted(effective_models.items())),
        "system_fingerprint_appearances": dict(sorted(fingerprints.items())),
        "runs_without_system_fingerprint": sum(not run.get("api", {}).get("system_fingerprints") for run in runs),
        "all_attempts": len(attempts),
        "attempt_status_counts": dict(sorted(Counter(a.get("status", "unknown") for a in attempts).items())),
        "retry_attempts_after_first": sum(int(a.get("attempt", 1)) > 1 for a in attempts),
    }


def _profile_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("provider", "model", "base_url", "dimensions", "encoding_format")
    return {field: profile.get(field) for field in fields}


def _load_vectors(path: Path, expected_texts: set[str]) -> tuple[dict[str, Sequence[float]], dict[str, Any]]:
    artifact = _read_json(path)
    texts, vectors = artifact.get("texts"), artifact.get("vectors")
    if not isinstance(texts, list) or not isinstance(vectors, list) or len(texts) != len(vectors):
        raise ValueError("Shared embedding artifact must contain equal-length texts and vectors lists.")
    if len(texts) != len(set(texts)) or set(texts) != expected_texts:
        raise ValueError("Shared embedding artifact texts must exactly cover the Luna/Terra canonical union.")
    return dict(zip(texts, vectors)), artifact.get("profile", {})


def _verify_embedding_trace(
    embeddings_path: Path,
    audit_path: Path,
    expected_texts: set[str],
    expected_profile: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _read_json(embeddings_path)
    audit = _read_json(audit_path)
    artifact_profile = artifact.get("profile", {})
    if _profile_projection(artifact_profile) != _profile_projection(expected_profile):
        raise ValueError("Shared embedding artifact profile differs from the cohort grouping profile.")

    trace_dir = audit_path.parent / "api"
    calls_path = trace_dir / "calls.jsonl"
    if not calls_path.is_file():
        raise ValueError("Shared embedding raw API trace calls.jsonl is missing.")
    events = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = {event["call_id"]: event for event in events if event.get("event") == "start"}
    finishes = {event["call_id"]: event for event in events if event.get("event") == "finish"}
    if not starts or set(starts) != set(finishes):
        raise ValueError("Shared embedding audit has unmatched API request/response events.")
    if any(event.get("status") != "received" or event.get("retries") != 0 for event in finishes.values()):
        raise ValueError("Shared embedding audit contains failed or retried API calls.")

    stage_names = {event.get("context", {}).get("stage") for event in starts.values()}
    if stage_names != {EXPECTED_ANALYSIS_STAGE}:
        raise ValueError("Shared embedding API calls were not recorded under the expected analysis stage.")

    requested_by_text: dict[str, tuple[float, ...]] = {}
    request_records = []
    expected_model = expected_profile.get("model")
    expected_base_url = str(expected_profile.get("base_url", "")).rstrip("/")
    expected_dimensions = expected_profile.get("dimensions")
    for call_id in sorted(starts):
        start = starts[call_id]
        finish = finishes[call_id]
        stem = f"call_{call_id:04d}"
        request_path = trace_dir / f"{stem}.request.json"
        response_path = trace_dir / f"{stem}.response.json"
        if not request_path.is_file() or not response_path.is_file():
            raise ValueError(f"Shared embedding API trace files are missing for call {call_id}.")
        request_bytes = request_path.read_bytes()
        response_bytes = response_path.read_bytes()
        if _sha256(request_path) != start.get("request_sha256"):
            raise ValueError(f"Shared embedding request hash mismatch for call {call_id}.")
        if _sha256(response_path) != finish.get("response_sha256"):
            raise ValueError(f"Shared embedding response hash mismatch for call {call_id}.")
        body = json.loads(request_bytes.decode("utf-8"))
        response = json.loads(response_bytes.decode("utf-8"))
        batch = body.get("input")
        if (not isinstance(batch, list) or not batch or any(not isinstance(text, str) for text in batch)
                or len(batch) != len(set(batch))):
            raise ValueError(f"Shared embedding request {call_id} has an invalid or duplicate text batch.")
        if body.get("model") != expected_model or body.get("encoding_format") != "float":
            raise ValueError(f"Shared embedding request {call_id} used an unexpected model or format.")
        if start.get("url") != expected_base_url + "/embeddings":
            raise ValueError(f"Shared embedding request {call_id} used an unexpected endpoint.")
        if expected_dimensions is None:
            if "dimensions" in body:
                raise ValueError(f"Shared embedding request {call_id} sent an unexpected dimensions override.")
        elif body.get("dimensions") != expected_dimensions:
            raise ValueError(f"Shared embedding request {call_id} used an unexpected dimensions value.")
        if finish.get("model") != expected_model or response.get("model") != expected_model:
            raise ValueError(f"Shared embedding response {call_id} returned an unexpected model.")
        if finish.get("context") != start.get("context"):
            raise ValueError(f"Shared embedding request/response context mismatch for call {call_id}.")

        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise ValueError(f"Shared embedding response {call_id} does not cover its requested batch.")
        vectors_by_index: dict[int, tuple[float, ...]] = {}
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if type(index) is not int or not 0 <= index < len(batch) or index in vectors_by_index:
                raise ValueError(f"Shared embedding response {call_id} has invalid indices.")
            vector = item.get("embedding")
            try:
                vectors_by_index[index] = tuple(canonical_float32(vector))
            except Exception as error:
                raise ValueError(f"Shared embedding response {call_id} contains an invalid vector.") from error
        if set(vectors_by_index) != set(range(len(batch))):
            raise ValueError(f"Shared embedding response {call_id} indices do not cover the batch.")
        for index, text in enumerate(batch):
            if text in requested_by_text:
                raise ValueError("Shared embedding trace requested a canonical text more than once.")
            requested_by_text[text] = vectors_by_index[index]
        request_records.append({
            "call_id": call_id,
            "request_sha256": start["request_sha256"],
            "response_sha256": finish["response_sha256"],
            "input_count": len(batch),
        })

    if set(requested_by_text) != expected_texts:
        raise ValueError("Shared embedding API traces do not exactly cover the accepted catalog union.")
    artifact_texts = artifact["texts"]
    artifact_vectors = artifact["vectors"]
    for text, vector in zip(artifact_texts, artifact_vectors):
        try:
            canonical = tuple(canonical_float32(vector))
        except Exception as error:
            raise ValueError("Shared embedding artifact contains an invalid vector.") from error
        if canonical != requested_by_text[text]:
            raise ValueError("Shared embedding artifact vector does not match its hashed raw API response.")

    stage_counts = Counter(event["context"]["stage"] for event in starts.values())
    effective_models = sorted({event.get("model") for event in finishes.values() if isinstance(event.get("model"), str)})
    if (audit.get("attempted_calls") != len(starts) or audit.get("finished_calls") != len(finishes)
            or audit.get("failed_http_or_transport_calls") != 0 or audit.get("retries") != 0
            or audit.get("calls_by_stage") != dict(stage_counts) or audit.get("effective_models") != effective_models):
        raise ValueError("Shared embedding audit.json summary does not match its raw API call trace.")
    if effective_models != [expected_model]:
        raise ValueError("Shared embedding trace did not use exactly the expected embedding model.")
    return {
        "verified_calls": len(starts),
        "verified_text_vectors": len(requested_by_text),
        "verified_stage": EXPECTED_ANALYSIS_STAGE,
        "trace_path": str(calls_path.resolve()),
        "trace_sha256": _sha256(calls_path),
        "request_response_hashes": request_records,
        "audit_summary_matches_trace": True,
        "artifact_vectors_match_hashed_responses": True,
    }


def _validate_shared_manifest(
    path: Path,
    luna: Mapping[str, Any],
    terra: Mapping[str, Any],
    embeddings_path: Path,
    embedding_audit_path: Path,
    expected_text_count: int,
    analysis_profile: Mapping[str, Any],
) -> dict[str, Any]:
    shared_manifest = _read_json(path)
    if shared_manifest.get("schema_version") != 1:
        raise ValueError("Unsupported shared analysis manifest schema.")
    cohort_inputs = {"luna": luna, "terra": terra}
    listed_cohorts = shared_manifest.get("cohorts", {})
    if set(listed_cohorts) != set(cohort_inputs):
        raise ValueError("Shared analysis manifest must bind exactly the Luna and Terra cohorts.")
    for key, cohort in cohort_inputs.items():
        entry = listed_cohorts.get(key, {})
        root = cohort["root"].resolve()
        manifest_path = cohort["manifest_path"].resolve()
        config_path = cohort["config_path"].resolve()
        if (Path(entry.get("root", "")).resolve() != root
                or Path(entry.get("manifest_path", "")).resolve() != manifest_path
                or Path(entry.get("configuration_path", "")).resolve() != config_path):
            raise ValueError(f"Shared analysis manifest paths do not bind to the {key} cohort.")
        manifest_hash = _sha256(manifest_path)
        config_hash = _sha256(config_path)
        if (entry.get("manifest_sha256") != manifest_hash or entry.get("configuration_sha256") != config_hash
                or entry.get("specification_configuration_sha256") != cohort["manifest"]["specification"]["configuration_sha256"]
                or entry.get("input_sha256") != cohort["manifest"]["specification"]["input_sha256"]
                or entry.get("code_sha256") != cohort["manifest"]["specification"]["code_sha256"]
                or entry.get("run_count") != len(cohort["runs"])):
            raise ValueError(f"Shared analysis manifest hashes/count do not bind to the {key} cohort.")
        catalog_entries = entry.get("catalogs", [])
        expected_catalogs = [
            {
                "run": run["run"],
                "attempt": run["attempt"],
                "path": str((cohort["root"] / run["directory"] / "global.json").resolve()),
                "sha256": run["artifact_sha256"]["global.json"],
            }
            for run in cohort["runs"]
        ]
        if catalog_entries != expected_catalogs:
            raise ValueError(f"Shared analysis manifest catalog list does not match accepted {key} artifacts.")

    shared = shared_manifest.get("shared", {})
    embeddings_path = embeddings_path.resolve()
    embedding_audit_path = embedding_audit_path.resolve()
    trace_path = embedding_audit_path.parent / "api" / "calls.jsonl"
    if (Path(shared.get("embeddings_path", "")).resolve() != embeddings_path
            or Path(shared.get("audit_path", "")).resolve() != embedding_audit_path
            or Path(shared.get("trace_path", "")).resolve() != trace_path.resolve()):
        raise ValueError("Shared analysis manifest paths do not bind to the embedding audit artifacts.")
    if (shared.get("embeddings_sha256") != _sha256(embeddings_path)
            or shared.get("audit_sha256") != _sha256(embedding_audit_path)
            or shared.get("trace_sha256") != _sha256(trace_path)
            or shared.get("stage") != EXPECTED_ANALYSIS_STAGE
            or shared.get("text_count") != expected_text_count
            or shared.get("profile") != analysis_profile):
        raise ValueError("Shared analysis manifest hashes/profile/count do not match the artifacts.")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "cohorts_bound": sorted(listed_cohorts),
        "catalogs_bound": sum(len(listed_cohorts[key]["catalogs"]) for key in cohort_inputs),
        "shared_artifacts_bound": ["embeddings.json", "audit.json", "api/calls.jsonl"],
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _within_csv_row(label: str, cohort: dict[str, Any], exact_frequency: dict[str, Any], pair_summary: dict[str, Any]) -> dict[str, Any]:
    counts = _count_summary(cohort["texts_by_run"])
    exact = pair_summary["exact"]
    semantic = pair_summary["semantic"]
    return {
        "model": label,
        "n_runs": counts["n"],
        "count_mean": counts["mean"],
        "count_sample_variance_ddof1_items_squared": counts["sample_variance_ddof1"],
        "count_sample_sd_items": counts["sample_standard_deviation"],
        "count_min": counts["min"],
        "count_max": counts["max"],
        "count_range": counts["range"],
        "within_pair_count": pair_summary["pair_count"],
        "exact_mean_matched": exact["mean_matched"],
        "exact_mean_total_unmatched_per_pair": exact["mean_total_unmatched"],
        "exact_mean_jaccard": exact["mean_jaccard"],
        "exact_identical_pair_count": exact["identical_pair_count"],
        "exact_complete_match_rate": exact["complete_match_rate"],
        "exact_mean_matched_fraction_per_catalog": exact["mean_matched_fraction_per_catalog"],
        "exact_mean_unmatched_fraction_per_catalog": exact["mean_unmatched_fraction_per_catalog"],
        "semantic_mean_matched": semantic["mean_matched"],
        "semantic_mean_total_unmatched_per_pair": semantic["mean_total_unmatched"],
        "semantic_mean_jaccard": semantic["mean_jaccard"],
        "semantic_identical_pair_count": semantic["identical_pair_count"],
        "semantic_complete_match_rate": semantic["complete_match_rate"],
        "semantic_mean_matched_fraction_per_catalog": semantic["mean_matched_fraction_per_catalog"],
        "semantic_mean_unmatched_fraction_per_catalog": semantic["mean_unmatched_fraction_per_catalog"],
        "literal_100_run_intersection_multiset": exact_frequency["multiset_intersection_count"],
        "literal_100_run_union_distinct_texts": exact_frequency["union_distinct_text_count"],
    }


def _write_figure(
    path: Path,
    luna: dict[str, Any],
    terra: dict[str, Any],
    luna_rows: Sequence[dict[str, Any]],
    terra_rows: Sequence[dict[str, Any]],
    cross_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cohorts = (luna, terra)
    all_counts = [len(texts) for cohort in cohorts for texts in cohort["texts_by_run"]]
    low, high = min(all_counts), max(all_counts)
    bins = np.arange(low - 0.5, high + 1.5, 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    count_ax, primary_ax, support_ax, cross_ax = axes.ravel()
    for cohort, color in zip(cohorts, ("#4C78A8", "#F58518")):
        counts = [len(texts) for texts in cohort["texts_by_run"]]
        stats = _count_summary(cohort["texts_by_run"])
        count_ax.hist(counts, bins=bins, alpha=0.52, color=color, edgecolor="white",
                      label=f"{cohort['label']} (média {stats['mean']:.2f}, DP {stats['sample_standard_deviation']:.2f})")
    count_ax.set_title("Itens canônicos globais por modelo")
    count_ax.set_xlabel("Número de itens por repetição")
    count_ax.set_ylabel("Repetições (de 100)")
    count_ax.set_xticks(range(low, high + 1))
    count_ax.legend(frameon=False, fontsize=9)
    count_ax.grid(axis="y", alpha=0.2)

    luna_jackknife = _jackknife_pair_mean(luna_rows, len(luna["runs"]))
    terra_jackknife = _jackknife_pair_mean(terra_rows, len(terra["runs"]))
    delta = terra_jackknife["estimate"] - luna_jackknife["estimate"]
    delta_se = math.hypot(luna_jackknife["standard_error"], terra_jackknife["standard_error"])
    delta_ci = (delta - 1.96 * delta_se, delta + 1.96 * delta_se)
    primary_data = [
        [row["semantic"]["jaccard"] for row in luna_rows],
        [row["semantic"]["jaccard"] for row in terra_rows],
    ]
    primary = primary_ax.boxplot(primary_data, patch_artist=True, showmeans=True, showfliers=False)
    for box, color in zip(primary["boxes"], ("#4C78A8", "#F58518")):
        box.set_facecolor(color)
        box.set_alpha(0.62)
    primary_ax.set_xticks((1, 2), ("gpt-6-luna", "gpt-6.1-terra"))
    primary_ax.set_ylim(-0.02, 1.02)
    primary_ax.set_ylabel("Jaccard semântico (0–1)")
    primary_ax.set_title("Principal: estabilidade semântica dentro de cada modelo")
    primary_ax.text(
        0.5, 0.96,
        f"Terra − Luna = {100 * delta:+.2f} p.p.\nIC 95% jackknife ≈ [{100 * delta_ci[0]:+.2f}, {100 * delta_ci[1]:+.2f}] p.p.",
        transform=primary_ax.transAxes, ha="center", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    primary_ax.grid(axis="y", alpha=0.2)

    support_data, support_labels, support_colors = [], [], []
    for rows, label in ((luna_rows, "Luna"), (terra_rows, "Terra")):
        for metric, metric_label, color in (
            ("exact", "literal", "#4C78A8"),
            ("semantic", "semântico proxy", "#F58518"),
        ):
            support_data.append([row[metric]["jaccard"] for row in rows])
            support_labels.append(f"{label}\n{metric_label}")
            support_colors.append(color)
    support = support_ax.boxplot(support_data, patch_artist=True, showmeans=True, showfliers=False)
    for box, color in zip(support["boxes"], support_colors):
        box.set_facecolor(color)
        box.set_alpha(0.62)
    support_ax.set_xticks(range(1, 5), support_labels)
    support_ax.set_ylim(-0.02, 1.02)
    support_ax.set_ylabel("Jaccard (0–1)")
    support_ax.set_title("Apoio: igualdade literal e proxy semântico")
    support_ax.grid(axis="y", alpha=0.2)

    cross_data = [
        [row["exact"]["jaccard"] for row in cross_rows],
        [row["semantic"]["jaccard"] for row in cross_rows],
    ]
    cross = cross_ax.boxplot(cross_data, patch_artist=True, showmeans=True, showfliers=False)
    for box, color in zip(cross["boxes"], ("#4C78A8", "#F58518")):
        box.set_facecolor(color)
        box.set_alpha(0.62)
    cross_ax.set_xticks((1, 2), ("literal", "semântico proxy"))
    cross_ax.set_ylim(-0.02, 1.02)
    cross_ax.set_ylabel("Jaccard (0–1)")
    cross_ax.set_title("Auxiliar: cruzamento Luna × Terra (10.000 pares)")
    cross_ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Estabilidade do catálogo global — gpt-6-luna × gpt-6.1-terra\n"
        "Comparações por pares reutilizam execuções e são dependentes; Jaccard semântico é um proxy, não uma medida de acurácia",
        fontsize=13,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare(
    luna_root: Path,
    terra_root: Path,
    embeddings_path: Path,
    embedding_audit_path: Path,
    output_dir: Path,
    expected_runs: int = 100,
    threshold: float = 0.85,
    overwrite: bool = False,
    shared_manifest_path: Path | None = None,
) -> dict[str, Path]:
    if threshold != 0.85:
        raise ValueError("The comparison semantic threshold is fixed at 0.85.")
    output_names = ("summary.json", "within_model.csv", "per_run_counts.csv", "cross_model.csv", "comparison.png")
    if not overwrite and output_dir.exists() and any((output_dir / name).exists() for name in output_names):
        raise FileExistsError("Comparison outputs already exist; pass --overwrite to replace them.")

    luna = _load_cohort("gpt-6-luna", luna_root, "gpt-6-luna", expected_runs)
    terra = _load_cohort("gpt-6.1-terra", terra_root, "gpt-6.1-terra", expected_runs)
    if luna["manifest"]["specification"]["input_sha256"] != terra["manifest"]["specification"]["input_sha256"]:
        raise ValueError("Luna and Terra cohorts do not use the same raw input hash.")
    if luna["configuration"].get("grouping") != terra["configuration"].get("grouping"):
        raise ValueError("Grouping configuration differs between cohorts.")

    extraction_luna = luna["configuration"].get("extraction", {})
    extraction_terra = terra["configuration"].get("extraction", {})
    extraction_luna_nonmodel = {key: value for key, value in extraction_luna.items() if key != "model"}
    extraction_terra_nonmodel = {key: value for key, value in extraction_terra.items() if key != "model"}
    if extraction_luna_nonmodel != extraction_terra_nonmodel:
        raise ValueError("Luna and Terra non-model extraction settings differ.")
    if extraction_luna.get("model") != "gpt-6-luna" or extraction_terra.get("model") != "gpt-6.1-terra":
        raise ValueError("Cohort configuration model names do not match the requested Luna/Terra comparison.")
    extraction_differences = {
        key: {"luna": extraction_luna.get(key), "terra": extraction_terra.get(key)}
        for key in sorted(set(extraction_luna) | set(extraction_terra))
        if extraction_luna.get(key) != extraction_terra.get(key)
    }
    model_only_configuration_change = set(extraction_differences) == {"model"}

    all_texts = {
        text
        for cohort in (luna, terra)
        for texts in cohort["texts_by_run"]
        for text in texts
    }
    vectors_by_text, analysis_profile = _load_vectors(embeddings_path, all_texts)
    normalized_texts, normalized_vectors = _validated_vectors(all_texts, vectors_by_text)
    text_index = {text: index for index, text in enumerate(normalized_texts)}
    similarities = _cosine_matrix(normalized_vectors)
    profile_projections = {
        json.dumps(_profile_projection(catalog["metadata"]["embedding_profile"]), sort_keys=True)
        for cohort in (luna, terra)
        for catalog in cohort["catalogs"]
    }
    analysis_projection = json.dumps(_profile_projection(analysis_profile), sort_keys=True)
    if len(profile_projections) != 1 or analysis_projection not in profile_projections:
        raise ValueError("Shared post-hoc embeddings do not match the cohorts' common embedding profile.")
    grouping_profile = luna["catalogs"][0]["metadata"]["embedding_profile"]
    shared_manifest_path = shared_manifest_path or (embeddings_path.parent / "manifest.json")
    shared_manifest_info = _validate_shared_manifest(
        shared_manifest_path, luna, terra, embeddings_path, embedding_audit_path,
        len(all_texts), analysis_profile,
    )
    embedding_trace_info = _verify_embedding_trace(
        embeddings_path, embedding_audit_path, all_texts, grouping_profile,
    )

    luna_within = _pair_rows(luna, luna, text_index, similarities, threshold, cartesian=False)
    terra_within = _pair_rows(terra, terra, text_index, similarities, threshold, cartesian=False)
    cross = _pair_rows(luna, terra, text_index, similarities, threshold, cartesian=True)
    if len(luna_within) != expected_runs * (expected_runs - 1) // 2:
        raise AssertionError("Unexpected Luna within-model pair count.")
    if len(terra_within) != expected_runs * (expected_runs - 1) // 2:
        raise AssertionError("Unexpected Terra within-model pair count.")
    if len(cross) != expected_runs * expected_runs:
        raise AssertionError("Unexpected cross-model Cartesian pair count.")

    luna_pair_summary = _summarize_pairs(luna_within)
    terra_pair_summary = _summarize_pairs(terra_within)
    cross_pair_summary = _summarize_pairs(cross)
    luna_frequency = _exact_frequency(luna["texts_by_run"])
    terra_frequency = _exact_frequency(terra["texts_by_run"])
    luna_count_summary = _count_summary(luna["texts_by_run"])
    terra_count_summary = _count_summary(terra["texts_by_run"])
    primary = _primary_outcome(
        luna_within, terra_within, expected_runs, expected_runs,
        luna_count_summary, terra_count_summary,
    )

    original_luna_summary_path = luna_root / "analysis" / "summary.json"
    original_luna_embeddings_path = luna_root / "analysis" / "embeddings.json"
    original_luna_summary = _read_json(original_luna_summary_path)
    original_luna_embeddings = _read_json(original_luna_embeddings_path)
    original_count_summary = original_luna_summary["global_canonical_count"]
    original_exact_summary = original_luna_summary["pairwise_summary"]["exact"]
    original_semantic_summary = original_luna_summary["pairwise_summary"]["semantic"]
    if [row["global_canonical_count"] for row in original_luna_summary["per_run"]] != [
        len(texts) for texts in luna["texts_by_run"]
    ]:
        raise ValueError("Historical Luna summary counts do not match the accepted Luna catalog artifacts.")
    jointly_embedded_luna_comparison = {
        "original_analysis": {
            "summary_path": str(original_luna_summary_path.resolve()),
            "summary_sha256": _sha256(original_luna_summary_path),
            "embedding_artifact_path": str(original_luna_embeddings_path.resolve()),
            "embedding_artifact_sha256": _sha256(original_luna_embeddings_path),
            "embedding_profile": original_luna_embeddings.get("profile"),
            "semantic_mean_jaccard": original_semantic_summary["mean_jaccard"],
            "semantic_mean_matched": original_semantic_summary["mean_matched"],
            "semantic_mean_total_unmatched": original_semantic_summary["mean_total_unmatched"],
            "exact_mean_jaccard": original_exact_summary["mean_jaccard"],
            "count_summary": original_count_summary,
        },
        "joint_union_embedding_analysis": {
            "semantic_mean_jaccard": luna_pair_summary["semantic"]["mean_jaccard"],
            "semantic_mean_matched": luna_pair_summary["semantic"]["mean_matched"],
            "semantic_mean_total_unmatched": luna_pair_summary["semantic"]["mean_total_unmatched"],
            "exact_mean_jaccard": luna_pair_summary["exact"]["mean_jaccard"],
            "count_summary": luna_count_summary,
        },
        "joint_minus_original": {
            "semantic_mean_jaccard": luna_pair_summary["semantic"]["mean_jaccard"] - original_semantic_summary["mean_jaccard"],
            "exact_mean_jaccard": luna_pair_summary["exact"]["mean_jaccard"] - original_exact_summary["mean_jaccard"],
        },
        "interpretation": "Primary model comparison uses the jointly embedded Luna baseline because every Luna and Terra text was embedded together in one fresh pass. Differences from the prior Luna report reflect a separate post-hoc embedding pass and are disclosed here.",
    }

    code_changes = {
        path: {"luna": luna["code_hashes"].get(path), "terra": terra["code_hashes"].get(path)}
        for path in sorted(set(luna["code_hashes"]) | set(terra["code_hashes"]))
        if luna["code_hashes"].get(path) != terra["code_hashes"].get(path)
    }
    if set(luna["code_hashes"]) != set(terra["code_hashes"]):
        raise ValueError("Luna/Terra package hash inventories differ.")
    if set(code_changes) != ALLOWED_CODE_HASH_CHANGES:
        raise ValueError(
            "Only the extraction model default may differ in the TOML and adapter configuration files; "
            f"actual changed package paths: {sorted(code_changes)}."
        )
    summary = {
        "schema_version": 1,
        "cohorts": {
            "gpt-6-luna": {
                "root": str(luna_root.resolve()),
                "configuration": luna["configuration"],
                "audit": _effective_model_audit(luna),
                "input_sha256": luna["manifest"]["specification"]["input_sha256"],
                "code_sha256": luna["manifest"]["specification"]["code_sha256"],
                "global_canonical_count": luna_count_summary,
                "exact_raw_text_across_runs": luna_frequency,
                "within_model_pairs": luna_pair_summary,
            },
            "gpt-6.1-terra": {
                "root": str(terra_root.resolve()),
                "configuration": terra["configuration"],
                "audit": _effective_model_audit(terra),
                "input_sha256": terra["manifest"]["specification"]["input_sha256"],
                "code_sha256": terra["manifest"]["specification"]["code_sha256"],
                "global_canonical_count": terra_count_summary,
                "exact_raw_text_across_runs": terra_frequency,
                "within_model_pairs": terra_pair_summary,
            },
        },
        "comparison": {
            "primary_outcome": primary,
            "luna_joint_embedding_vs_original_analysis": jointly_embedded_luna_comparison,
            "raw_input_hash_equal": True,
            "extraction_parameter_differences": extraction_differences,
            "configuration_differs_only_by_model": model_only_configuration_change,
            "code_hash_differences": code_changes,
            "all_non_model_code_hashes_equal": True,
            "shared_grouping_configuration": luna["configuration"]["grouping"],
            "shared_fresh_embedding_profile": analysis_profile,
            "common_posthoc_embedding_artifact": {
                "path": str(embeddings_path.resolve()),
                "sha256": _sha256(embeddings_path),
                "distinct_text_count": len(all_texts),
            },
            "common_embedding_audit": {
                "path": str(embedding_audit_path.resolve()),
                "sha256": _sha256(embedding_audit_path),
                "summary": _read_json(embedding_audit_path),
                "raw_trace_verification": embedding_trace_info,
            },
            "shared_analysis_manifest": shared_manifest_info,
            "semantic_threshold": threshold,
            "semantic_method": "cosine >= threshold; deterministic maximum-cardinality one-to-one matching with augmenting paths; ties follow catalog order, then descending cosine and right index",
            "semantic_caveat": "Semantic matches are embedding-based repeatability proxies, not verified factual equivalence.",
            "exact_method": "literal raw info string equality, multiset counts, with one-to-one occurrence matching",
            "within_pair_design": "all unordered pairs within each 100-run cohort",
            "cross_pair_design": "full Cartesian product of 100 Luna and 100 Terra catalogs; no run-ID pairing",
            "unmatched_definition": "total_unmatched = unmatched_luna_or_a + unmatched_terra_or_b = n_a + n_b - 2*matched",
            "inference_note": "Pairwise catalogs are dependent and are not treated as independent observations. The primary interval uses a leave-one-run-out jackknife over within-cohort U-statistics and assumes exchangeable independent sequential runs; it is approximate and exploratory. No accuracy or causal model effect is measured.",
            "pair_counts": {
                "within_each_model": expected_runs * (expected_runs - 1) // 2,
                "cross_model": expected_runs * expected_runs,
            },
            "cross_model_pairs": cross_pair_summary,
        },
    }

    within_rows = [
        _within_csv_row("gpt-6-luna", luna, luna_frequency, luna_pair_summary),
        _within_csv_row("gpt-6.1-terra", terra, terra_frequency, terra_pair_summary),
    ]
    run_rows = [
        {"model": cohort["label"], "run": run["run"], "global_canonical_count": len(texts)}
        for cohort in (luna, terra)
        for run, texts in zip(cohort["runs"], cohort["texts_by_run"])
    ]
    cross_rows = []
    for row in cross:
        exact, semantic = row["exact"], row["semantic"]
        cross_rows.append({
            "luna_run": row["run_a"],
            "terra_run": row["run_b"],
            "luna_count": row["count_a"],
            "terra_count": row["count_b"],
            "exact_matched": exact["matched"],
            "exact_unmatched_luna": exact["unmatched_a"],
            "exact_unmatched_terra": exact["unmatched_b"],
            "exact_total_unmatched": exact["unmatched_a"] + exact["unmatched_b"],
            "exact_jaccard": exact["jaccard"],
            "exact_luna_matched_fraction": exact["coverage_a"],
            "exact_terra_matched_fraction": exact["coverage_b"],
            "exact_mean_matched_fraction_per_catalog": math.fsum(
                value for value in (exact["coverage_a"], exact["coverage_b"]) if value is not None
            ) / sum(value is not None for value in (exact["coverage_a"], exact["coverage_b"]))
            if exact["coverage_a"] is not None or exact["coverage_b"] is not None else None,
            "exact_mean_unmatched_fraction_per_catalog": math.fsum(
                value for value in (
                    exact["unmatched_a"] / row["count_a"] if row["count_a"] else None,
                    exact["unmatched_b"] / row["count_b"] if row["count_b"] else None,
                ) if value is not None
            ) / sum(value is not None for value in (
                exact["unmatched_a"] / row["count_a"] if row["count_a"] else None,
                exact["unmatched_b"] / row["count_b"] if row["count_b"] else None,
            )) if row["count_a"] or row["count_b"] else None,
            "exact_total_unmatched_fraction_of_memberships": (
                exact["unmatched_a"] + exact["unmatched_b"]
            ) / (row["count_a"] + row["count_b"]) if row["count_a"] + row["count_b"] else None,
            "exact_fully_matched": exact["unmatched_a"] == exact["unmatched_b"] == 0,
            "semantic_matched": semantic["matched"],
            "semantic_unmatched_luna": semantic["unmatched_a"],
            "semantic_unmatched_terra": semantic["unmatched_b"],
            "semantic_total_unmatched": semantic["unmatched_a"] + semantic["unmatched_b"],
            "semantic_jaccard": semantic["jaccard"],
            "semantic_luna_matched_fraction": semantic["coverage_a"],
            "semantic_terra_matched_fraction": semantic["coverage_b"],
            "semantic_mean_matched_fraction_per_catalog": math.fsum(
                value for value in (semantic["coverage_a"], semantic["coverage_b"]) if value is not None
            ) / sum(value is not None for value in (semantic["coverage_a"], semantic["coverage_b"]))
            if semantic["coverage_a"] is not None or semantic["coverage_b"] is not None else None,
            "semantic_mean_unmatched_fraction_per_catalog": math.fsum(
                value for value in (
                    semantic["unmatched_a"] / row["count_a"] if row["count_a"] else None,
                    semantic["unmatched_b"] / row["count_b"] if row["count_b"] else None,
                ) if value is not None
            ) / sum(value is not None for value in (
                semantic["unmatched_a"] / row["count_a"] if row["count_a"] else None,
                semantic["unmatched_b"] / row["count_b"] if row["count_b"] else None,
            )) if row["count_a"] or row["count_b"] else None,
            "semantic_total_unmatched_fraction_of_memberships": (
                semantic["unmatched_a"] + semantic["unmatched_b"]
            ) / (row["count_a"] + row["count_b"]) if row["count_a"] + row["count_b"] else None,
            "semantic_fully_matched": semantic["unmatched_a"] == semantic["unmatched_b"] == 0,
        })

    paths = {
        "summary": output_dir / "summary.json",
        "within_model": output_dir / "within_model.csv",
        "per_run_counts": output_dir / "per_run_counts.csv",
        "cross_model": output_dir / "cross_model.csv",
        "figure": output_dir / "comparison.png",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    within_fields = list(within_rows[0])
    _write_csv(paths["within_model"], within_fields, within_rows)
    _write_csv(paths["per_run_counts"], ("model", "run", "global_canonical_count"), run_rows)
    _write_csv(paths["cross_model"], tuple(cross_rows[0]), cross_rows)
    _write_figure(paths["figure"], luna, terra, luna_within, terra_within, cross)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna-root", type=Path, required=True)
    parser.add_argument("--terra-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True, help="Shared fresh post-hoc embeddings artifact.")
    parser.add_argument("--embedding-audit", type=Path, required=True)
    parser.add_argument("--shared-manifest", type=Path, help="Shared analysis manifest; defaults to embeddings/../manifest.json.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = compare(
        args.luna_root, args.terra_root, args.embeddings, args.embedding_audit,
        args.output_dir, args.runs, args.threshold, args.overwrite, args.shared_manifest,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
