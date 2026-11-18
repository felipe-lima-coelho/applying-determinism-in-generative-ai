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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cohort(label: str, root: Path, expected_runs: int) -> dict[str, Any]:
    manifest_path = root / "experiment.json"
    config_path = root / "configuration.json"
    hashes_path = root / "code_hashes.json"
    manifest, config, code_hashes = map(_read_json, (manifest_path, config_path, hashes_path))
    if manifest.get("status") != "completed":
        raise ValueError(f"{label} experiment is not completed.")
    accepted = manifest.get("runs")
    if not isinstance(accepted, list) or len(accepted) != expected_runs:
        raise ValueError(f"{label} must have exactly {expected_runs} accepted runs.")
    accepted = sorted(accepted, key=lambda row: row.get("run", -1))
    run_ids = [row.get("run") for row in accepted]
    if run_ids != list(range(1, expected_runs + 1)):
        raise ValueError(f"{label} accepted runs must have unique IDs 1..{expected_runs}.")

    catalogs: list[dict[str, Any]] = []
    texts_by_run: list[list[str]] = []
    for run in accepted:
        if run.get("status") != "completed":
            raise ValueError(f"{label} run {run['run']} is not completed.")
        directory = root / run["directory"]
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
    return {
        "pair_count": len(rows),
        "exact": _pairwise_summary(rows, "exact"),
        "semantic": _pairwise_summary(rows, "semantic"),
        "total_unmatched_definition": "unmatched_a + unmatched_b = n_a + n_b - 2 * matched",
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
        "semantic_mean_matched": semantic["mean_matched"],
        "semantic_mean_total_unmatched_per_pair": semantic["mean_total_unmatched"],
        "semantic_mean_jaccard": semantic["mean_jaccard"],
        "semantic_identical_pair_count": semantic["identical_pair_count"],
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    count_ax, pair_ax = axes
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

    datasets = []
    labels = []
    colors = []
    for rows, label in ((luna_rows, "Luna"), (terra_rows, "Terra"), (cross_rows, "Luna × Terra")):
        for metric, metric_label, color in (
            ("exact", "literal", "#4C78A8"),
            ("semantic", "semântico proxy", "#F58518"),
        ):
            datasets.append([row[metric]["jaccard"] for row in rows])
            labels.append(f"{label}\n{metric_label}")
            colors.append(color)
    bp = pair_ax.boxplot(datasets, patch_artist=True, showmeans=True, showfliers=False)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.62)
    pair_ax.set_xticks(range(1, len(labels) + 1), labels)
    pair_ax.set_ylim(-0.02, 1.02)
    pair_ax.set_ylabel("Jaccard (0–1)")
    pair_ax.set_title("Sobreposição entre catálogos")
    pair_ax.grid(axis="y", alpha=0.2)
    pair_ax.text(
        0.5, -0.22,
        "Dentro de cada modelo: 4.950 pares; cruzado Luna × Terra: 10.000 pares.\n"
        "Os pares reutilizam execuções e são dependentes; caixas e médias são descritivas.",
        transform=pair_ax.transAxes, ha="center", va="top", fontsize=8,
    )

    fig.suptitle(
        "Variação do catálogo global — gpt-6-luna × gpt-6.1-terra",
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
) -> dict[str, Path]:
    if threshold != 0.85:
        raise ValueError("The comparison semantic threshold is fixed at 0.85.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = ("summary.json", "within_model.csv", "per_run_counts.csv", "cross_model.csv", "comparison.png")
    if not overwrite and any((output_dir / name).exists() for name in output_names):
        raise FileExistsError("Comparison outputs already exist; pass --overwrite to replace them.")

    luna = _load_cohort("gpt-6-luna", luna_root, expected_runs)
    terra = _load_cohort("gpt-6.1-terra", terra_root, expected_runs)
    if luna["manifest"]["specification"]["input_sha256"] != terra["manifest"]["specification"]["input_sha256"]:
        raise ValueError("Luna and Terra cohorts do not use the same raw input hash.")
    if luna["configuration"].get("grouping") != terra["configuration"].get("grouping"):
        raise ValueError("Grouping configuration differs between cohorts.")

    extraction_luna = luna["configuration"].get("extraction", {})
    extraction_terra = terra["configuration"].get("extraction", {})
    extraction_differences = {
        key: {"luna": extraction_luna.get(key), "terra": extraction_terra.get(key)}
        for key in sorted(set(extraction_luna) | set(extraction_terra))
        if extraction_luna.get(key) != extraction_terra.get(key)
    }
    model_only_configuration_change = set(extraction_differences) <= {"model"}

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

    code_changes = {
        path: {"luna": luna["code_hashes"].get(path), "terra": terra["code_hashes"].get(path)}
        for path in sorted(set(luna["code_hashes"]) | set(terra["code_hashes"]))
        if luna["code_hashes"].get(path) != terra["code_hashes"].get(path)
    }
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
            "raw_input_hash_equal": True,
            "extraction_parameter_differences": extraction_differences,
            "configuration_differs_only_by_model": model_only_configuration_change,
            "code_hash_differences": code_changes,
            "all_non_model_code_hashes_equal": not code_changes or set(code_changes) <= {
                "experiments/configs/semantic_extraction.toml"
            },
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
            },
            "semantic_threshold": threshold,
            "semantic_method": "cosine >= threshold; deterministic maximum-cardinality one-to-one matching with augmenting paths; ties follow catalog order, then descending cosine and right index",
            "semantic_caveat": "Semantic matches are embedding-based repeatability proxies, not verified factual equivalence.",
            "exact_method": "literal raw info string equality, multiset counts, with one-to-one occurrence matching",
            "within_pair_design": "all unordered pairs within each 100-run cohort",
            "cross_pair_design": "full Cartesian product of 100 Luna and 100 Terra catalogs; no run-ID pairing",
            "unmatched_definition": "total_unmatched = unmatched_luna_or_a + unmatched_terra_or_b = n_a + n_b - 2*matched",
            "inference_note": "All pairwise results are descriptive; pairs share catalogs and are not independent observations. No pair-level confidence intervals or p-values are computed.",
            "pair_counts": {
                "within_each_model": expected_runs * (expected_runs - 1) // 2,
                "cross_model": expected_runs * expected_runs,
            },
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
            "exact_fully_matched": exact["unmatched_a"] == exact["unmatched_b"] == 0,
            "semantic_matched": semantic["matched"],
            "semantic_unmatched_luna": semantic["unmatched_a"],
            "semantic_unmatched_terra": semantic["unmatched_b"],
            "semantic_total_unmatched": semantic["unmatched_a"] + semantic["unmatched_b"],
            "semantic_jaccard": semantic["jaccard"],
            "semantic_fully_matched": semantic["unmatched_a"] == semantic["unmatched_b"] == 0,
        })

    paths = {
        "summary": output_dir / "summary.json",
        "within_model": output_dir / "within_model.csv",
        "per_run_counts": output_dir / "per_run_counts.csv",
        "cross_model": output_dir / "cross_model.csv",
        "figure": output_dir / "comparison.png",
    }
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = compare(
        args.luna_root, args.terra_root, args.embeddings, args.embedding_audit,
        args.output_dir, args.runs, args.threshold, args.overwrite,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
