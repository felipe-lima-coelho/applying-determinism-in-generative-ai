"""Descriptive repeatability analysis for global canonical information.

The semantic comparison is a proxy: it uses cosine similarity between freshly
embedded canonical texts and a deterministic maximum-cardinality one-to-one
matching at the configured threshold. It does not establish factual identity.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _catalog_texts(catalog: Mapping[str, Any], run_number: int) -> list[str]:
    canonical = catalog.get("canonical_information")
    if not isinstance(canonical, list):
        raise ValueError(f"Run {run_number} has no canonical_information list.")
    texts: list[str] = []
    for item_number, item in enumerate(canonical, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Run {run_number} canonical item {item_number} is not an object.")
        text = item.get("info")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Run {run_number} canonical item {item_number} has invalid info text.")
        texts.append(text)
    return texts


def _validated_vectors(
    texts: set[str], vectors_by_text: Mapping[str, Sequence[float]],
) -> tuple[list[str], list[tuple[float, ...]]]:
    missing = texts - set(vectors_by_text)
    extra = set(vectors_by_text) - texts
    if missing or extra:
        raise ValueError(
            "Post-hoc vectors must cover exactly the distinct canonical texts "
            f"(missing={len(missing)}, extra={len(extra)})."
        )

    unique_texts = sorted(texts)
    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for text in unique_texts:
        raw = vectors_by_text[text]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
            raise ValueError("Each post-hoc embedding must be a non-empty numeric sequence.")
        if dimension is not None and len(raw) != dimension:
            raise ValueError("Post-hoc embeddings must all have the same dimension.")
        vector = []
        for component in raw:
            if isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(component):
                raise ValueError("Post-hoc embeddings must contain only finite numeric values.")
            vector.append(float(component))
        norm = math.hypot(*vector)
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("Post-hoc embeddings must have finite, non-zero norms.")
        dimension = len(vector)
        vectors.append(tuple(component / norm for component in vector))
    return unique_texts, vectors


def _cosine_matrix(vectors: list[tuple[float, ...]]) -> list[list[float]]:
    """Materialize a single cosine matrix for the analysis-only text set."""
    if not vectors:
        return []
    try:
        import numpy as np
    except ImportError:
        # The project has no required numerical dependency. This fallback is
        # adequate for small catalogs; normal experiment installs use NumPy.
        return [
            [
                1.0 if left == right else min(1.0, max(-1.0, math.fsum(a * b for a, b in zip(left, right))))
                for right in vectors
            ]
            for left in vectors
        ]

    normalized = np.asarray(vectors, dtype=np.float64)
    similarities = normalized @ normalized.T
    np.clip(similarities, -1.0, 1.0, out=similarities)
    return similarities.tolist()


def _match_metrics(matched: int, count_a: int, count_b: int) -> dict[str, int | float | None]:
    unmatched_a = count_a - matched
    unmatched_b = count_b - matched
    union = count_a + count_b - matched
    return {
        "matched": matched,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "jaccard": matched / union if union else 1.0,
        "coverage_a": matched / count_a if count_a else None,
        "coverage_b": matched / count_b if count_b else None,
    }


def _exact_match_count(texts_a: list[str], texts_b: list[str]) -> int:
    counts_a, counts_b = Counter(texts_a), Counter(texts_b)
    return sum(min(count, counts_b.get(text, 0)) for text, count in counts_a.items())


def _maximum_cardinality_matching(adjacency: list[list[int]], right_count: int) -> list[tuple[int, int]]:
    """Return a deterministic maximum matching using augmenting paths.

    Left nodes are visited in canonical list order. Neighbor lists are already
    sorted by descending similarity, then right-node index. Reassignment along
    augmenting paths prevents the many-to-one inflation of greedy matching.
    """
    matched_left_by_right: list[int | None] = [None] * right_count

    def augment(left: int, seen_right: list[bool]) -> bool:
        for right in adjacency[left]:
            if seen_right[right]:
                continue
            seen_right[right] = True
            incumbent = matched_left_by_right[right]
            if incumbent is None or augment(incumbent, seen_right):
                matched_left_by_right[right] = left
                return True
        return False

    for left in range(len(adjacency)):
        augment(left, [False] * right_count)
    result = [
        (left, right)
        for right, left in enumerate(matched_left_by_right)
        if left is not None
    ]
    return sorted(result)


def _semantic_match_count(
    texts_a: list[str],
    texts_b: list[str],
    text_index: Mapping[str, int],
    similarities: list[list[float]],
    threshold: float,
) -> int:
    adjacency: list[list[int]] = []
    for text_a in texts_a:
        left_vector = text_index[text_a]
        neighbors = [
            (similarities[left_vector][text_index[text_b]], right)
            for right, text_b in enumerate(texts_b)
            if similarities[left_vector][text_index[text_b]] >= threshold
        ]
        neighbors.sort(key=lambda item: (-item[0], item[1]))
        adjacency.append([right for _, right in neighbors])
    return len(_maximum_cardinality_matching(adjacency, len(texts_b)))


def _pairwise_summary(pairs: list[dict[str, Any]], criterion: str) -> dict[str, int | float | None]:
    metrics = [pair[criterion] for pair in pairs]
    if not metrics:
        return {
            "pair_count": 0,
            "mean_matched": None,
            "mean_total_unmatched": None,
            "mean_jaccard": None,
            "min_jaccard": None,
            "max_jaccard": None,
            "identical_pair_count": 0,
        }
    jaccards = [metric["jaccard"] for metric in metrics]
    return {
        "pair_count": len(metrics),
        "mean_matched": math.fsum(metric["matched"] for metric in metrics) / len(metrics),
        "mean_total_unmatched": math.fsum(
            metric["unmatched_a"] + metric["unmatched_b"] for metric in metrics
        ) / len(metrics),
        "mean_jaccard": math.fsum(jaccards) / len(jaccards),
        "min_jaccard": min(jaccards),
        "max_jaccard": max(jaccards),
        "identical_pair_count": sum(
            metric["matched"] == metric["matched"] + metric["unmatched_a"]
            and metric["matched"] == metric["matched"] + metric["unmatched_b"]
            for metric in metrics
        ),
    }


def analyze_catalogs(
    catalogs: Sequence[Mapping[str, Any]],
    vectors_by_text: Mapping[str, Sequence[float]],
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Compare catalogs in input order and return JSON-serializable statistics.

    `vectors_by_text` must contain exactly one fresh post-hoc embedding for each
    distinct raw `info` string across all catalogs. The vectors are used only
    in this in-memory analysis and are not loaded from a pipeline vector store.
    """
    if not catalogs:
        raise ValueError("At least one global catalog is required.")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between -1 and 1.")

    texts_by_run = [_catalog_texts(catalog, run + 1) for run, catalog in enumerate(catalogs)]
    all_texts = {text for texts in texts_by_run for text in texts}
    if len(texts_by_run) > 1 or all_texts:
        unique_texts, normalized_vectors = _validated_vectors(all_texts, vectors_by_text)
    else:
        unique_texts, normalized_vectors = [], []
    text_index = {text: index for index, text in enumerate(unique_texts)}
    similarities = _cosine_matrix(normalized_vectors)

    counts = [len(texts) for texts in texts_by_run]
    count_mean = math.fsum(counts) / len(counts)
    sample_variance = (
        math.fsum((count - count_mean) ** 2 for count in counts) / (len(counts) - 1)
        if len(counts) > 1 else None
    )
    count_summary = {
        "n": len(counts),
        "mean": count_mean,
        "sample_variance_ddof1": sample_variance,
        "sample_standard_deviation": math.sqrt(sample_variance) if sample_variance is not None else None,
        "min": min(counts),
        "max": max(counts),
        "range": max(counts) - min(counts),
    }

    exact_counts_by_run = [Counter(texts) for texts in texts_by_run]
    exact_texts = sorted({text for counts_for_run in exact_counts_by_run for text in counts_for_run})
    exact_items = []
    frequency_distribution = [0] * len(texts_by_run)
    all_run_intersection = 0
    all_run_intersection_distinct = 0
    multiset_union = 0
    for text in exact_texts:
        occurrences = [counts_for_run.get(text, 0) for counts_for_run in exact_counts_by_run]
        runs_present = sum(count > 0 for count in occurrences)
        frequency_distribution[runs_present - 1] += 1
        all_run_intersection += min(occurrences)
        all_run_intersection_distinct += min(occurrences) > 0
        multiset_union += max(occurrences)
        exact_items.append({
            "info": text,
            "runs_present": runs_present,
            "total_occurrences": sum(occurrences),
            "occurrences_per_run": occurrences,
        })

    per_run = [
        {"run": run + 1, "global_canonical_count": count}
        for run, count in enumerate(counts)
    ]
    pairs = []
    for run_a in range(len(texts_by_run)):
        for run_b in range(run_a + 1, len(texts_by_run)):
            texts_a, texts_b = texts_by_run[run_a], texts_by_run[run_b]
            exact_matched = _exact_match_count(texts_a, texts_b)
            semantic_matched = _semantic_match_count(
                texts_a, texts_b, text_index, similarities, float(threshold),
            )
            pairs.append({
                "run_a": run_a + 1,
                "run_b": run_b + 1,
                "exact": _match_metrics(exact_matched, len(texts_a), len(texts_b)),
                "semantic": _match_metrics(semantic_matched, len(texts_a), len(texts_b)),
            })

    return {
        "schema_version": 1,
        "run_count": len(texts_by_run),
        "semantic_threshold": float(threshold),
        "semantic_measure": "cosine similarity of fresh post-hoc embeddings; one-to-one maximum-cardinality matching",
        "semantic_caveat": "Similarity above threshold is a repeatability proxy, not verified factual equivalence.",
        "matching_tie_policy": "Visit left items in catalog order; visit eligible neighbors by descending cosine then right item index; use augmenting paths.",
        "pairwise_summary": {
            "exact": _pairwise_summary(pairs, "exact"),
            "semantic": _pairwise_summary(pairs, "semantic"),
            "dependence_note": "Pairwise values share runs and are descriptive; no pair-level confidence intervals or p-values are computed.",
        },
        "global_canonical_count": count_summary,
        "exact_raw_text_across_runs": {
            "comparison": "literal Python string equality; occurrences are treated as a multiset",
            "multiset_intersection_count": all_run_intersection,
            "multiset_union_count": multiset_union,
            "distinct_raw_text_count": len(exact_texts),
            "union_distinct_text_count": len(exact_texts),
            "intersection_distinct_text_count": all_run_intersection_distinct,
            "run_frequency_distribution": [
                {"runs_present": frequency, "distinct_texts": number}
                for frequency, number in enumerate(frequency_distribution, start=1)
            ],
            "items": exact_items,
        },
        "per_run": per_run,
        "pairwise": pairs,
    }


def write_analysis_artifacts(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write summary JSON, run/pair CSVs, and a static PNG figure."""
    destination = Path(output_dir)
    if not destination.is_dir():
        raise ValueError("Analysis output directory must already exist.")

    summary_path = destination / "summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    per_run_path = destination / "per_run.csv"
    with per_run_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "global_canonical_count"))
        writer.writeheader()
        writer.writerows(summary["per_run"])

    pairwise_path = destination / "pairwise.csv"
    pair_fields = (
        "run_a", "run_b",
        "exact_matched", "exact_unmatched_a", "exact_unmatched_b", "exact_jaccard",
        "exact_coverage_a", "exact_coverage_b",
        "semantic_matched", "semantic_unmatched_a", "semantic_unmatched_b", "semantic_jaccard",
        "semantic_coverage_a", "semantic_coverage_b",
    )
    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for pair in summary["pairwise"]:
            writer.writerow({
                "run_a": pair["run_a"],
                "run_b": pair["run_b"],
                **{f"exact_{key}": value for key, value in pair["exact"].items()},
                **{f"semantic_{key}": value for key, value in pair["semantic"].items()},
            })

    figure_path = destination / "canonical_variance.png"
    _write_figure(summary, figure_path)
    return {
        "summary": summary_path,
        "per_run": per_run_path,
        "pairwise": pairwise_path,
        "figure": figure_path,
    }


def _write_figure(summary: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Writing the PNG figure requires matplotlib and NumPy.") from error

    per_run = summary["per_run"]
    pairs = summary["pairwise"]
    run_count = len(per_run)
    counts = [row["global_canonical_count"] for row in per_run]
    count_summary = summary["global_canonical_count"]
    exact_matrix = np.eye(run_count, dtype=float)
    semantic_matrix = np.eye(run_count, dtype=float)
    for pair in pairs:
        first, second = pair["run_a"] - 1, pair["run_b"] - 1
        exact_matrix[first, second] = exact_matrix[second, first] = pair["exact"]["jaccard"]
        semantic_matrix[first, second] = semantic_matrix[second, first] = pair["semantic"]["jaccard"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    count_axis, histogram_axis, exact_axis, semantic_axis = axes.ravel()

    run_numbers = list(range(1, run_count + 1))
    count_axis.plot(run_numbers, counts, marker="o", markersize=3, linewidth=0.8, color="#1f77b4")
    count_axis.axhline(count_summary["mean"], color="#d62728", linestyle="--", linewidth=1.3, label="Média")
    count_axis.set_title("Itens canônicos globais por repetição")
    count_axis.set_xlabel("Repetição")
    count_axis.set_ylabel("Número de itens")
    count_axis.legend(frameon=False)
    if run_count > 10:
        tick_step = max(1, math.ceil(run_count / 10))
        count_axis.set_xticks(run_numbers[::tick_step])
    count_axis.grid(axis="y", alpha=0.2)

    low, high = min(counts), max(counts)
    histogram_bins = [low - 0.5 + offset for offset in range(high - low + 2)]
    histogram_axis.hist(counts, bins=histogram_bins, color="#1f77b4", edgecolor="white")
    histogram_axis.set_title("Distribuição da contagem entre repetições")
    histogram_axis.set_xlabel("Número de itens canônicos globais")
    histogram_axis.set_ylabel("Repetições")
    histogram_axis.set_xticks(list(range(low, high + 1)))
    histogram_axis.grid(axis="y", alpha=0.2)

    sample_labels = [1, *range(1, run_count + 1, max(1, math.ceil(run_count / 10)))]
    tick_positions = sorted(set(label - 1 for label in sample_labels if 1 <= label <= run_count))
    tick_labels = [str(position + 1) for position in tick_positions]
    for axis, matrix, title in (
        (exact_axis, exact_matrix, "Jaccard de texto literal"),
        (semantic_axis, semantic_matrix, f"Jaccard semântico proxy (cosseno ≥ {summary['semantic_threshold']:.2f})"),
    ):
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Repetição")
        axis.set_ylabel("Repetição")
        axis.set_xticks(tick_positions, tick_labels, rotation=45)
        axis.set_yticks(tick_positions, tick_labels)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Jaccard")

    figure.suptitle(
        f"Repetibilidade do catálogo global — {run_count} repetições\n"
        "Comparações entre pares são dependentes e apresentadas de forma descritiva",
        fontsize=13,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)
