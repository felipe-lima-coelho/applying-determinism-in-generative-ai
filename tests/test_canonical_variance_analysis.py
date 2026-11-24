import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from determinism.experiments.canonical_variance_analysis import (
    analyze_catalogs,
    write_analysis_artifacts,
)


def catalog(*texts):
    return {"canonical_information": [{"info": text} for text in texts]}


def unit_vector(degrees):
    radians = math.radians(degrees)
    return (math.cos(radians), math.sin(radians))


class CanonicalVarianceAnalysisTests(unittest.TestCase):
    def test_count_summary_uses_sample_variance_with_ddof_one(self):
        summary = analyze_catalogs(
            [catalog("one"), catalog("one", "two", "three")],
            {"one": (1, 0), "two": (0, 1), "three": (-1, 0)},
        )

        self.assertEqual(summary["global_canonical_count"], {
            "n": 2,
            "mean": 2.0,
            "sample_variance_ddof1": 2.0,
            "sample_standard_deviation": math.sqrt(2.0),
            "min": 1,
            "max": 3,
            "range": 2,
        })

    def test_many_to_one_semantic_similarity_cannot_inflate_matches(self):
        summary = analyze_catalogs(
            [catalog("left-1", "left-2"), catalog("right")],
            {"left-1": (1, 0), "left-2": (0.99, 0.1), "right": (1, 0)},
        )
        match = summary["pairwise"][0]["semantic"]

        self.assertEqual(match["matched"], 1)
        self.assertEqual(match["unmatched_a"], 1)
        self.assertEqual(match["unmatched_b"], 0)
        self.assertEqual(match["jaccard"], 0.5)
        self.assertEqual(match["coverage_a"], 0.5)
        self.assertEqual(match["coverage_b"], 1.0)

    def test_augmenting_path_finds_maximum_cardinality_not_greedy_match(self):
        # A0 prefers R0, while A1 can match only R0. A greedy assignment gets
        # one pair; the augmenting path moves A0 to R1 and gets two.
        summary = analyze_catalogs(
            [catalog("A0", "A1"), catalog("R0", "R1")],
            {"A0": unit_vector(0), "A1": unit_vector(30), "R0": unit_vector(15), "R1": unit_vector(-25)},
        )

        self.assertEqual(summary["pairwise"][0]["semantic"]["matched"], 2)
        self.assertEqual(summary["pairwise"][0]["semantic"]["jaccard"], 1.0)

    def test_similarity_at_threshold_is_included(self):
        other_component = math.sqrt(1.0 - 0.85**2)
        vectors = {"x": (1, 0), "y": (0.85, other_component)}
        at_boundary = analyze_catalogs([catalog("x"), catalog("y")], vectors, threshold=0.85)
        above_boundary = analyze_catalogs(
            [catalog("x"), catalog("y")], vectors, threshold=math.nextafter(0.85, 1.0),
        )

        self.assertEqual(at_boundary["pairwise"][0]["semantic"]["matched"], 1)
        self.assertEqual(above_boundary["pairwise"][0]["semantic"]["matched"], 0)

    def test_exact_all_run_multiset_intersection_union_and_frequency(self):
        catalogs = [
            catalog("dup", "dup", "only-first"),
            catalog("dup", "only-second"),
            catalog("dup", "dup", "only-second"),
        ]
        texts = {text for item in catalogs for entry in item["canonical_information"] for text in (entry["info"],)}
        vectors = {text: (1, 0) for text in texts}
        summary = analyze_catalogs(catalogs, vectors)
        exact = summary["exact_raw_text_across_runs"]

        self.assertEqual(exact["multiset_intersection_count"], 1)
        self.assertEqual(exact["multiset_union_count"], 4)
        self.assertEqual(exact["union_distinct_text_count"], 3)
        self.assertEqual(exact["intersection_distinct_text_count"], 1)
        self.assertEqual(exact["run_frequency_distribution"], [
            {"runs_present": 1, "distinct_texts": 1},
            {"runs_present": 2, "distinct_texts": 1},
            {"runs_present": 3, "distinct_texts": 1},
        ])
        self.assertEqual(summary["pairwise_summary"]["semantic"]["pair_count"], 3)
        self.assertIn("descriptive", summary["pairwise_summary"]["dependence_note"])

    def test_empty_catalogues_define_empty_jaccard_and_null_coverage(self):
        summary = analyze_catalogs([catalog(), catalog()], {})
        pair = summary["pairwise"][0]

        self.assertEqual(pair["exact"]["jaccard"], 1.0)
        self.assertEqual(pair["semantic"]["jaccard"], 1.0)
        self.assertIsNone(pair["exact"]["coverage_a"])
        self.assertIsNone(pair["exact"]["coverage_b"])
        self.assertEqual(summary["pairwise_summary"]["semantic"]["mean_jaccard"], 1.0)
        self.assertEqual(summary["global_canonical_count"]["sample_variance_ddof1"], 0.0)

    def test_raw_text_vector_set_must_match_canonical_texts_exactly(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            analyze_catalogs([catalog("actual"), catalog("actual")], {"stale": (1, 0)})

    def test_writer_emits_summary_csv_and_png_artifacts(self):
        summary = analyze_catalogs(
            [catalog("same"), catalog("same")],
            {"same": (1, 0)},
        )
        with TemporaryDirectory() as directory:
            artifacts = write_analysis_artifacts(summary, directory)

            self.assertEqual(set(artifacts), {"summary", "per_run", "pairwise", "figure"})
            self.assertTrue(all(path.is_file() for path in artifacts.values()))
            self.assertEqual(json.loads(artifacts["summary"].read_text(encoding="utf-8")), summary)
            self.assertEqual(artifacts["pairwise"].read_text(encoding="utf-8").count("\n"), 2)
            self.assertGreater(artifacts["figure"].stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
