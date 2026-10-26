import copy
import json
import math
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from determinism.entrypoints.cli import main
from determinism.experiments import SemanticGroupingExperiment, group_dataset, grouping_pipeline, run_grouping
from determinism.modules.semantic_grouping.adapters import GroupingConfig
from determinism.modules.semantic_grouping.adapters.postgres_vectors import canonical_float32
from determinism.modules.semantic_grouping.domain import (
    CompleteLinkageGrouping, GroupingError, cosine_matrix,
    member_mean, validate_information_dataset, validate_partition,
)
from determinism.modules.semantic_grouping.ports import EmbeddingError, VectorStoreError
from determinism.pipelines import Pipeline


# Este caso entrava em ciclo com realocação; complete linkage só faz fusões.
PREVIOUS_CYCLE_VECTORS = (
    (1.0, -0.07871239057564743, 0.09367055784785899, -0.23204513472594998, 0.14729004528473894),
    (1.0, -0.2268006655763914, 0.02861044447852092, 0.06588306718453929, -0.16145721369184335),
    (1.0, 0.33335443637621154, 0.3690262763595806, 0.09284515899167771, 0.06783925275781338),
    (1.0, -0.187827669915082, 0.15499103556411156, 0.13600167212142028, -0.06757741551852987),
    (1.0, 0.26556889294790975, -0.13450797141497492, -0.03325406475065125, -0.14568460949733758),
    (1.0, -0.15931913167747389, -0.0181338659067785, -0.2911311896581515, -0.05710393529746976),
    (1.0, 0.10079973576323517, 0.3248209038148395, -0.20421819079303696, -0.17070214353091595),
)


def dataset(*segments):
    return [{"source": "preserved", "metadata": {"open": [1, 2]}, "segments": [
        {"content": "Original content.", "map": {"0": {"token": 0, "word": "Original", "count": 7}},
         "metadata": {"segment": index}, "information": [
             {"info": text, "context": f"context-{index}-{position}", "tokens": [0], "extra": "kept"}
             for position, text in enumerate(texts)
         ]}
        for index, texts in enumerate(segments)
    ]}]


class MemoryVectorStore:
    def __init__(self):
        self.rows = {}
        self.fetches = []
        self.puts = []

    def fetch(self, profile, texts):
        self.fetches.append((profile.key, tuple(texts)))
        return {text: self.rows[(profile.key, text)] for text in texts if (profile.key, text) in self.rows}

    def put(self, profile, vectors):
        self.puts.append((profile.key, dict(vectors)))
        for text, vector in vectors.items():
            self.rows.setdefault((profile.key, text), canonical_float32(vector))


class FakeEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(tuple(texts))
        return tuple(self.vectors[text] for text in texts)


class GroupingDomainTests(unittest.TestCase):
    def test_inclusive_threshold_boundary_singleton_empty_and_identical_occurrences(self):
        strategy = CompleteLinkageGrouping()
        self.assertEqual(strategy.partition((), .85), ())
        self.assertEqual(strategy.partition(((1.0,),), .85), ((0,),))
        self.assertEqual(strategy.partition(((1, .85), (.85, 1)), .85), ((0, 1),))
        below = math.nextafter(.85, -math.inf)
        self.assertEqual(strategy.partition(((1, below), (below, 1)), .85), ((0,), (1,)))
        self.assertEqual(strategy.partition(cosine_matrix(((1, 0), (1, 0), (1, 0))), .85), ((0, 1, 2),))
        self.assertEqual(member_mean(0, (0,), ((1.0,),)), 1.0)

    def test_chaining_is_rejected_and_ties_follow_original_occurrence_order(self):
        matrix = ((1, .9, .7), (.9, 1, .9), (.7, .9, 1))
        self.assertEqual(CompleteLinkageGrouping().partition(matrix, .85), ((0, 1), (2,)))
        for _ in range(5):
            self.assertEqual(CompleteLinkageGrouping().partition(matrix, .85), ((0, 1), (2,)))
        with self.assertRaises(GroupingError):
            validate_partition(((0, 1, 2),), matrix, .85)

    def test_every_mean_can_pass_while_one_pair_blocks_the_merge(self):
        matrix = ((1, .96, .90), (.96, 1, .84), (.90, .84, 1))
        self.assertTrue(all(member_mean(index, (0, 1, 2), matrix) >= .85 for index in range(3)))
        # A decisão não consulta a média, que continua apenas descritiva.
        with patch("determinism.modules.semantic_grouping.domain.contracts.member_mean", side_effect=AssertionError("membership cannot use means")):
            result = CompleteLinkageGrouping().partition(matrix, .85)
        self.assertEqual(result, ((0, 1), (2,)))
        with self.assertRaisesRegex(GroupingError, "par abaixo"):
            validate_partition(((0, 1, 2),), matrix, .85)

    def test_merge_priority_is_maximum_cross_minimum_rather_than_cross_mean(self):
        vectors = tuple((math.cos(math.radians(angle)), math.sin(math.radians(angle))) for angle in (0, 5, 12, 23))
        matrix = cosine_matrix(vectors)
        self.assertGreater((matrix[0][2] + matrix[1][2]) / 2, matrix[2][3])
        self.assertLess(min(matrix[0][2], matrix[1][2]), matrix[2][3])
        self.assertEqual(CompleteLinkageGrouping().partition(matrix, .95), ((0, 1), (2, 3)))

    def test_previous_cycle_fixture_finishes_with_complete_linkage(self):
        for vectors in (PREVIOUS_CYCLE_VECTORS, tuple(canonical_float32(vector) for vector in PREVIOUS_CYCLE_VECTORS)):
            matrix = cosine_matrix(vectors)
            result = CompleteLinkageGrouping().partition(matrix, .85)
            self.assertEqual(sorted(index for group in result for index in group), list(range(len(vectors))))
            for group in result:
                for position, first in enumerate(group):
                    self.assertTrue(all(matrix[first][second] >= .85 for second in group[position + 1:]))

    def test_final_partition_cannot_leave_an_eligible_complete_linkage_merge(self):
        matrix = ((1, .9, .7), (.9, 1, .7), (.7, .7, 1))
        with self.assertRaisesRegex(GroupingError, "fusão complete linkage elegível"):
            validate_partition(((0,), (1,), (2,)), matrix, .85)
        self.assertEqual(validate_partition(((0, 1), (2,)), matrix, .85), ((0, 1), (2,)))

    def test_each_cosine_pair_is_computed_once_and_strategy_reuses_the_matrix(self):
        vectors = ((1, 0), (1, 0), (0, 1), (-1, 0))
        with patch("determinism.modules.semantic_grouping.domain.contracts.math.fsum", wraps=math.fsum) as sums:
            matrix = cosine_matrix(vectors)
        self.assertEqual(sums.call_count, len(vectors) * (len(vectors) - 1) // 2)
        with patch("determinism.modules.semantic_grouping.domain.contracts.math.fsum", side_effect=AssertionError("cosines and means must not be recomputed")):
            self.assertEqual(CompleteLinkageGrouping().partition(matrix, .85), ((0, 1), (2,), (3,)))

    def test_partition_validation_rejects_dropped_duplicated_empty_and_bad_members(self):
        matrix = ((1, 0), (0, 1))
        for partition in (((0,),), ((0,), (0, 1)), ((0,), ()), ((True,), (0,)), [[0], [1]], ((0,), (2,))):
            with self.subTest(partition=partition), self.assertRaises(GroupingError):
                validate_partition(partition, matrix, .85)


class GroupingApplicationTests(unittest.TestCase):
    def setUp(self):
        self.profile = GroupingConfig().profile
        self.store = MemoryVectorStore()
        self.provider = FakeEmbeddings({"A": (1, 0), "B": (1, 0), "C": (0, 1)})

    def group(self, value, **options):
        return group_dataset(value, provider=self.provider, vector_store=self.store, profile=self.profile, **options)

    def test_all_sources_segments_duplicates_contexts_counts_and_metadata_are_preserved(self):
        value = dataset(["A", "B", "A", "C"], ["A"])
        value.append(copy.deepcopy(value[0]))
        original = copy.deepcopy(value)
        result = self.group(value)
        self.assertEqual(value, original)
        self.assertEqual(self.provider.calls, [("A", "B", "C")])
        for source_index, source in enumerate(result):
            for segment_index, segment in enumerate(source["segments"]):
                groups = segment.pop("set_different_information")
                self.assertEqual(segment, original[source_index]["segments"][segment_index])
                expected = original[source_index]["segments"][segment_index]["information"]
                actual = [item for group in groups for item in group]
                self.assertEqual(Counter((item["info"], item["context"]) for item in actual), Counter((item["info"], item["context"]) for item in expected))
                self.assertTrue(all(set(item) == {"info", "context", "avg_similarity"} for item in actual))
                self.assertEqual(len(groups[0]), 3 if segment_index == 0 else 1)
                self.assertTrue(all(item["avg_similarity"] == 1.0 for item in actual))
        self.assertEqual(result, original)

    def test_cache_reuse_and_model_endpoint_revision_dimension_invalidation(self):
        value = dataset(["A", "A", "C"])
        first = self.group(value)
        second = self.group(first)
        self.assertEqual(first, second)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(len(self.store.rows), 2)
        for values in ({"model": "other"}, {"base_url": "http://localhost:1234/v1"}, {"cache_revision": "2"}, {"dimensions": 2}):
            group_dataset(value, provider=self.provider, vector_store=self.store, profile=replace(self.profile, **values))
        self.assertEqual(len(self.provider.calls), 5)

    def test_raw_and_cached_threshold_results_use_same_persisted_float32_representation(self):
        below = .8499999999
        vectors = {"A": (1, 0), "B": (below, math.sqrt(1 - below**2))}
        provider = FakeEmbeddings(vectors)
        self.assertLess(cosine_matrix(tuple(vectors.values()))[0][1], .85)
        self.assertGreater(cosine_matrix(tuple(canonical_float32(vector) for vector in vectors.values()))[0][1], .85)
        first = group_dataset(dataset(["A", "B"]), provider=provider, vector_store=self.store, profile=self.profile)
        second = group_dataset(first, provider=provider, vector_store=self.store, profile=self.profile)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]["segments"][0]["set_different_information"]), 1)
        self.assertEqual(len(provider.calls), 1)

    def test_empty_information_is_empty_grouping_and_does_not_use_external_ports(self):
        value = dataset([], [])
        value[0]["segments"][0]["set_different_information"] = [["stale"]]
        result = self.group(value)
        self.assertEqual([segment["set_different_information"] for segment in result[0]["segments"]], [[], []])
        self.assertFalse(self.provider.calls)
        self.assertFalse(self.store.fetches)

    def test_each_segment_builds_one_pair_matrix_reused_for_grouping_and_output(self):
        value = dataset(["A", "B", "A", "C"], ["A"])
        with patch("determinism.modules.semantic_grouping.application.grouping_stage.cosine_matrix", wraps=cosine_matrix) as matrices:
            result = self.group(value)
        self.assertEqual(matrices.call_count, 2)
        self.assertEqual([len(call.args[0]) for call in matrices.call_args_list], [4, 1])
        self.assertEqual([len(segment["set_different_information"]) for segment in result[0]["segments"]], [2, 1])

    def test_all_information_is_prevalidated_before_any_cache_or_provider_call(self):
        for invalid in (None, {}, [{"info": "", "context": "ok"}], [{"info": "A"}], [{"info": "A", "context": 3}], [{"info": "A\x00", "context": "ok"}], [{"info": "A", "context": "\ud800"}]):
            value = dataset(["A"], [])
            value[0]["segments"][1]["information"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(GroupingError):
                self.group(value)
        missing = dataset(["A"])
        del missing[0]["segments"][0]["information"]
        with self.assertRaisesRegex(GroupingError, "information"):
            self.group(missing)
        self.assertFalse(self.provider.calls)
        self.assertFalse(self.store.fetches)

    def test_malformed_provider_and_cache_vectors_leave_caller_untouched(self):
        value = dataset(["A", "B"])
        original = copy.deepcopy(value)
        for returned in ((), ((0, 0), (1, 0)), ((1, 0), (1, 0, 1)), ((float("nan"), 1), (1, 0))):
            class BadProvider:
                def embed(self, texts):
                    return returned
            with self.subTest(returned=returned), self.assertRaises(GroupingError):
                group_dataset(value, provider=BadProvider(), vector_store=MemoryVectorStore(), profile=self.profile)
        self.assertEqual(value, original)
        self.store.rows[(self.profile.key, "A")] = (0, 0)
        with self.assertRaises(GroupingError):
            self.group(value)
        self.assertFalse(self.provider.calls)

    def test_missing_persisted_embedding_fails_after_put_before_grouping(self):
        class NonDurableStore(MemoryVectorStore):
            def put(self, profile, vectors):
                pass
        with self.assertRaisesRegex(VectorStoreError, "todos"):
            group_dataset(dataset(["A"]), provider=self.provider, vector_store=NonDurableStore(), profile=self.profile)

    def test_custom_strategy_is_injected_and_its_output_is_independently_validated(self):
        class ReversedGroups:
            def partition(self, matrix, threshold):
                return ((1, 0), (2,))
        result = self.group(dataset(["A", "B", "C"]), strategy=ReversedGroups())
        self.assertEqual([item["info"] for item in result[0]["segments"][0]["set_different_information"][0]], ["A", "B"])
        class Invalid:
            def partition(self, matrix, threshold):
                return ((0,), (1,), (2,))
        with self.assertRaisesRegex(GroupingError, "fusão complete linkage elegível"):
            self.group(dataset(["A", "B", "C"]), strategy=Invalid())


class GroupingPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryVectorStore()
        self.provider = FakeEmbeddings({"A": (1, 0), "B": (0, 1)})

    def test_in_place_and_separate_output_rebuild_only_grouping(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.json"
            output = Path(directory) / "output.json"
            source.write_text(json.dumps(dataset(["A", "A", "B"])), encoding="utf-8")
            before = source.read_bytes()
            first = run_grouping(source, output, provider=self.provider, vector_store=self.store)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)
            second = run_grouping(source, provider=self.provider, vector_store=self.store)
            self.assertEqual(second, first)
            self.assertEqual(len(self.provider.calls), 1)

    def test_provider_store_strategy_and_atomic_write_failure_preserve_both_files(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "data.json"
            output = Path(directory) / "output.json"
            source.write_text(json.dumps(dataset(["A", "B"])), encoding="utf-8")
            output.write_text("existing-output", encoding="utf-8")
            before = source.read_bytes(), output.read_bytes()
            with patch.object(self.provider, "embed", side_effect=EmbeddingError("failed")):
                with self.assertRaises(EmbeddingError):
                    run_grouping(source, output, provider=self.provider, vector_store=self.store)
            with patch.object(self.store, "fetch", side_effect=VectorStoreError("failed")):
                with self.assertRaises(VectorStoreError):
                    run_grouping(source, output, provider=self.provider, vector_store=self.store)
            with patch("os.replace", side_effect=OSError("failed")):
                with self.assertRaises(OSError):
                    run_grouping(source, output, provider=self.provider, vector_store=self.store)
            self.assertEqual((source.read_bytes(), output.read_bytes()), before)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["data.json", "output.json"])
            # O cache pode sobreviver à falha da escrita, evitando nova inferência.
            self.assertTrue(self.store.rows)

    def test_invalid_complete_linkage_partition_preserves_json_and_durable_computation(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "invalid-grouping.json"
            texts = ["A", "B", "C"]
            source.write_text(json.dumps(dataset(texts)), encoding="utf-8")
            before = source.read_bytes()
            third_y = (.84 - .96 * .90) / .28
            provider = FakeEmbeddings({"A": (1, 0, 0), "B": (.96, .28, 0), "C": (.90, third_y, math.sqrt(1 - .90**2 - third_y**2))})
            class InvalidMeanBasedStrategy:
                def partition(self, matrix, threshold):
                    return ((0, 1, 2),)
            with self.assertRaisesRegex(GroupingError, "par abaixo"):
                run_grouping(source, provider=provider, vector_store=self.store, strategy=InvalidMeanBasedStrategy())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(self.store.rows), len(texts))

    def test_empty_cli_works_without_api_key_or_database(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "empty.json"
            source.write_text(json.dumps(dataset([])), encoding="utf-8")
            with patch("builtins.print") as printing:
                self.assertEqual(main(["group", str(source), "--env-file", str(Path(directory) / "missing.env")]), 0)
            self.assertIn("grupos: 0", printing.call_args.args[0])
            self.assertEqual(json.loads(source.read_text(encoding="utf-8"))[0]["segments"][0]["set_different_information"], [])

    def test_experiment_pipeline_repository_and_provider_are_replaceable(self):
        seen = []
        value = dataset(["A"])
        class Repository:
            def load(self, path):
                seen.append(path)
                return value
            def save(self, path, result):
                seen.append((path, result))
        def alternative(current):
            result = copy.deepcopy(current)
            result[0]["experiment"] = "custom"
            return result
        experiment = SemanticGroupingExperiment(Pipeline((alternative,)), Repository())
        result = experiment.run("input.json", "output.json")
        self.assertEqual(seen, [Path("input.json"), (Path("output.json"), result)])
        self.assertEqual(result[0]["experiment"], "custom")
        self.assertNotIn("experiment", value[0])


if __name__ == "__main__":
    unittest.main()
