import copy
import json
import math
import unittest
from collections import Counter
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from determinism.entrypoints.cli import main
from determinism.experiments import SemanticGroupingExperiment, group_dataset, grouping_pipeline, run_grouping, run_grouping_with_catalog
from determinism.modules.semantic_grouping.adapters import GroupingConfig
from determinism.modules.semantic_grouping.adapters.postgres_vectors import canonical_float32
from determinism.modules.semantic_grouping.domain import (
    CompleteLinkageGrouping, GroupingError, InformationOccurrence, canonical_representative, cosine_matrix,
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
                canonical = segment.pop("canonical_information")
                self.assertEqual(segment, original[source_index]["segments"][segment_index])
                expected = original[source_index]["segments"][segment_index]["information"]
                actual = [item for group in groups for item in group]
                self.assertEqual(Counter((item["info"], item["context"]) for item in actual), Counter((item["info"], item["context"]) for item in expected))
                self.assertTrue(all(set(item) == {"info", "context", "avg_similarity"} for item in actual))
                self.assertEqual(len(groups[0]), 3 if segment_index == 0 else 1)
                self.assertTrue(all(item["avg_similarity"] == 1.0 for item in actual))
                self.assertEqual(len(canonical), len(groups))
                self.assertEqual(sorted(index for item in canonical for index in item["member_indices"]), list(range(len(expected))))
                for item, group in zip(canonical, groups):
                    representative = expected[item["representative_index"]]
                    self.assertIn(item["representative_index"], item["member_indices"])
                    self.assertEqual((item["info"], item["context"]), (representative["info"], representative["context"]))
                    self.assertEqual([expected[index]["info"] for index in item["member_indices"]], [member["info"] for member in group])
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

    def test_extraction_triples_are_preserved_in_history_and_only_info_is_embedded(self):
        value = dataset(["A", "B"])
        items = value[0]["segments"][0]["information"]
        items[0]["syntagm"] = {"subject": "Ana", "relation": "dormiu", "object": ""}
        items[1]["syntagm"] = {"subject": "Bruno", "relation": "comprou", "object": "um livro"}
        original = copy.deepcopy(value)
        result = self.group(value)
        self.assertEqual(value, original)
        self.assertEqual(result[0]["segments"][0]["information"], items)
        self.assertEqual(self.provider.calls, [("A", "B")])
        self.assertTrue(all(set(item) == {"info", "context", "avg_similarity"}
                            for group in result[0]["segments"][0]["set_different_information"] for item in group))
        items[0]["syntagm"]["subject"] = "Outra pessoa"
        self.assertEqual(result[0]["segments"][0]["information"], original[0]["segments"][0]["information"])

    def test_raw_and_cached_threshold_results_use_same_persisted_float32_representation(self):
        below = .8499999999
        vectors = {"A": (1, 0), "B": (below, math.sqrt(1 - below**2))}
        provider = FakeEmbeddings(vectors)
        self.assertLess(cosine_matrix(tuple(vectors.values()))[0][1], .85)
        self.assertGreater(cosine_matrix(tuple(canonical_float32(vector) for vector in vectors.values()))[0][1], .85)
        first = group_dataset(dataset(["A", "B"]), provider=provider, vector_store=self.store, profile=self.profile, threshold=.85)
        second = group_dataset(first, provider=provider, vector_store=self.store, profile=self.profile, threshold=.85)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]["segments"][0]["set_different_information"]), 1)
        self.assertEqual(len(provider.calls), 1)

    def test_empty_information_is_empty_grouping_and_does_not_use_external_ports(self):
        value = dataset([], [])
        value[0]["segments"][0]["set_different_information"] = [["stale"]]
        value[0]["segments"][0]["canonical_information"] = {"stale": True}
        result = self.group(value)
        self.assertEqual([segment["set_different_information"] for segment in result[0]["segments"]], [[], []])
        self.assertEqual([segment["canonical_information"] for segment in result[0]["segments"]], [[], []])
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


class CanonicalizationTests(unittest.TestCase):
    def group(self, value, vectors, **options):
        self.store = MemoryVectorStore()
        self.provider = FakeEmbeddings(vectors)
        return group_dataset(value, provider=self.provider, vector_store=self.store,
                             profile=GroupingConfig().profile, **options)

    def test_nearest_normalized_frequency_weighted_centroid_matches_canonical_selection(self):
        texts = ["A", "B", "C", "C"]
        vectors = {text: tuple(scale * component for component in
                              (math.cos(math.radians(angle)), math.sin(math.radians(angle))))
                   for text, angle, scale in (("A", 0, 1000), ("B", 10, .001), ("C", 20, 1))}
        result = self.group(dataset(texts), vectors, threshold=.90)
        segment = result[0]["segments"][0]
        stored = [self.store.rows[(GroupingConfig().profile.key, text)] for text in texts]
        units = [tuple(component / math.hypot(*vector) for component in vector) for vector in stored]
        centroid = tuple(math.fsum(components) / len(units) for components in zip(*units))
        # Oráculo independente: calcula o centroide e seus cossenos diretamente.
        similarities = [math.fsum(a * b for a, b in zip(unit, centroid)) / math.hypot(*centroid) for unit in units]
        expected_index = max(range(len(units)), key=similarities.__getitem__)
        canonical = segment["canonical_information"][0]
        self.assertEqual(expected_index, 1)
        self.assertEqual(canonical["representative_index"], expected_index)
        self.assertEqual(canonical["info"], "B")
        self.assertEqual(canonical["member_indices"], [0, 1, 2, 3])
        self.assertEqual(canonical["avg_similarity"], segment["set_different_information"][0][expected_index]["avg_similarity"])
        # As normas arbitrárias do provedor não viram pesos semânticos.
        unscaled = {text: (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
                    for text, angle in (("A", 0), ("B", 10), ("C", 20))}
        other = self.group(dataset(texts), unscaled, threshold=.90)
        self.assertEqual(other[0]["segments"][0]["canonical_information"][0]["info"], "B")

    def test_frequency_of_occurrences_changes_representative_without_deduplication(self):
        vectors = {"A": (1, 0), "B": (.98, .2)}
        tied = self.group(dataset(["B", "A"]), vectors)[0]["segments"][0]
        repeated = self.group(dataset(["A", "B", "B"]), vectors)[0]["segments"][0]
        self.assertEqual(tied["canonical_information"][0]["info"], "A")
        self.assertEqual(repeated["canonical_information"][0]["info"], "B")
        self.assertEqual(repeated["canonical_information"][0]["member_indices"], [0, 1, 2])
        self.assertEqual(len(repeated["set_different_information"][0]), 3)
        self.assertEqual(self.provider.calls, [("A", "B")])

    def test_exact_ties_select_same_text_and_context_under_all_input_permutations(self):
        entries = [{"info": "Z", "context": "a", "tokens": [0]},
                   {"info": "A", "context": "z", "tokens": [99]},
                   {"info": "A", "context": "a", "tokens": [],
                    "syntagm": {"subject": "ignored", "relation": "ignored", "object": "ignored"}}]
        for order in permutations(entries):
            with self.subTest(order=order):
                value = dataset([])
                value[0]["segments"][0]["information"] = list(order)
                result = self.group(value, {"Z": (1, 0), "A": (1, 0)})
                canonical = result[0]["segments"][0]["canonical_information"][0]
                self.assertEqual((canonical["info"], canonical["context"]), ("A", "a"))
                self.assertEqual(order[canonical["representative_index"]]["context"], "a")
                self.assertEqual(self.provider.calls, [tuple(dict.fromkeys(item["info"] for item in order))])
        identical = tuple(InformationOccurrence("A", "same") for _ in range(2))
        self.assertEqual(canonical_representative(identical, {1: 1.0, 0: 1.0}), 0)
        # Médias próximas não são arredondadas para criar empates artificiais.
        self.assertEqual(canonical_representative(identical, {0: .9, 1: math.nextafter(.9, math.inf)}), 1)

    def test_complete_linkage_is_called_once_and_canonicalization_keeps_its_partition(self):
        class CountingStrategy:
            def __init__(self):
                self.calls = 0
            def partition(self, matrix, threshold):
                self.calls += 1
                return CompleteLinkageGrouping().partition(matrix, threshold)
        strategy = CountingStrategy()
        vectors = {text: (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
                   for text, angle in (("A", 0), ("B", 10), ("C", 35))}
        result = self.group(dataset(["A", "B", "C"]), vectors, threshold=.95, strategy=strategy)
        segment = result[0]["segments"][0]
        self.assertEqual(strategy.calls, 1)
        self.assertEqual([item["member_indices"] for item in segment["canonical_information"]], [[0, 1], [2]])
        self.assertEqual([len(group) for group in segment["set_different_information"]], [2, 1])
        singleton = segment["canonical_information"][1]
        self.assertEqual((singleton["info"], singleton["representative_index"], singleton["avg_similarity"]), ("C", 2, 1.0))

    def test_zero_centroid_has_deterministic_cosine_distance_fallback(self):
        result = self.group(dataset(["Z", "A"]), {"Z": (1, 0), "A": (-1, 0)}, threshold=-1)
        canonical = result[0]["segments"][0]["canonical_information"]
        self.assertEqual(canonical, [{"info": "A", "context": "context-0-1", "representative_index": 1,
                                      "member_indices": [0, 1], "avg_similarity": -1.0}])
        with self.assertRaises(GroupingError):
            canonical_representative((), {})

    def test_canonical_indices_stay_local_to_each_source_and_segment(self):
        value = dataset(["Z", "A"], ["Z"])
        value.append(dataset(["A"])[0])
        result = self.group(value, {"Z": (1, 0), "A": (1, 0)})
        actual = [segment["canonical_information"][0] for source in result for segment in source["segments"]]
        self.assertEqual([item["representative_index"] for item in actual], [1, 0, 0])
        self.assertEqual([item["member_indices"] for item in actual], [[0, 1], [0], [0]])
        for empty in ([], [{"segments": []}]):
            self.assertEqual(self.group(empty, {}), empty)
            self.assertFalse(self.provider.calls)
            self.assertFalse(self.store.fetches)


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
            with patch("determinism.modules.semantic_grouping.application.grouping_stage.canonical_representative", side_effect=GroupingError("failed")):
                with self.assertRaises(GroupingError):
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
            value = dataset([])
            value[0]["segments"][0]["canonical_information"] = ["stale"]
            source.write_text(json.dumps(value), encoding="utf-8")
            with patch("builtins.print") as printing:
                self.assertEqual(main(["group", str(source), "--env-file", str(Path(directory) / "missing.env")]), 0)
            self.assertIn("grupos: 0", printing.call_args_list[0].args[0])
            self.assertEqual(json.loads(source.read_text(encoding="utf-8"))[0]["segments"][0]["set_different_information"], [])
            self.assertEqual(json.loads(source.read_text(encoding="utf-8"))[0]["segments"][0]["canonical_information"], [])

    def test_cli_rebuilds_both_artifacts_and_cached_repetition_is_byte_identical(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = Path(directory) / "output.json"
            value = dataset(["A", "A", "B"])
            value[0]["segments"][0].update({"set_different_information": ["stale"], "canonical_information": {"old": True}})
            source.write_text(json.dumps(value), encoding="utf-8")
            before = source.read_bytes()
            def injected_run(input_path, output_path, **options):
                return run_grouping_with_catalog(input_path, output_path, config=options["config"],
                                    provider=self.provider, vector_store=self.store)
            with patch("determinism.entrypoints.cli.run_grouping_with_catalog", side_effect=injected_run), patch("builtins.print"):
                self.assertEqual(main(["group", str(source), "--output", str(output)]), 0)
                first = output.read_bytes()
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(main(["group", str(output)]), 0)
            self.assertEqual(output.read_bytes(), first)
            self.assertEqual(len(self.provider.calls), 1)
            canonical = json.loads(first)[0]["segments"][0]["canonical_information"]
            self.assertEqual([item["member_indices"] for item in canonical], [[0, 1], [2]])

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
