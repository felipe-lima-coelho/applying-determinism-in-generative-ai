import copy
import io
import json
import math
import os
import tempfile
import threading
import unittest
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from determinism.entrypoints.cli import main
from determinism.experiments import (
    SingletonRefinementExperiment, extract_dataset, refine_dataset, refinement_pipeline, run_refinement,
)
from determinism.modules.semantic_extraction.adapters import ExtractionConfig, OpenAICompatibleProvider
from determinism.modules.semantic_extraction.adapters.prompts import coverage_message, structured_output, system_prompt, user_message
from determinism.modules.semantic_extraction.application import ExtractionStage
from determinism.modules.semantic_extraction.domain import CoverageFeedback, InferenceResult, MapValidationError, ResponseValidationError, validate_mapped_dataset
from determinism.modules.semantic_extraction.ports import InferenceError
from determinism.modules.semantic_grouping.adapters import GroupingConfig, PostgresVectorStore, read_postgres_settings
from determinism.modules.semantic_grouping.adapters.postgres_vectors import canonical_float32
from determinism.modules.semantic_grouping.application import GroupingStage
from determinism.modules.semantic_grouping.domain import CompleteLinkageGrouping, cosine_matrix
from determinism.modules.semantic_grouping.ports import EmbeddingError
from determinism.modules.singleton_refinement.adapters import RefinementConfig, load_refinement_config
from determinism.modules.singleton_refinement.application import SingletonRefinementStage
from determinism.modules.singleton_refinement.domain import (
    IncompleteRefinementError, RefinementError, RefinementInputError, validate_grouped_dataset,
)
from determinism.pipelines import Pipeline


def item(text, *, tokens=(0, 1), context=None):
    return {"info": text, "tokens": list(tokens), "context": context or f"Contexto de {text}"}


def grouped_dataset(*sources):
    result = []
    for source_index, segments in enumerate(sources):
        source = {"name": f"source {source_index}", "metadata": {"source": [True, None]}, "segments": []}
        for segment_index, texts in enumerate(segments):
            information = [item(text) for text in texts]
            groups = {}
            for occurrence in information:
                groups.setdefault(occurrence["info"], []).append({
                    "info": occurrence["info"], "context": occurrence["context"], "avg_similarity": 1.0,
                })
            source["segments"].append({
                "content": f"Trecho original {source_index} {segment_index}.",
                "map": {
                    str(index): {"token": index, "word": word, "count": len(information), "metadata": [index]}
                    for index, word in enumerate(("Trecho", "original"))
                },
                "information": information, "set_different_information": list(groups.values()),
                "metadata": {"segment": segment_index}, "start": segment_index / 2,
            })
        result.append(source)
    return result


class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.fetches = []
        self.puts = []
        self.initializations = 0

    def initialize(self):
        self.initializations += 1

    def fetch(self, profile, texts):
        self.fetches.append(tuple(texts))
        return {text: self.rows[(profile.key, text)] for text in texts if (profile.key, text) in self.rows}

    def put(self, profile, vectors):
        self.puts.append(dict(vectors))
        for text, vector in vectors.items():
            self.rows.setdefault((profile.key, text), canonical_float32(vector))


class Embeddings:
    def __init__(self, vectors=None):
        self.vectors = vectors or {
            "A": (1, 0, 0), "B": (1, 0, 0), "C": (0, 1, 0), "C2": (0, 1, 0),
            "D": (1, 0, 0), "E": (0, 1, 0), "X": (0, 0, 1), "Y": (-1, 0, 0),
        }
        self.calls = []
        self.fail_on = set()

    def embed(self, texts):
        self.calls.append(tuple(texts))
        if self.fail_on.intersection(texts):
            raise EmbeddingError("embedding failed")
        return tuple(self.vectors[text] for text in texts)


class Generations:
    def __init__(self, payloads):
        self.payloads = {key: iter(value) for key, value in payloads.items()}
        self.calls = []
        self.lock = threading.Lock()

    def extract(self, segment):
        with self.lock:
            self.calls.append(segment)
            payload = next(self.payloads[(segment.source_index, segment.segment_index)])
        if isinstance(payload, Exception):
            raise payload
        return InferenceResult({"informacoes": payload})


class RefinementFixture:
    def setUp(self):
        self.store = MemoryStore()
        self.embeddings = Embeddings()
        self.profile = GroupingConfig().profile

    def stages(self, provider, *, cap=10, concurrency=4, vectors=None):
        return (ExtractionStage(provider, concurrency, cap),
                GroupingStage(vectors or self.embeddings, self.store, self.profile, CompleteLinkageGrouping()))

    def refine(self, dataset, provider, **options):
        extractor, grouper = self.stages(provider)
        return refine_dataset(dataset, extractor=extractor, grouper=grouper, **options)


class RefinementTests(RefinementFixture, unittest.TestCase):
    def test_exact_occurrences_append_new_unique_embedding_and_reuse_identical_texts(self):
        value = grouped_dataset([["A", "A", "C"]])
        value[0]["segments"][0]["map"]["0"]["count"] = 999
        before = copy.deepcopy(value)
        self.store.put(self.profile, {"A": (1, 0, 0), "C": (0, 1, 0)})
        provider = Generations({(0, 0): [[item("C"), item("C2"), item("C2", context="Outro contexto de C2")]]})
        result = self.refine(value, provider)
        segment = result[0]["segments"][0]
        self.assertEqual(segment["information"], before[0]["segments"][0]["information"] + [item("C"), item("C2"), item("C2", context="Outro contexto de C2")])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0].coverage.pending, ())
        self.assertEqual(self.embeddings.calls, [("C2",)])
        self.assertEqual([entry["count"] for entry in segment["map"].values()], [6, 6])
        expected = Counter((entry["info"], entry["context"]) for entry in segment["information"])
        actual = Counter((entry["info"], entry["context"]) for group in segment["set_different_information"] for entry in group)
        self.assertEqual(actual, expected)
        self.assertEqual([len(group) for group in segment["set_different_information"]], [2, 4])
        for key in ("content", "metadata", "start"):
            self.assertEqual(segment[key], before[0]["segments"][0][key])
        self.assertEqual(segment["map"]["0"]["metadata"], [0])
        self.assertEqual(value, before)

    def test_independent_finish_across_sources_and_segments_stalled_round_is_not_success(self):
        value = grouped_dataset([["A", "A"], ["B", "B", "C"]], [["D", "D", "E"]])
        before = copy.deepcopy(value)
        provider = Generations({(0, 1): [[item("C")]], (1, 0): [[], [item("E")]]})
        checkpoints = []
        fetch_counts = []
        def observe(current):
            checkpoints.append(current)
            fetch_counts.append(len(self.store.fetches))
        extractor, grouper = self.stages(provider)
        observed_grouping = Mock(wraps=grouper)
        result = refine_dataset(value, extractor=extractor, grouper=observed_grouping, on_progress=observe)
        self.assertEqual(Counter((call.source_index, call.segment_index) for call in provider.calls), {(0, 1): 1, (1, 0): 2})
        self.assertEqual([call.coverage.generation for call in provider.calls if call.source_index == 1], [1, 11])
        self.assertEqual(result[0]["segments"][0], before[0]["segments"][0])
        self.assertEqual(result[0]["segments"][1], checkpoints[0][0]["segments"][1])
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual([entry.singleton_groups for entry in validate_grouped_dataset(checkpoints[0])], [1])
        self.assertFalse(validate_grouped_dataset(result))
        self.assertEqual(self.embeddings.calls, [("B", "C", "D", "E")])
        self.assertTrue(all(set(texts) == {"D", "E"} for texts in self.store.fetches[fetch_counts[0]:]))
        self.assertTrue(all("A" not in texts for texts in self.store.fetches))
        self.assertEqual([call.args[1] for call in observed_grouping.group_segments.call_args_list],
                         [frozenset({(0, 1), (1, 0)}), frozenset({(1, 0)})])
        self.assertEqual(value, before)

    def test_forced_generations_share_the_extraction_global_concurrency_bound(self):
        value = grouped_dataset([["A", "A", "C"], ["A", "A", "C"]],
                                [["A", "A", "C"], ["A", "A", "C"]])
        barrier, lock = threading.Barrier(2), threading.Lock()
        active = peak = 0
        calls = []
        class ConcurrentProvider:
            def extract(self, segment):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    calls.append(segment.path)
                try:
                    barrier.wait(timeout=3)
                    return InferenceResult({"informacoes": [item("C")]})
                finally:
                    with lock:
                        active -= 1
        extractor, grouper = self.stages(ConcurrentProvider(), concurrency=2)
        result = refine_dataset(value, extractor=extractor, grouper=grouper)
        self.assertEqual(peak, 2)
        self.assertEqual(set(calls), {"$[0].segments[0]", "$[0].segments[1]", "$[1].segments[0]", "$[1].segments[1]"})
        self.assertFalse(validate_grouped_dataset(result))

    def test_normal_extract_skips_covered_but_selected_extract_forces_once_with_original_indices(self):
        value = grouped_dataset([["A", "A"]], [["C", "C", "E"]])
        before = copy.deepcopy(value)
        provider = Generations({(1, 0): [[item("E")]]})
        self.assertEqual(extract_dataset(value, provider=provider), value)
        self.assertFalse(provider.calls)
        extractor, _ = self.stages(provider)
        result = extractor.extract_segments(value, frozenset({(1, 0)}))
        self.assertEqual(provider.calls[0].path, "$[1].segments[0]")
        self.assertEqual(len(result[1]["segments"][0]["information"]), 4)
        self.assertEqual(result[0], before[0])
        self.assertEqual(value, before)

    def test_complete_linkage_rejects_chain_and_means_describe_final_groups(self):
        vectors = {text: (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
                   for text, angle in (("A", 0), ("B", 10), ("C", 40), ("C2", 40))}
        value = grouped_dataset([["A", "B", "C"]])
        members = [member for group in value[0]["segments"][0]["set_different_information"] for member in group]
        value[0]["segments"][0]["set_different_information"] = [members[:2], members[2:]]
        provider = Generations({(0, 0): [[item("C2")]]})
        extractor, grouper = self.stages(provider, vectors=Embeddings(vectors))
        result = refine_dataset(value, extractor=extractor, grouper=grouper)
        groups = result[0]["segments"][0]["set_different_information"]
        self.assertEqual([[entry["info"] for entry in group] for group in groups], [["A", "B"], ["C", "C2"]])
        for group in groups:
            matrix = cosine_matrix(tuple(self.store.rows[(self.profile.key, entry["info"])] for entry in group))
            for index, entry in enumerate(group):
                others = [similarity for other, similarity in enumerate(matrix[index]) if other != index]
                self.assertTrue(all(similarity >= .85 for similarity in others))
                self.assertAlmostEqual(entry["avg_similarity"], sum(others) / len(others))

    def test_new_singleton_after_original_is_resolved_requires_another_round(self):
        value = grouped_dataset([["A", "A", "C"]])
        provider = Generations({(0, 0): [[item("C"), item("X")], [item("X")]]})
        checkpoints = []
        result = self.refine(value, provider, on_progress=checkpoints.append)
        self.assertEqual(len(provider.calls), 2)
        first_groups = checkpoints[0][0]["segments"][0]["set_different_information"]
        self.assertEqual([group[0]["info"] for group in first_groups if len(group) == 1], ["X"])
        for checkpoint in checkpoints:
            segment = checkpoint[0]["segments"][0]
            self.assertEqual(Counter((entry["info"], entry["context"]) for entry in segment["information"]),
                             Counter((entry["info"], entry["context"]) for group in segment["set_different_information"] for entry in group))
            for group in segment["set_different_information"]:
                matrix = cosine_matrix(tuple(self.store.rows[(self.profile.key, entry["info"])] for entry in group))
                self.assertTrue(all(matrix[left][right] >= .85 for left in range(len(group)) for right in range(left)))
        self.assertFalse(validate_grouped_dataset(result))

    def test_no_singletons_empty_information_and_one_large_group_use_no_external_ports(self):
        values = ([], [{"segments": []}], grouped_dataset([[], ["A", "A", "A"]]))
        for value in values:
            with self.subTest(value=value):
                provider = Generations({})
                before = copy.deepcopy(value)
                result = self.refine(value, provider)
                self.assertEqual(result, value)
                self.assertEqual(value, before)
                self.assertFalse(provider.calls)
        self.assertFalse(self.embeddings.calls)
        self.assertFalse(self.store.fetches)

    def test_every_segment_partition_is_prevalidated_including_duplicates_contexts_empty_groups(self):
        def missing(segment):
            del segment["set_different_information"]
        modifications = [missing,
            lambda segment: segment.update(set_different_information=None),
            lambda segment: segment["set_different_information"].append([]),
            lambda segment: segment["set_different_information"][0].pop(),
            lambda segment: segment["set_different_information"][0].append(copy.deepcopy(segment["set_different_information"][0][0])),
            lambda segment: segment["set_different_information"][0][0].update(context="contexto desatualizado"),
            lambda segment: segment["set_different_information"].append(["invalid"]),
        ]
        for modify in modifications:
            value = grouped_dataset([["C"]], [["A", "A"]])
            modify(value[1]["segments"][0])
            provider = Generations({})
            with self.subTest(modify=modify), self.assertRaisesRegex(RefinementInputError, r"\$\[1\].*determinism group"):
                self.refine(value, provider)
            self.assertFalse(provider.calls)
        self.assertFalse(self.store.fetches)
        self.assertFalse(self.embeddings.calls)

    def test_whole_dataset_maps_and_histories_are_validated_before_calls(self):
        for field in ("map", "information"):
            value = grouped_dataset([["C"]], [["A", "A"]])
            if field == "map":
                value[1]["segments"][0]["map"]["0"]["count"] = False
                error = MapValidationError
            else:
                value[1]["segments"][0]["information"][0]["tokens"] = [0, 99]
                error = ResponseValidationError
            provider = Generations({})
            with self.subTest(field=field), self.assertRaises(error):
                self.refine(value, provider)
            self.assertFalse(provider.calls)
        self.assertFalse(self.store.fetches)

    def test_limit_returns_only_coherent_deepcopied_progress_and_resumes_without_recounting_history_twice(self):
        value = grouped_dataset([["A", "A", "C"]])
        provider = Generations({(0, 0): [[], [item("C")]]})
        checkpoints = []
        with self.assertRaises(IncompleteRefinementError) as raised:
            self.refine(value, provider, max_rounds=1, on_progress=checkpoints.append)
        error = raised.exception
        self.assertEqual(error.completed_rounds, 1)
        self.assertEqual(error.pending[0].path, "$[0].segments[0]")
        self.assertEqual(error.pending[0].singleton_groups, 1)
        self.assertIn("max_rounds=1", str(error))
        self.assertEqual(error.partial_dataset, checkpoints[0])
        result = self.refine(error.partial_dataset, provider, max_rounds=1)
        self.assertEqual(len(result[0]["segments"][0]["information"]), 4)
        self.assertEqual(result[0]["segments"][0]["map"]["0"]["count"], 4)
        error.partial_dataset[0]["metadata"]["source"].append("changed")
        self.assertNotEqual(error.partial_dataset, checkpoints[0])
        self.assertEqual(value[0]["metadata"]["source"], [True, None])

    def test_callbacks_cannot_modify_the_next_round_input(self):
        value = grouped_dataset([["A", "A", "C"]])
        provider = Generations({(0, 0): [[], [item("C")]]})
        def observer(current):
            current[0]["segments"][0]["information"].clear()
        result = self.refine(value, provider, on_progress=observer)
        self.assertEqual(len(result[0]["segments"][0]["information"]), 4)

    def test_selected_extraction_failure_reports_original_path_and_discards_ungrouped_partial(self):
        value = grouped_dataset([["A", "A"]], [["C"]])
        segment = value[1]["segments"][0]
        segment["map"]["2"] = {"token": 2, "word": "mais", "count": 1}
        segment["information"][0]["tokens"] = [0, 1, 2]
        provider = Generations({(1, 0): [[item("C2")]]})
        extractor, grouper = self.stages(provider, cap=1)
        checkpoints = []
        with self.assertRaisesRegex(IncompleteRefinementError, r"\$\[1\].segments\[0\].*cobertura incompleta") as raised:
            refine_dataset(value, extractor=extractor, grouper=grouper, on_progress=checkpoints.append)
        self.assertEqual(raised.exception.partial_dataset, value)
        self.assertEqual(raised.exception.completed_rounds, 0)
        self.assertFalse(checkpoints)
        self.assertFalse(self.embeddings.calls)
        self.assertFalse(self.store.fetches)

    def test_covered_requests_use_shared_builder_base_schema_sampling_and_real_empty_pending(self):
        value = grouped_dataset([["A", "A", "C"]])
        provider = Generations({(0, 0): [[item("C")]]})
        self.refine(value, provider)
        segment = provider.calls[0]
        adapter = OpenAICompatibleProvider(ExtractionConfig(), "test-key")
        body = adapter.build_request_body(segment)
        base = adapter.build_request_body(replace(segment, coverage=None))
        self.assertEqual({key: val for key, val in body.items() if key != "messages"},
                         {key: val for key, val in base.items() if key != "messages"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": user_message(segment)})
        self.assertEqual(body["messages"][0]["content"], system_prompt() + "\n\n" + coverage_message(segment))
        prompt = body["messages"][0]["content"]
        self.assertIn("Execute novamente a extração completa", prompt)
        self.assertNotIn("concentre a revisão", prompt)
        self.assertNotIn("singleton", prompt)
        self.assertNotIn(segment.content, prompt)
        state = json.loads(prompt.split("<coverage_state>\n", 1)[1].split("\n</coverage_state>", 1)[0])
        self.assertEqual(state, {"generation": 1, "minimum_count": 2, "pending": []})
        self.assertEqual((body["model"], body["temperature"], body["seed"], body["reasoning_effort"]), ("gpt-6-luna", 0, 42, "none"))
        self.assertEqual(body["response_format"]["json_schema"]["schema"], structured_output()["schema"])
        self.assertEqual(CoverageFeedback(1, ()).pending, ())


class RefinementPersistenceTests(RefinementFixture, unittest.TestCase):
    def run_file(self, source, provider, output=None, **options):
        return run_refinement(source, output, provider=provider, embedding_provider=self.embeddings, vector_store=self.store, **options)

    def test_provider_or_embedding_failure_keeps_last_completed_round_and_output_override(self):
        for failure in ("provider", "embedding"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                self.setUp()
                source = Path(directory) / "source.json"
                output = Path(directory) / "result.json"
                value = grouped_dataset([["A", "A", "C"]])
                source.write_text(json.dumps(value), encoding="utf-8")
                before = source.read_bytes()
                output.write_text("old-output", encoding="utf-8")
                second = InferenceError("provider failed") if failure == "provider" else [item("Y")]
                self.embeddings.fail_on = {"Y"}
                provider = Generations({(0, 0): [[item("X")], second]})
                with self.assertRaises(IncompleteRefinementError) as raised:
                    self.run_file(source, provider, output)
                saved = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(raised.exception.partial_dataset, saved)
                self.assertEqual(raised.exception.completed_rounds, 1)
                self.assertIn(str(output), str(raised.exception))
                self.assertEqual([entry["info"] for entry in saved[0]["segments"][0]["information"]], ["A", "A", "C", "X"])
                self.assertEqual(validate_grouped_dataset(saved)[0].singleton_groups, 2)
                resume = Generations({(0, 0): [[item("C"), item("X")]]})
                result = self.run_file(output, resume)
                self.assertFalse(validate_grouped_dataset(result))
                self.assertEqual(result[0]["segments"][0]["map"]["0"]["count"], 6)
                self.assertEqual(len(result[0]["segments"][0]["information"]), 6)

    def test_failed_first_phase_never_saves_old_groups_with_appended_information(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "result.json"
            value = grouped_dataset([["A", "A", "C"]])
            source.write_text(json.dumps(value), encoding="utf-8")
            output.write_text("old-output", encoding="utf-8")
            before = source.read_bytes(), output.read_bytes()
            self.embeddings.fail_on = {"X"}
            provider = Generations({(0, 0): [[item("X")]]})
            with self.assertRaises(IncompleteRefinementError) as raised:
                self.run_file(source, provider, output)
            self.assertEqual((source.read_bytes(), output.read_bytes()), before)
            self.assertEqual(raised.exception.partial_dataset, value)
            self.assertEqual(raised.exception.completed_rounds, 0)

    def test_atomic_checkpoint_failure_keeps_previous_round_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            value = grouped_dataset([["A", "A", "C"]])
            source.write_text(json.dumps(value), encoding="utf-8")
            provider = Generations({(0, 0): [[item("X")], [item("C"), item("X")]]})
            import os
            real_replace = os.replace
            replacements = 0
            def replace_once(*args):
                nonlocal replacements
                replacements += 1
                if replacements == 2:
                    raise OSError("atomic checkpoint failed")
                return real_replace(*args)
            with patch("os.replace", side_effect=replace_once), self.assertRaises(OSError):
                self.run_file(source, provider)
            saved = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual(len(saved[0]["segments"][0]["information"]), 4)
            self.assertEqual(validate_grouped_dataset(saved)[0].singleton_groups, 2)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["source.json"])

    def test_preflight_and_noop_precede_credential_reads_adapter_construction_and_database_initialize(self):
        external = {name: Mock(side_effect=AssertionError("external interaction")) for name in (
            "OpenAICompatibleProvider", "OpenAIEmbeddingProvider", "PostgresVectorStore", "read_api_key",
            "read_embedding_api_key", "read_postgres_settings",
        )}
        with tempfile.TemporaryDirectory() as directory, patch.multiple("determinism.experiments.singleton_refinement", **external):
            source = Path(directory) / "source.json"
            output = Path(directory) / "output.json"
            for invalid in ("missing", "stale", "map", "history"):
                value = grouped_dataset([["C"]], [["A", "A"]])
                last = value[1]["segments"][0]
                if invalid == "missing":
                    del last["set_different_information"]
                elif invalid == "stale":
                    last["set_different_information"][0].pop()
                elif invalid == "map":
                    last["map"]["0"]["count"] = False
                else:
                    last["information"][0]["tokens"] = [0, 99]
                source.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaises((RefinementError, MapValidationError, ResponseValidationError)):
                    run_refinement(source)
            value = grouped_dataset([[], ["A", "A"]])
            source.write_text(json.dumps(value), encoding="utf-8")
            before = source.read_bytes()
            self.assertEqual(run_refinement(source, output), value)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), value)
            for dependency in external.values():
                dependency.assert_not_called()

    def test_limit_is_nonzero_cli_and_saves_coherent_checkpoint_then_noop_cli_succeeds_without_services(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps(grouped_dataset([["A", "A", "C"]])), encoding="utf-8")
            provider = Generations({(0, 0): [[]]})
            def configured(input_path, output_path, **options):
                return self.run_file(input_path, provider, output_path, config=options["config"])
            with patch("determinism.entrypoints.cli.run_refinement", side_effect=configured), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main(["refine", str(source), "--max-rounds", "1"])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn("max_rounds=1", stderr.getvalue())
            self.assertIn("Checkpoint coerente salvo", stderr.getvalue())
            self.assertEqual(len(json.loads(source.read_text(encoding="utf-8"))[0]["segments"][0]["information"]), 3)
            source.write_text(json.dumps(grouped_dataset([["A", "A"]])), encoding="utf-8")
            with patch("builtins.print") as printing:
                self.assertEqual(main(["refine", str(source), "--env-file", str(Path(directory) / "missing.env")]), 0)
            self.assertIn("grupos singleton: 0", printing.call_args.args[0])

    def test_experiment_accepts_alternative_pipeline_repository_and_progress_observer(self):
        value = grouped_dataset([["A", "A", "C"]])
        saved = []
        class Repository:
            def load(self, path):
                return value
            def save(self, path, current):
                saved.append((path, copy.deepcopy(current)))
        provider = Generations({(0, 0): [[], [item("C")]]})
        extractor, grouper = self.stages(provider)
        observed = []
        stage = SingletonRefinementStage(extractor, grouper, on_progress=observed.append)
        experiment = SingletonRefinementExperiment(Pipeline((stage,)), Repository())
        result = experiment.run("source.json", "result.json")
        self.assertEqual(len(saved), 2)
        self.assertEqual(len(observed), 2)
        self.assertEqual(saved[-1], (Path("result.json"), result))
        alternative = SingletonRefinementExperiment(Pipeline((lambda current: copy.deepcopy(current),)), Repository())
        self.assertEqual(alternative.run("source.json"), value)
        self.assertEqual(saved[-1][0], Path("source.json"))


@contextmanager
def local_ai_server():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            if self.path == "/v1/chat/completions":
                response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"informacoes": [item("C2")]})}}]}
            else:
                response = {"data": [{"index": index, "embedding": [1, 0] if text == "A" else [0, 1]}
                                     for index, text in enumerate(body["input"])]}
            raw = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class RefinementConfigurationTests(unittest.TestCase):
    def test_packaged_loop_config_and_explicit_overrides_are_independent(self):
        self.assertEqual(load_refinement_config(), RefinementConfig(max_rounds=10))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refine.toml"
            path.write_text("max_rounds=2", encoding="utf-8")
            self.assertEqual(load_refinement_config(path).max_rounds, 2)
            for content in ("max_rounds=0", "max_rounds=true", 'max_rounds="2"', 'model="private-input"', "not-valid-toml"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(RefinementError) as raised:
                    load_refinement_config(path)
                self.assertNotIn("private-input", str(raised.exception))

    def test_cli_uses_existing_http_adapters_and_all_three_explicit_configs(self):
        with tempfile.TemporaryDirectory() as directory, local_ai_server() as (url, calls):
            root = Path(directory)
            source, output = root / "source.json", root / "output.json"
            source.write_text(json.dumps(grouped_dataset([["A", "A", "C"]])), encoding="utf-8")
            before = source.read_bytes()
            extraction, grouping, loop = root / "extract.toml", root / "group.toml", root / "loop.toml"
            extraction.write_text(f'provider="openai-compatible"\nmodel="local-extraction"\nbase_url="{url}"\nreasoning_effort="omit"\ntemperature=0.1\nseed=24\nmax_concurrency=2\nmax_generations_per_segment=1\nmax_completion_tokens=900\napi_key_env="LOCAL_REFINEMENT_KEY"', encoding="utf-8")
            grouping.write_text(f'provider="openai-compatible"\nmodel="local-embeddings"\nbase_url="{url}"\nthreshold=0.9\ndimensions=2\napi_key_env="LOCAL_REFINEMENT_KEY"\ncache_revision="test-v2"', encoding="utf-8")
            loop.write_text("max_rounds=1", encoding="utf-8")
            store = MemoryStore()
            with patch("determinism.experiments.singleton_refinement.PostgresVectorStore", return_value=store), patch("determinism.experiments.singleton_refinement.read_postgres_settings", return_value=object()), patch("builtins.print") as printing:
                self.assertEqual(main(["refine", str(source), "--output", str(output), "--config", str(loop),
                                       "--extraction-config", str(extraction), "--grouping-config", str(grouping), "--env-file", str(root / "missing.env")]), 0)
            self.assertEqual(source.read_bytes(), before)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(validate_grouped_dataset(result))
            self.assertEqual(store.initializations, 1)
            self.assertEqual([path for path, _ in calls], ["/v1/chat/completions", "/v1/embeddings"])
            request = calls[0][1]
            self.assertEqual((request["model"], request["temperature"], request["seed"], request["max_completion_tokens"]), ("local-extraction", .1, 24, 900))
            self.assertNotIn("reasoning_effort", request)
            self.assertTrue(request["messages"][0]["content"].startswith(system_prompt()))
            self.assertEqual(request["response_format"]["json_schema"]["schema"], structured_output()["schema"])
            self.assertEqual(calls[1][1]["model"], "local-embeddings")
            self.assertEqual(calls[1][1]["dimensions"], 2)
            self.assertEqual(calls[1][1]["input"], ["A", "C", "C2"])
            self.assertIn("grupos singleton: 0", printing.call_args.args[0])


@unittest.skipUnless(os.environ.get("GROUPING_TEST_POSTGRES") == "1", "Integração PostgreSQL opt-in: GROUPING_TEST_POSTGRES=1")
class RefinementPostgresIntegrationTests(unittest.TestCase):
    def test_real_cache_new_embeddings_repeat_hits_and_resolved_resume_with_local_http(self):
        store = PostgresVectorStore(read_postgres_settings())
        store.initialize()
        with tempfile.TemporaryDirectory() as directory, local_ai_server() as (url, calls):
            extraction = ExtractionConfig(provider="openai-compatible", model="local-extraction", base_url=url,
                                          reasoning_effort="omit", api_key_env="LOCAL_REFINEMENT_INTEGRATION_KEY",
                                          max_concurrency=1, max_generations_per_segment=1)
            grouping = GroupingConfig(provider="openai-compatible", model="local-embeddings", base_url=url,
                                      api_key_env="LOCAL_REFINEMENT_INTEGRATION_KEY", dimensions=2,
                                      cache_revision="refinement-integration-" + uuid.uuid4().hex)
            source, output = Path(directory) / "source.json", Path(directory) / "output.json"
            source.write_text(json.dumps(grouped_dataset([["A", "A", "C"]])), encoding="utf-8")
            source_before = source.read_bytes()
            try:
                first = run_refinement(source, output, config=RefinementConfig(1), extraction_config=extraction,
                                       grouping_config=grouping, vector_store=store, env_file=None)
                self.assertFalse(validate_grouped_dataset(first))
                self.assertEqual(source.read_bytes(), source_before)
                self.assertEqual(set(store.fetch(grouping.profile, ("A", "C", "C2"))), {"A", "C", "C2"})
                self.assertEqual([path for path, _ in calls], ["/v1/chat/completions", "/v1/embeddings"])
                output_before = output.read_bytes()
                second = run_refinement(source, output, config=RefinementConfig(1), extraction_config=extraction,
                                        grouping_config=grouping, vector_store=store, env_file=None)
                self.assertEqual(first, second)
                self.assertEqual(output.read_bytes(), output_before)
                self.assertEqual([path for path, _ in calls], ["/v1/chat/completions", "/v1/embeddings", "/v1/chat/completions"])
                resolved = run_refinement(output, config=RefinementConfig(1), extraction_config=extraction,
                                          grouping_config=grouping, env_file=None)
                self.assertEqual(resolved, first)
                self.assertEqual(output.read_bytes(), output_before)
                self.assertEqual(len(calls), 3)
            finally:
                with store._connect() as connection:
                    connection.execute("DELETE FROM semantic_grouping.embeddings WHERE profile_key = %s", (grouping.profile.key,))
                    connection.execute("DELETE FROM semantic_grouping.embedding_profiles WHERE profile_key = %s", (grouping.profile.key,))


if __name__ == "__main__":
    unittest.main()
