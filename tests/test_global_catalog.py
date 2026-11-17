"""Catálogo global dos representantes locais, proveniência e persistência pareada."""

import copy
import io
import json
import math
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from determinism.entrypoints.cli import main
from determinism.experiments import (
    build_global_catalog, group_dataset, group_dataset_with_catalog, run_grouping, run_grouping_with_catalog,
)
from determinism.modules.semantic_grouping.adapters import GroupingConfig, global_catalog_output_path
from determinism.modules.semantic_grouping.domain import GroupingError, cosine_matrix
from determinism.modules.semantic_grouping.ports import VectorStoreError
from test_semantic_grouping import FakeEmbeddings, MemoryVectorStore, dataset


class GlobalCatalogTests(unittest.TestCase):
    def setUp(self):
        self.profile = GroupingConfig().profile
        self.store = MemoryVectorStore()
        self.provider = FakeEmbeddings({"A": (1, 0), "alias": (1, 0), "B": (1, 0), "C": (0, 1)})

    def group(self, value, **options):
        return group_dataset_with_catalog(value, provider=self.provider, vector_store=self.store,
                                          profile=self.profile, **options)

    def build(self, value, **options):
        return build_global_catalog(value, vector_store=self.store, profile=self.profile, **options)

    def test_cross_source_and_same_source_duplicates_keep_local_results_and_all_provenance(self):
        value = dataset(["A", "alias"], ["A"])
        value.append(dataset(["B", "C"])[0])
        value[0]["name"], value[1]["name"] = "source one", "source two"
        original = copy.deepcopy(value)
        result = self.group(value)
        self.assertEqual(value, original)
        self.assertEqual(result.dataset, group_dataset(value, provider=self.provider, vector_store=self.store, profile=self.profile))
        catalog = result.global_catalog
        self.assertEqual(catalog["metadata"]["counts"], {
            "sources": 2, "segments": 3, "original_information": 5,
            "local_canonical_information": 4, "global_canonical_information": 2,
        })
        self.assertEqual([item["info"] for item in catalog["information"]], ["A", "A", "B", "C"])
        self.assertEqual([item["member_indices"] for item in catalog["canonical_information"]], [[0, 1, 2], [3]])
        self.assertEqual([len(group) for group in catalog["set_different_information"]], [3, 1])
        self.assertEqual([source["metadata"]["name"] for source in catalog["sources"]], ["source one", "source two"])
        self.assertEqual(catalog["sources"][0]["segments"][1]["metadata"], {"metadata": {"segment": 1}})
        covered = []
        for item in catalog["information"]:
            provenance = item["provenance"]
            segment = result.dataset[provenance["source_index"]]["segments"][provenance["segment_index"]]
            canonical = segment["canonical_information"][provenance["canonical_index"]]
            self.assertEqual(item["info"], canonical["info"])
            self.assertEqual(provenance["representative_index"], canonical["representative_index"])
            self.assertEqual(provenance["member_indices"], canonical["member_indices"])
            covered.extend((provenance["source_index"], provenance["segment_index"], index) for index in provenance["member_indices"])
        self.assertEqual(sorted(covered), [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1)])
        self.assertEqual(self.provider.calls, [("A", "alias", "B", "C")])

    def test_global_matrix_contains_only_local_representatives_with_one_vote_each(self):
        self.provider.vectors = {
            text: (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            for text, angle in (("A", 0), ("B", 10), ("C", 20))
        }
        value = dataset(["A"] * 20, ["B"], ["C"])
        with patch("determinism.modules.semantic_grouping.application.global_catalog.cosine_matrix", wraps=cosine_matrix) as matrix:
            result = self.group(value)
        self.assertEqual(matrix.call_count, 1)
        self.assertEqual(len(matrix.call_args.args[0]), 3)
        catalog = result.global_catalog
        self.assertEqual(catalog["canonical_information"][0]["info"], "B")
        self.assertEqual(catalog["canonical_information"][0]["member_indices"], [0, 1, 2])
        self.assertEqual(catalog["information"][0]["provenance"]["member_indices"], list(range(20)))
        self.assertEqual(self.store.fetches[-1][1], ("A", "B", "C"))
        # O centroide direto usa três representantes, não as 22 folhas originais.
        vectors = matrix.call_args.args[0]
        unit = [tuple(component / math.hypot(*vector) for component in vector) for vector in vectors]
        centroid = tuple(math.fsum(vector[dimension] for vector in unit) / 3 for dimension in range(2))
        closest = max(range(3), key=lambda index: math.fsum(a * b for a, b in zip(unit[index], centroid)))
        self.assertEqual(catalog["canonical_information"][0]["representative_index"], closest)

    def test_global_complete_linkage_blocks_chain_and_honors_inclusive_threshold(self):
        value = dataset(["A"], ["B"], ["C"])
        matrix = ((1, .90, .70), (.90, 1, .90), (.70, .90, 1))
        with patch("determinism.modules.semantic_grouping.application.global_catalog.cosine_matrix", return_value=matrix):
            catalog = self.group(value).global_catalog
        self.assertEqual(catalog["metadata"]["threshold"], .90)
        self.assertEqual([item["member_indices"] for item in catalog["canonical_information"]], [[0, 1], [2]])
        for item in catalog["canonical_information"]:
            for first in item["member_indices"]:
                self.assertTrue(all(matrix[first][second] >= .90 for second in item["member_indices"]))

    def test_explicit_threshold_is_shared_by_local_and_global_stages(self):
        self.provider.vectors["B"] = (math.cos(math.radians(30)), math.sin(math.radians(30)))
        value = dataset(["A"], ["B"])
        self.assertEqual(len(self.group(value).global_catalog["canonical_information"]), 2)
        catalog = self.group(value, threshold=.85).global_catalog
        self.assertEqual(catalog["metadata"]["threshold"], .85)
        self.assertEqual(catalog["canonical_information"][0]["member_indices"], [0, 1])

    def test_context_tokens_syntagm_and_metadata_do_not_enter_global_scores(self):
        value = dataset(["A"], ["B"])
        for index, segment in enumerate(value[0]["segments"]):
            segment["information"][0].update({"context": "unrelated " * (index + 1), "tokens": [999 + index],
                                                "syntagm": {"subject": str(index)}, "extra": {"index": index}})
        result = self.group(value)
        self.assertEqual(self.provider.calls, [("A", "B")])
        self.assertEqual(len(result.global_catalog["canonical_information"]), 1)
        self.assertEqual(result.global_catalog["canonical_information"][0]["avg_similarity"], 1.0)
        self.assertEqual(result.dataset[0]["segments"][1]["information"], value[0]["segments"][1]["information"])
        self.assertTrue(all(set(item) == {"info", "context", "provenance"} for item in result.global_catalog["information"]))

    def test_deterministic_global_representative_ties_and_original_index_fallback(self):
        for order in permutations((("B", "context-z"), ("A", "context-z"), ("A", "context-a"))):
            value = dataset(*([text] for text, _ in order))
            for segment, (_, context) in zip(value[0]["segments"], order):
                segment["information"][0]["context"] = context
            item = self.group(value).global_catalog["canonical_information"][0]
            self.assertEqual((item["info"], item["context"]), ("A", "context-a"))
            self.assertEqual(item["provenance"]["segment_index"], order.index(("A", "context-a")))
        value = dataset(["A"], ["A"])
        for segment in value[0]["segments"]:
            segment["information"][0]["context"] = "same"
        self.assertEqual(self.group(value).global_catalog["canonical_information"][0]["representative_index"], 0)

    def test_empty_and_singleton_catalogues_do_not_require_external_dependencies(self):
        for value in ([], [{"segments": []}], dataset([], [])):
            catalog = self.group(value).global_catalog
            self.assertEqual(catalog["information"], [])
            self.assertEqual(catalog["set_different_information"], [])
            self.assertEqual(catalog["canonical_information"], [])
        self.assertFalse(self.provider.calls)
        self.assertFalse(self.store.fetches)
        catalog = self.group(dataset(["A"])).global_catalog
        self.assertEqual(catalog["canonical_information"][0]["member_indices"], [0])
        self.assertEqual(catalog["canonical_information"][0]["avg_similarity"], 1.0)

    def test_malformed_or_stale_local_provenance_is_rejected_before_cache_access(self):
        valid = self.group(dataset(["A", "alias"])).dataset
        before = len(self.store.fetches)
        for change in ({"member_indices": [0]}, {"member_indices": [0, 0, 1]}, {"member_indices": [False, 1]},
                       {"member_indices": [0, 2]}, {"representative_index": 2}, {"info": "stale"}):
            value = copy.deepcopy(valid)
            value[0]["segments"][0]["canonical_information"][0].update(change)
            with self.subTest(change=change), self.assertRaises(GroupingError):
                self.build(value)
        value = copy.deepcopy(valid)
        del value[0]["segments"][0]["canonical_information"]
        with self.assertRaisesRegex(GroupingError, "local primeiro"):
            self.build(value)
        self.assertEqual(len(self.store.fetches), before)

    def test_missing_or_invalid_persisted_canonical_vectors_and_invalid_partition_fail(self):
        value = self.group(dataset(["A"], ["C"])).dataset
        original = copy.deepcopy(value)
        with patch.object(self.store, "fetch", return_value={}):
            with self.assertRaises(VectorStoreError):
                self.build(value)
        with patch.object(self.store, "fetch", return_value={"A": (0, 0), "C": (0, 1)}):
            with self.assertRaises(GroupingError):
                self.build(value)
        class InvalidStrategy:
            def partition(self, matrix, threshold):
                return ((0, 1),)
        with self.assertRaisesRegex(GroupingError, "par abaixo"):
            self.build(value, strategy=InvalidStrategy())
        self.assertEqual(value, original)
        self.assertEqual(len(self.provider.calls), 1)


class GlobalCatalogPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryVectorStore()
        self.provider = FakeEmbeddings({"A": (1, 0), "B": (0, 1)})

    def run_group(self, source, output=None, **options):
        return run_grouping_with_catalog(source, output, provider=self.provider, vector_store=self.store, **options)

    def test_default_sidecar_and_both_cached_outputs_are_byte_identical(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "grouped.json"
            source.write_text(json.dumps(dataset(["A", "A"], ["A", "B"])), encoding="utf-8")
            before = source.read_bytes()
            result = self.run_group(source, output)
            catalog_path = global_catalog_output_path(output)
            self.assertEqual(catalog_path.name, "grouped.global.json")
            self.assertEqual(json.loads(catalog_path.read_text(encoding="utf-8")), result.global_catalog)
            self.assertEqual(result.global_catalog["metadata"]["dataset_file"], "grouped.json")
            first = output.read_bytes(), catalog_path.read_bytes()
            # API legada continua retornando uma lista; a repetição recompõe os dois artefatos.
            returned = run_grouping(output, provider=self.provider, vector_store=self.store)
            self.assertIsInstance(returned, list)
            self.assertEqual(returned, result.dataset)
            self.assertEqual((output.read_bytes(), catalog_path.read_bytes()), first)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(self.provider.calls), 1)

    def test_cli_explicit_sidecar_and_empty_input_work_without_credentials(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, catalog = root / "input.json", root / "grouped.json", root / "catalog.json"
            source.write_text(json.dumps(dataset([], [])), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["group", str(source), "--output", str(output), "--global-output", str(catalog),
                                       "--env-file", str(root / "absent.env")]), 0)
            value = json.loads(catalog.read_text(encoding="utf-8"))
            self.assertEqual(value["information"], [])
            self.assertEqual(value["metadata"]["counts"]["global_canonical_information"], 0)
            self.assertIn(str(catalog), stdout.getvalue())
            self.assertIn("representantes locais: 0", stdout.getvalue())
            self.assertFalse((root / "grouped.global.json").exists())

    def test_explicit_sidecar_in_another_directory_records_relative_primary_reference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalogs").mkdir()
            source, catalog = root / "input.json", root / "catalogs" / "catalog.json"
            source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
            result = self.run_group(source, global_output_path=catalog)
            self.assertEqual(result.global_catalog["metadata"]["dataset_file"], "../input.json")

    def test_catalogue_cannot_collide_with_input_or_primary_even_through_aliases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, alias = root / "input.json", root / "output.json", root / "alias.json"
            source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
            os.link(source, alias)
            collisions = [source, root / "." / source.name, output, alias]
            if os.name == "nt":
                collisions.append(root / "INPUT.JSON")
            for collision in collisions:
                with self.subTest(collision=collision), self.assertRaisesRegex(GroupingError, "destino diferente"):
                    self.run_group(source, output, global_output_path=collision)
            before = source.read_bytes()
            with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as raised:
                main(["group", str(source), "--global-output", str(source)])
            self.assertEqual(raised.exception.code, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(self.provider.calls)
            self.assertFalse(output.exists())

    def test_symbolic_link_destinations_preserve_links_and_reference_the_actual_primary(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, catalog = root / "input.json", root / "alias.json", root / "catalog-alias.json"
            primary_target, catalog_target = root / "actual.json", root / "catalogs" / "actual-catalog.json"
            catalog_target.parent.mkdir()
            source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
            primary_target.write_bytes(b"previous-primary")
            catalog_target.write_bytes(b"previous-catalog")
            try:
                output.symlink_to(primary_target)
                catalog.symlink_to(catalog_target)
            except OSError:
                self.skipTest("Criação de symlink não permitida neste ambiente.")
            result = self.run_group(source, output, global_output_path=catalog)
            self.assertTrue(output.is_symlink())
            self.assertTrue(catalog.is_symlink())
            self.assertEqual(json.loads(primary_target.read_text(encoding="utf-8")), result.dataset)
            self.assertEqual(json.loads(catalog_target.read_text(encoding="utf-8")), result.global_catalog)
            reference = result.global_catalog["metadata"]["dataset_file"]
            self.assertEqual(reference, "../actual.json")
            self.assertEqual((catalog_target.parent / reference).resolve(), primary_target.resolve())

    def test_failed_second_replacement_restores_existing_or_absent_primary_and_cleans_staging(self):
        for existing in (False, True):
            with self.subTest(existing=existing), TemporaryDirectory() as directory:
                root = Path(directory)
                source, output, catalog = root / "input.json", root / "output.json", root / "output.global.json"
                source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
                if existing:
                    output.write_bytes(b"original-primary")
                    catalog.write_bytes(b"original-catalog")
                before = {path.name: path.read_bytes() for path in root.iterdir()}
                replace = os.replace
                def fail_catalog(first, second):
                    if Path(second) == catalog:
                        raise OSError("simulated catalog replacement failure")
                    return replace(first, second)
                with patch("os.replace", side_effect=fail_catalog), self.assertRaises(OSError):
                    self.run_group(source, output)
                self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)

    def test_catalog_failure_and_serialization_error_preserve_both_existing_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, catalog = root / "input.json", root / "output.json", root / "output.global.json"
            source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
            output.write_bytes(b"old-primary")
            catalog.write_bytes(b"old-catalog")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with patch("determinism.experiments.semantic_grouping.build_global_catalog", side_effect=GroupingError("bad catalogue")):
                with self.assertRaises(GroupingError):
                    self.run_group(source, output)
            with patch("determinism.experiments.semantic_grouping.build_global_catalog", return_value={"metadata": {}, "bad": {1}}):
                with self.assertRaisesRegex(GroupingError, "serializar"):
                    self.run_group(source, output)
            self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)

    def test_failed_catalog_replace_restores_in_place_input(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            catalog = global_catalog_output_path(source)
            source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
            before = source.read_bytes()
            replace = os.replace
            def fail_catalog(first, second):
                if Path(second) == catalog:
                    raise OSError("catalog replacement failed")
                return replace(first, second)
            with patch("os.replace", side_effect=fail_catalog), self.assertRaises(OSError):
                self.run_group(source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_failed_rollback_retains_previous_primary_for_recovery(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.json", root / "output.json"
            catalog = global_catalog_output_path(output)
            source.write_text(json.dumps(dataset(["A"])), encoding="utf-8")
            output.write_bytes(b"previous-primary")
            catalog.write_bytes(b"previous-catalog")
            replace, primary_replaced = os.replace, False
            def fail_catalog_and_rollback(first, second):
                nonlocal primary_replaced
                target = Path(second)
                if target == catalog or (target == output and primary_replaced):
                    raise OSError("simulated persistent write failure")
                if target == output:
                    primary_replaced = True
                return replace(first, second)
            with patch("os.replace", side_effect=fail_catalog_and_rollback), self.assertRaisesRegex(GroupingError, "recuperação") as caught:
                self.run_group(source, output)
            backups = list(root.glob("*.tmp"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"previous-primary")
            self.assertIn(str(backups[0]), str(caught.exception))
            self.assertEqual(catalog.read_bytes(), b"previous-catalog")


if __name__ == "__main__":
    unittest.main()
