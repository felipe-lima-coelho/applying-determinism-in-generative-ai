import copy
import unittest
from pathlib import Path

from determinism.experiments import map_dataset, run_mapping
from determinism.modules.text_mapping.application import MappingStage
from determinism.modules.text_mapping.domain import DatasetFormatError
from determinism.modules.text_mapping.domain.tokenization import UnicodeWordTokenizer
from determinism.pipelines import Pipeline


class CountingTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize(self, text):
        self.calls.append(text)
        return UnicodeWordTokenizer().tokenize(text)


class MappingTests(unittest.TestCase):
    def test_required_example_has_eight_ordered_occurrences(self):
        dataset = [{"segments": [{"content": "A casa é verde, porém não tem porta."}]}]

        mapped = map_dataset(dataset)

        self.assertEqual(
            mapped[0]["segments"][0]["map"],
            {
                "0": {"token": 0, "word": "A", "count": 0},
                "1": {"token": 1, "word": "casa", "count": 0},
                "2": {"token": 2, "word": "é", "count": 0},
                "3": {"token": 3, "word": "verde", "count": 0},
                "4": {"token": 4, "word": "porém", "count": 0},
                "5": {"token": 5, "word": "não", "count": 0},
                "6": {"token": 6, "word": "tem", "count": 0},
                "7": {"token": 7, "word": "porta", "count": 0},
            },
        )

    def test_all_sources_and_segments_reset_maps_and_preserve_metadata(self):
        dataset = [
            {
                "name": "primeira",
                "metadata": {"language": "pt-BR", "weights": [0.5, 1]},
                "segments": [
                    {
                        "id": "a",
                        "content": "Olá, Olá!",
                        "start": 1.25,
                        "map": {"8": {"token": 8, "word": "antiga", "count": 42}},
                    },
                    {"content": "Outra casa.", "map": "mapa antigo"},
                ],
            },
            {"name": "segunda", "segments": [{"content": "É a casa!"}]},
        ]
        original = copy.deepcopy(dataset)

        mapped = map_dataset(dataset)

        self.assertEqual(dataset, original)
        self.assertIsNot(mapped, dataset)
        expected_words = [["Olá", "Olá"], ["Outra", "casa"], ["É", "a", "casa"]]
        segments = [segment for source in mapped for segment in source["segments"]]
        for segment, words in zip(segments, expected_words, strict=True):
            self.assertEqual(list(segment["map"]), [str(i) for i in range(len(words))])
            self.assertEqual([entry["word"] for entry in segment["map"].values()], words)
            self.assertEqual([entry["token"] for entry in segment["map"].values()], list(range(len(words))))
            self.assertTrue(all(entry["count"] == 0 for entry in segment["map"].values()))
        for source, previous in zip(mapped, original, strict=True):
            self.assertEqual(
                {key: value for key, value in source.items() if key != "segments"},
                {key: value for key, value in previous.items() if key != "segments"},
            )
            for segment, previous_segment in zip(source["segments"], previous["segments"], strict=True):
                self.assertEqual(
                    {key: value for key, value in segment.items() if key != "map"},
                    {key: value for key, value in previous_segment.items() if key != "map"},
                )
        mapped[0]["metadata"]["weights"].append(2)
        self.assertEqual(dataset, original)

    def test_explicit_execution_always_rebuilds_map_and_resets_count(self):
        mapped = map_dataset([{"segments": [{"content": "a a"}]}])
        previous_map = mapped[0]["segments"][0]["map"]
        previous_map["0"]["count"] = 9
        previous_map["extra"] = {"token": 99, "word": "obsoleta", "count": 12}

        rebuilt = map_dataset(mapped)

        word_map = rebuilt[0]["segments"][0]["map"]
        self.assertIsNot(word_map, previous_map)
        self.assertEqual(list(word_map), ["0", "1"])
        self.assertEqual([entry["count"] for entry in word_map.values()], [0, 0])
        self.assertEqual(previous_map["0"]["count"], 9)

    def test_empty_dataset_sources_and_contents(self):
        self.assertEqual(map_dataset([]), [])
        self.assertEqual(map_dataset([{"segments": []}]), [{"segments": []}])
        for content in ("", " \n\t ", "... , : ; — - ' ’ _ 🙂", "\u0301\u0308"):
            with self.subTest(content=content):
                mapped = map_dataset([{"segments": [{"content": content, "map": {"old": 1}}]}])
                self.assertEqual(mapped[0]["segments"][0]["map"], {})

    def test_malformed_structure_reports_location_before_tokenizing(self):
        invalid_values = [
            (None, "lista de fontes"),
            ({}, "lista de fontes"),
            ([None], "$[0]"),
            ([{}], "$[0].segments"),
            ([{"segments": None}], "$[0].segments"),
            ([{"segments": {}}], "$[0].segments"),
            ([{"segments": [None]}], "$[0].segments[0]"),
            ([{"segments": [{}]}], "$[0].segments[0].content"),
            ([{"segments": [{"content": 123}]}], "$[0].segments[0].content"),
            ([{"segments": [{"content": None}]}], "$[0].segments[0].content"),
            ([{"segments": [{"content": "válido"}]}, {"segments": [{}]}], "$[1].segments[0].content"),
        ]
        for dataset, location in invalid_values:
            with self.subTest(dataset=dataset):
                tokenizer = CountingTokenizer()
                original = copy.deepcopy(dataset)
                with self.assertRaises(DatasetFormatError) as raised:
                    map_dataset(dataset, tokenizer=tokenizer)
                self.assertIn(location, str(raised.exception))
                self.assertEqual(tokenizer.calls, [])
                self.assertEqual(dataset, original)

    def test_pipeline_reuses_same_dataset_and_map_without_tokenizing_again(self):
        tokenizer = CountingTokenizer()
        observed = []

        def observe(dataset):
            observed.append((dataset, dataset[0]["segments"][0]["map"]))
            return dataset

        pipeline = Pipeline((MappingStage(tokenizer), observe, observe))
        mapped = pipeline.run([{"segments": [{"content": "Uma casa."}]}])
        word_map = mapped[0]["segments"][0]["map"]

        self.assertEqual(tokenizer.calls, ["Uma casa."])
        self.assertEqual(len(observed), 2)
        for dataset, observed_map in observed:
            self.assertIs(dataset, mapped)
            self.assertIs(observed_map, word_map)
        self.assertEqual(word_map["0"]["word"], "Uma")
        self.assertEqual(word_map["1"]["word"], "casa")
        self.assertEqual(tokenizer.calls, ["Uma casa."])

    def test_tokenizer_is_replaceable(self):
        class WholeContentTokenizer:
            def tokenize(self, text):
                return [text]

        mapped = map_dataset(
            [{"segments": [{"content": "A casa."}]}],
            tokenizer=WholeContentTokenizer(),
        )
        self.assertEqual(
            mapped[0]["segments"][0]["map"],
            {"0": {"token": 0, "word": "A casa.", "count": 0}},
        )

    def test_repository_is_replaceable_and_saved_result_is_returned(self):
        class MemoryRepository:
            def load(self, path):
                self.input_path = path
                return [{"name": "memória", "segments": [{"content": "Olá."}]}]

            def save(self, path, dataset):
                self.output_path = path
                self.saved = dataset

        repository = MemoryRepository()

        mapped = run_mapping("entrada.json", "saida.json", repository=repository)

        self.assertEqual(repository.input_path, Path("entrada.json"))
        self.assertEqual(repository.output_path, Path("saida.json"))
        self.assertIs(repository.saved, mapped)
        self.assertEqual(mapped[0]["segments"][0]["map"]["0"]["word"], "Olá")


class UnicodeTokenizerTests(unittest.TestCase):
    def test_case_accents_repetitions_and_combining_marks_are_preserved(self):
        text = "É, é! É? café cafe\u0301 ÁGUA água; 2026 ١٢ 中文."
        self.assertEqual(
            list(UnicodeWordTokenizer().tokenize(text)),
            ["É", "é", "É", "café", "cafe\u0301", "ÁGUA", "água", "2026", "١٢", "中文"],
        )

    def test_punctuation_symbols_hyphens_and_apostrophes_separate_words(self):
        text = "Mountbatten-Windsor d'água d’Água guarda_chuva 12,5 🙂 @Olá\t‘fim’—sim"
        self.assertEqual(
            list(UnicodeWordTokenizer().tokenize(text)),
            ["Mountbatten", "Windsor", "d", "água", "d", "Água", "guarda", "chuva", "12", "5", "Olá", "fim", "sim"],
        )


if __name__ == "__main__":
    unittest.main()
