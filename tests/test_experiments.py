import unittest
from pathlib import Path

from determinism.experiments import WordMappingExperiment
from determinism.modules.text_mapping.application import MappingStage
from determinism.pipelines import Pipeline


class ExperimentTests(unittest.TestCase):
    def test_alternative_pipeline_and_repository_are_used_as_configured(self):
        class AlternativeTokenizer:
            def __init__(self):
                self.calls = []

            def tokenize(self, text):
                self.calls.append(text)
                return [text]

        class MemoryRepository:
            def __init__(self):
                self.loaded = []
                self.saved = []

            def load(self, path):
                self.loaded.append(path)
                return [{"name": path.stem, "segments": [{"content": "Uma expressão inteira."}]}]

            def save(self, path, dataset):
                self.saved.append((path, dataset))

        observed = []

        def observe(dataset):
            observed.append(dataset)
            return dataset

        tokenizer = AlternativeTokenizer()
        repository = MemoryRepository()
        pipeline = Pipeline((MappingStage(tokenizer), observe))
        experiment = WordMappingExperiment(pipeline=pipeline, repository=repository)

        first = experiment.run("primeiro.json", "resultado.json")
        second = experiment.run("segundo.json")

        self.assertIs(experiment.pipeline, pipeline)
        self.assertIs(experiment.repository, repository)
        self.assertEqual(repository.loaded, [Path("primeiro.json"), Path("segundo.json")])
        self.assertEqual(repository.saved[0][0], Path("resultado.json"))
        self.assertEqual(repository.saved[1][0], Path("segundo.json"))
        self.assertIs(repository.saved[0][1], first)
        self.assertIs(repository.saved[1][1], second)
        self.assertIs(observed[0], first)
        self.assertIs(observed[1], second)
        self.assertEqual(tokenizer.calls, ["Uma expressão inteira.", "Uma expressão inteira."])
        self.assertEqual(
            first[0]["segments"][0]["map"],
            {"0": {"token": 0, "word": "Uma expressão inteira.", "count": 0}},
        )
        self.assertEqual(second[0]["name"], "segundo")


if __name__ == "__main__":
    unittest.main()
