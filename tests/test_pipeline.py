import unittest

from determinism.pipelines import Pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_is_generic_and_applies_stages_in_order(self):
        pipeline = Pipeline[int]((lambda value: value + 1, lambda value: value * 2))

        self.assertEqual(pipeline.run(2), 6)

    def test_empty_pipeline_preserves_input_identity(self):
        value = {"qualquer": "contrato"}

        self.assertIs(Pipeline(()).run(value), value)


if __name__ == "__main__":
    unittest.main()
