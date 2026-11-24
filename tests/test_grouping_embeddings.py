import io
import json
import math
import os
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from determinism.modules.semantic_grouping.adapters.configuration import (
    GroupingConfig, GroupingConfigurationError, PostgresSettings, load_grouping_config,
    read_embedding_api_key, read_environment, read_postgres_settings,
)
from determinism.modules.semantic_grouping.adapters.openai_embeddings import MAX_RESPONSE_BYTES, OpenAIEmbeddingProvider
from determinism.modules.semantic_grouping.adapters.postgres_vectors import canonical_float32
from determinism.modules.semantic_grouping.domain import EmbeddingProfile, GroupingError, cosine_matrix, validate_vector
from determinism.modules.semantic_grouping.ports import EmbeddingError, VectorStoreError


class GroupingConfigurationTests(unittest.TestCase):
    def test_packaged_defaults_and_explicit_model_local_endpoint_and_dimensions(self):
        default = load_grouping_config()
        self.assertEqual((default.provider, default.model, default.threshold), ("openai", "text-embedding-3-small", 0.85))
        self.assertEqual(GroupingConfig().threshold, .85)
        self.assertIsNone(default.dimensions)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local.toml"
            path.write_text('provider="openai-compatible"\nmodel="local-model"\nbase_url="http://127.0.0.1:1234/v1"\ndimensions=12\nbatch_size=2\ncache_revision="new-weights"', encoding="utf-8")
            config = load_grouping_config(path)
        self.assertEqual(config.dimensions, 12)
        self.assertEqual(config.profile.cache_revision, "new-weights")

    def test_invalid_config_and_unknown_fields_are_rejected_without_echoing_toml(self):
        for values in (
            {"provider": "other"}, {"model": ""}, {"model": 1}, {"base_url": "http://api.openai.com/v1"},
            {"base_url": "https://user:secret@api.openai.com/v1"}, {"base_url": "https://api.openai.com/v1?secret=x"},
            {"base_url": None}, {"threshold": float("nan")}, {"threshold": True}, {"threshold": 1.01},
            {"dimensions": 0}, {"dimensions": True}, {"dimensions": 16001}, {"batch_size": 2049}, {"batch_size": 0},
            {"timeout_seconds": float("inf")}, {"api_key_env": "secret-value!"}, {"cache_revision": ""},
        ):
            with self.subTest(values=values), self.assertRaises(GroupingConfigurationError):
                GroupingConfig(**values)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for text in ('unknown="secret-value"', 'model = "secret-value'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(GroupingConfigurationError) as caught:
                    load_grouping_config(path)
                self.assertNotIn("secret-value", str(caught.exception))

    def test_environment_precedence_and_secret_representations(self):
        with TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text('OPENAI_API_KEY="file-secret"\nGROUPING_DB_PASSWORD=database-secret\nIGNORED=not-selected\n', encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-secret"}, clear=True):
                self.assertEqual(read_embedding_api_key(GroupingConfig(), env), "environment-secret")
                settings = read_postgres_settings(env)
            self.assertEqual(settings.password, "database-secret")
            self.assertNotIn("database-secret", repr(settings))
            self.assertEqual(read_environment(("OPENAI_API_KEY",), env, environ={}), {"OPENAI_API_KEY": "file-secret"})
        provider = OpenAIEmbeddingProvider(GroupingConfig(), "provider-secret")
        self.assertNotIn("provider-secret", repr(provider))

    def test_missing_key_local_key_optional_and_invalid_database_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GroupingConfigurationError):
                read_embedding_api_key(GroupingConfig(), None)
            self.assertIsNone(read_embedding_api_key(GroupingConfig(provider="openai-compatible", base_url="http://localhost:1234/v1"), None))
            with self.assertRaises(GroupingConfigurationError):
                read_postgres_settings(None)
        with patch.dict(os.environ, {"GROUPING_DB_PORT": "credential-looking-text"}, clear=True):
            with self.assertRaises(GroupingConfigurationError) as caught:
                read_postgres_settings(None)
            self.assertNotIn("credential-looking-text", str(caught.exception))
        for values in ({"port": 0}, {"password": ""}, {"password": "x\ny"}, {"connect_timeout": 0}):
            with self.subTest(values=values), self.assertRaises(GroupingConfigurationError):
                PostgresSettings(**values)

    def test_cache_identity_includes_model_provider_endpoint_dimensions_and_revision(self):
        profile = GroupingConfig().profile
        alternatives = [replace(profile, **values) for values in (
            {"provider": "local"}, {"model": "other"}, {"base_url": "http://localhost:1234/v1"},
            {"dimensions": 256}, {"cache_revision": "2"},
        )]
        self.assertEqual(len({profile.key, *(item.key for item in alternatives)}), 6)
        self.assertEqual(profile.key, replace(profile, base_url=profile.base_url + "/").key)
        self.assertEqual(profile.metadata["encoding_format"], "float")


class EmbeddingAdapterTests(unittest.TestCase):
    def provider(self, **values):
        return OpenAIEmbeddingProvider(GroupingConfig(**values), "test-secret")

    def response(self, vectors):
        entries = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({"data": list(reversed(entries))}).encode()
        return response

    def test_exact_request_batching_and_out_of_order_response_indices(self):
        opener = Mock()
        opener.open.side_effect = [self.response([[1, 0], [0, 1]]), self.response([[0.5, 0.5]])]
        provider = self.provider(batch_size=2, dimensions=2, timeout_seconds=3)
        with patch("urllib.request.build_opener", return_value=opener):
            result = provider.embed(["primeiro", 'segundo " ç', "terceiro"])
        self.assertEqual(result, ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)))
        self.assertEqual(opener.open.call_count, 2)
        request = opener.open.call_args_list[0].args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(json.loads(request.data), {"model": "text-embedding-3-small", "input": ["primeiro", 'segundo " ç'], "encoding_format": "float", "dimensions": 2})
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(opener.open.call_args_list[0].kwargs, {"timeout": 3})

    def test_default_omits_dimensions_and_local_provider_needs_no_key(self):
        provider = OpenAIEmbeddingProvider(GroupingConfig(provider="openai-compatible", base_url="http://127.0.0.1:1234/v1", model="local"))
        opener = Mock()
        opener.open.return_value = self.response([[1, 2, 3]])
        with patch("urllib.request.build_opener", return_value=opener):
            provider.embed(["text"])
        request = opener.open.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertNotIn("dimensions", json.loads(request.data))

    def test_every_text_is_prevalidated_before_any_request_and_empty_is_noop(self):
        provider = self.provider(batch_size=1)
        with patch("urllib.request.build_opener") as opening:
            for texts in (["valid", " "], ["valid", "\x00"], ["valid", "\ud800"], ["valid", None], "text"):
                with self.subTest(texts=texts), self.assertRaises(EmbeddingError):
                    provider.embed(texts)
            self.assertEqual(provider.embed([]), ())
            opening.assert_not_called()

    def test_invalid_indices_and_vectors_are_rejected(self):
        invalid = [None, {"data": []}, {"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [1]}]},
                   {"data": [{"index": True, "embedding": [1]}, {"index": 1, "embedding": [1]}]},
                   {"data": [{"index": 2, "embedding": [1]}, {"index": 1, "embedding": [1]}]}]
        for vector in ([], [0, 0], [float("nan"), 1], [float("inf"), 1], [True, 1], ["1", 0], [1], [1, 0, 1]):
            invalid.append({"data": [{"index": 0, "embedding": vector}, {"index": 1, "embedding": [1, 0]}]})
        for envelope in invalid:
            with self.subTest(envelope=envelope), self.assertRaises(EmbeddingError):
                OpenAIEmbeddingProvider._parse_response(envelope, 2, 2)

    def test_dimension_changes_across_batches_are_rejected(self):
        opener = Mock()
        opener.open.side_effect = [self.response([[1, 0]]), self.response([[1, 0, 1]])]
        with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(EmbeddingError):
            self.provider(batch_size=1).embed(["one", "two"])

    def test_http_transport_and_non_json_errors_are_sanitized_without_retry(self):
        secret = "test-secret"
        errors = [urllib.error.HTTPError("https://server/" + secret, status, secret, {}, io.BytesIO(secret.encode())) for status in (401, 429, 500)]
        errors += [OSError(secret), ValueError(secret)]
        for error in errors:
            opener = Mock()
            opener.open.side_effect = error
            with self.subTest(error=type(error).__name__), patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaises(EmbeddingError) as caught:
                    self.provider().embed(["text"])
                self.assertNotIn(secret, str(caught.exception))
                self.assertEqual(opener.open.call_count, 1)
        for raw in (b"not-json test-secret", b'{"data": [], "data": []}', b'{"data": NaN}'):
            response = self.response([[1]])
            response.read.return_value = raw
            with patch("urllib.request.build_opener", return_value=Mock(open=Mock(return_value=response))):
                with self.assertRaises(EmbeddingError):
                    self.provider().embed(["text"])

    def test_redirects_are_never_followed(self):
        from determinism.modules.semantic_grouping.adapters.openai_embeddings import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 307, "redirect", {}, "https://other"))


class VectorValidationTests(unittest.TestCase):
    def test_zero_nonfinite_wrong_dimension_boolean_and_float32_overflow(self):
        for vector in ([], [0, 0], [True, 1], [float("nan")], [float("inf")], [10**1000], ["1"]):
            with self.subTest(vector=vector), self.assertRaises(GroupingError):
                validate_vector(vector)
        with self.assertRaises(GroupingError):
            validate_vector([1, 0], 3)
        for vector in ((1e100,), (1e-100,)):
            with self.subTest(vector=vector), self.assertRaises(VectorStoreError):
                canonical_float32(vector)

    def test_all_pairs_include_duplicate_occurrences_and_stable_cosine(self):
        matrix = cosine_matrix(((1, 0), (1, 0), (0, 1), (-1, 0)))
        self.assertEqual(matrix, ((1, 1, 0, -1), (1, 1, 0, -1), (0, 0, 1, 0), (-1, -1, 0, 1)))
        self.assertEqual(cosine_matrix(()), ())
        self.assertAlmostEqual(cosine_matrix(((1e100, 0), (0.85e100, math.sqrt(1 - 0.85**2) * 1e100)))[0][1], 0.85)


if __name__ == "__main__":
    unittest.main()
