import json
import math
import os
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from determinism.entrypoints.cli import main
from determinism.experiments import group_dataset
from determinism.modules.semantic_grouping.adapters import GroupingConfig, PostgresSettings, PostgresVectorStore, read_postgres_settings
from determinism.modules.semantic_grouping.adapters.postgres_vectors import _text_hash, canonical_float32
from determinism.modules.semantic_grouping.domain import EmbeddingProfile
from determinism.modules.semantic_grouping.ports import VectorStoreError


class PostgresAdapterTests(unittest.TestCase):
    def setUp(self):
        self.profile = GroupingConfig().profile
        self.store = PostgresVectorStore(PostgresSettings(password="never-print-this"))

    def connection(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        return connection

    def test_connection_and_sql_errors_do_not_expose_password_dsn_or_driver_details(self):
        import psycopg
        with patch("psycopg.connect", side_effect=psycopg.OperationalError("postgresql://never-print-this@server")):
            with self.assertRaises(VectorStoreError) as caught:
                self.store.initialize()
        self.assertNotIn("never-print-this", str(caught.exception))
        connection = self.connection()
        connection.execute.side_effect = RuntimeError("never-print-this")
        with patch.object(PostgresVectorStore, "_connect", return_value=connection):
            with self.assertRaises(VectorStoreError) as caught:
                self.store.initialize()
        self.assertNotIn("never-print-this", str(caught.exception))
        self.assertNotIn("never-print-this", repr(self.store))

    def test_fetch_verifies_metadata_text_dimensions_and_uses_parameterized_identity(self):
        connection = self.connection()
        profile_result = MagicMock()
        profile_result.fetchone.return_value = (self.profile.metadata, 2)
        vector_result = MagicMock()
        vector_result.fetchall.return_value = [(self.profile.metadata, _text_hash("exact text"), "exact text", 2, "[0.85,0.5]")]
        connection.execute.side_effect = [profile_result, vector_result]
        with patch.object(PostgresVectorStore, "_connect", return_value=connection):
            self.assertEqual(self.store.fetch(self.profile, ["exact text"]), {"exact text": canonical_float32((.85, .5))})
        query, parameters = connection.execute.call_args.args
        self.assertNotIn("exact text", query)
        self.assertEqual(parameters, (self.profile.key, [_text_hash("exact text")]))
        for row in (
            ({"wrong": "profile"}, _text_hash("exact text"), "exact text", 2, "[1,0]"),
            (self.profile.metadata, _text_hash("exact text"), "wrong text", 2, "[1,0]"),
            (self.profile.metadata, _text_hash("exact text"), "exact text", 3, "[1,0]"),
            (self.profile.metadata, _text_hash("exact text"), "exact text", 2, "[0,0]"),
        ):
            connection.execute.side_effect = [profile_result, vector_result]
            vector_result.fetchall.return_value = [row]
            with self.subTest(row=row), patch.object(PostgresVectorStore, "_connect", return_value=connection):
                with self.assertRaises(ValueError):
                    self.store.fetch(self.profile, ["exact text"])

    def test_idempotent_put_is_one_transaction_and_refuses_profile_dimension_drift(self):
        connection = self.connection()
        profile_result = MagicMock()
        profile_result.fetchone.return_value = (self.profile.metadata, 2)
        connection.execute.side_effect = [MagicMock(), profile_result]
        with patch.object(PostgresVectorStore, "_connect", return_value=connection):
            self.store.put(self.profile, {"text with 'quotes'": (1, .1)})
        cursor = connection.cursor.return_value.__enter__.return_value
        query, rows = cursor.executemany.call_args.args
        self.assertIn("ON CONFLICT (profile_key, text_hash) DO NOTHING", query)
        self.assertNotIn("quotes", query)
        self.assertEqual(rows[0][3], "text with 'quotes'")
        self.assertEqual(json.loads(rows[0][-1]), list(canonical_float32((1, .1))))
        connection.execute.side_effect = [MagicMock(), profile_result]
        profile_result.fetchone.return_value = (self.profile.metadata, 3)
        with patch.object(PostgresVectorStore, "_connect", return_value=connection):
            with self.assertRaisesRegex(VectorStoreError, "dimensão"):
                self.store.put(self.profile, {"another": (1, .1)})
        self.assertIs(connection.__exit__.call_args.args[0], VectorStoreError)

    def test_empty_requests_do_not_open_database_connections(self):
        with patch.object(PostgresVectorStore, "_connect") as connect:
            self.assertEqual(self.store.fetch(self.profile, []), {})
            self.store.put(self.profile, {})
            connect.assert_not_called()


@unittest.skipUnless(os.environ.get("GROUPING_TEST_POSTGRES") == "1", "Integração PostgreSQL opt-in: GROUPING_TEST_POSTGRES=1")
class PostgresIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PostgresVectorStore(read_postgres_settings())
        cls.store.initialize()

    def setUp(self):
        self.owned_profiles = []
        self.revision = "integration-" + uuid.uuid4().hex

    def own(self, profile):
        self.owned_profiles.append(profile.key)
        return profile

    def tearDown(self):
        # Cada perfil pertence exclusivamente a este teste (revision UUID).
        if self.owned_profiles:
            with self.store._connect() as connection:
                connection.execute("DELETE FROM semantic_grouping.embeddings WHERE profile_key = ANY(%s)", (self.owned_profiles,))
                connection.execute("DELETE FROM semantic_grouping.embedding_profiles WHERE profile_key = ANY(%s)", (self.owned_profiles,))

    def test_real_schema_idempotence_dynamic_dimensions_and_durable_upsert(self):
        profile = self.own(EmbeddingProfile("integration-test", "fixture", "memory://postgres", cache_revision=self.revision))
        vectors = {"persisted-a": (1, .1), "persisted-b": (0, 1)}
        self.store.initialize()
        self.store.put(profile, vectors)
        self.store.put(profile, {"persisted-a": (1, .1)})
        reopened = PostgresVectorStore(read_postgres_settings())
        self.assertEqual(reopened.fetch(profile, list(vectors)), {text: canonical_float32(value) for text, value in vectors.items()})
        with reopened._connect() as connection:
            count = connection.execute("SELECT count(*) FROM semantic_grouping.embeddings WHERE profile_key=%s", (profile.key,)).fetchone()[0]
        self.assertEqual(count, 2)
        with self.assertRaisesRegex(VectorStoreError, "dimensão"):
            reopened.put(profile, {"wrong-dimension": (1, 0, 0)})
        other = self.own(EmbeddingProfile("integration-test", "fixture", "memory://postgres", dimensions=3, cache_revision=self.revision))
        reopened.put(other, {"persisted-a": (1, 0, 0)})
        self.assertEqual(reopened.fetch(other, ["persisted-a"]), {"persisted-a": (1, 0, 0)})

    def test_real_float32_canonicalization_and_cached_run_are_identical(self):
        profile = self.own(EmbeddingProfile("integration-test", "precision", "memory://postgres", cache_revision=self.revision))
        value = [{"segments": [{"content": "preserve", "information": [
            {"info": "precision-a", "context": "first"}, {"info": "precision-b", "context": "second"},
        ]}]}]
        class Provider:
            def __init__(self):
                self.calls = 0
            def embed(self, texts):
                self.calls += 1
                vectors = {"precision-a": (1, 0), "precision-b": (.8499999999, math.sqrt(1 - .8499999999**2))}
                return tuple(vectors[text] for text in texts)
        provider = Provider()
        first = group_dataset(value, provider=provider, vector_store=self.store, profile=profile)
        second = group_dataset(first, provider=provider, vector_store=self.store, profile=profile)
        self.assertEqual(first, second)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(first[0]["segments"][0]["set_different_information"]), 1)

    def test_full_cli_local_embeddings_real_postgres_and_repeated_cache_hit(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                payload = {"data": [{"index": index, "embedding": [1, 0] if text == "cli-a" else [0, 1]} for index, text in reversed(list(enumerate(body["input"])))]}
                encoded = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with TemporaryDirectory() as directory:
                source = Path(directory) / "input.json"
                config = Path(directory) / "local.toml"
                source.write_text(json.dumps([{"segments": [{"content": "same", "information": [
                    {"info": "cli-a", "context": "a"}, {"info": "cli-a", "context": "a2"}, {"info": "cli-b", "context": "b"},
                ]}]}]), encoding="utf-8")
                endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                self.own(GroupingConfig(provider="openai-compatible", model="local-integration-fixture", base_url=endpoint, cache_revision=self.revision).profile)
                config.write_text(f'provider="openai-compatible"\nmodel="local-integration-fixture"\nbase_url="{endpoint}"\ncache_revision="{self.revision}"', encoding="utf-8")
                with patch("builtins.print"):
                    self.assertEqual(main(["group", str(source), "--config", str(config)]), 0)
                    first = source.read_bytes()
                    self.assertEqual(main(["group", str(source), "--config", str(config)]), 0)
                self.assertEqual(first, source.read_bytes())
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["input"], ["cli-a", "cli-b"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
