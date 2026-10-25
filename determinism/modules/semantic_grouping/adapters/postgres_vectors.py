"""Cache durável pgvector, sem índice ANN: similaridades são locais ao segmento."""

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..domain import EmbeddingProfile, GroupingError, Vector, validate_vector
from ..ports import VectorStoreError
from .configuration import PostgresSettings


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS semantic_grouping;
CREATE TABLE IF NOT EXISTS semantic_grouping.embedding_profiles (
    profile_key text PRIMARY KEY,
    profile_metadata jsonb NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions BETWEEN 1 AND 16000)
);
CREATE TABLE IF NOT EXISTS semantic_grouping.embeddings (
    profile_key text NOT NULL REFERENCES semantic_grouping.embedding_profiles(profile_key),
    profile_metadata jsonb NOT NULL,
    text_hash text NOT NULL,
    text_content text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions BETWEEN 1 AND 16000),
    embedding vector NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (profile_key, text_hash),
    CHECK (vector_dims(embedding) = dimensions)
);
"""


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_float32(vector: Vector) -> Vector:
    """Mesmo domínio numérico de pgvector; rejeita overflow e underflow a zero."""
    try:
        converted = tuple(struct.unpack("!f", struct.pack("!f", component))[0] for component in vector)
    except (OverflowError, struct.error):
        raise VectorStoreError("Embedding não pode ser representado em float32 no pgvector.") from None
    try:
        return validate_vector(converted, len(vector))
    except GroupingError:
        raise VectorStoreError("Embedding inválido após conversão para float32 do pgvector.") from None


@dataclass(frozen=True)
class PostgresVectorStore:
    settings: PostgresSettings

    def _connect(self):
        try:
            import psycopg
        except ImportError:
            raise VectorStoreError("Adaptador PostgreSQL requer psycopg[binary]; instale o projeto com pip install -e .") from None
        try:
            return psycopg.connect(
                host=self.settings.host, port=self.settings.port, dbname=self.settings.dbname,
                user=self.settings.user, password=self.settings.password, connect_timeout=self.settings.connect_timeout,
            )
        except psycopg.Error:
            raise VectorStoreError("Não foi possível conectar ao cache PostgreSQL; verifique serviço e variáveis GROUPING_DB_*.") from None

    def initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(SCHEMA_SQL)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError("Não foi possível inicializar extensão vector, schema e tabela de embeddings.") from None

    def fetch(self, profile: EmbeddingProfile, texts: Sequence[str]) -> dict[str, Vector]:
        if not texts:
            return {}
        requested = {_text_hash(text): text for text in texts}
        result: dict[str, Vector] = {}
        try:
            with self._connect() as connection:
                stored_profile = connection.execute(
                    "SELECT profile_metadata, dimensions FROM semantic_grouping.embedding_profiles WHERE profile_key = %s",
                    (profile.key,),
                ).fetchone()
                if stored_profile is None:
                    return {}
                metadata, dimension = stored_profile
                if metadata != profile.metadata or (profile.dimensions is not None and dimension != profile.dimensions):
                    raise VectorStoreError("Perfil ou dimensão do cache não corresponde à configuração solicitada.")
                rows = connection.execute(
                    "SELECT profile_metadata, text_hash, text_content, dimensions, embedding::text "
                    "FROM semantic_grouping.embeddings WHERE profile_key = %s AND text_hash = ANY(%s)",
                    (profile.key, list(requested)),
                ).fetchall()
            for metadata, text_hash, text, stored_dimension, encoded in rows:
                if metadata != profile.metadata or requested.get(text_hash) != text:
                    raise VectorStoreError("Metadados ou identidade textual do cache não correspondem ao perfil solicitado.")
                vector = canonical_float32(validate_vector(json.loads(encoded), dimension))
                if len(vector) != stored_dimension:
                    raise VectorStoreError("Dimensão armazenada não corresponde ao vetor do cache.")
                dimension = dimension or stored_dimension
                result[text] = vector
        except GroupingError:
            raise
        except Exception:
            raise VectorStoreError("Não foi possível ler embeddings válidos do cache PostgreSQL.") from None
        return result

    def put(self, profile: EmbeddingProfile, vectors: Mapping[str, Vector]) -> None:
        if not vectors:
            return
        rows = []
        dimension = profile.dimensions
        for text, value in vectors.items():
            vector = canonical_float32(validate_vector(value, dimension))
            dimension = dimension or len(vector)
            rows.append((profile.key, json.dumps(profile.metadata, ensure_ascii=False), _text_hash(text), text, len(vector), json.dumps(vector, allow_nan=False)))
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO semantic_grouping.embedding_profiles (profile_key, profile_metadata, dimensions) "
                    "VALUES (%s, %s::jsonb, %s) ON CONFLICT (profile_key) DO NOTHING",
                    (profile.key, json.dumps(profile.metadata, ensure_ascii=False), dimension),
                )
                stored_profile = connection.execute(
                    "SELECT profile_metadata, dimensions FROM semantic_grouping.embedding_profiles WHERE profile_key = %s",
                    (profile.key,),
                ).fetchone()
                if stored_profile != (profile.metadata, dimension):
                    raise VectorStoreError("Perfil já persistido tem metadados ou dimensão diferentes; altere cache_revision.")
                with connection.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO semantic_grouping.embeddings "
                        "(profile_key, profile_metadata, text_hash, text_content, dimensions, embedding) "
                        "VALUES (%s, %s::jsonb, %s, %s, %s, %s::vector) "
                        "ON CONFLICT (profile_key, text_hash) DO NOTHING", rows,
                    )
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError("Não foi possível persistir embeddings no cache PostgreSQL; a transação foi revertida.") from None
