"""Cossenos globais, hierarquia complete linkage e representante medoid."""

import math
from itertools import combinations

from .contracts import CatalogError


APPLICATION_REVISION = "catalog-v2-global-complete-link-medoid-1"


def validate_threshold(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
        raise CatalogError("similarity_threshold deve ser um número finito entre -1 e 1.")
    return float(value)


def validate_embeddings(payload, count, dimensions=None):
    if not isinstance(payload, dict) or set(payload) != {"embeddings"}:
        raise CatalogError("Resposta de embeddings deve conter somente embeddings.")
    vectors = payload["embeddings"]
    if not isinstance(vectors, list) or len(vectors) != count:
        raise CatalogError("Embeddings deve conter exatamente um vetor por texto.")
    result = []
    for vector in vectors:
        if not isinstance(vector, list) or not 1 <= len(vector) <= 16000:
            raise CatalogError("Embedding deve ser um vetor não vazio com até 16000 dimensões.")
        dimensions = dimensions or len(vector)
        if len(vector) != dimensions:
            raise CatalogError("Dimensão de embedding incompatível com o perfil ou os demais vetores.")
        try:
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
                raise CatalogError("Embedding deve conter somente números finitos.")
            checked = tuple(float(value) for value in vector)
            norm = math.hypot(*checked)
        except OverflowError:
            raise CatalogError("Embedding fora do domínio numérico finito.") from None
        if not norm or not math.isfinite(norm):
            raise CatalogError("Embedding deve possuir norma finita e não nula.")
        result.append(checked)
    return tuple(result)


def cosine_matrix(vectors):
    normalized = []
    for vector in vectors:
        norm = math.hypot(*vector)
        normalized.append(tuple(value / norm for value in vector))
    rows = [[1.0 if first == second else 0.0 for second in range(len(vectors))]
            for first in range(len(vectors))]
    for first, second in combinations(range(len(vectors)), 2):
        cosine = math.fsum(a * b for a, b in zip(normalized[first], normalized[second]))
        rows[first][second] = rows[second][first] = max(-1.0, min(1.0, cosine))
    return tuple(tuple(row) for row in rows)


def complete_linkage(candidate_ids, comparisons, threshold):
    """Maior mínimo elegível; empates usam índices do snapshot, nunca strings c10/c2."""
    validate_threshold(threshold)
    position = {key: index for index, key in enumerate(candidate_ids)}
    matrix = [[1.0 if first == second else 0.0 for second in range(len(position))]
              for first in range(len(position))]
    eligible = set()
    for pair in comparisons:
        first, second = position[pair["left_id"]], position[pair["right_id"]]
        matrix[first][second] = matrix[second][first] = pair["similarity"]
        if pair["relation"] == "equivalent" and pair["similarity"] >= threshold:
            eligible.add(frozenset((first, second)))
    groups = tuple((index,) for index in range(len(position)))
    merges = []
    while True:
        best = None
        for first, second in combinations(range(len(groups)), 2):
            left, right = groups[first], groups[second]
            if not all(frozenset((a, b)) in eligible for a in left for b in right):
                continue
            similarity = min(matrix[a][b] for a in left for b in right)
            choice = (-similarity, left, right, first, second)
            if best is None or choice < best:
                best = choice
        if best is None:
            break
        negative_score, left, right, first, second = best
        union = tuple(sorted((*left, *right)))
        merges.append({"left_candidate_ids": [candidate_ids[index] for index in left],
                       "right_candidate_ids": [candidate_ids[index] for index in right],
                       "similarity": -negative_score,
                       "candidate_ids": [candidate_ids[index] for index in union]})
        groups = tuple(sorted([group for index, group in enumerate(groups) if index not in (first, second)] + [union]))
    representatives = []
    for group in groups:
        # Inclui a diagonal (igual para todos). fsum evita dependência da ordem da soma.
        medoid = min(group, key=lambda index: (-math.fsum(matrix[index][other] for other in group), index))
        representatives.append(candidate_ids[medoid])
    return [[candidate_ids[index] for index in group] for group in groups], merges, representatives
