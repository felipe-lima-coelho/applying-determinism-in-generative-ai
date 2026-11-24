"""Escolha de uma variante original para representar cada grupo validado."""

from collections.abc import Mapping

from .contracts import GroupingError, InformationOccurrence


def canonical_representative(
    occurrences: tuple[InformationOccurrence, ...], member_scores: Mapping[int, float],
) -> int:
    """Maior média cosseno; empates preservam texto e contexto lexicais exatos.

    Para n vetores unitários e centroide c não nulo, cos(u_i, c) é
    [1 + (n - 1) * member_mean(i)] / [n * ||c||]. Assim, a escolha
    equivale à variante mais próxima por cosseno do centroide de ocorrências,
    sem recalcular vetores. Repetições representam sua própria frequência.
    Com centroide nulo, a soma das distâncias cosseno continua definida.
    O contexto só desempata proveniências de um mesmo texto, nunca o score.
    """
    if not member_scores:
        raise GroupingError("Um grupo canônico deve conter ao menos uma ocorrência.")
    return min(member_scores, key=lambda index: (
        -member_scores[index], occurrences[index].info, occurrences[index].context, index,
    ))
