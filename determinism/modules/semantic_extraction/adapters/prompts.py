"""Sistema base fixo, feedback numérico, dados do segmento e schema da API."""

import json
from importlib.resources import files
from xml.sax.saxutils import escape

from ..domain import MIN_TOKEN_COVERAGE, SegmentInput


def system_prompt() -> str:
    return files("determinism.modules.semantic_extraction").joinpath("assets/system_prompt.v1.txt").read_text(encoding="utf-8")


def structured_output() -> dict:
    resource = files("determinism.modules.semantic_extraction").joinpath("assets/structured_output.v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def coverage_message(segment: SegmentInput) -> str | None:
    feedback = segment.coverage
    if feedback is None:
        return None
    if any(token not in segment.token_ids or count >= MIN_TOKEN_COVERAGE for token, count in feedback.pending):
        raise ValueError("O feedback deve referenciar somente IDs conhecidos com cobertura pendente.")
    state = json.dumps(
        {"generation": feedback.generation, "minimum_count": MIN_TOKEN_COVERAGE,
         "pending": [{"token": token, "count": count, "needed": MIN_TOKEN_COVERAGE - count}
                     for token, count in feedback.pending]},
        ensure_ascii=True, separators=(",", ":"),
    )
    return (
        "Esta é uma geração de um processo de extração com cobertura acumulada. "
        "O estado numérico abaixo foi calculado a partir de referências já validadas; "
        "count é o número de informações acumuladas que citam cada ID, minimum_count é a meta "
        "e needed é o número de referências que ainda faltam. Essa meta é acumulada e pode "
        "ser cumprida em gerações diferentes; não crie duas afirmações artificiais só para atingi-la. "
        "Nesta geração, concentre a revisão nas afirmações relacionadas aos IDs pendentes, "
        "em vez de repetir afirmações sem relação com eles. Para cada ID pendente, localize a "
        "ocorrência exata no mapa e revise a oração original inteira no trecho da mensagem do usuário. "
        "Extraia afirmações completas e fiéis que sejam sustentadas por essas ocorrências, "
        "associando todas as palavras que de fato sustentam cada afirmação. "
        "A associação refere-se à passagem original e aos IDs de cada ocorrência, não a uma "
        "correspondência lexical com a paráfrase produzida. "
        "Inclua palavras funcionais e ocorrências repetidas quando participarem da afirmação; "
        "não omita um ID só porque outra ocorrência da mesma palavra já foi citada. "
        "Revise também verbos de ligação, auxiliares e construções de realce: seus IDs podem "
        "sustentar uma identificação, circunstância ou evento da afirmação completa, mesmo "
        "quando a reformulação não reproduz literalmente essas palavras. "
        "Se uma tentativa anterior não avançou, reexamine a associação dos IDs pendentes às afirmações. "
        "A meta nunca autoriza inventar fatos, criar fragmentos gramaticais, fabricar associações "
        "ou repetir IDs dentro de uma informação. Preserve a fidelidade semântica e retorne uma "
        "lista vazia ou parcial quando não houver outras associações sustentadas pelo trecho. "
        "Siga o sistema base e o schema da API.\n\n"
        f"<coverage_state>\n{state}\n</coverage_state>"
    )


def user_message(segment: SegmentInput) -> str:
    # Entidades protegem os delimitadores. No JSON, escapes Unicode preservam
    # exatamente as palavras e continuam sendo JSON diretamente decodificável.
    mapping = json.dumps(segment.projected_map(), ensure_ascii=False, indent=2)
    mapping = mapping.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return (
        "Extraia as informações semânticas presentes no trecho abaixo, seguindo todas as definições e regras estabelecidas no sistema.\n\n"
        "Para cada informação, preencha info, tokens, context e syntagm. A tripla em syntagm deve representar a mesma informação de info, "
        "e tokens deve conter exclusivamente os IDs das palavras originais que sustentam essa informação.\n\n"
        f"<trecho>\n{escape(segment.content)}\n</trecho>\n\n"
        f"<mapeamento_tokens>\n{mapping}\n</mapeamento_tokens>"
    )
