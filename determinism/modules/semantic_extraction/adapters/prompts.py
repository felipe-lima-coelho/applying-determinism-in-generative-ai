"""Os três canais do contrato: sistema fixo, dados do segmento e schema da API."""

import json
from importlib.resources import files
from xml.sax.saxutils import escape

from ..domain import SegmentInput


def system_prompt() -> str:
    return files("determinism.modules.semantic_extraction").joinpath("assets/system_prompt.v1.txt").read_text(encoding="utf-8")


def structured_output() -> dict:
    resource = files("determinism.modules.semantic_extraction").joinpath("assets/structured_output.v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def user_message(segment: SegmentInput) -> str:
    # Entidades protegem os delimitadores. No JSON, escapes Unicode preservam
    # exatamente as palavras e continuam sendo JSON diretamente decodificável.
    mapping = json.dumps(segment.projected_map(), ensure_ascii=False, indent=2)
    mapping = mapping.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return (
        "Extraia as informações semânticas presentes no trecho abaixo, seguindo todas as definições e regras estabelecidas no sistema.\n\n"
        f"<trecho>\n{escape(segment.content)}\n</trecho>\n\n"
        f"<mapeamento_tokens>\n{mapping}\n</mapeamento_tokens>"
    )
