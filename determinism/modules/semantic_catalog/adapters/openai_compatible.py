"""Inferência estruturada para extração, equivalência e suporte entre segmentos."""

import copy
import json
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..ports import CatalogInferenceError, CatalogResponse
from ..domain.requirements import REQUIREMENT_KINDS
from .configuration import CatalogConfig, ConfigurationError
from .prompts import structured_output, system_prompt


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("invalid JSON constant")


def _decode_json(raw: bytes | str) -> Any:
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _safe_identifier(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.\[\]-]{1,100}", value) else None


def _requirements_contract(template: dict, data: dict) -> dict:
    """Exige uma atribuição para cada token no próprio schema da resposta.

    O template permanece imutável; cada requisição tem seu schema independente,
    inclusive quando vários workers constroem contratos de tamanhos diferentes.
    """
    candidate = data.get("candidate") if isinstance(data, dict) else None
    tokens = candidate.get("statement_tokens") if isinstance(candidate, dict) else None
    if (not isinstance(tokens, list) or not tokens
            or any(not isinstance(token, dict) or type(token.get("id")) is not int
                   or token["id"] != index for index, token in enumerate(tokens))):
        raise CatalogInferenceError("requirements exige statement_tokens não vazios com IDs consecutivos a partir de zero.")
    assignment = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": sorted(REQUIREMENT_KINDS)},
            "group": {"type": "integer", "minimum": 0, "maximum": len(tokens) - 1},
        },
        "required": ["kind", "group"],
    }
    contract = copy.deepcopy(template)
    contract["schema"]["$defs"] = {"requirement_assignment": assignment}
    schema = contract["schema"]["properties"]["requirements_by_token"]
    token_assignments = {
        "type": "array", "minItems": 1, "maxItems": len(tokens),
        "items": {"$ref": "#/$defs/requirement_assignment"},
    }
    schema["properties"] = {f"t{index}": copy.deepcopy(token_assignments) for index in range(len(tokens))}
    schema["required"] = list(schema["properties"])
    return contract


def _records_by_id(value: object, field: str) -> dict:
    if not isinstance(value, list) or not value:
        raise CatalogInferenceError("O contrato dinâmico exige uma lista não vazia de IDs.")
    result = {}
    for item in value:
        identifier = item.get(field) if isinstance(item, dict) else None
        if not isinstance(identifier, str) or not identifier.strip() or identifier in result:
            raise CatalogInferenceError("O contrato dinâmico exige IDs textuais, não vazios e distintos.")
        result[identifier] = item
    return result


def _required_mapping(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def _comparison_contract(template: dict, data: dict) -> dict:
    pairs = _records_by_id(data.get("pairs") if isinstance(data, dict) else None, "pair_id")
    contract = copy.deepcopy(template)
    result = _required_mapping({
        "relation": {"type": "string", "enum": ["equivalent", "different", "uncertain"]},
        "reason": {"type": "string", "minLength": 1},
    })
    mapping = _required_mapping({identifier: {"$ref": "#/$defs/comparison"} for identifier in pairs})
    contract["schema"] = _required_mapping({"comparisons": mapping})
    contract["schema"]["$defs"] = {"comparison": result}
    return contract


def _assessment_contract(template: dict, data: dict) -> dict:
    facts = _records_by_id(data.get("facts") if isinstance(data, dict) else None, "id")
    contract = copy.deepcopy(template)
    result = _required_mapping({
        "status": {"type": "string", "enum": ["entailed", "absent", "contradicted", "ambiguous"]},
        "evidence_quotes": {"type": "array", "items": {"type": "string", "minLength": 1}},
    })
    fact_schemas = {}
    for identifier, fact in facts.items():
        requirements = _records_by_id(fact.get("requirements"), "id")
        fact_schemas[identifier] = _required_mapping({
            "reference_status": {"type": "string", "enum": ["compatible", "ambiguous", "different"]},
            "requirements": _required_mapping({key: {"$ref": "#/$defs/requirement_assessment"} for key in requirements}),
        })
    contract["schema"] = _required_mapping({"assessments": _required_mapping(fact_schemas)})
    contract["schema"]["$defs"] = {"requirement_assessment": result}
    return contract


def _normalize_wire_response(operation: str, response: CatalogResponse) -> CatalogResponse:
    """Converte mapas completos de IDs para as listas da porta de aplicação.

    Não preenche entradas ausentes e não descarta campos inesperados. Servidores
    compatíveis que ainda devolvem listas continuam sujeitos à validação normal.
    """
    payload = response.payload
    if not isinstance(payload, dict):
        return response
    if operation == "compare" and set(payload) == {"comparisons"} and isinstance(payload["comparisons"], dict):
        comparisons = []
        for identifier, item in payload["comparisons"].items():
            if not isinstance(item, dict) or set(item) != {"relation", "reason"}:
                raise CatalogInferenceError("Campos inválidos no mapa de comparações do provedor.")
            comparisons.append({"pair_id": identifier, **item})
        payload = {"comparisons": comparisons}
    elif operation == "assess_requirements" and set(payload) == {"assessments"} and isinstance(payload["assessments"], dict):
        assessments = []
        for identifier, item in payload["assessments"].items():
            if (not isinstance(item, dict) or set(item) != {"reference_status", "requirements"}
                    or not isinstance(item["requirements"], dict)):
                raise CatalogInferenceError("Campos inválidos no mapa de avaliações do provedor.")
            requirements = []
            for key, requirement in item["requirements"].items():
                if not isinstance(requirement, dict) or set(requirement) != {"status", "evidence_quotes"}:
                    raise CatalogInferenceError("Campos inválidos no mapa de requisitos avaliados do provedor.")
                requirements.append({"requirement_id": key, **requirement})
            assessments.append({"fact_id": identifier, "reference_status": item["reference_status"], "requirements": requirements})
        payload = {"assessments": assessments}
    else:
        return response
    return CatalogResponse(payload, response.model, response.system_fingerprint)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class OpenAICompatibleCatalogProvider:
    config: CatalogConfig
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.config.provider == "openai" and not self.api_key:
            raise ConfigurationError("Credencial OpenAI ausente.")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or any(char.isspace() or ord(char) < 32 for char in self.api_key)
        ):
            raise ConfigurationError("Formato de credencial inválido.")

    def build_request_body(self, operation: str, data: dict) -> dict[str, Any]:
        contract = structured_output(operation)
        if operation == "requirements":
            contract = _requirements_contract(contract, data)
        elif operation == "compare":
            contract = _comparison_contract(contract, data)
        elif operation == "assess_requirements":
            contract = _assessment_contract(contract, data)
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt(operation)},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False, allow_nan=False)},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_completion_tokens": self.config.max_completion_tokens,
            "response_format": {
                "type": contract["type"],
                "json_schema": {key: contract[key] for key in ("name", "strict", "schema")},
            },
            "store": False,
            **({"reasoning_effort": self.config.reasoning_effort} if self.config.reasoning_effort != "omit" else {}),
        }

    def extract(self, segment: dict) -> CatalogResponse:
        return self._request("extract", {"segment": segment})

    def compare(self, pairs: Sequence[dict]) -> CatalogResponse:
        return self._request("compare", {"pairs": list(pairs)})

    def assess(self, segment: dict, facts: Sequence[dict]) -> CatalogResponse:
        return self._request("assess", {"segment": segment, "facts": list(facts)})

    def requirements(self, candidate: dict) -> CatalogResponse:
        return self._request("requirements", {"candidate": candidate})

    def assess_requirements(self, segment: dict, facts: Sequence[dict]) -> CatalogResponse:
        return self._request("assess_requirements", {"segment": segment, "facts": list(facts)})

    def _request(self, operation: str, data: dict) -> CatalogResponse:
        body = json.dumps(self.build_request_body(operation, data), ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST",
        )
        try:
            # Cada worker possui seu opener; não há retries ou redirecionamentos.
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            code = param = None
            try:
                detail = _decode_json(error.read(MAX_RESPONSE_BYTES)).get("error", {})
                if isinstance(detail, dict):
                    code = _safe_identifier(detail.get("code"))
                    param = _safe_identifier(detail.get("param"))
                    if self.api_key:
                        code = code if code is None or self.api_key not in code else None
                        param = param if param is None or self.api_key not in param else None
            except (ValueError, UnicodeError, AttributeError, OSError):
                pass
            finally:
                error.close()
            details = ", ".join(value for value in (
                f"code={code}" if code else None, f"param={param}" if param else None,
            ) if value)
            suffix = f" ({details})" if details else ""
            raise CatalogInferenceError(
                f"Provedor recusou a operação {operation}: HTTP {error.code}{suffix}. "
                "Nenhum parâmetro foi omitido nem modelo substituído.",
            ) from None
        except (OSError, ValueError):
            raise CatalogInferenceError(
                "Falha de conexão/timeout com o provedor; detalhes suprimidos para proteger a credencial.",
            ) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CatalogInferenceError("Resposta do provedor excedeu o limite de 8 MiB.")
        try:
            envelope = _decode_json(raw)
        except (ValueError, UnicodeError):
            raise CatalogInferenceError("O provedor retornou uma resposta HTTP que não é JSON válido.") from None
        return _normalize_wire_response(operation, self._parse_response(envelope))

    @staticmethod
    def _parse_response(envelope: object) -> CatalogResponse:
        if not isinstance(envelope, dict):
            raise CatalogInferenceError("Envelope Chat Completions inválido.")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise CatalogInferenceError("O provedor deve retornar exatamente uma resposta em choices.")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise CatalogInferenceError("Mensagem Chat Completions ausente ou inválida.")
        if message.get("refusal"):
            raise CatalogInferenceError("O provedor recusou a avaliação (refusal); o texto da recusa não foi registrado.")
        if choice.get("finish_reason") != "stop":
            raise CatalogInferenceError("Geração incompleta ou interrompida pelo provedor; nenhuma resposta parcial será aceita.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CatalogInferenceError("O provedor retornou conteúdo vazio ou não textual.")
        try:
            payload = _decode_json(content)
        except (ValueError, UnicodeError):
            raise CatalogInferenceError("O conteúdo da resposta não é JSON válido.") from None
        model, fingerprint = envelope.get("model"), envelope.get("system_fingerprint")
        return CatalogResponse(
            payload, model if isinstance(model, str) else None,
            fingerprint if isinstance(fingerprint, str) else None,
        )
