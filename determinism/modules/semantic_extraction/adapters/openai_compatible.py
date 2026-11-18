"""Chat Completions HTTP: OpenAI e servidores locais explicitamente compatíveis."""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..domain import InferenceResult, SegmentInput
from ..ports import InferenceError
from .configuration import ConfigurationError, ExtractionConfig
from .prompts import coverage_message, structured_output, system_prompt, user_message


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
    if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.\[\]-]{1,100}", value):
        return value
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Nunca encaminhe a credencial a outro destino nem repita uma geração.
        return None


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    config: ExtractionConfig
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.config.provider == "openai" and not self.api_key:
            raise ConfigurationError("Credencial OpenAI ausente.")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or any(char.isspace() or ord(char) < 32 for char in self.api_key)
        ):
            raise ConfigurationError("Formato de credencial inválido.")

    def build_request_body(self, segment: SegmentInput) -> dict[str, Any]:
        contract = structured_output()
        prompt = system_prompt()
        feedback = coverage_message(segment)
        if feedback is not None:
            prompt += "\n\n" + feedback
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message(segment)},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_completion_tokens": self.config.max_completion_tokens,
            "response_format": {
                "type": contract["type"],
                "json_schema": {key: contract[key] for key in ("name", "strict", "schema")},
            },
            "store": False,
        }
        if self.config.reasoning_effort != "omit":
            body["reasoning_effort"] = self.config.reasoning_effort
        return body

    def extract(self, segment: SegmentInput) -> InferenceResult:
        body = json.dumps(self.build_request_body(segment), ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST",
        )
        try:
            # Um opener por chamada evita estado compartilhado entre workers.
            # urllib não faz retries de POST; redirecionamentos também são recusados.
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
            details = ", ".join(value for value in (f"code={code}" if code else None, f"param={param}" if param else None) if value)
            suffix = f" ({details})" if details else ""
            raise InferenceError(f"Provedor recusou a requisição: HTTP {error.code}{suffix}. Nenhum parâmetro foi omitido nem modelo substituído.") from None
        except (OSError, ValueError):
            raise InferenceError("Falha de conexão/timeout com o provedor; detalhes de transporte suprimidos para proteger a credencial.") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InferenceError("Resposta do provedor excedeu o limite de 8 MiB.")
        try:
            envelope = _decode_json(raw)
        except (ValueError, UnicodeError):
            raise InferenceError("O provedor retornou uma resposta HTTP que não é JSON válido.") from None
        return self._parse_response(envelope)

    @staticmethod
    def _parse_response(envelope: object) -> InferenceResult:
        if not isinstance(envelope, dict):
            raise InferenceError("Envelope Chat Completions inválido.")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise InferenceError("O provedor deve retornar exatamente uma resposta em choices.")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise InferenceError("Mensagem Chat Completions ausente ou inválida.")
        if message.get("refusal"):
            raise InferenceError("O provedor recusou a extração (refusal); o texto da recusa não foi registrado.")
        if choice.get("finish_reason") != "stop":
            raise InferenceError("Geração incompleta ou interrompida pelo provedor; nenhuma informação será salva.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise InferenceError("O provedor retornou conteúdo vazio ou não textual.")
        try:
            payload = _decode_json(content)
        except (ValueError, UnicodeError):
            raise InferenceError("O conteúdo da resposta não é JSON válido; nenhuma informação será salva.") from None
        model = envelope.get("model")
        fingerprint = envelope.get("system_fingerprint")
        return InferenceResult(
            payload,
            model if isinstance(model, str) else None,
            fingerprint if isinstance(fingerprint, str) else None,
        )
