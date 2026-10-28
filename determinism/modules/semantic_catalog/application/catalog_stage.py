"""Extração de unidades, equivalência conservadora e avaliação de suporte exaustiva."""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import ClassVar

from ..domain import CatalogError, catalog_digest, dataset_digest, validate_catalog
from ..ports.cache import ResponseCache
from ..ports.provider import CatalogProvider, CatalogResponse
from ...text_mapping.domain import validate_dataset


def _text(value, label):
    if not isinstance(value, str) or not any(character.isalnum() for character in value):
        raise CatalogError(f"{label} deve ser texto não vazio.")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise CatalogError(f"{label} deve conter texto UTF-8 válido.") from None
    return value


def _records(payload, key):
    if not isinstance(payload, dict) or set(payload) != {key} or not isinstance(payload[key], list):
        raise CatalogError(f"Resposta deve conter somente a lista {key}.")
    return payload[key]


def _fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CatalogError("Campos inválidos na resposta semântica.")


def _evidence(quotes, segment, required=False):
    if not isinstance(quotes, list) or (required and not quotes):
        raise CatalogError("Evidências devem ser uma lista de citações literais; suporte confirmado exige evidência.")
    evidence = []
    seen = set()
    for quote in quotes:
        _text(quote, "evidence_quotes")
        start = segment["content"].find(quote)
        if start < 0:
            raise CatalogError(f"Evidência não encontrada literalmente em {segment['id']}.")
        # Todas as ocorrências literais são registradas; não se escolhe silenciosamente uma repetição.
        while start >= 0:
            key = (start, start + len(quote))
            if key not in seen:
                evidence.append({"segment_id": segment["id"], "quote": quote, "start": key[0], "end": key[1]})
                seen.add(key)
            start = segment["content"].find(quote, start + 1)
    return sorted(evidence, key=lambda item: (item["start"], item["end"]))


def _extraction(payload, segment):
    facts = []
    for item in _records(payload, "facts"):
        _fields(item, ("statement", "evidence_quotes", "information_indices"))
        statement = _text(item["statement"], "statement").strip()
        indices = item["information_indices"]
        if (not isinstance(indices, list) or any(type(index) is not int or not 0 <= index < len(segment["information"]) for index in indices)
                or indices != sorted(set(indices))):
            raise CatalogError("information_indices deve referenciar ocorrências distintas em ordem crescente.")
        facts.append({"statement": statement, "evidence": _evidence(item["evidence_quotes"], segment, True),
                      "information_indices": list(indices)})
    return facts


def _comparisons(payload, pairs):
    expected = {pair["pair_id"]: pair for pair in pairs}
    result = {}
    for item in _records(payload, "comparisons"):
        _fields(item, ("pair_id", "relation", "reason"))
        key = item["pair_id"]
        if not isinstance(key, str) or key not in expected or key in result:
            raise CatalogError("Comparação possui ID desconhecido ou repetido.")
        if item["relation"] not in ("equivalent", "different", "uncertain"):
            raise CatalogError("Relação semântica inválida.")
        pair = expected[key]
        result[key] = {"pair_id": key, "left_id": pair["left"]["id"], "right_id": pair["right"]["id"],
                       "relation": item["relation"], "reason": _text(item["reason"], "reason")}
    if set(result) != set(expected):
        raise CatalogError("O provedor omitiu comparações; ausência não é diferença semântica.")
    return [result[key] for key in expected]


def _assessments(payload, segment, facts):
    expected = {fact["id"] for fact in facts}
    result = {}
    for item in _records(payload, "assessments"):
        _fields(item, ("fact_id", "status", "evidence_quotes", "reason"))
        key = item["fact_id"]
        if not isinstance(key, str) or key not in expected or key in result:
            raise CatalogError("Avaliação possui informação desconhecida ou repetida.")
        if item["status"] not in ("supported", "uncertain", "not_supported"):
            raise CatalogError("Estado de suporte inválido.")
        result[key] = {"segment_ids": [segment["id"]], "status": item["status"],
                       "evidence": _evidence(item["evidence_quotes"], segment, item["status"] == "supported"),
                       "reason": _text(item["reason"], "reason")}
    if set(result) != expected:
        raise CatalogError("O provedor omitiu avaliações; ausência não é falta de suporte.")
    return [(fact["id"], result[fact["id"]]) for fact in facts]


def _snapshot(dataset):
    validate_dataset(dataset)
    dataset_digest(dataset)  # Também rejeita dados que não possam ser versionados em JSON.
    segments = []
    for source_index, source in enumerate(dataset):
        for segment_index, segment in enumerate(source["segments"]):
            information = segment.get("information", [])
            if not isinstance(information, list) or any(not isinstance(item, dict) or not isinstance(item.get("info"), str) for item in information):
                raise CatalogError("information, quando presente, deve ser uma lista de objetos com info textual.")
            if "set_different_information" in segment and not isinstance(segment["set_different_information"], list):
                raise CatalogError("set_different_information deve ser uma lista.")
            source_name = source.get("name", f"Fonte {source_index}")
            if not isinstance(source_name, str):
                raise CatalogError("O nome da fonte deve ser textual.")
            segments.append({
                "id": f"s{source_index}.seg{segment_index}", "source_id": f"s{source_index}",
                "source_index": source_index, "segment_index": segment_index, "source_name": source_name,
                "content": segment["content"], "information": copy.deepcopy(information),
                "map": copy.deepcopy(segment.get("map", {})),
                "set_different_information": copy.deepcopy(segment.get("set_different_information", [])),
                "source_metadata": copy.deepcopy({key: value for key, value in source.items() if key != "segments"}),
                "segment_metadata": copy.deepcopy({key: value for key, value in segment.items()
                                                     if key not in ("content", "information", "map", "set_different_information")}),
            })
    return segments


def _batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _partition(candidates, comparisons):
    """Ligação completa sobre equivalências: nunca usa fechamento transitivo."""
    equivalent = {frozenset((pair["left_id"], pair["right_id"])) for pair in comparisons if pair["relation"] == "equivalent"}
    groups = [(index,) for index in range(len(candidates))]
    while True:
        merge = next(((i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))
                      if all(frozenset((candidates[a]["id"], candidates[b]["id"])) in equivalent
                             for a in groups[i] for b in groups[j])), None)
        if merge is None:
            return groups
        i, j = merge
        union = tuple(sorted((*groups[i], *groups[j])))
        groups = sorted([group for position, group in enumerate(groups) if position not in (i, j)] + [union])


@dataclass(frozen=True)
class CatalogStage:
    application_revision: ClassVar[str] = "catalog-v1"
    provider: CatalogProvider
    max_concurrency: int = 4
    comparison_batch_size: int = 24
    assessment_batch_size: int = 16
    cache: ResponseCache | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.provenance, dict):
            raise CatalogError("provenance deve ser um objeto JSON.")
        if self.cache is not None and not self.provenance:
            raise CatalogError("Cache exige provenance com identidade e revisão do avaliador.")
        try:
            json.dumps(self.provenance, allow_nan=False)
        except (TypeError, ValueError):
            raise CatalogError("provenance deve conter somente dados JSON finitos.") from None
        for name in ("max_concurrency", "comparison_batch_size", "assessment_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 256:
                raise CatalogError(f"{name} deve estar entre 1 e 256.")

    def _invoke(self, operation, arguments, validate):
        request = {"operation": operation, "arguments": arguments, "provenance": self.provenance,
                   "application_revision": self.application_revision}
        key = hashlib.sha256(json.dumps(request, ensure_ascii=True, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()
        response = self.cache.get(key) if self.cache is not None else None
        cached = response is not None
        if response is None:
            response = getattr(self.provider, operation)(*copy.deepcopy(arguments))
        if not isinstance(response, CatalogResponse):
            raise CatalogError("O provedor deve retornar CatalogResponse.")
        for value in (response.model, response.system_fingerprint):
            if value is not None and not isinstance(value, str):
                raise CatalogError("Metadados inválidos na resposta do provedor.")
            if value is not None:
                try:
                    value.encode("utf-8")
                except UnicodeError:
                    raise CatalogError("Metadados do provedor devem conter texto UTF-8 válido.") from None
        result = validate(response.payload)
        if self.cache is not None and not cached:
            self.cache.put(key, response)
        # O uso de cache não muda o artefato: mesmos resultados => mesmo checksum.
        trace = {"request_id": key, "operation": operation, "model": response.model,
                 "system_fingerprint": response.system_fingerprint}
        return result, trace

    def _run_jobs(self, jobs):
        # Batches limitam tarefas em voo e preservam ordem independente da conclusão.
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for batch in _batches(jobs, self.max_concurrency):
                futures = [executor.submit(self._invoke, *job) for job in batch]
                for future in futures:
                    yield future.result()

    def __call__(self, dataset):
        segments = _snapshot(dataset)
        traces, candidates = [], []
        jobs = [("extract", (segment,), lambda payload, segment=segment: _extraction(payload, segment))
                for segment in segments if segment["content"].strip()]
        nonempty = [segment for segment in segments if segment["content"].strip()]
        for segment, (items, trace) in zip(nonempty, self._run_jobs(jobs)):
            traces.append(trace)
            # Extrações idênticas no mesmo segmento não viram unidades duplicadas.
            seen = {}
            for item in items:
                if item["statement"] in seen:
                    existing = seen[item["statement"]]
                    existing["information_indices"] = sorted(set(existing["information_indices"] + item["information_indices"]))
                    existing["evidence"] = sorted({(e["start"], e["end"]): e for e in existing["evidence"] + item["evidence"]}.values(), key=lambda e: (e["start"], e["end"]))
                    continue
                candidate = {"id": f"c{len(candidates)}", "segment_id": segment["id"], **item}
                candidates.append(candidate)
                seen[item["statement"]] = candidate

        by_segment = {segment["id"]: segment for segment in segments}
        pairs = []
        for index, (left, right) in enumerate(combinations(candidates, 2)):
            def contextual(candidate):
                segment = by_segment[candidate["segment_id"]]
                return {**candidate, "content": segment["content"], "source_metadata": segment["source_metadata"],
                        "segment_metadata": segment["segment_metadata"]}
            pairs.append({"pair_id": f"p{index}", "left": contextual(left), "right": contextual(right)})
        jobs = [("compare", (batch,), lambda payload, batch=batch: _comparisons(payload, batch))
                for batch in _batches(pairs, self.comparison_batch_size)]
        comparisons = []
        for items, trace in self._run_jobs(jobs):
            comparisons.extend(items)
            traces.append(trace)
        # A coocorrência no mesmo trecho não pode tornar um fato e seu detalhe
        # equivalentes. Confirma propostas de fusão usando somente as afirmações.
        proposed = {item["pair_id"] for item in comparisons if item["relation"] == "equivalent"}
        isolated_pairs = [
            {"pair_id": pair["pair_id"], "mode": "statement_only_verification",
             **{side: {"id": pair[side]["id"], "segment_id": pair[side]["segment_id"],
                       "statement": pair[side]["statement"], "content": pair[side]["statement"],
                       "source_metadata": {}, "segment_metadata": {}} for side in ("left", "right")}}
            for pair in pairs if pair["pair_id"] in proposed
        ]
        jobs = [("compare", (batch,), lambda payload, batch=batch: _comparisons(payload, batch))
                for batch in _batches(isolated_pairs, self.comparison_batch_size)]
        verification = {}
        for items, trace in self._run_jobs(jobs):
            verification.update((item["pair_id"], item) for item in items)
            traces.append(trace)
        for item in comparisons:
            checked = verification.get(item["pair_id"])
            if checked is not None:
                item["relation"] = checked["relation"]
                item["reason"] = f"Contexto: {item['reason']} Verificação das afirmações isoladas: {checked['reason']}"
        facts = []
        for group in _partition(candidates, comparisons):
            members = [candidates[index] for index in group]
            # Representante real, sem síntese que possa apagar qualificações.
            facts.append({"id": f"f{len(facts)}", "statement": members[0]["statement"],
                          "candidate_ids": [member["id"] for member in members], "assessments": []})
        by_candidate = {candidate["id"]: candidate for candidate in candidates}
        assessments_input = [{"id": fact["id"], "statement": fact["statement"],
                              "origin_contexts": [{"segment_id": by_candidate[key]["segment_id"],
                                                   "statement": by_candidate[key]["statement"],
                                                   "content": by_segment[by_candidate[key]["segment_id"]]["content"],
                                                   "source_metadata": by_segment[by_candidate[key]["segment_id"]]["source_metadata"],
                                                   "segment_metadata": by_segment[by_candidate[key]["segment_id"]]["segment_metadata"]}
                                                  for key in fact["candidate_ids"]]}
                             for fact in facts]
        jobs = [("assess", (segment, batch),
                 lambda payload, segment=segment, batch=batch: _assessments(payload, segment, batch))
                for segment in segments for batch in _batches(assessments_input, self.assessment_batch_size)]
        by_fact = {fact["id"]: fact for fact in facts}
        for items, trace in self._run_jobs(jobs):
            traces.append(trace)
            for key, assessment in items:
                by_fact[key]["assessments"].append(assessment)
        accepted, rejected = [], []
        for fact in facts:
            (accepted if any(item["status"] != "not_supported" for item in fact["assessments"]) else rejected).append(fact)
        catalog = {"schema_version": 1, "catalog_id": "", "dataset_fingerprint": dataset_digest(dataset),
                   "policy": {"equivalence": "exhaustive_complete_link", "support": "independent_segments",
                              "uncertainty": "explicit", "canonical_statement": "first_candidate",
                              "merge_verification": "context_and_statement_only_agreement",
                              "scope": "transcript_claims_not_external_fact_checking"},
                   "provenance": {**copy.deepcopy(self.provenance), "application_revision": "catalog-v1", "calls": traces},
                   "segments": segments, "candidates": candidates, "comparisons": comparisons,
                   "facts": accepted, "rejected_facts": rejected}
        catalog["catalog_id"] = catalog_digest(catalog)
        return validate_catalog(catalog, dataset=dataset)
