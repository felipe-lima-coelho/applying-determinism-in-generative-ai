"""Catálogo de unidades fixas: requisitos literais e decisão de suporte em código."""

import copy
from dataclasses import dataclass
from itertools import combinations

from .catalog_stage import CatalogStage, _batches, _comparisons, _evidence, _fields, _partition, _records, _snapshot
from ..domain.contracts import CatalogError, _digest, catalog_digest, dataset_digest
from ..domain.requirements import (
    REFERENCE_STATUSES, REQUIREMENT_STATUSES, candidate_identifier, compile_requirements,
    normalize_statement, reduce_requirement_status, statement_tokens,
)


def fixed_candidates(dataset, segments):
    """Preserva todas as formulações distintas; grupos por embedding não as eliminam."""
    result = []
    for segment in segments:
        original = dataset[segment["source_index"]]["segments"][segment["segment_index"]]
        if "information" not in original and segment["content"].strip():
            raise CatalogError("O catálogo por requisitos exige information: execute a extração antes de gerar o catálogo.")
        seen = {}
        for index, occurrence in enumerate(segment["information"]):
            statement = normalize_statement(occurrence["info"])
            if not segment["content"].strip():
                raise CatalogError("Informações extraídas exigem conteúdo de origem não vazio.")
            if statement not in seen:
                seen[statement] = {
                    "id": candidate_identifier(segment["id"], segment["content"], statement),
                    "segment_id": segment["id"], "statement": statement,
                    # Esta âncora é proveniência; só a avaliação confirma suporte.
                    "evidence": _evidence([segment["content"]], segment, True),
                    "information_indices": [],
                }
            seen[statement]["information_indices"].append(index)
        result.extend(seen[key] for key in sorted(seen))
    return result


def _union_evidence(items):
    found = {(entry["segment_id"], entry["start"], entry["end"]): entry
             for item in items for entry in item["evidence"]}
    return [found[key] for key in sorted(found)]


def requirement_assessments(payload, segment, facts):
    """Valida cada obrigação antes de reduzir; ausência de resposta nunca vira ausência de suporte."""
    expected = {fact["id"]: fact for fact in facts}
    result = {}
    for item in _records(payload, "assessments"):
        _fields(item, ("fact_id", "reference_status", "requirements"))
        key = item["fact_id"]
        if not isinstance(key, str) or key not in expected or key in result:
            raise CatalogError("Avaliação possui informação desconhecida ou repetida.")
        reference = item["reference_status"]
        if not isinstance(reference, str) or reference not in REFERENCE_STATUSES:
            raise CatalogError("reference_status inválido.")
        requirements = {r["id"]: r for r in expected[key]["requirements"]}
        checked = {}
        if not isinstance(item["requirements"], list):
            raise CatalogError("requirements da avaliação deve ser uma lista.")
        for raw in item["requirements"]:
            _fields(raw, ("requirement_id", "status", "evidence_quotes"))
            identifier = raw["requirement_id"]
            if not isinstance(identifier, str) or identifier not in requirements or identifier in checked:
                raise CatalogError("Avaliação possui requisito desconhecido ou repetido.")
            status = raw["status"]
            if not isinstance(status, str) or status not in REQUIREMENT_STATUSES:
                raise CatalogError("Estado de requisito inválido.")
            if status == "absent" and raw["evidence_quotes"] != []:
                raise CatalogError("Requisito ausente deve ter evidências vazias.")
            checked[identifier] = {"requirement_id": identifier, "status": status,
                                   "evidence": _evidence(raw["evidence_quotes"], segment, status != "absent")}
        if set(checked) != set(requirements):
            raise CatalogError("O provedor omitiu requisitos; nenhuma decisão parcial será aceita.")
        ordered = [checked[requirement["id"]] for requirement in expected[key]["requirements"]]
        status = reduce_requirement_status(reference, [entry["status"] for entry in ordered])
        reason = {
            "supported": "Todos os requisitos são sustentados e a referência é compatível.",
            "uncertain": "Há ambiguidade em requisito ou referência, sem ausência ou contradição confirmada.",
            "not_supported": "Há requisito ausente ou contradito, ou referência incompatível.",
        }[status]
        result[key] = {"segment_ids": [segment["id"]], "status": status,
                       "evidence": _union_evidence(ordered), "reason": reason,
                       "reference_status": reference, "requirement_results": ordered}
    if set(result) != set(expected):
        raise CatalogError("O provedor omitiu avaliações; nenhuma informação pode ser ignorada.")
    return [(fact["id"], result[fact["id"]]) for fact in facts]


@dataclass(frozen=True)
class RequirementCatalogStage(CatalogStage):
    """Não faz extração livre; contratos de requisitos são a única geração estrutural."""

    application_revision = "catalog-requirements-v2"
    assessment_batch_size: int = 1

    def __call__(self, dataset):
        # Import tardio evita acoplamento entre implementação do estágio e validação.
        from ..domain import decision_digest, validate_catalog

        segments = _snapshot(dataset)
        candidates = fixed_candidates(dataset, segments)
        by_segment = {segment["id"]: segment for segment in segments}
        traces = []
        jobs = [("requirements", ({"id": candidate["id"], "statement": candidate["statement"],
                                    "statement_tokens": statement_tokens(candidate["statement"])},),
                 lambda payload, statement=candidate["statement"]: compile_requirements(payload, statement))
                for candidate in candidates]
        for candidate, (requirements, trace) in zip(candidates, self._run_jobs(jobs)):
            candidate["requirements"] = requirements
            traces.append(trace)

        def contextual(candidate):
            segment = by_segment[candidate["segment_id"]]
            return {**candidate, "content": segment["content"], "source_metadata": segment["source_metadata"],
                    "segment_metadata": segment["segment_metadata"]}

        pairs = [{"pair_id": "p_" + _digest(sorted([left["id"], right["id"]])),
                  "left": contextual(left), "right": contextual(right)}
                 for left, right in combinations(candidates, 2)]
        jobs = [("compare", (batch,), lambda payload, batch=batch: _comparisons(payload, batch))
                for batch in _batches(pairs, self.comparison_batch_size)]
        comparisons = []
        for items, trace in self._run_jobs(jobs):
            comparisons.extend(items)
            traces.append(trace)

        proposed = {item["pair_id"] for item in comparisons if item["relation"] == "equivalent"}
        isolated_pairs = [
            {"pair_id": pair["pair_id"], "mode": "statement_only_verification",
             **{side: {"id": pair[side]["id"], "segment_id": pair[side]["segment_id"],
                       "statement": pair[side]["statement"], "content": pair[side]["statement"],
                       "requirements": pair[side]["requirements"], "source_metadata": {}, "segment_metadata": {}}
                for side in ("left", "right")}}
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
                item["reason"] = f"Contexto: {item['reason']} Verificação isolada: {checked['reason']}"

        facts = []
        by_candidate = {candidate["id"]: candidate for candidate in candidates}
        for group in _partition(candidates, comparisons):
            members = [candidates[index] for index in group]
            member_ids = [member["id"] for member in members]
            facts.append({"id": "f_" + _digest(sorted(member_ids)), "statement": members[0]["statement"],
                          "candidate_ids": member_ids, "assessments": []})
        assessment_input = [
            {"id": fact["id"], "statement": fact["statement"],
             "requirements": by_candidate[fact["candidate_ids"][0]]["requirements"],
             "origin_contexts": [{"segment_id": by_candidate[key]["segment_id"],
                                  "statement": by_candidate[key]["statement"],
                                  "content": by_segment[by_candidate[key]["segment_id"]]["content"],
                                  "source_metadata": by_segment[by_candidate[key]["segment_id"]]["source_metadata"],
                                  "segment_metadata": by_segment[by_candidate[key]["segment_id"]]["segment_metadata"]}
                                 for key in fact["candidate_ids"]]}
            for fact in facts
        ]
        jobs = [("assess_requirements", (segment, batch),
                 lambda payload, segment=segment, batch=batch: requirement_assessments(payload, segment, batch))
                for segment in segments for batch in _batches(assessment_input, self.assessment_batch_size)]
        by_fact = {fact["id"]: fact for fact in facts}
        for items, trace in self._run_jobs(jobs):
            for key, assessment in items:
                by_fact[key]["assessments"].append(assessment)
            traces.append(trace)
        accepted, rejected = [], []
        for fact in facts:
            (accepted if any(item["status"] != "not_supported" for item in fact["assessments"]) else rejected).append(fact)
        catalog = {
            "schema_version": 2, "catalog_id": "", "decision_id": "", "dataset_fingerprint": dataset_digest(dataset),
            "policy": {"equivalence": "exhaustive_complete_link", "support": "independent_segments",
                       "uncertainty": "explicit", "canonical_statement": "first_fixed_candidate",
                       "merge_verification": "context_and_statement_only_agreement",
                       "scope": "extracted_units_validated_against_transcripts", "candidate_source": "existing_information",
                       "requirements": "literal_token_coverage_v1", "assessment": "requirement_reduction_v1",
                       "granularity": "existing_information_with_requirement_breakdown"},
            "provenance": {**copy.deepcopy(self.provenance), "application_revision": self.application_revision, "calls": traces},
            "segments": segments, "candidates": candidates, "comparisons": comparisons,
            "facts": accepted, "rejected_facts": rejected,
        }
        catalog["decision_id"] = decision_digest(catalog)
        catalog["catalog_id"] = catalog_digest(catalog)
        return validate_catalog(catalog, dataset)
