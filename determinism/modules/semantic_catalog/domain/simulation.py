"""Simulação pura de remoções conjuntas sobre um catálogo congelado."""

import copy
from collections.abc import Iterable, Set
from dataclasses import dataclass
from types import MappingProxyType

from .contracts import CatalogError, validate_catalog


def _selection(values: Iterable[str], known: Set[str], name: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise CatalogError(f"{name} deve ser uma coleção de identificadores, não uma string.")
    try:
        selected = list(values)
    except TypeError:
        raise CatalogError(f"{name} deve ser uma coleção de identificadores.") from None
    if any(not isinstance(value, str) or value not in known for value in selected):
        raise CatalogError(f"{name} contém um identificador desconhecido.")
    return set(selected)


@dataclass(frozen=True)
class _Evidence:
    segment_id: str
    quote: str
    start: int
    end: int

    def record(self) -> dict:
        return {"segment_id": self.segment_id, "quote": self.quote, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class _Support:
    segment_ids: tuple[str, ...]
    members: frozenset[str]
    status: str
    evidence: tuple[_Evidence, ...]
    reason: str

    def record(self) -> dict:
        return {"segment_ids": list(self.segment_ids), "status": self.status,
                "evidence": [item.record() for item in self.evidence], "reason": self.reason}


@dataclass(frozen=True)
class _Fact:
    id: str
    statement: str
    supports: tuple[_Support, ...]
    baseline_supported: bool
    baseline_source_ids: tuple[str, ...]


class RemovalSimulator:
    """Valida e compila um snapshot uma vez para explorar muitos cenários.

    O índice interno contém apenas estruturas imutáveis e não expõe o catálogo.
    Relatórios são construídos novamente a cada chamada. Nem mudanças no objeto
    de entrada nem alterações nos relatórios afetam os cenários posteriores.
    """

    __slots__ = ("__segments", "__source_segments", "__known_segments", "__facts",
                 "__catalog_id", "__dataset_fingerprint", "__rejected_count")

    def __init__(self, catalog: object, dataset: object = None):
        snapshot = copy.deepcopy(catalog)
        validated = validate_catalog(snapshot, dataset)
        self.__catalog_id = validated["catalog_id"]
        self.__dataset_fingerprint = validated["dataset_fingerprint"]
        self.__rejected_count = len(validated["rejected_facts"])
        # Conteúdo e pares de comparação não são necessários após a validação.
        self.__segments = MappingProxyType({
            item["id"]: (item["source_id"], item["source_index"], item["segment_index"])
            for item in validated["segments"]
        })
        self.__known_segments = frozenset(self.__segments)
        source_segments: dict[str, set[str]] = {}
        for segment_id, (source_id, _, _) in self.__segments.items():
            source_segments.setdefault(source_id, set()).add(segment_id)
        self.__source_segments = MappingProxyType({key: frozenset(ids) for key, ids in source_segments.items()})
        facts = []
        for fact in sorted(validated["facts"], key=lambda item: item["id"]):
            supports = tuple(
                _Support(tuple(item["segment_ids"]), frozenset(item["segment_ids"]), item["status"],
                         tuple(_Evidence(**evidence) for evidence in item["evidence"]), item["reason"])
                for item in fact["assessments"] if item["status"] != "not_supported"
            )
            confirmed = tuple(support for support in supports if support.status == "supported")
            facts.append(_Fact(fact["id"], fact["statement"], supports, bool(confirmed), self.__sources_for(confirmed)))
        self.__facts = tuple(facts)

    def __segment_order(self, segment_id):
        return self.__segments[segment_id][1:]

    def __sources_for(self, supports):
        return tuple(sorted({self.__segments[segment_id][0] for support in supports for segment_id in support.segment_ids},
                            key=lambda source_id: int(source_id[1:])))

    def simulate(
        self,
        removed_segment_ids: Iterable[str] = (),
        removed_source_ids: Iterable[str] = (),
    ) -> dict:
        """Consulta o índice já validado; não percorre a matriz de comparações."""
        removed = _selection(removed_segment_ids, self.__known_segments, "removed_segment_ids")
        removed_sources = _selection(removed_source_ids, self.__source_segments.keys(), "removed_source_ids")
        for source_id in removed_sources:
            removed.update(self.__source_segments[source_id])
        retained = self.__known_segments - removed
        result = {
            "schema_version": 1,  # Contrato do relatório permanece igual para catálogos v1 e v2.
            "catalog_id": self.__catalog_id,
            "dataset_fingerprint": self.__dataset_fingerprint,
            "removed_segment_ids": sorted(removed, key=self.__segment_order),
            "retained_segment_ids": sorted(retained, key=self.__segment_order),
            "lost_information": [], "preserved_information": [], "uncertain_information": [],
        }
        for fact in self.__facts:
            surviving = [support for support in fact.supports if support.members.issubset(retained)]
            remaining = [support for support in surviving if support.status == "supported"]
            possible = [support for support in surviving if support.status == "uncertain"]
            baseline_status = "supported" if fact.baseline_supported else "uncertain"
            remaining_status = "supported" if remaining else "uncertain" if possible else "not_supported"
            classification = (
                "uncertain_information" if not fact.baseline_supported else
                "preserved_information" if remaining else
                "uncertain_information" if possible else "lost_information"
            )
            removed_supports = []
            for support in fact.supports:
                missing = support.members.intersection(removed)
                if missing:
                    removed_supports.append({**support.record(), "missing_segment_ids": sorted(missing, key=self.__segment_order)})
            entry = {
                "fact_id": fact.id, "statement": fact.statement,
                "baseline_status": baseline_status, "remaining_status": remaining_status,
                "remaining_supports": [support.record() for support in remaining],
                "possible_supports": [support.record() for support in possible],
                "removed_supports": removed_supports,
                "baseline_source_ids": list(fact.baseline_source_ids), "remaining_source_ids": list(self.__sources_for(remaining)),
            }
            result[classification].append(entry)
        result["summary"] = {
            "total_information": len(self.__facts),
            "lost": len(result["lost_information"]),
            "preserved": len(result["preserved_information"]),
            "uncertain": len(result["uncertain_information"]),
            "rejected": self.__rejected_count,
            "removed_segments": len(removed), "remaining_segments": len(retained),
        }
        return result


def simulate_removal(
    catalog: object,
    removed_segment_ids: Iterable[str] = (),
    removed_source_ids: Iterable[str] = (),
    dataset: object = None,
) -> dict:
    """Classifica informações sem novas inferências nem alterações no catálogo.

    Um conjunto de suporte sobrevive somente se TODOS os seus segmentos ficam.
    Uma informação com origem incerta não vira perda confirmada por uma remoção.
    """
    return RemovalSimulator(catalog, dataset).simulate(removed_segment_ids, removed_source_ids)
