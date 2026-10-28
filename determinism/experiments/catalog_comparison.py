"""Compara decisões de catálogos v2 sem inferência nem equivalência por posição."""

import copy
from pathlib import Path

from ..modules.semantic_catalog.adapters.json_artifacts import read_json, write_json
from ..modules.semantic_catalog.domain import CatalogError, validate_catalog
from .semantic_catalog import _destination


def _changes(first, second, identity):
    """Preserva ambos os valores; ausência é explícita, nunca uma decisão negativa."""
    return [
        {identity: key, "first": copy.deepcopy(first.get(key)), "second": copy.deepcopy(second.get(key))}
        for key in sorted(first.keys() | second.keys())
        if first.get(key) != second.get(key)
    ]


def _inventory(catalog):
    return {item["id"]: {key: item[key] for key in ("segment_id", "statement", "information_indices")}
            for item in catalog["candidates"]}


def _requirements(catalog):
    return {item["id"]: sorted(item["requirements"], key=lambda requirement: requirement["id"])
            for item in catalog["candidates"]}


def _relations(catalog):
    return {tuple(sorted((item["left_id"], item["right_id"]))): item["relation"]
            for item in catalog["comparisons"]}


def _facts(catalog):
    return {tuple(sorted(item["candidate_ids"])): (field, item)
            for field in ("facts", "rejected_facts") for item in catalog[field]}


def _supports(catalog):
    result = {}
    for members, (_, fact) in _facts(catalog).items():
        for assessment in fact["assessments"]:
            key = (members, tuple(sorted(assessment["segment_ids"])))
            result[key] = {
                "status": assessment["status"], "reference_status": assessment["reference_status"],
                "requirements": sorted(
                    ({"requirement_id": item["requirement_id"], "status": item["status"]}
                     for item in assessment["requirement_results"]),
                    key=lambda item: item["requirement_id"],
                ),
            }
    return result


def compare_catalogs(first: object, second: object) -> dict:
    """Compara unidades, requisitos, relações, grupos e suportes de um snapshot.

    Requer v2, cujos candidatos são identificados pelo conteúdo e pela origem.
    Não alinha IDs posicionais de catálogos v1 e não reclassifica diferenças
    entre inferências como ambiguidade textual. O resultado é independente dos
    objetos recebidos e não modifica catálogos ou decisões.
    """
    left, right = validate_catalog(first), validate_catalog(second)
    if left["schema_version"] != 2 or right["schema_version"] != 2:
        raise CatalogError("A comparação de decisões exige dois catálogos v2; IDs posicionais v1 não identificam a mesma informação entre execuções.")
    if left["dataset_fingerprint"] != right["dataset_fingerprint"]:
        raise CatalogError("Os catálogos usam datasets diferentes; a comparação de instabilidade exige o mesmo snapshot.")

    left_groups, right_groups = _facts(left), _facts(right)
    groups = {
        "removed": [list(group) for group in sorted(left_groups.keys() - right_groups.keys())],
        "added": [list(group) for group in sorted(right_groups.keys() - left_groups.keys())],
    }
    support_changes = _changes(_supports(left), _supports(right), "location")
    for item in support_changes:
        members, segment_ids = item.pop("location")
        item["candidate_ids"] = list(members)
        item["segment_ids"] = list(segment_ids)
    relation_changes = _changes(_relations(left), _relations(right), "candidate_ids")
    for item in relation_changes:
        item["candidate_ids"] = list(item["candidate_ids"])
    disposition_changes = _changes(
        {members: category for members, (category, _) in left_groups.items()},
        {members: category for members, (category, _) in right_groups.items()}, "candidate_ids",
    )
    for item in disposition_changes:
        item["candidate_ids"] = list(item["candidate_ids"])
    representatives = _changes(
        {members: {"canonical_candidate_id": fact["candidate_ids"][0], "statement": fact["statement"]}
         for members, (_, fact) in left_groups.items()},
        {members: {"canonical_candidate_id": fact["candidate_ids"][0], "statement": fact["statement"]}
         for members, (_, fact) in right_groups.items()}, "candidate_ids",
    )
    for item in representatives:
        item["candidate_ids"] = list(item["candidate_ids"])
    differences = {
        "inventory": _changes(_inventory(left), _inventory(right), "candidate_id"),
        "requirements": _changes(_requirements(left), _requirements(right), "candidate_id"),
        "relations": relation_changes,
        "groups": groups,
        "disposition": disposition_changes,
        "representatives": representatives,
        "support": support_changes,
        "policy": None if left["policy"] == right["policy"] else {
            "first": copy.deepcopy(left["policy"]), "second": copy.deepcopy(right["policy"]),
        },
    }
    same_decisions = left["decision_id"] == right["decision_id"]
    same_catalog = left["catalog_id"] == right["catalog_id"]
    affected_ids = {
        item["candidate_id"] for field in ("inventory", "requirements") for item in differences[field]
    }
    affected_ids.update(candidate_id for field in ("relations", "disposition", "support", "representatives")
                        for item in differences[field] for candidate_id in item["candidate_ids"])
    affected_ids.update(candidate_id for field in ("removed", "added") for group in groups[field] for candidate_id in group)
    inventory = {**_inventory(left), **_inventory(right)}
    return {
        "schema_version": 1,
        "comparison_kind": "inference_stability",
        "dataset_fingerprint": left["dataset_fingerprint"],
        "first": {key: left[key] for key in ("catalog_id", "decision_id")},
        "second": {key: right[key] for key in ("catalog_id", "decision_id")},
        "same_catalog_content": same_catalog,
        "same_decisions": same_decisions,
        "difference_kind": "none" if same_catalog else "audit_only" if same_decisions else "inference_instability",
        "differences": differences,
        "affected_candidates": [{"candidate_id": key, **copy.deepcopy(inventory[key])} for key in sorted(affected_ids)],
        "summary": {
            "inventory_changes": len(differences["inventory"]),
            "requirement_changes": len(differences["requirements"]),
            "relation_changes": len(relation_changes),
            "removed_groups": len(groups["removed"]), "added_groups": len(groups["added"]),
            "disposition_changes": len(disposition_changes),
            "representative_changes": len(representatives),
            "support_changes": len(support_changes),
            "policy_changed": differences["policy"] is not None,
        },
    }


def run_catalog_comparison(first_path, second_path, output_path) -> dict:
    """Persiste diagnóstico separado; nunca promove ou substitui um catálogo."""
    destination = _destination(output_path, first_path, second_path)
    report = compare_catalogs(read_json(first_path), read_json(second_path))
    report["same_file_bytes"] = Path(first_path).read_bytes() == Path(second_path).read_bytes()
    if report["same_catalog_content"] and not report["same_file_bytes"]:
        report["difference_kind"] = "formatting_only"
    write_json(destination, report)
    return report
