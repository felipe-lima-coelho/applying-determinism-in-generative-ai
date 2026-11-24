"""Contrato estrutural do dataset textual, preservando metadados abertos."""

from typing import Any, TypeAlias, cast


Dataset: TypeAlias = list[dict[str, Any]]


class DatasetFormatError(ValueError):
    """A estrutura recebida não atende ao contrato mínimo do dataset."""


def validate_dataset(value: object) -> Dataset:
    """Valida todos os segmentos sem modificar nem descartar metadados."""
    if not isinstance(value, list):
        raise DatasetFormatError("O dataset deve ser uma lista de fontes.")

    for source_index, source in enumerate(value):
        source_path = f"$[{source_index}]"
        if not isinstance(source, dict):
            raise DatasetFormatError(f"{source_path} deve ser um objeto de fonte.")
        if "segments" not in source or not isinstance(source["segments"], list):
            raise DatasetFormatError(f"{source_path}.segments deve ser uma lista.")

        for segment_index, segment in enumerate(source["segments"]):
            segment_path = f"{source_path}.segments[{segment_index}]"
            if not isinstance(segment, dict):
                raise DatasetFormatError(f"{segment_path} deve ser um objeto de segmento.")
            if "content" not in segment or not isinstance(segment["content"], str):
                raise DatasetFormatError(f"{segment_path}.content deve ser uma string.")

    return cast(Dataset, value)
