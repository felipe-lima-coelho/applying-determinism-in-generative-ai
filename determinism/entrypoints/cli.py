"""Adaptador de linha de comando do experimento de mapeamento."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..experiments import run_mapping
from ..modules.text_mapping.adapters import DatasetFileError
from ..modules.text_mapping.domain import DatasetFormatError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="determinism")
    commands = parser.add_subparsers(dest="command", required=True)
    mapping = commands.add_parser("map", help="Reconstrói map em todos os segmentos.")
    mapping.add_argument("input", type=Path, help="Dataset JSON de entrada.")
    mapping.add_argument(
        "--output",
        type=Path,
        help="Arquivo de destino. Por padrão, substitui a entrada após concluir o mapeamento.",
    )
    arguments = parser.parse_args(argv)

    try:
        dataset = run_mapping(arguments.input, arguments.output)
    except (DatasetFormatError, DatasetFileError, OSError) as error:
        parser.exit(status=1, message=f"Erro: {error}\n")

    segments = sum(len(source["segments"]) for source in dataset)
    tokens = sum(len(segment["map"]) for source in dataset for segment in source["segments"])
    destination = arguments.output if arguments.output is not None else arguments.input
    print(f"Mapeamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, palavras: {tokens})")
    return 0
