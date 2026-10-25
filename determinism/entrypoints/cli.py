"""Linha de comando dos experimentos explícitos de mapeamento e extração."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..experiments import run_extraction, run_mapping
from ..modules.semantic_extraction.adapters import ConfigurationError, load_extraction_config
from ..modules.semantic_extraction.domain import ExtractionError
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
    extraction = commands.add_parser("extract", help="Extrai information usando os mapas existentes, sem remapear.")
    extraction.add_argument("input", type=Path, help="Dataset JSON já mapeado.")
    extraction.add_argument("--output", type=Path, help="Destino; por padrão, substitui a entrada após sucesso total.")
    extraction.add_argument("--config", type=Path, help="Configuração TOML; padrão em experiments/configs/semantic_extraction.toml.")
    extraction.add_argument("--env-file", type=Path, default=Path(".env"), help="Credencial opcional em .env; ambiente tem precedência.")
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "map":
            dataset = run_mapping(arguments.input, arguments.output)
        else:
            dataset = run_extraction(
                arguments.input, arguments.output,
                config=load_extraction_config(arguments.config), env_file=arguments.env_file,
            )
    except (DatasetFormatError, DatasetFileError, ExtractionError, ConfigurationError, OSError) as error:
        parser.exit(status=1, message=f"Erro: {error}\n")

    segments = sum(len(source["segments"]) for source in dataset)
    destination = arguments.output if arguments.output is not None else arguments.input
    if arguments.command == "map":
        tokens = sum(len(segment["map"]) for source in dataset for segment in source["segments"])
        print(f"Mapeamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, palavras: {tokens})")
    else:
        information = sum(len(segment["information"]) for source in dataset for segment in source["segments"])
        print(f"Extração salva em {destination} (fontes: {len(dataset)}, segmentos: {segments}, informações: {information})")
    return 0
