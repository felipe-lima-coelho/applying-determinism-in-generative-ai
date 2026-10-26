"""Linha de comando dos experimentos explícitos de mapeamento, extração e agrupamento."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..experiments import run_extraction, run_grouping, run_mapping
from ..modules.semantic_extraction.adapters import ConfigurationError, load_extraction_config
from ..modules.semantic_extraction.domain import ExtractionError
from ..modules.semantic_grouping.adapters import load_grouping_config
from ..modules.semantic_grouping.domain import GroupingError
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
    extraction = commands.add_parser("extract", help="Acumula information até cada token ter pelo menos duas referências.")
    extraction.add_argument("input", type=Path, help="Dataset JSON já mapeado.")
    extraction.add_argument("--output", type=Path, help="Destino; por padrão, substitui a entrada após sucesso total.")
    extraction.add_argument("--config", type=Path, help="Configuração TOML; padrão em experiments/configs/semantic_extraction.toml.")
    extraction.add_argument("--env-file", type=Path, default=Path(".env"), help="Credencial opcional em .env; ambiente tem precedência.")
    grouping = commands.add_parser("group", help="Agrupa information por complete linkage dentro de cada segmento.")
    grouping.add_argument("input", type=Path, help="Dataset JSON com information em cada segmento.")
    grouping.add_argument("--output", type=Path, help="Destino; por padrão, substitui a entrada após sucesso total.")
    grouping.add_argument("--config", type=Path, help="Configuração TOML; padrão em experiments/configs/semantic_grouping.toml.")
    grouping.add_argument("--env-file", type=Path, default=Path(".env"), help="Credenciais de embeddings/PostgreSQL; ambiente tem precedência.")
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "map":
            dataset = run_mapping(arguments.input, arguments.output)
        elif arguments.command == "extract":
            dataset = run_extraction(
                arguments.input, arguments.output,
                config=load_extraction_config(arguments.config), env_file=arguments.env_file,
            )
        else:
            dataset = run_grouping(
                arguments.input, arguments.output,
                config=load_grouping_config(arguments.config), env_file=arguments.env_file,
            )
    except (DatasetFormatError, DatasetFileError, ExtractionError, ConfigurationError, GroupingError, OSError) as error:
        parser.exit(status=1, message=f"Erro: {error}\n")

    segments = sum(len(source["segments"]) for source in dataset)
    destination = arguments.output if arguments.output is not None else arguments.input
    if arguments.command == "map":
        tokens = sum(len(segment["map"]) for source in dataset for segment in source["segments"])
        print(f"Mapeamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, palavras: {tokens})")
    elif arguments.command == "extract":
        information = sum(len(segment["information"]) for source in dataset for segment in source["segments"])
        print(f"Extração salva em {destination} (fontes: {len(dataset)}, segmentos: {segments}, informações acumuladas: {information})")
    else:
        groups = sum(len(segment["set_different_information"]) for source in dataset for segment in source["segments"])
        print(f"Agrupamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, grupos: {groups})")
    return 0
