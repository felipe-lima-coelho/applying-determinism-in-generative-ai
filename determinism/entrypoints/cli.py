"""Linha de comando dos experimentos explícitos de processamento do dataset."""

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ..experiments import run_extraction, run_grouping, run_mapping, run_refinement
from ..modules.semantic_extraction.adapters import ConfigurationError, load_extraction_config
from ..modules.semantic_extraction.domain import ExtractionError
from ..modules.semantic_grouping.adapters import load_grouping_config
from ..modules.semantic_grouping.domain import GroupingError
from ..modules.singleton_refinement.adapters import load_refinement_config
from ..modules.singleton_refinement.domain import RefinementError
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
    extraction.add_argument("--output", type=Path, help="Destino; por padrão, substitui a entrada. Salva progresso válido se atingir o limite de cobertura.")
    extraction.add_argument("--config", type=Path, help="Configuração TOML; padrão em experiments/configs/semantic_extraction.toml.")
    extraction.add_argument("--env-file", type=Path, default=Path(".env"), help="Credencial opcional em .env; ambiente tem precedência.")
    grouping = commands.add_parser("group", help="Agrupa information por complete linkage dentro de cada segmento.")
    grouping.add_argument("input", type=Path, help="Dataset JSON com information em cada segmento.")
    grouping.add_argument("--output", type=Path, help="Destino; por padrão, substitui a entrada após sucesso total.")
    grouping.add_argument("--config", type=Path, help="Configuração TOML; padrão em experiments/configs/semantic_grouping.toml.")
    grouping.add_argument("--env-file", type=Path, default=Path(".env"), help="Credenciais de embeddings/PostgreSQL; ambiente tem precedência.")
    refinement = commands.add_parser("refine", help="Reextrai e reagrupa somente segmentos com grupos singleton, até não restar nenhum.")
    refinement.add_argument("input", type=Path, help="Dataset com mapas, information e grupos válidos em todos os segmentos.")
    refinement.add_argument("--output", type=Path, help="Destino; por padrão, substitui a entrada. Salva checkpoints após cada rodada completa.")
    refinement.add_argument("--config", type=Path, help="TOML do loop; padrão em experiments/configs/singleton_refinement.toml.")
    refinement.add_argument("--max-rounds", type=int, help="Sobrescreve explicitamente o limite de rodadas do loop.")
    refinement.add_argument("--extraction-config", type=Path, help="Mesmo TOML usado por extract; padrão atual de semantic_extraction.toml.")
    refinement.add_argument("--grouping-config", type=Path, help="Mesmo TOML usado por group; padrão atual de semantic_grouping.toml.")
    refinement.add_argument("--env-file", type=Path, default=Path(".env"), help="Credenciais dos provedores e PostgreSQL; ambiente tem precedência.")
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "map":
            dataset = run_mapping(arguments.input, arguments.output)
        elif arguments.command == "extract":
            dataset = run_extraction(
                arguments.input, arguments.output,
                config=load_extraction_config(arguments.config), env_file=arguments.env_file,
            )
        elif arguments.command == "group":
            dataset = run_grouping(
                arguments.input, arguments.output,
                config=load_grouping_config(arguments.config), env_file=arguments.env_file,
            )
        else:
            refinement_config = load_refinement_config(arguments.config)
            if arguments.max_rounds is not None:
                refinement_config = replace(refinement_config, max_rounds=arguments.max_rounds)
            dataset = run_refinement(
                arguments.input, arguments.output, config=refinement_config,
                extraction_config=load_extraction_config(arguments.extraction_config),
                grouping_config=load_grouping_config(arguments.grouping_config), env_file=arguments.env_file,
            )
    except (DatasetFormatError, DatasetFileError, ExtractionError, ConfigurationError, GroupingError, RefinementError, OSError) as error:
        parser.exit(status=1, message=f"Erro: {error}\n")

    segments = sum(len(source["segments"]) for source in dataset)
    destination = arguments.output if arguments.output is not None else arguments.input
    if arguments.command == "map":
        tokens = sum(len(segment["map"]) for source in dataset for segment in source["segments"])
        print(f"Mapeamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, palavras: {tokens})")
    elif arguments.command == "extract":
        information = sum(len(segment["information"]) for source in dataset for segment in source["segments"])
        print(f"Extração salva em {destination} (fontes: {len(dataset)}, segmentos: {segments}, informações acumuladas: {information})")
    elif arguments.command == "group":
        groups = sum(len(segment["set_different_information"]) for source in dataset for segment in source["segments"])
        print(f"Agrupamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, grupos: {groups})")
    else:
        information = sum(len(segment["information"]) for source in dataset for segment in source["segments"])
        print(f"Refinamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, informações acumuladas: {information}, grupos singleton: 0)")
    return 0
