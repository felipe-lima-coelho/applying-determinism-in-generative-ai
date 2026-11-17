"""Linha de comando dos experimentos explícitos de mapeamento, extração e agrupamento."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..experiments import run_catalog, run_extraction, run_grouping, run_mapping, run_simulation
from ..experiments.word_mapping import mapping_output_path
from ..modules.semantic_catalog.adapters import load_catalog_config
from ..modules.semantic_catalog.domain import CatalogError
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
    mapping.add_argument("input", type=Path, help="Transcrição JSON de entrada em data/transcriptions/; a fonte é preservada.")
    mapping.add_argument(
        "--output",
        type=Path,
        help="Arquivo de destino, diferente da entrada. Padrão: data/processed/<nome da transcrição>.",
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
    catalog = commands.add_parser("catalog", help="Constrói catálogo global com evidências e suporte por segmento.")
    catalog.add_argument("input", type=Path, help="Dataset JSON; information e grupos existentes são usados como pistas.")
    catalog.add_argument("--output", type=Path, required=True, help="Catálogo JSON separado; nunca substitui o dataset.")
    catalog.add_argument("--config", type=Path, help="Configuração TOML de semantic_catalog.")
    catalog.add_argument("--env-file", type=Path, default=Path(".env"), help="Credencial do avaliador; ambiente tem precedência.")
    cache_options = catalog.add_mutually_exclusive_group()
    cache_options.add_argument("--cache-dir", type=Path, default=Path(".cache/semantic_catalog"), help="Cache persistente das avaliações; padrão .cache/semantic_catalog.")
    cache_options.add_argument("--no-cache", action="store_true", help="Executa novas avaliações sem consultar ou gravar cache.")
    simulation = commands.add_parser("simulate", help="Simula remoção de segmentos sem chamadas ao modelo.")
    simulation.add_argument("input", type=Path, help="Catálogo JSON congelado.")
    simulation.add_argument("--output", type=Path, required=True, help="Relatório JSON separado.")
    simulation.add_argument("--remove", action="append", default=[], metavar="SEGMENT_ID", help="ID do catálogo, por exemplo s0.seg0; repetível.")
    simulation.add_argument("--remove-source", action="append", default=[], metavar="SOURCE_ID", help="Remove todos os segmentos de uma fonte, por exemplo s0; repetível.")
    simulation.add_argument("--dataset", type=Path, help="Verifica se o catálogo ainda corresponde exatamente a este dataset.")
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
        elif arguments.command == "catalog":
            result = run_catalog(arguments.input, arguments.output, config=load_catalog_config(arguments.config),
                                 env_file=arguments.env_file, cache_dir=None if arguments.no_cache else arguments.cache_dir)
            print(f"Catálogo salvo em {arguments.output} (informações: {len(result['facts'])}, rejeitadas: {len(result['rejected_facts'])}, segmentos: {len(result['segments'])})")
            for segment in result["segments"]:
                print(f"  {segment['id']} | {segment['source_name']} | segmento {segment['segment_index']}")
            return 0
        else:
            report = run_simulation(arguments.input, arguments.output, removed_segment_ids=arguments.remove,
                                    removed_source_ids=arguments.remove_source, dataset_path=arguments.dataset)
            print(f"Simulação salva em {arguments.output}")
            for key, label in (("lost_information", "Perdidas"), ("preserved_information", "Preservadas"), ("uncertain_information", "Incertas")):
                print(f"{label}: {len(report[key])}")
                for item in report[key]:
                    print(f"  {item['fact_id']}: {item['statement']}")
            return 0
    except (DatasetFormatError, DatasetFileError, ExtractionError, ConfigurationError, GroupingError, CatalogError, OSError) as error:
        parser.exit(status=1, message=f"Erro: {error}\n")

    segments = sum(len(source["segments"]) for source in dataset)
    destination = arguments.output if arguments.output is not None else arguments.input
    if arguments.command == "map":
        destination = mapping_output_path(arguments.input, arguments.output)
        tokens = sum(len(segment["map"]) for source in dataset for segment in source["segments"])
        print(f"Mapeamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, palavras: {tokens})")
    elif arguments.command == "extract":
        information = sum(len(segment["information"]) for source in dataset for segment in source["segments"])
        print(f"Extração salva em {destination} (fontes: {len(dataset)}, segmentos: {segments}, informações acumuladas: {information})")
    else:
        groups = sum(len(segment["set_different_information"]) for source in dataset for segment in source["segments"])
        print(f"Agrupamento salvo em {destination} (fontes: {len(dataset)}, segmentos: {segments}, grupos: {groups})")
    return 0
