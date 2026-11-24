# Determinismo aplicado ao processamento de texto

Repositório de experimentos em Python sobre processamento de texto com IA generativa. O fluxo combina mapeamento lexical determinístico, extração semântica por API e agrupamento de informações com embeddings e PostgreSQL/pgvector.

## Instalação

Requer Python 3.11 ou superior:

```powershell
python -m pip install -e .
```

## Fluxo de uso

1. Crie o diretório de saída usado pelo mapeamento:

   ```powershell
   python -c "from pathlib import Path; Path('data/processed').mkdir(parents=True, exist_ok=True)"
   ```

2. Gere os mapas de palavras do dataset de exemplo. Essa etapa não usa API:

   ```powershell
   python -m determinism map data/transcriptions/bebe_real_sources.json
   ```

   O resultado é salvo em `data/processed/bebe_real_sources.json`; a transcrição original é preservada.

3. Configure `OPENAI_API_KEY` no ambiente ou em `.env` e execute a extração semântica:

   ```powershell
   python -m determinism extract data/processed/bebe_real_sources.json
   ```

4. Para agrupar as informações, configure também as variáveis `GROUPING_DB_*` em `.env` e inicie o PostgreSQL com pgvector:

   ```powershell
   docker compose -f compose.semantic-grouping.yaml up -d
   python -m determinism group data/processed/bebe_real_sources.json
   ```

`extract` e `group` usam o arquivo de entrada como destino por padrão. Passe `--output <arquivo.json>` para salvar em outro caminho. `group` também gera um catálogo global ao lado do resultado. Consulte `python -m determinism --help` para as opções da CLI.

## Configurações e dados

- [Exemplo de variáveis de ambiente](.env.example)
- [Configuração da extração](determinism/experiments/configs/semantic_extraction.toml)
- [Configuração do agrupamento](determinism/experiments/configs/semantic_grouping.toml)
- [Serviço PostgreSQL/pgvector](compose.semantic-grouping.yaml)
- [Dataset de exemplo](data/transcriptions/bebe_real_sources.json)
