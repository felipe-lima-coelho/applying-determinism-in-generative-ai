# Determinismo aplicado ao processamento de texto

Aplicação Python para experimentos científicos de mestrado. A primeira etapa constrói um mapa de **ocorrências de palavras por segmento**, independente de modelos de linguagem. Cada execução explícita reconstrói `map` em todos os segmentos de todas as fontes e inicializa `count` com zero. Não há dependências de execução além da biblioteca padrão do Python 3.11 ou superior.

## Executar o experimento

Na raiz do repositório:

```powershell
python -m determinism map data/processed/bebe_real_segments.json
```

O comando salva o resultado na própria entrada. Para gravar em outro arquivo:

```powershell
python -m determinism map data/processed/bebe_real_segments.json --output data/processed/bebe_real_mapped.json
```

O diretório de destino deve existir. O arquivo fornecido já contém o mapeamento inicial: **1 fonte, 1 segmento e 77 ocorrências**. Reexecutar o comando substitui o mapa anterior, inclusive quaisquer valores de `count` existentes. O mesmo conteúdo e a mesma política de tokenização produzem o mesmo resultado.

Opcionalmente, `python -m pip install -e .` disponibiliza o comando equivalente `determinism map <arquivo>`. Nenhuma chave de API é necessária para esta etapa.

## Contrato do dataset e das palavras

A entrada é uma lista de objetos de fonte, cada um com uma lista `segments`. Cada segmento deve ser um objeto com `content` do tipo string. Os demais campos, incluindo `name`, metadados e o texto original de `content`, são preservados. Fontes sem segmentos, listas vazias e textos vazios são válidos; textos sem palavras recebem `"map": {}`.

Cada mapa é um objeto JSON com chaves `"0"`, `"1"`, ... em ordem de ocorrência. O campo inteiro `token` coincide com a chave convertida para inteiro, e o índice recomeça em zero a cada segmento. Repetições ocupam entradas distintas; `count` não representa frequência nesta etapa e permanece zero.

Para `A casa é verde, porém não tem porta.`, o mapa contém exatamente oito entradas:

```json
{
    "0": {"token": 0, "word": "A", "count": 0},
    "1": {"token": 1, "word": "casa", "count": 0},
    "2": {"token": 2, "word": "é", "count": 0},
    "3": {"token": 3, "word": "verde", "count": 0},
    "4": {"token": 4, "word": "porém", "count": 0},
    "5": {"token": 5, "word": "não", "count": 0},
    "6": {"token": 6, "word": "tem", "count": 0},
    "7": {"token": 7, "word": "porta", "count": 0}
}
```

A política padrão considera palavra uma sequência de letras ou números Unicode e marcas combinantes anexas. Espaços, pontuação e símbolos separam sequências e não recebem entradas. Assim, `Mountbatten-Windsor` gera `Mountbatten` e `Windsor`; `d'água` e `d’água` geram `d` e `água`; `12,5` gera `12` e `5`. Sublinhados também separam palavras e emojis são ignorados. Marcas combinantes isoladas são ignoradas, mas uma palavra como `café` mantém sua representação original. Não há conversão de caixa, remoção de acentos, normalização Unicode, deduplicação ou tokenização de LLM. A definição é lexical, não uma análise linguística de cada idioma. Para comparar experimentos, mantenha a mesma versão do Python e de sua base Unicode (`unicodedata.unidata_version`).

## API e reuso em memória

```python
from determinism.experiments import map_dataset

dataset = [{"name": "exemplo", "segments": [{"content": "A casa é verde, porém não tem porta."}]}]
mapped = map_dataset(dataset)
word_map = mapped[0]["segments"][0]["map"]

assert word_map["2"] == {"token": 2, "word": "é", "count": 0}
assert "map" not in dataset[0]["segments"][0]
```

`map_dataset` retorna uma cópia independente do dataset, alterando somente `map`. Guarde `mapped` e reutilize seus mapas em leituras e etapas posteriores. O acesso ao mapa pronto não tokeniza novamente e não existe cache global. Uma nova chamada ao mapeamento reconstrói os mapas, mesmo que já existam.

O pipeline passa o mesmo objeto retornado por uma etapa à seguinte, sem serialização entre elas:

```python
from determinism.modules.text_mapping.application import MappingStage
from determinism.modules.text_mapping.domain.tokenization import UnicodeWordTokenizer
from determinism.pipelines import Pipeline

def observe_maps(dataset):
    for source in dataset:
        for segment in source["segments"]:
            print(segment["map"])
    return dataset

pipeline = Pipeline((MappingStage(UnicodeWordTokenizer()), observe_maps))
mapped = pipeline.run(dataset)
```

Etapas são funções ou objetos chamáveis que recebem e retornam o dataset. Um tokenizador alternativo implementa `tokenize(text)`, devolvendo as palavras em ordem, e pode ser injetado em `MappingStage` ou em `map_dataset(dataset, tokenizer=...)`. `run_mapping(input_path, output_path=None, repository=..., tokenizer=...)` executa entrada, mapeamento e persistência e retorna o próprio resultado em memória. Um repositório alternativo implementa `load(path)` e `save(path, dataset)`.

## Organização e extensão

O monólito modular segue uma arquitetura orientada a experimentos científicos (Experiment-Driven Architecture). O módulo funcional contém suas regras, caso de uso, portas e adaptadores em diretórios específicos. Os experimentos são unidades explícitas de composição e execução, e o pipeline é reutilizável fora desse módulo:

```text
determinism/
├── modules/
│   └── text_mapping/
│       ├── domain/
│       │   ├── dataset.py          # Estrutura e validação do dataset
│       │   ├── word_map.py         # Mapa de ocorrências
│       │   └── tokenization.py     # Política Unicode pura
│       ├── ports/
│       │   ├── tokenizer.py        # Contrato de tokenização
│       │   └── dataset_repository.py
│       ├── application/
│       │   └── mapping_stage.py    # Caso de uso, independente da entrada/saída
│       └── adapters/
│           └── json_repository.py # Persistência JSON
├── pipelines/
│   └── sequential.py              # Pipeline genérico e contrato de etapa
├── experiments/
│   └── word_mapping.py            # Definição e execução do experimento
├── entrypoints/
│   └── cli.py                    # Interface de linha de comando
└── __main__.py
```

As dependências apontam para as regras internas: `application` usa `domain` e a porta `Tokenizer`; `adapters` implementa a porta `DatasetRepository` usando o contrato de `domain`; `ports` depende somente de tipos internos e da biblioteca padrão. O domínio não conhece arquivos, CLI, experimentos ou adaptadores. `pipelines` só conhece um tipo genérico e objetos chamáveis, sem importar módulos funcionais ou JSON. `experiments` reúne essas peças e `entrypoints` aciona o experimento. O pacote raiz não carrega adaptadores ao importar um componente interno.

`WordMappingExperiment` é uma definição executável com `pipeline` e `repository` injetáveis. Sua configuração padrão usa `MappingStage`, `UnicodeWordTokenizer` e `JsonDatasetRepository`; a mesma definição pode ser reutilizada em diferentes datasets. A configuração em Python registra quais implementações e em qual ordem o experimento executa. Não há framework de configuração ou registro automático de resultados nesta primeira etapa.

Este exemplo define outro experimento de mapeamento com uma política alternativa (somente espaços separam palavras), outra sequência de etapas e o adaptador JSON existente:

```python
from determinism.experiments import WordMappingExperiment
from determinism.modules.text_mapping.adapters import JsonDatasetRepository
from determinism.modules.text_mapping.application import MappingStage
from determinism.pipelines import Pipeline

class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()

def observe_maps(dataset):
    print([len(segment["map"]) for source in dataset for segment in source["segments"]])
    return dataset

experiment = WordMappingExperiment(
    pipeline=Pipeline((MappingStage(WhitespaceTokenizer()), observe_maps)),
    repository=JsonDatasetRepository(),
)
mapped = experiment.run(
    "data/processed/bebe_real_segments.json",
    "data/processed/bebe_real_whitespace.json",
)
```

A política alternativa é apenas uma demonstração de substituição; a CLI mantém a política Unicode documentada. Para novos experimentos, defina a composição em `experiments/`; para novas capacidades funcionais, crie um módulo em `modules/`. O experimento atual inclui somente o mapeamento. Nenhuma contagem posterior ou integração com LLM foi implementada.

Erros de estrutura indicam a localização do campo, por exemplo `$[1].segments[0].content`. JSON inválido, chaves duplicadas, constantes como `NaN` e arquivos que não sejam UTF-8 são rejeitados. A leitura aceita BOM UTF-8; a saída usa UTF-8 sem BOM. Todo o dataset é validado e mapeado antes da escrita; o JSON é serializado integralmente e gravado em arquivo temporário no diretório de destino, seguido por substituição atômica. Falhas anteriores à substituição deixam o arquivo existente intacto. A formatação JSON é padronizada, preservando os valores e campos dos dados.

## Verificação

```powershell
python -m unittest discover -s tests -v
```

Os testes cobrem o exemplo de oito palavras, múltiplas fontes e segmentos, substituição de mapas, reinicialização de `count`, pontuação, caixa, acentos, Unicode, repetições, vazios, preservação dos dados, reuso por identidade em memória, pipeline genérico, experimento com pipeline/tokenizador/repositório alternativos, execução da CLI e proteção de arquivos em falhas de leitura, tokenização, serialização e escrita.
