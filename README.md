# Determinismo aplicado ao processamento de texto

Aplicação Python para experimentos científicos de mestrado. O experimento de mapeamento constrói um mapa de **ocorrências de palavras por segmento**, independente de modelos de linguagem. Cada execução explícita de `map` reconstrói `map` em todos os segmentos de todas as fontes e inicializa `count` com zero. O experimento separado de extração usa os mapas existentes para gerar informações semânticas. Não há dependências de execução além da biblioteca padrão do Python 3.11 ou superior.

## Executar o experimento

Na raiz do repositório:

```powershell
python -m determinism map data/processed/bebe_real_segments.json
```

O comando salva o resultado na própria entrada. Para gravar em outro arquivo:

```powershell
python -m determinism map data/processed/bebe_real_segments.json --output data/processed/bebe_real_mapped.json
```

O diretório de destino deve existir. O arquivo fornecido já contém o mapeamento inicial: **1 fonte, 1 segmento e 21 ocorrências**, além da extração de exemplo. Reexecutar `map` substitui o mapa anterior, inclusive quaisquer valores de `count` existentes. O mesmo conteúdo e a mesma política de tokenização produzem o mesmo resultado. Depois de alterar conteúdo ou política de mapeamento, execute novamente a extração para substituir informações que referenciem o mapa antigo.

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

## Extração semântica concorrente

Com `OPENAI_API_KEY` no ambiente ou no arquivo `.env` da raiz (consulte `.env.example`), execute:

```powershell
python -m determinism extract data/processed/bebe_real_segments.json
```

O comando processa todos os segmentos de todas as fontes e substitui a entrada somente após sucesso de toda a extração. `--output <arquivo.json>` escolhe outro destino; `--config <arquivo.toml>` escolhe outra configuração; `--env-file <arquivo>` escolhe outro arquivo de credencial. O ambiente tem precedência sobre `.env`, inclusive quando a variável estiver vazia. O leitor de `.env` aceita `NOME=valor`, aspas simples/duplas e linhas de comentário; não executa expansão ou código. A credencial não faz parte do TOML nem dos resultados.

A configuração versionada está em [`determinism/experiments/configs/semantic_extraction.toml`](determinism/experiments/configs/semantic_extraction.toml). Esses valores são carregados também após instalar o pacote:

| Campo | Padrão | Uso |
| --- | --- | --- |
| `provider` | `openai` | `openai` ou `openai-compatible` |
| `model` | `gpt-6-luna` | Identificador exato, sem substituição automática |
| `base_url` | `https://api.openai.com/v1` | Endpoint base de Chat Completions |
| `max_concurrency` | `4` | Limite de chamadas e tarefas pendentes (1 a 256) |
| `temperature` | `0.0` | Temperatura enviada explicitamente |
| `seed` | `42` | Semente inteira fixa enviada explicitamente |
| `reasoning_effort` | `none` | Permite temperatura explícita no modelo padrão |
| `timeout_seconds` | `60.0` | Timeout de transporte por chamada |
| `max_completion_tokens` | `4096` | Limite de saída por segmento |
| `api_key_env` | `OPENAI_API_KEY` | Nome da variável que contém a credencial |

Cada segmento precisa ter `map` válido **antes** da execução. Não há remapeamento implícito: conteúdo, mapas, contagens e metadados são preservados. Todos os segmentos e mapas são validados antes da primeira chamada, incluindo IDs consecutivos a partir de zero, correspondência entre chave e `token`, palavras não vazias e `count` inteiro não negativo. Isso valida a estrutura; a correspondência semântica entre um mapa antigo e um texto posteriormente editado é responsabilidade de quem prepara o experimento.

O adaptador mantém três partes separadas:

1. `modules/semantic_extraction/assets/system_prompt.v1.txt`: texto de sistema fixo fornecido para o experimento.
2. Mensagem de usuário: somente a introdução fixa, `<trecho>` e `<mapeamento_tokens>`, projetando o mapa para `token` e `word`. O texto é escapado como dados XML; caracteres delimitadores dentro das palavras usam escapes JSON reversíveis. Nenhuma regra ou schema é repetido nessa mensagem.
3. `modules/semantic_extraction/assets/structured_output.v1.json`: schema fixo enviado em `response_format` como JSON Schema estrito.

A resposta da API tem raiz `{"informacoes": [...]}`. Somente a lista retornada é salva em `segment["information"]`; não são salvos envelope da API, modelo, fingerprint ou credenciais. Uma nova execução substitui a lista anterior, sem incrementar `count`. A validação adicional rejeita campos extras/ausentes, strings vazias ou sem letras/números, menos de dois IDs por informação, IDs booleanos/desconhecidos/repetidos ou fora de ordem. Listas vazias e cobertura parcial são permitidas para priorizar fidelidade semântica. A aplicação não fabrica informações para cobrir tokens; a qualidade e o suporte factual das afirmações ainda precisam de avaliação científica.

Há no máximo `max_concurrency` tarefas em voo. A posição de cada resultado é guardada separadamente de sua ordem de conclusão, preservando fontes e segmentos. As respostas só são aplicadas a uma cópia após todas passarem na validação. Recusa, truncamento, erro de autenticação/rate limit/conexão, JSON inválido ou resposta fora do contrato interrompem o experimento; tarefas em curso podem concluir, mas não há novas submissões após a falha observada. Não há retries automáticos, troca de modelo ou remoção de parâmetros para contornar incompatibilidades. A escrita usa o mesmo repositório JSON com substituição atômica do mapeamento e deixa o arquivo anterior intacto em caso de falha anterior à substituição. Diagnósticos HTTP incluem somente status e identificadores de erro sanitizados, sem corpo, headers de autorização ou texto de recusa.

### Provedor compatível ou local

Copie o TOML padrão para um arquivo de experimento e altere explicitamente, por exemplo:

```toml
provider = "openai-compatible"
model = "identificador-do-modelo-local"
base_url = "http://127.0.0.1:1234/v1"
max_concurrency = 2
temperature = 0.0
seed = 42
reasoning_effort = "omit"
timeout_seconds = 60.0
max_completion_tokens = 4096
api_key_env = "LOCAL_AI_API_KEY"
```

Execute com `--config <arquivo.toml>`. A credencial é opcional nesse provedor. `reasoning_effort = "omit"` omite **somente esse campo por escolha explícita**, para servidores que não o aceitam. O servidor precisa oferecer `/chat/completions` e aceitar `temperature`, `seed`, `max_completion_tokens`, `store = false` e Structured Outputs estrito; essa compatibilidade depende do servidor e do modelo. Os parâmetros exigidos não são descartados automaticamente. A porta `InferenceProvider` permite implementar outros provedores, inclusive locais sem HTTP, sem modificar o domínio ou a aplicação:

```python
from determinism.experiments import extract_dataset
from determinism.modules.semantic_extraction.domain import InferenceResult

class OfflineExample:
    def extract(self, segment):
        # Ilustração da porta; um extrator real deve analisar segment.content/tokens.
        return InferenceResult({"informacoes": []})

extracted = extract_dataset(mapped, provider=OfflineExample(), max_concurrency=2)
```

Implementações dessa porta devem ser seguras para chamadas simultâneas. `SemanticExtractionExperiment(pipeline, repository=...)` permite trocar o pipeline ou a persistência; `extraction_pipeline(provider, max_concurrency=...)` compõe apenas a extração. `run_extraction(input_path, output_path=None, config=..., provider=..., repository=...)` executa a composição e retorna o resultado salvo. A injeção de um provedor não lê credenciais nem acessa APIs.

### Reprodutibilidade e verificação da API

Temperatura zero e semente fixa não garantem determinismo. A documentação oficial descreve `seed` como parâmetro depreciado de melhor esforço; diferenças de infraestrutura e do modelo ainda podem mudar a saída. O adaptador usa Chat Completions, preserva `seed` e usa `reasoning_effort = "none"` para permitir `temperature` no GPT-6 Luna. Consulte as fontes oficiais: [modelo](https://developers.openai.com/api/docs/models/gpt-6-luna), [parâmetros de GPT-6](https://developers.openai.com/api/docs/guides/latest-model), [Chat Completions e seed](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat) e [limites de reprodutibilidade](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter).

Em 25/10/2026, uma consulta de disponibilidade e **uma chamada de geração**, com o segmento atual de 21 palavras, confirmaram acesso a `gpt-6-luna` e aceitação do JSON Schema, temperatura `0`, semente `42` e reasoning `none`. A resposta informou modelo `gpt-6-luna`, sem `system_fingerprint`, e gerou as três informações presentes no dataset. O conteúdo, mapa, contagens e demais metadados foram comparados antes/depois e preservados. Foi uma verificação de integração e estrutura JSON; não houve benchmark nem repetição para medir determinismo. Para comparações futuras, registre versão do código, configuração, dataset, modelo retornado e `system_fingerprint` quando disponível em `InferenceResult`; o experimento padrão não cria um registro automático.

## Organização e extensão

O monólito modular segue uma arquitetura orientada a experimentos científicos (Experiment-Driven Architecture). O módulo funcional contém suas regras, caso de uso, portas e adaptadores em diretórios específicos. Os experimentos são unidades explícitas de composição e execução, e o pipeline é reutilizável fora desse módulo:

```text
determinism/
├── modules/
│   ├── text_mapping/
│   │   ├── domain/
│   │   │   ├── dataset.py          # Estrutura e validação do dataset
│   │   │   ├── word_map.py         # Mapa de ocorrências
│   │   │   └── tokenization.py     # Política Unicode pura
│   │   ├── ports/
│   │   │   ├── tokenizer.py        # Contrato de tokenização
│   │   │   └── dataset_repository.py
│   │   ├── application/
│   │   │   └── mapping_stage.py    # Caso de uso, independente da entrada/saída
│   │   └── adapters/
│   │       └── json_repository.py # Persistência JSON
│   └── semantic_extraction/
│       ├── domain/                # Entradas imutáveis e validação da resposta/mapa
│       ├── ports/                 # Inferência e persistência substituíveis
│       ├── application/           # Etapa concorrente com sucesso integral
│       ├── adapters/              # HTTP compatível, configuração e mensagens
│       └── assets/                # Sistema e JSON Schema versionados separados
├── pipelines/
│   └── sequential.py              # Pipeline genérico e contrato de etapa
├── experiments/
│   ├── word_mapping.py            # Definição do mapeamento
│   ├── semantic_extraction.py     # Composição explícita da extração
│   └── configs/
│       └── semantic_extraction.toml
├── entrypoints/
│   └── cli.py                    # Interface de linha de comando
└── __main__.py
```

As dependências apontam para as regras internas: `application` usa seu `domain` e suas portas; `adapters` implementa essas portas; `ports` depende somente de tipos internos e da biblioteca padrão. O domínio não conhece arquivos, CLI, experimentos ou adaptadores. Os domínios e aplicações dos dois módulos não importam um ao outro. A composição da extração reutiliza o adaptador JSON existente por compatibilidade estrutural com sua própria porta de persistência, sem acoplar a aplicação ao módulo de mapeamento. `pipelines` só conhece um tipo genérico e objetos chamáveis, sem importar módulos funcionais ou JSON. `experiments` reúne essas peças e `entrypoints` aciona o experimento. O pacote raiz não carrega adaptadores ao importar um componente interno.

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

A política alternativa é apenas uma demonstração de substituição; a CLI mantém a política Unicode documentada. Para novos experimentos, defina a composição em `experiments/`; para novas capacidades funcionais, crie um módulo em `modules/`. Os experimentos de mapeamento e extração são executados separadamente. Nenhuma etapa de incremento posterior de contagens foi implementada.

Erros de estrutura indicam a localização do campo, por exemplo `$[1].segments[0].content`. JSON inválido, chaves duplicadas, constantes como `NaN` e arquivos que não sejam UTF-8 são rejeitados. A leitura aceita BOM UTF-8; a saída usa UTF-8 sem BOM. Todo o dataset é validado e mapeado antes da escrita; o JSON é serializado integralmente e gravado em arquivo temporário no diretório de destino, seguido por substituição atômica. Falhas anteriores à substituição deixam o arquivo existente intacto. A formatação JSON é padronizada, preservando os valores e campos dos dados.

## Verificação

```powershell
python -m unittest discover -s tests -v
```

Os testes são offline e não requerem chave de API. Cobrem o exemplo de oito palavras, múltiplas fontes e segmentos, substituição de mapas, reinicialização de `count`, pontuação, caixa, acentos, Unicode, repetições, vazios, preservação dos dados, reuso por identidade em memória e pipeline genérico. A extração cobre concorrência realmente simultânea com conclusão fora de ordem, limite de tarefas, associação correta entre fontes/segmentos, imutabilidade dos dados, mapas inválidos antes de qualquer chamada, substituição de informações, validação estrita dos IDs/respostas, separação sistema/dados/schema, escapes, propagação da configuração, respostas recusadas/truncadas/inválidas, erros HTTP sanitizados sem retries, provedores/pipelines/repositórios alternativos e proteção atômica de arquivos. Um servidor HTTP local simulado verifica o fluxo completo da CLI com configuração compatível.
