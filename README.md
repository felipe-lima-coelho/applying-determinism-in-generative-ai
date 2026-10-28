# Determinismo aplicado ao processamento de texto

Aplicação Python 3.11 ou superior para experimentos científicos de mestrado. O experimento de mapeamento constrói um mapa de **ocorrências de palavras por segmento**, independente de modelos de linguagem. Cada execução explícita de `map` reconstrói `map` em todos os segmentos de todas as fontes e inicializa `count` com zero. A extração usa os mapas existentes para gerar informações semânticas. O agrupamento separado compara essas informações usando embeddings persistidos em PostgreSQL. Mapeamento e extração usam a biblioteca padrão; o adaptador PostgreSQL requer `psycopg[binary]`, instalado com `python -m pip install -e .`.

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

O comando processa todos os segmentos de todas as fontes. Ao concluir, substitui a entrada; se atingir o limite de cobertura, salva o progresso validado nesse mesmo destino e termina com erro, indicando as pendências. `--output <arquivo.json>` escolhe outro destino, preservando a entrada; `--config <arquivo.toml>` escolhe outra configuração; `--env-file <arquivo>` escolhe outro arquivo de credencial. O ambiente tem precedência sobre `.env`, inclusive quando a variável estiver vazia. O leitor de `.env` aceita `NOME=valor`, aspas simples/duplas e linhas de comentário; não executa expansão ou código. A credencial não faz parte do TOML nem dos resultados.

A configuração versionada está em [`determinism/experiments/configs/semantic_extraction.toml`](determinism/experiments/configs/semantic_extraction.toml). Esses valores são carregados também após instalar o pacote:

| Campo | Padrão | Uso |
| --- | --- | --- |
| `provider` | `openai` | `openai` ou `openai-compatible` |
| `model` | `gpt-6-luna` | Identificador exato, sem substituição automática |
| `base_url` | `https://api.openai.com/v1` | Endpoint base de Chat Completions |
| `max_concurrency` | `4` | Limite de chamadas e tarefas pendentes (1 a 256) |
| `max_generations_per_segment` | `10` | Limite de novas gerações por segmento nesta execução; cobertura incompleta causa erro |
| `temperature` | `0.0` | Temperatura enviada explicitamente |
| `seed` | `42` | Semente inteira fixa enviada explicitamente |
| `reasoning_effort` | `none` | Permite temperatura explícita no modelo padrão |
| `timeout_seconds` | `60.0` | Timeout de transporte por chamada |
| `max_completion_tokens` | `4096` | Limite de saída por geração |
| `api_key_env` | `OPENAI_API_KEY` | Nome da variável que contém a credencial |

Cada segmento precisa ter `map` válido **antes** da execução. Não há remapeamento implícito: conteúdo, IDs, palavras e metadados são preservados. A extração altera somente `information` e os respectivos `count`. Todos os segmentos, mapas e informações anteriores são validados antes da primeira chamada, incluindo IDs consecutivos a partir de zero, correspondência entre chave e `token`, palavras não vazias e `count` inteiro não negativo. O histórico usa o mesmo contrato estrito das respostas novas; `information` ausente equivale a uma lista vazia. Isso valida a estrutura; a correspondência semântica entre um mapa antigo e um texto posteriormente editado é responsabilidade de quem prepara o experimento.

O adaptador mantém estas partes separadas:

1. `modules/semantic_extraction/assets/system_prompt.v1.txt`: texto de sistema fixo fornecido para o experimento.
2. Mensagem de usuário: somente a introdução fixa, `<trecho>` e `<mapeamento_tokens>`, projetando o mapa para `token` e `word`. O texto é escapado como dados XML; caracteres delimitadores dentro das palavras usam escapes JSON reversíveis. Nenhuma regra ou schema é repetido nessa mensagem.
3. `modules/semantic_extraction/assets/structured_output.v1.json`: schema fixo enviado em `response_format` como JSON Schema estrito.
4. Um bloco suplementar com instruções fixas de revisão da cobertura e estado numérico dinâmico: ordinal da geração nesta execução, meta, IDs/contagens ainda abaixo de dois e número de referências que faltam (`needed`). Ele é anexado ao texto base, preservado literalmente, na mesma mensagem de sistema; a chamada continua com exatamente duas mensagens, uma de sistema e uma de usuário. O estado é derivado somente das referências validadas, ordenado por ID e recalculado antes de cada chamada, inclusive a primeira quando há histórico. Nenhuma palavra, informação extraída ou outro texto do dataset é promovido a instrução de sistema. A orientação pede a revisão das orações completas, incluindo palavras funcionais, ocorrências repetidas, verbos de ligação/auxiliares e construções de realce quando sustentam uma afirmação; a fidelidade semântica continua prioritária. A mensagem de usuário continua contendo somente texto e mapa.

A resposta da API tem raiz `{"informacoes": [...]}`. Cada lista validada é **acrescentada**, em ordem, a `segment["information"]`, preservando todo o histórico. Cada referência a um token contribui uma unidade para seu `count`, por informação: duas informações na mesma geração que incluem o token contribuem duas unidades. Informações iguais, inclusive entre gerações, são mantidas e contam novamente; `count` mede referências acumuladas, sem avaliar se os fatos são semanticamente distintos. Não são salvos envelope da API, modelo, fingerprint, credenciais ou registros separados por geração.

No início, os contadores são recalculados a partir de **todas** as informações existentes, sem somar os `count` antigos. Assim, um histórico válido com contagens antigas ou zeradas pode ser retomado sem perder informações nem contar duas vezes. Após cada geração validada, o código acrescenta a lista, incrementa as referências e verifica se **todos** os tokens do mapa têm `count >= 2`. Por exemplo, as listas de IDs `[0, 1]` e `[0, 2]` produzem contagens `2, 1, 1`; uma próxima informação com `[1, 2]` completa a meta `2, 2, 2`. Contagens acima de dois são preservadas. Um segmento já coberto não faz novas chamadas. Um mapa vazio satisfaz a condição sem inferência; um mapa com somente um token é rejeitado antes de qualquer chamada, pois o contrato de cada informação exige pelo menos dois IDs distintos conhecidos.

A validação adicional rejeita campos extras/ausentes, strings vazias ou sem letras/números, menos de dois IDs por informação, IDs booleanos/desconhecidos/repetidos ou fora de ordem. Uma geração pode retornar lista vazia ou cobertura parcial; nesse caso o segmento continua. A aplicação não fabrica informações para cobrir tokens; a qualidade e o suporte factual das afirmações ainda precisam de avaliação científica.

`max_generations_per_segment` é uma proteção finita contra omissões persistentes, inclusive gerações vazias. O limite conta somente novas chamadas desta execução. Cada nova chamada recebe o estado atual dos IDs pendentes; o ordinal muda mesmo se a resposta anterior não avançar, sem alterar modelo, temperatura, seed ou schema. Se a última geração permitida ainda deixar algum token abaixo de dois, a execução falha com `IncompleteCoverageError`, identificando **todos** os segmentos incompletos, IDs pendentes e suas contagens, inclusive segmentos que ainda não iniciaram. O limite nunca transforma cobertura incompleta em sucesso.

Nessa falha específica, `run_extraction` e a CLI salvam atomicamente as informações validadas e suas contagens no destino escolhido e continuam retornando erro (código 1 na CLI), com a mensagem `Progresso parcial salvo em ...`. Para retomar, execute `extract` usando **esse arquivo salvo como entrada**; com `--output`, isso significa usar o arquivo de saída na próxima execução. O histórico será revalidado e recontado uma única vez, os segmentos já cobertos serão ignorados e somente novas respostas serão acrescentadas. Não execute `map` para retomar, pois ele reinicializa o mapa. O limite de novas gerações recomeça por execução. A API em memória `extract_dataset`/`ExtractionStage` mantém o chamador intacto e disponibiliza a cópia validada em `error.partial_dataset`; ela não grava arquivos. Um mapa de um token continua sendo rejeitado antes de inferências, sem snapshot ou salvamento. Respostas perdidas por execuções anteriores à implementação desse salvamento não são recuperáveis.

Há no máximo `max_concurrency` tarefas em voo, com uma geração por vez para cada segmento. Cada segmento encerra suas chamadas assim que atinge a cobertura, enquanto os demais continuam; segmentos incompletos voltam à fila. A posição de cada resultado é guardada separadamente de sua ordem de conclusão, preservando fontes e segmentos. Ao observar o limite de um segmento, a aplicação para novas submissões e aguarda **todas** as chamadas já submetidas, validando e retendo suas respostas antes de montar o snapshot parcial. Recusa, truncamento, erro de autenticação/rate limit/conexão, JSON inválido ou resposta fora do contrato, inclusive durante essa espera, abortam sem salvar o progresso dessa execução. Esses erros e entradas inválidas mantêm os arquivos anteriores intactos. Falhas de inferência não têm retries automáticos; as novas gerações de cobertura ocorrem somente depois de respostas válidas. Não há troca de modelo ou remoção de parâmetros para contornar incompatibilidades. A escrita, completa ou parcial por limite, usa o mesmo repositório JSON com substituição atômica do mapeamento; uma falha de escrita deixa o destino anterior intacto e não anuncia salvamento. Somente o sucesso com cobertura completa imprime a mensagem `Extração salva em ...`, com a quantidade total de informações acumuladas, incluindo o histórico; esse total e a quantidade de segmentos do dataset não representam o número de gerações. Diagnósticos HTTP incluem somente status e identificadores de erro sanitizados, sem corpo, headers de autorização ou texto de recusa.

### Provedor compatível ou local

Copie o TOML padrão para um arquivo de experimento e altere explicitamente, por exemplo:

```toml
provider = "openai-compatible"
model = "identificador-do-modelo-local"
base_url = "http://127.0.0.1:1234/v1"
max_concurrency = 2
max_generations_per_segment = 10
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
        # Simulação da porta: repete o trecho inteiro, sem decomposição semântica.
        return InferenceResult({"informacoes": [{
            "info": segment.content,
            "tokens": [entry.token for entry in segment.tokens],
            "context": "O que foi afirmado no trecho?",
        }]})

extracted = extract_dataset(
    mapped, provider=OfflineExample(), max_concurrency=2, max_generations_per_segment=10,
)
```

Implementações dessa porta devem ser seguras para chamadas simultâneas e podem usar `segment.coverage.generation` e `segment.coverage.pending` (tupla de pares ID/contagem) para orientar novas gerações; o texto e os tokens originais continuam completos e imutáveis. A meta é acumulada entre gerações e não exige criar afirmações artificiais para obter duas referências na mesma resposta. As associações se referem à passagem original, mesmo quando a paráfrase usa outras palavras. `SemanticExtractionExperiment(pipeline, repository=...)` permite trocar o pipeline ou a persistência; `extraction_pipeline(provider, max_concurrency=..., max_generations_per_segment=...)` compõe apenas a extração. `run_extraction(input_path, output_path=None, config=..., provider=..., repository=...)` executa a composição e retorna o resultado salvo quando a cobertura está completa. A injeção de um provedor não lê credenciais nem acessa APIs.

### Reprodutibilidade e verificação da API

Temperatura zero e semente fixa não garantem determinismo. A documentação oficial descreve `seed` como parâmetro depreciado de melhor esforço; diferenças de infraestrutura e do modelo ainda podem mudar a saída. O adaptador usa Chat Completions, preserva `seed` e usa `reasoning_effort = "none"` para permitir `temperature` no GPT-6 Luna. Consulte as fontes oficiais: [modelo](https://developers.openai.com/api/docs/models/gpt-6-luna), [parâmetros de GPT-6](https://developers.openai.com/api/docs/guides/latest-model), [Chat Completions e seed](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat) e [limites de reprodutibilidade](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter).

Em 25/10/2026, uma consulta de disponibilidade e **uma chamada de geração**, com o segmento atual de 21 palavras, confirmaram acesso a `gpt-6-luna` e aceitação do JSON Schema, temperatura `0`, semente `42` e reasoning `none`. A resposta informou modelo `gpt-6-luna`, sem `system_fingerprint`, e gerou as três informações presentes no dataset. O conteúdo, mapa, contagens e demais metadados foram comparados antes/depois e preservados. Foi uma verificação de integração e estrutura JSON; não houve benchmark nem repetição para medir determinismo. Para comparações futuras, registre versão do código, configuração, dataset, modelo retornado e `system_fingerprint` quando disponível em `InferenceResult`; o experimento padrão não cria um registro automático.

## Experimento separado de agrupamento semântico

`group` lê **cada ocorrência** de `information[*].info` em cada segmento de cada fonte e reconstrói somente `set_different_information`. Não executa mapeamento, extração nem geração de texto. `content`, `map`, `count`, `information` e todos os metadados continuam iguais. Após uma nova extração ou qualquer alteração/concatenação em `information`, execute `group` novamente para atualizar esse resultado derivado.

O padrão de [semantic_grouping.toml](determinism/experiments/configs/semantic_grouping.toml) é OpenAI, `text-embedding-3-small`, limiar `0.85`, lotes de 32 textos e dimensão padrão do modelo (1536 para esse modelo). O banco é uma instância nova e exclusiva do experimento:

```powershell
python -m pip install -e .
# Complete o .env existente seguindo .env.example; use uma senha local aleatória.
docker compose -f compose.semantic-grouping.yaml up -d --wait
python -m determinism group data/processed/bebe_real_segments.json
```

O `.env` ignorado contém `OPENAI_API_KEY` e as variáveis `GROUPING_DB_HOST=127.0.0.1`, `GROUPING_DB_PORT=55432`, `GROUPING_DB_NAME=semantic_grouping`, `GROUPING_DB_USER=grouping` e `GROUPING_DB_PASSWORD`. A senha de exemplo deve ser substituída; não versione credenciais. O ambiente tem precedência sobre `.env`. A CLI aceita `--output <arquivo.json>`, `--config <arquivo.toml>` e `--env-file <arquivo.env>`; sem `--output`, substitui a entrada atomicamente após sucesso integral. O diretório de destino deve existir.

O Compose usa a imagem oficial `pgvector/pgvector:0.8.1-pg17`, fixada pelo digest, com container `determinism-grouping-postgres`, volume `determinism-grouping-embeddings` e porta vinculada somente a `127.0.0.1:55432`. Para usar outra porta livre, altere `GROUPING_DB_PORT`. A aplicação inicializa `CREATE EXTENSION IF NOT EXISTS vector` e seu schema/tabelas de forma idempotente. Para interromper e retomar, preservando o volume:

```powershell
docker compose -f compose.semantic-grouping.yaml stop
docker compose -f compose.semantic-grouping.yaml start --wait
```

### Regras e formato

Todo o dataset é validado antes de consultar o cache ou o provedor. Cada segmento precisa de `information` como lista; cada item precisa de `info` não vazio e `context` string UTF-8. Contexto vazio é permitido, e somente `info` é enviado ao provedor. `information: []` produz `set_different_information: []`, sem inferência; a ausência da lista é um erro explícito. `tokens` e campos adicionais permanecem no histórico original, mas não entram na saída do agrupamento:

```json
"set_different_information": [
    [
        {"info": "A casa é verde.", "context": "Qual é a cor da casa?", "avg_similarity": 1.0},
        {"info": "A casa é verde.", "context": "Como a casa foi descrita?", "avg_similarity": 1.0}
    ],
    [{"info": "A casa não tem porta.", "context": "A casa tem porta?", "avg_similarity": 1.0}]
]
```

Os grupos formam uma partição: cada ocorrência original aparece exatamente uma vez, inclusive strings idênticas com contextos iguais ou diferentes. Nunca se agrupam informações de segmentos ou fontes diferentes. O cache pode compartilhar o vetor de um texto idêntico, mas a matriz de cossenos compara **todos os pares de ocorrências dentro de cada segmento**. Essa matriz é calculada uma vez e guardada em memória para as fusões, validação e descrição dos grupos. `avg_similarity` é somente uma medida descritiva: a média dos cossenos do membro com os demais membros de seu grupo, excluindo ele mesmo. Para singleton, `1.0` é uma convenção explícita, pois não há pares.

A estratégia padrão é o **agrupamento hierárquico aglomerativo por complete linkage**. Cada ocorrência começa em seu próprio singleton. A similaridade entre dois grupos é o **menor cosseno entre todos os seus pares cruzados**; em cada etapa, funde-se o par de grupos com a maior dessas similaridades, desde que ela seja maior ou igual ao limiar. Assim, todo par dentro de um grupo final precisa atingir `0.85` na configuração padrão. Uma média alta nunca permite incluir um par abaixo do limiar, e `avg_similarity` não controla a participação no grupo.

O processo termina quando nenhuma fusão elegível resta. Cada fusão reduz o número de grupos em um; com `n` ocorrências, há no máximo `n - 1` fusões. Empates usam as tuplas ordenadas de índices originais; grupos e membros são apresentados em ordem de primeira ocorrência. Um validador independente verifica a partição exata, todos os pares internos e a ausência de fusões complete linkage elegíveis entre os grupos finais. A sequência hierárquica é determinística para uma mesma matriz e limiar, sem prometer o menor número possível de grupos.

### Cache e outros provedores

O schema `semantic_grouping` armazena perfis em `embedding_profiles` e vetores em `embeddings`. A identidade inclui provider, model, endpoint, dimensão solicitada, `encoding_format="float"`, `cache_revision` e texto exato. Metadados e dimensão efetiva são verificados na leitura e na escrita. `vector` sem dimensão fixa permite modelos com dimensões diferentes; um perfil mantém uma única dimensão efetiva. Upserts são transacionais e idempotentes; não há índice ANN, pois a comparação exaustiva acontece por segmento em Python.

pgvector armazena componentes em float32. A aplicação persiste e **relê todos os vetores solicitados antes de calcular similaridades**, inclusive na primeira execução. Assim, a repetição com cache usa a mesma representação numérica; valores no limiar não são arredondados para forçar agrupamentos. Vetores nulos, não finitos, malformados ou com dimensão incorreta são rejeitados.

Para um servidor local com `/v1/embeddings`, copie o TOML e altere:

```toml
provider = "openai-compatible"
model = "modelo-local-de-embeddings"
base_url = "http://127.0.0.1:1234/v1"
api_key_env = "LOCAL_AI_API_KEY"
threshold = 0.85
batch_size = 32
cache_revision = "pesos-v1"
# dimensions = 768 # Opcional; o servidor precisa aceitar esse parâmetro.
```

A credencial é opcional para esse provedor. Altere `cache_revision` ao trocar pesos locais mantendo nome e URL. `dimensions`, quando informado, é enviado sem fallback; caso contrário, usa-se a dimensão padrão retornada pelo modelo. As respostas são restauradas por `data[*].index`, mesmo fora de ordem. O adaptador não trunca textos, omite parâmetros automaticamente, segue redirects ou repete requisições. A API valida os limites de tokens (OpenAI: até 8192 por texto, 300000 por lote e 2048 inputs); excesso gera erro HTTP explícito sem salvar o JSON. O lote padrão de 32 mantém o teto agregado quando cada texto respeita seu limite.

`EmbeddingProvider`, `VectorStore`, `GroupingStrategy` e `DatasetRepository` são portas independentes. Para outro protocolo ou uma biblioteca local, implemente `embed(texts)` retornando uma sequência de vetores numéricos na mesma ordem, atribua um `EmbeddingProfile` fiel ao modelo e injete as implementações:

```python
from determinism.experiments import group_dataset
from determinism.modules.semantic_grouping.domain import CompleteLinkageGrouping, EmbeddingProfile

grouped = group_dataset(
    extracted,
    provider=local_embedder,
    vector_store=persistent_vector_store,
    profile=EmbeddingProfile("local-python", "meu-modelo", "local://meus-pesos", cache_revision="1"),
    strategy=CompleteLinkageGrouping(),
    threshold=0.85,
)
```

Um `PostgresVectorStore` construído diretamente deve receber `initialize()` antes do primeiro uso. `grouping_pipeline(...)` compõe somente essa etapa; `SemanticGroupingExperiment(pipeline, repository=...)` permite trocar o pipeline inteiro. `run_grouping(..., config=..., provider=..., vector_store=..., strategy=..., repository=...)` oferece a composição com arquivos. Dependências injetadas não exigem credenciais da OpenAI nem um banco real nos testes offline.

Fontes oficiais: [Embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create), [guia de embeddings](https://developers.openai.com/api/docs/guides/embeddings), [pgvector e Docker](https://github.com/pgvector/pgvector#docker), [instalação do Psycopg](https://www.psycopg.org/install/).

Em 25/10/2026, uma única requisição real de embeddings processou os **3 textos únicos das 6 ocorrências** do dataset atual: vetores de 1536 dimensões persistidos e grupos com 4 e 2 membros. A igualdade profunda dos campos originais foi verificada. Após reiniciar somente o serviço desse experimento, a repetição usou zero chamadas ao provedor e produziu os mesmos bytes JSON. O banco ficou com os 3 vetores reais; as integrações removem somente seus perfis temporários exclusivos.

Com complete linkage e esses mesmos vetores, o resultado continua contendo **6 ocorrências em grupos de 4 e 2**: o cosseno entre os textos A e B é `0.8853809907660802`, acima de `0.85`; A–C e B–C são `0.7220780303650856` e `0.6829254158505015`. Portanto, A e B e suas repetições podem compartilhar um grupo; C e sua repetição ficam no outro. O limiar e os dados originais permanecem iguais.

## Catálogo global e remoção de segmentos

As etapas `map`, `extract` e `group` continuam independentes e com seus contratos anteriores. A etapa `catalog` constrói um **artefato global separado**, sem modificar o dataset nem substituir `set_different_information`. A unidade de remoção é o segmento; a unidade de cobertura é uma afirmação do catálogo. Um grupo local pode conter vários detalhes e não é tratado como uma informação indivisível.

```powershell
python -m determinism catalog data/processed/bebe_real_segments.json --output data/processed/bebe_real_catalog.json
python -m determinism simulate data/processed/bebe_real_catalog.json --remove s0.seg0 --output data/processed/bebe_real_removal_band_seg0.json --dataset data/processed/bebe_real_segments.json
```

`catalog` imprime os IDs dos segmentos: `s0.seg0` significa o segmento de índice 0 da fonte de índice 0, com índices base zero. O nome da fonte é apenas descritivo; fontes com nomes iguais continuam distintas. Os IDs são válidos **dentro do snapshot identificado pelo hash do dataset**; não devem ser reutilizados em outro catálogo sem verificar essa identidade.

Para remover vários segmentos simultaneamente, repita `--remove`. Para remover todos os segmentos de uma fonte, use `--remove-source s0`. Os seletores podem ser combinados; repetições contam uma única vez. Nenhuma remoção altera fisicamente o dataset:

```powershell
python -m determinism simulate data/processed/bebe_real_catalog.json --remove s0.seg0 --remove s1.seg0 --output data/processed/bebe_real_removal_primeiros_segmentos.json
python -m determinism simulate data/processed/bebe_real_catalog.json --remove-source s0 --output data/processed/bebe_real_removal_band.json
```

Sem seletores, `simulate` relata a cobertura original. IDs desconhecidos são erros. `--dataset` é opcional: quando informado, verifica o hash integral do dataset, incluindo ordem e metadados, e rejeita catálogos desatualizados. Sem essa opção, o cenário se refere ao snapshot embutido no catálogo. Os destinos dos dois comandos são obrigatórios e devem ser diferentes das entradas, inclusive quando caminhos apontam ao mesmo arquivo por hardlink. A gravação é atômica e o diretório de destino deve existir.

### Construção do catálogo

O método padrão `method="requirements"` produz catálogos de versão 2. Ele usa as unidades já disponíveis em `information`, sem uma segunda extração livre de `content`. A fonte original continua sendo a referência para validar suporte; uma informação extraída anteriormente não se torna verdadeira por estar no inventário.

1. Valida toda a entrada e registra os segmentos, conteúdo, mapas, informações, grupos e metadados. O inventário preserva as formulações e suas referências de origem, unindo apenas duplicatas textuais do mesmo segmento. Os candidatos recebem IDs derivados de conteúdo e origem. Nenhum grupo local por embedding elimina automaticamente qualificadores ou unidades candidatas.
2. Extrai automaticamente os requisitos de cada afirmação fixa, classificando participantes, predicado, atribuição, polaridade, modalidade e qualificadores. O modelo seleciona IDs de palavras da própria afirmação; o código constrói o texto e os IDs dos requisitos. A cobertura integral desses tokens é validada, mas cobertura lexical não prova que a decomposição semântica está correta.
3. Compara todos os pares de candidatos, inclusive dentro da mesma fonte. O avaliador classifica cada par como `equivalent`, `different` ou `uncertain`. Equivalência exige o mesmo conteúdo factual nas duas direções. Similaridade temática ou implicação somente em uma direção não autoriza a fusão.
4. Confirma cada proposta de equivalência em uma segunda comparação, usando somente as afirmações isoladas. Só então forma grupos por ligação completa sobre as decisões de equivalência, independentemente do limiar de embeddings da etapa `group`. Todo par de um grupo deve ser equivalente nas duas verificações. Relações incertas não fundem unidades; cadeias A–B–C não implicam A–C. A formulação canônica vem de um candidato real, sem nova síntese.
5. Avalia cada unidade global contra cada segmento, requisito por requisito, com evidências literais. Cada requisito recebe `entailed`, `absent`, `contradicted` ou `ambiguous`; a compatibilidade das referências é avaliada separadamente. O código calcula o suporte final: ausência/contradição ou referência incompatível produz `not_supported`; caso contrário, ambiguidade produz `uncertain`; todos os requisitos satisfeitos e referências compatíveis produzem `supported`. Um detalhe ausente não é compensado por uma dúvida de referência.

O artefato v2 contém `segments`, `candidates`, `comparisons`, `facts`, `rejected_facts`, `policy`, `provenance`, `dataset_fingerprint`, `catalog_id` e `decision_id`. As avaliações preservam os resultados de cada requisito, as evidências e a decisão derivada. As evidências têm offsets de caracteres Unicode, com início inclusivo e fim exclusivo. O contexto literal ligado a um candidato identifica sua origem; a confirmação de suporte depende das avaliações, não dessa ligação de origem.

`catalog_id` identifica o artefato completo. `decision_id` identifica as unidades, requisitos, relações, agrupamentos e decisões de suporte, ignorando evidências, justificativas e metadados de execução. Hashes detectam alterações e identificam snapshots; não são assinaturas nem certificam correção semântica. Unidades sem suporte confirmado ou possível ficam em `rejected_facts`, preservadas para auditoria e excluídas das contagens de perda.

`method="legacy"` permite executar o método anterior, com extração livre no catálogo e schema v1. Catálogos v1 salvos continuam aceitos pelo simulador. As versões representam políticas diferentes e não devem ser misturadas na mesma comparação de estabilidade.

### Perda, preservação e incerteza

Uma informação originalmente confirmada fica **preservada** se algum suporte confirmado sobreviver; fica **incerta** se só restar suporte possível; fica **perdida** se nenhum suporte confirmado ou possível sobreviver. Uma informação cuja própria presença original era incerta permanece na categoria incerta, mesmo após remover todos os seus possíveis suportes; `baseline_status` e `remaining_status` explicam esse caso.

Os relatórios contêm `lost_information`, `preserved_information`, `uncertain_information` e `summary`. Cada entrada inclui a afirmação, evidências restantes confirmadas e possíveis, suportes interrompidos, segmentos faltantes e fontes antes/depois. A redução de fontes é registrada separadamente da perda de conteúdo. Não se somam perdas individuais: remover A ou B pode preservar um fato, enquanto remover ambos pode eliminá-lo.

Nos relatórios v2, `requirement_coverage` mostra também quais componentes de cada afirmação permanecem preservados, perdidos ou incertos, com seus segmentos de suporte. Uma afirmação composta pode deixar de ser sustentada integralmente enquanto parte de seu conteúdo permanece: em “A ponte abriu na terça-feira”, a abertura pode continuar presente e somente o dia ser perdido. Esses componentes não são novas afirmações globais autônomas e não aumentam `total_information`; a contagem principal continua medindo as unidades extraídas. A CLI exibe os componentes das afirmações perdidas para tornar essa perda parcial visível.

A construção automática verifica **suporte independente por segmento**. Referências que só possam ser resolvidas com outro segmento devem produzir incerteza; o contexto de origem de uma informação não pode preencher lacunas do segmento avaliado. O simulador continua aceitando os suportes conjuntos do formato v1: todos os `segment_ids` daquela evidência precisam sobreviver. O catálogo v2 utiliza avaliações independentes de um segmento por vez.

### Configuração, cache e execução local

`--config` seleciona um TOML; o padrão é `determinism/experiments/configs/semantic_catalog.toml`. O avaliador padrão usa `gpt-6-luna`, temperatura 0, seed 42, concorrência 4, lotes de 24 comparações e uma avaliação, até 8192 tokens de saída e timeout de 120 segundos. `--env-file` e `api_key_env` seguem as regras existentes, com precedência do ambiente. Para servidor local, configure `provider="openai-compatible"`, `model`, `base_url` e, se necessário, `reasoning_effort="omit"`. O catálogo não depende do PostgreSQL nem faz chamadas de embeddings; esta implementação é uma referência semântica exaustiva.

O cache padrão da CLI fica em `.cache/semantic_catalog`, ignorado pelo Git. `--cache-dir` escolhe outro local e `--no-cache` força novas avaliações. Cada resposta validada é persistida atomicamente, identificada pela operação, dados, configuração, revisão e hash dos prompts/schemas. Falhas não substituem o catálogo anterior; respostas já validadas permitem retomar a execução. Não há retries silenciosos. `cache_revision` deve mudar ao trocar pesos locais com o mesmo nome e endpoint. O cache contém textos e decisões do corpus, mas não credenciais.

Repetir com o mesmo cache preserva o catálogo byte a byte; reutilização integral não exige credencial nem rede. Novas inferências não têm garantia de determinismo. Prompts, schemas, configuração, modelo retornado e `system_fingerprint`, quando disponível, ficam identificados na proveniência.

### Comparar decisões de duas execuções

Para avaliar inferências novas, gere os catálogos em arquivos separados usando `--no-cache`. Em seguida, compare as decisões do mesmo snapshot, sem novas chamadas ao modelo:

```powershell
python -m determinism catalog data/processed/bebe_real_segments.json --output data/processed/catalog_first.json --no-cache
python -m determinism catalog data/processed/bebe_real_segments.json --output data/processed/catalog_second.json --no-cache
python -m determinism compare-catalogs data/processed/catalog_first.json data/processed/catalog_second.json --output data/processed/catalog_comparison.json --fail-on-difference
```

`compare-catalogs` exige dois catálogos v2 do mesmo dataset. O relatório separa diferenças de inventário, requisitos, equivalências, grupos, representantes e suporte. Ele também distingue igualdade das decisões (`same_decisions`), igualdade do artefato (`same_catalog_content`) e igualdade dos bytes dos arquivos (`same_file_bytes`). Não alinha IDs posicionais de versões antigas nem confunde a variação de uma explicação com uma nova decisão semântica.

Com `--fail-on-difference`, o comando grava o diagnóstico e retorna código 2 se as decisões mudarem; código 0 significa decisões iguais, mesmo quando citações ou formatação diferem. Código 1 indica erro de entrada ou execução. Esse controle pode interromper uma rotina antes de promover a nova versão: a comparação nunca substitui ou escolhe automaticamente um dos catálogos. `inference_instability` descreve divergência entre execuções; a categoria `uncertain` do simulador continua descrevendo suporte textual ambíguo.

A API equivalente é `compare_catalogs(first, second)` para objetos em memória ou `run_catalog_comparison(first_path, second_path, output_path)` para arquivos. As inferências automáticas de requisitos e equivalência continuam sujeitas a erro e variação. Reprodutibilidade exata de um resultado publicado exige preservar o catálogo e suas decisões; concordância entre poucas chamadas não garante determinismo geral.

### API Python e vários cenários

```python
from determinism.experiments import run_catalog, RemovalSimulator

catalog = run_catalog(
    "data/processed/bebe_real_segments.json",
    "data/processed/bebe_real_catalog.json",
    cache_dir=".cache/semantic_catalog",
)
simulator = RemovalSimulator(catalog)  # Valida e compila o índice uma única vez.
individual = simulator.simulate(removed_segment_ids=["s0.seg0"])
conjunta = simulator.simulate(removed_segment_ids=["s0.seg0", "s1.seg0"])
fonte = simulator.simulate(removed_source_ids=["s0"])
print(conjunta["lost_information"])
```

`simulate_removal(catalog, ...)` é a alternativa para um único cenário. Nenhuma simulação usa modelos. `RemovalSimulator` mantém um índice imutável e não revalida a matriz de candidatos a cada consulta. `build_catalog(dataset, provider=...)` permite trocar o avaliador. O método por requisitos usa as portas `requirements`, `compare` e `assess_requirements`; o método legado usa `extract`, `compare` e `assess`. Provedores precisam ser seguros para chamadas concorrentes. Cache injetado exige `provenance` explícita e versionada; `run_catalog` exige `provider_identity` ao combinar provedor injetado com cache em disco.

Com N candidatos, a referência compara N(N−1)/2 pares; com F unidades globais e S segmentos, avalia F×S relações de suporte. Batching, concorrência limitada e cache reduzem chamadas repetidas; não eliminam esse custo quadrático de construção. Para muitos cenários, reutilize o catálogo e o índice compilado. Uma busca aproximada futura deve medir sua recuperação contra essa referência, pois omissões podem causar falsas perdas.

O resultado estima cobertura de **afirmações presentes nas transcrições**, não verdade externa nem informação visual/sonora ausente do texto. A granularidade e a completude da extração afetam a contagem; a meta lexical `count >= 2` não certifica completude semântica. Equivalências incertas ficam registradas em `comparisons` e podem produzir unidades ainda duplicadas: a contagem não é uma prova de distinção semântica absoluta. Valide equivalências, detalhes, atribuições, datas relativas e cenários de remoção com anotação humana antes de interpretar as contagens como resultado científico.

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
│   ├── semantic_extraction/
│   │   ├── domain/                # Entradas imutáveis e validação da resposta/mapa
│   │   ├── ports/                 # Inferência e persistência substituíveis
│   │   ├── application/           # Etapa concorrente com sucesso integral
│   │   ├── adapters/              # HTTP compatível, configuração e mensagens
│   │   └── assets/                # Sistema e JSON Schema versionados separados
│   └── semantic_grouping/
│       ├── domain/                # Validação, cossenos e agrupamento determinístico
│       ├── ports/                 # Embeddings, cache, estratégia e repositório
│       ├── application/           # Etapa isolada por segmento, com releitura do cache
│       └── adapters/              # HTTP embeddings, TOML e PostgreSQL/pgvector
├── pipelines/
│   └── sequential.py              # Pipeline genérico e contrato de etapa
├── experiments/
│   ├── word_mapping.py            # Definição do mapeamento
│   ├── semantic_extraction.py     # Composição explícita da extração
│   ├── semantic_grouping.py       # Composição explícita do agrupamento
│   └── configs/
│       ├── semantic_extraction.toml
│       └── semantic_grouping.toml
├── entrypoints/
│   └── cli.py                    # Interface de linha de comando
└── __main__.py
```

As dependências apontam para as regras internas: `application` usa seu `domain` e suas portas; `adapters` implementa essas portas; `ports` depende somente de tipos internos e da biblioteca padrão. O domínio não conhece arquivos, CLI, experimentos ou adaptadores. Os domínios e aplicações dos três módulos não importam um ao outro. As composições da extração e do agrupamento reutilizam o adaptador JSON existente por compatibilidade estrutural com suas próprias portas de persistência, sem acoplar suas aplicações ao módulo de mapeamento. `pipelines` só conhece um tipo genérico e objetos chamáveis, sem importar módulos funcionais ou JSON. `experiments` reúne essas peças e `entrypoints` aciona o experimento. O pacote raiz não carrega adaptadores ao importar um componente interno.

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

A política alternativa é apenas uma demonstração de substituição; a CLI mantém a política Unicode documentada. Para novos experimentos, defina a composição em `experiments/`; para novas capacidades funcionais, crie um módulo em `modules/`. Os experimentos de mapeamento e extração são executados separadamente. O mapeamento inicializa contagens em zero; a extração as deriva das informações acumuladas e só conclui com a cobertura mínima de duas referências por token.

Erros de estrutura indicam a localização do campo, por exemplo `$[1].segments[0].content`. JSON inválido, chaves duplicadas, constantes como `NaN` e arquivos que não sejam UTF-8 são rejeitados. A leitura aceita BOM UTF-8; a saída usa UTF-8 sem BOM. Todo o dataset é validado e mapeado antes da escrita; o JSON é serializado integralmente e gravado em arquivo temporário no diretório de destino, seguido por substituição atômica. Falhas anteriores à substituição deixam o arquivo existente intacto. A formatação JSON é padronizada, preservando os valores e campos dos dados.

## Verificação

```powershell
python -m unittest discover -s tests -v
```

Para executar também as integrações contra o banco local configurado (sem chamadas externas de API):

```powershell
$env:GROUPING_TEST_POSTGRES = "1"
python -m unittest discover -s tests -p test_postgres_vectors.py -v
Remove-Item Env:GROUPING_TEST_POSTGRES
```

Os testes do agrupamento verificam limiar inclusivo, singletons, duplicatas, todos os pares calculados uma vez, prioridade pelo maior mínimo entre grupos, empates, maximalidade da partição, isolamento por fonte/segmento, preservação dos campos, rejeição de pares abaixo do limiar mesmo quando todas as médias passam, cache/invalidação, float32, batching/índices/erros do provedor e atomicidade do JSON. As integrações usam perfis com UUID e removem exclusivamente seus registros no final; verificam pgvector real, persistência, idempotência, dimensões e a CLI com servidor HTTP local.

Os testes são offline e não requerem chave de API. Cobrem o exemplo de oito palavras, múltiplas fontes e segmentos, substituição de mapas, reinicialização de `count`, pontuação, caixa, acentos, Unicode, repetições, vazios, preservação dos dados, reuso por identidade em memória e pipeline genérico. A extração cobre concorrência realmente simultânea com conclusão fora de ordem, limite de tarefas, parada independente por segmento, contagem por informação, concatenação de duplicatas, retomada do histórico com contagens antigas, segmentos já cobertos sem chamadas, imutabilidade dos dados, mapas/histórico inválidos antes de qualquer chamada, mapas vazios e de um token, feedback numérico e ordinal por geração, revisão das ocorrências 17/23, snapshots no limite com chamadas em voo e segmentos não iniciados, retomada sem dupla contagem, validação estrita dos IDs/respostas, separação sistema/dados/schema, escapes, propagação da configuração, respostas recusadas/truncadas/inválidas, erros HTTP sanitizados sem retries, provedores/pipelines/repositórios alternativos e proteção atômica de arquivos. Um servidor HTTP local simulado verifica sucesso completo e salvamento parcial com saída nãozero no fluxo da CLI.
