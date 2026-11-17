# Determinismo aplicado ao processamento de texto

Aplicação Python 3.11 ou superior para experimentos científicos de mestrado. O experimento de mapeamento constrói um mapa de **ocorrências de palavras por segmento**, independente de modelos de linguagem. Cada execução explícita de `map` reconstrói `map` em todos os segmentos de todas as fontes e inicializa `count` com zero. A extração usa os mapas existentes para gerar informações semânticas. O agrupamento separado compara essas informações usando embeddings persistidos em PostgreSQL. Mapeamento e extração usam a biblioteca padrão; o adaptador PostgreSQL requer `psycopg[binary]`, instalado com `python -m pip install -e .`.

## Executar o experimento

Na raiz do repositório:

```powershell
python -m determinism map data/transcriptions/bebe_real_sources.json
```

O comando lê a transcrição base em `data/transcriptions/` e salva a transcrição completa com os mapas em `data/processed/bebe_real_sources.json`, mantendo o mesmo nome e preservando o arquivo original. Para escolher outro destino:

```powershell
python -m determinism map data/transcriptions/bebe_real_sources.json --output data/processed/bebe_real_mapped.json
```

O diretório de destino deve existir. Para entradas em `transcriptions/`, o destino padrão fica no diretório irmão `processed/`, inclusive ao usar caminhos absolutos. Para entradas em outros diretórios, o padrão é `data/processed/<nome do arquivo>`, relativo ao diretório de execução. O destino não pode ser o próprio arquivo de entrada, mesmo com `--output`.

A transcrição base fornecida contém **2 fontes e 4 segmentos**. Reexecutar `map` reconstrói o resultado em `processed` a partir da transcrição base, substituindo o resultado anterior e inicializando todas as contagens em zero. O mesmo conteúdo e a mesma política de tokenização produzem o mesmo resultado. Execute a extração e o agrupamento sobre o arquivo resultante em `processed`; ao recriá-lo a partir da base, essas etapas precisam ser executadas novamente.

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

Etapas são funções ou objetos chamáveis que recebem e retornam o dataset. Um tokenizador alternativo implementa `tokenize(text)`, devolvendo as palavras em ordem, e pode ser injetado em `MappingStage` ou em `map_dataset(dataset, tokenizer=...)`. `run_mapping(input_path, output_path=None, repository=..., tokenizer=...)` executa entrada, mapeamento e persistência e retorna o próprio resultado em memória. Tanto essa função quanto `WordMappingExperiment.run` usam o mesmo destino padrão da CLI em `processed` e rejeitam a sobrescrita da transcrição de entrada. Um repositório alternativo implementa `load(path)` e `save(path, dataset)`.

## Extração semântica concorrente

Com `OPENAI_API_KEY` no ambiente ou no arquivo `.env` da raiz (consulte `.env.example`), execute:

```powershell
python -m determinism extract data/processed/bebe_real_sources.json
```

O comando processa todos os segmentos de todas as fontes. Ao concluir, substitui a entrada; se atingir o limite de cobertura, salva o progresso validado nesse mesmo destino e termina com erro, indicando as pendências. `--output <arquivo.json>` escolhe outro destino, preservando a entrada; `--config <arquivo.toml>` escolhe outra configuração; `--env-file <arquivo>` escolhe outro arquivo de credencial. O ambiente tem precedência sobre `.env`, inclusive quando a variável estiver vazia. O leitor de `.env` aceita `NOME=valor`, aspas simples/duplas e linhas de comentário; não executa expansão ou código. A credencial não faz parte do TOML nem dos resultados.

A configuração versionada está em [`determinism/experiments/configs/semantic_extraction.toml`](determinism/experiments/configs/semantic_extraction.toml). Esses valores são carregados também após instalar o pacote:

| Campo | Padrão | Uso |
| --- | --- | --- |
| `provider` | `openai` | `openai` ou `openai-compatible` |
| `model` | `gpt-6-luna` | Identificador exato, sem substituição automática |
| `base_url` | `https://api.openai.com/v1` | Endpoint base de Chat Completions |
| `max_concurrency` | `4` | Limite de chamadas e tarefas pendentes (1 a 256) |
| `max_generations_per_segment` | `20` | Limite de novas gerações por segmento nesta execução; cobertura incompleta causa erro |
| `temperature` | `0.0` | Temperatura enviada explicitamente |
| `seed` | `42` | Semente inteira fixa enviada explicitamente |
| `reasoning_effort` | `none` | Permite temperatura explícita no modelo padrão |
| `timeout_seconds` | `60.0` | Timeout de transporte por chamada |
| `max_completion_tokens` | `4096` | Limite de saída por geração |
| `api_key_env` | `OPENAI_API_KEY` | Nome da variável que contém a credencial |

Cada segmento precisa ter `map` válido **antes** da execução. Não há remapeamento implícito: conteúdo, IDs, palavras e metadados são preservados. Cada execução de `extract` refaz a extração de todos os segmentos, substitui `information` e calcula os respectivos `count` somente a partir das respostas novas. Os resultados anteriores em `set_different_information` e `canonical_information` são removidos, inclusive dos snapshots de progresso parcial, pois dependem das informações substituídas; execute `group` novamente após a extração. Todos os segmentos e mapas são validados antes da primeira chamada, incluindo IDs consecutivos a partir de zero, correspondência entre chave e `token`, palavras não vazias e `count` inteiro não negativo. Isso valida a estrutura; a correspondência semântica entre um mapa antigo e um texto posteriormente editado é responsabilidade de quem prepara o experimento.

O contrato atual exige `syntagm` em cada nova informação retornada pelo provedor. Extrações anteriores são ignoradas, inclusive formatos antigos sem `syntagm` ou listas inválidas, porque serão substituídas. Basta executar novamente `python -m determinism extract data/processed/bebe_real_sources.json`; não é necessário limpar `information`, zerar contagens manualmente ou executar `map`. Isso faz novas chamadas à API mesmo quando a extração anterior já tinha cobertura completa. Use `--output` para salvar o novo resultado em outro arquivo e preservar a entrada.

O adaptador mantém estas partes separadas:

1. `modules/semantic_extraction/assets/system_prompt.v1.txt`: texto de sistema fixo fornecido para o experimento.
2. Mensagem de usuário: introdução fixa e orientação para preencher `info`, `tokens`, `context` e `syntagm`, seguidas de `<trecho>` e `<mapeamento_tokens>`, projetando o mapa para `token` e `word`. O texto é escapado como dados XML; caracteres delimitadores dentro das palavras usam escapes JSON reversíveis. O schema permanece separado dessa mensagem.
3. `modules/semantic_extraction/assets/structured_output.v1.json`: schema fixo enviado em `response_format` como JSON Schema estrito.
4. Um bloco suplementar com instruções fixas de revisão da cobertura e estado numérico dinâmico: ordinal da geração nesta execução, meta, IDs/contagens ainda abaixo de dois e número de referências que faltam (`needed`). Ele é anexado ao texto base, preservado literalmente, na mesma mensagem de sistema; a chamada continua com exatamente duas mensagens, uma de sistema e uma de usuário. O estado é derivado somente das referências validadas, ordenado por ID e recalculado antes de cada chamada, começando com todos os IDs pendentes e contagens zero na primeira chamada. Nenhuma palavra, informação extraída ou outro texto do dataset é promovido a instrução de sistema. A orientação pede a revisão das orações completas, incluindo palavras funcionais, ocorrências repetidas, verbos de ligação/auxiliares e construções de realce quando sustentam uma afirmação; a fidelidade semântica continua prioritária. A mensagem de usuário continua contendo somente texto e mapa.

A resposta da API tem raiz `{"informacoes": [...]}`. As listas validadas são acumuladas em ordem **somente dentro da execução atual**, começando com uma lista vazia. O resultado substitui `segment["information"]`; informações de execuções anteriores não são acrescentadas. Cada referência a um token contribui uma unidade para seu `count`, por informação: duas informações na mesma geração que incluem o token contribuem duas unidades. Informações iguais, inclusive entre gerações da mesma execução, são mantidas e contam novamente; `count` mede referências acumuladas, sem avaliar se os fatos são semanticamente distintos. Não são salvos envelope da API, modelo, fingerprint, credenciais ou registros separados por geração.

Cada item persistido contém exatamente `info`, `tokens`, `context` e `syntagm`. A tripla tem exatamente `subject`, `relation` e `object`, todos strings, e deve expressar a mesma informação de `info` com as mesmas evidências. `subject` e `relation` precisam ser preenchidos; `object` aceita `""` para predicados completos sem complemento, como em `{"subject": "Ana", "relation": "dormiu", "object": ""}`. A validação copia também esse objeto para evitar compartilhamento mutável com o provedor. As regras do prompt preservam atribuição, modalidade, condição, negação, quantificação e argumentos necessários; a validação estrutural não comprova equivalência semântica ou suporte factual.

No início de cada execução, a cobertura de todos os tokens começa em zero, independentemente das informações e contagens gravadas anteriormente. Após cada geração validada, o código acrescenta a lista atual, incrementa as referências e verifica se **todos** os tokens do mapa têm `count >= 2`. Por exemplo, as listas de IDs `[0, 1]` e `[0, 2]` produzem contagens `2, 1, 1`; uma próxima informação com `[1, 2]` completa a meta `2, 2, 2`. Contagens acima de dois são preservadas dentro dessa execução. Um segmento encerra suas chamadas quando atinge a cobertura atual; executar o comando novamente reinicia o processo. Um mapa vazio produz `information: []` sem inferência; um mapa com somente um token é rejeitado antes de qualquer chamada, pois o contrato de cada informação exige pelo menos dois IDs distintos conhecidos.

A validação adicional rejeita campos extras/ausentes, inclusive dentro de `syntagm`, strings vazias ou sem letras/números (com a exceção de `object: ""`), menos de dois IDs por informação, IDs booleanos/desconhecidos/repetidos ou fora de ordem. Uma geração pode retornar lista vazia ou cobertura parcial; nesse caso o segmento continua. A aplicação não fabrica informações para cobrir tokens; a qualidade e o suporte factual das afirmações ainda precisam de avaliação científica.

`max_generations_per_segment` é uma proteção finita contra omissões persistentes, inclusive gerações vazias. O limite conta somente novas chamadas desta execução. Cada nova chamada recebe o estado atual dos IDs pendentes; o ordinal muda mesmo se a resposta anterior não avançar, sem alterar modelo, temperatura, seed ou schema. Se a última geração permitida ainda deixar algum token abaixo de dois, a execução falha com `IncompleteCoverageError`, identificando **todos** os segmentos incompletos, IDs pendentes e suas contagens, inclusive segmentos que ainda não iniciaram. O limite nunca transforma cobertura incompleta em sucesso.

Nessa falha específica, `run_extraction` e a CLI salvam atomicamente as informações validadas **da execução atual** e suas contagens no destino escolhido, substituindo a extração anterior, e continuam retornando erro (código 1 na CLI), com a mensagem `Progresso parcial salvo em ...`. Não há retomada automática: uma nova execução de `extract` recomeça do zero, inclusive para segmentos completos no resultado parcial. O limite de gerações recomeça por execução. A API em memória `extract_dataset`/`ExtractionStage` mantém o chamador intacto e disponibiliza a cópia validada em `error.partial_dataset`; ela não grava arquivos. Um mapa de um token continua sendo rejeitado antes de inferências, sem snapshot ou salvamento.

Há no máximo `max_concurrency` tarefas em voo, com uma geração por vez para cada segmento. Cada segmento encerra suas chamadas assim que atinge a cobertura, enquanto os demais continuam; segmentos incompletos voltam à fila. A posição de cada resultado é guardada separadamente de sua ordem de conclusão, preservando fontes e segmentos. Ao observar o limite de um segmento, a aplicação para novas submissões e aguarda **todas** as chamadas já submetidas, validando e retendo suas respostas antes de montar o snapshot parcial. Recusa, truncamento, erro de autenticação/rate limit/conexão, JSON inválido ou resposta fora do contrato, inclusive durante essa espera, abortam sem salvar o progresso dessa execução. Esses erros e entradas inválidas mantêm os arquivos anteriores intactos. Falhas de inferência não têm retries automáticos; as novas gerações de cobertura ocorrem somente depois de respostas válidas. Não há troca de modelo ou remoção de parâmetros para contornar incompatibilidades. A escrita, completa ou parcial por limite, usa o mesmo repositório JSON com substituição atômica do mapeamento; uma falha de escrita deixa o destino anterior intacto e não anuncia salvamento. Somente o sucesso com cobertura completa imprime a mensagem `Extração salva em ...`, com a quantidade total de informações acumuladas na execução atual; esse total e a quantidade de segmentos do dataset não representam o número de gerações. Diagnósticos HTTP incluem somente status e identificadores de erro sanitizados, sem corpo, headers de autorização ou texto de recusa.

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
from determinism.experiments import extract_dataset, map_dataset
from determinism.modules.semantic_extraction.domain import InferenceResult

class OfflineExample:
    def extract(self, segment):
        # Simulação da porta para o trecho fixo deste exemplo.
        return InferenceResult({"informacoes": [{
            "info": "Ana dormiu.",
            "tokens": [0, 1],
            "context": "O que Ana fez?",
            "syntagm": {"subject": "Ana", "relation": "dormiu", "object": ""},
        }]})

example_mapped = map_dataset([{"segments": [{"content": "Ana dormiu."}]}])
extracted = extract_dataset(
    example_mapped, provider=OfflineExample(), max_concurrency=2, max_generations_per_segment=10,
)
```

Implementações dessa porta devem ser seguras para chamadas simultâneas e podem usar `segment.coverage.generation` e `segment.coverage.pending` (tupla de pares ID/contagem) para orientar novas gerações; o texto e os tokens originais continuam completos e imutáveis. A meta é acumulada entre gerações e não exige criar afirmações artificiais para obter duas referências na mesma resposta. As associações se referem à passagem original, mesmo quando a paráfrase usa outras palavras. `SemanticExtractionExperiment(pipeline, repository=...)` permite trocar o pipeline ou a persistência; `extraction_pipeline(provider, max_concurrency=..., max_generations_per_segment=...)` compõe apenas a extração. `run_extraction(input_path, output_path=None, config=..., provider=..., repository=...)` executa a composição e retorna o resultado salvo quando a cobertura está completa. A injeção de um provedor não lê credenciais nem acessa APIs.

### Reprodutibilidade e verificação da API

Temperatura zero e semente fixa não garantem determinismo. A documentação oficial descreve `seed` como parâmetro depreciado de melhor esforço; diferenças de infraestrutura e do modelo ainda podem mudar a saída. O adaptador usa Chat Completions, preserva `seed` e usa `reasoning_effort = "none"` para permitir `temperature` no GPT-6 Luna. Consulte as fontes oficiais: [modelo](https://developers.openai.com/api/docs/models/gpt-6-luna), [parâmetros de GPT-6](https://developers.openai.com/api/docs/guides/latest-model), [Chat Completions e seed](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat) e [limites de reprodutibilidade](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter).

Em 25/10/2026, uma consulta de disponibilidade e **uma chamada de geração**, com o segmento atual de 21 palavras, confirmaram acesso a `gpt-6-luna` e aceitação do JSON Schema, temperatura `0`, semente `42` e reasoning `none`. A resposta informou modelo `gpt-6-luna`, sem `system_fingerprint`, e gerou as três informações presentes no dataset. O conteúdo, mapa, contagens e demais metadados foram comparados antes/depois e preservados. Foi uma verificação de integração e estrutura JSON; não houve benchmark nem repetição para medir determinismo. Para comparações futuras, registre versão do código, configuração, dataset, modelo retornado e `system_fingerprint` quando disponível em `InferenceResult`; o experimento padrão não cria um registro automático.

## Experimento separado de agrupamento semântico

`group` lê **cada ocorrência** de `information[*].info` em cada segmento de cada fonte, reconstrói `set_different_information` e, logo após validar os grupos, escolhe uma variante original por grupo em `canonical_information`. Depois reúne esses representantes locais de **todos os segmentos e fontes**, faz um segundo agrupamento e escolhe os representantes do catálogo global. Não executa mapeamento, extração nem geração de texto. `content`, `map`, `count`, `information` e os demais metadados continuam iguais. Após uma nova extração ou qualquer alteração/concatenação em `information`, execute `group` novamente para atualizar os resultados locais e o catálogo.

O padrão de [semantic_grouping.toml](determinism/experiments/configs/semantic_grouping.toml) é OpenAI, `text-embedding-3-small`, limiar `0.90` para as duas etapas, lotes de 32 textos e dimensão padrão do modelo (1536 para esse modelo). O limiar anterior era `0.85`; configurações TOML explícitas continuam podendo selecionar outro valor. O banco é uma instância exclusiva do experimento:

```powershell
python -m pip install -e .
# Complete o .env existente seguindo .env.example; use uma senha local aleatória.
docker compose -f compose.semantic-grouping.yaml up -d --wait
python -m determinism group data/processed/bebe_real_sources.json
```

O `.env` ignorado contém `OPENAI_API_KEY` e as variáveis `GROUPING_DB_HOST=127.0.0.1`, `GROUPING_DB_PORT=55432`, `GROUPING_DB_NAME=semantic_grouping`, `GROUPING_DB_USER=grouping` e `GROUPING_DB_PASSWORD`. A senha de exemplo deve ser substituída; não versione credenciais. O ambiente tem precedência sobre `.env`. A CLI aceita `--output <arquivo.json>`, `--global-output <catalogo.json>`, `--config <arquivo.toml>` e `--env-file <arquivo.env>`. Sem `--output`, atualiza o dataset de entrada; o catálogo padrão é um irmão chamado `<nome sem extensão>.global.json`, neste exemplo `data/processed/bebe_real_sources.global.json`. Com `--output resultado.json`, o padrão passa a ser `resultado.global.json`. O destino global deve diferir da entrada e do dataset agrupado, inclusive por aliases de caminho. Os diretórios de destino devem existir.

O dataset principal mantém a raiz `list[source]` e o formato local anterior. Os dois JSONs são calculados, validados, serializados e preparados antes das substituições; uma falha comum de escrita restaura os arquivos anteriores. Se a própria restauração falhar, a cópia anterior permanece no caminho de recuperação informado no erro. Cada substituição é atômica, mas dois arquivos não constituem uma transação de filesystem: uma interrupção abrupta entre elas pode exigir executar `group` novamente. A CLI informa separadamente o total de grupos locais e o total de informações globais.

O Compose usa a imagem oficial `pgvector/pgvector:0.8.1-pg17`, fixada pelo digest, com container `determinism-grouping-postgres`, volume `determinism-grouping-embeddings` e porta vinculada somente a `127.0.0.1:55432`. Para usar outra porta livre, altere `GROUPING_DB_PORT`. A aplicação inicializa `CREATE EXTENSION IF NOT EXISTS vector` e seu schema/tabelas de forma idempotente. Para interromper e retomar, preservando o volume:

```powershell
docker compose -f compose.semantic-grouping.yaml stop
docker compose -f compose.semantic-grouping.yaml start --wait
```

### Regras e formato

Todo o dataset é validado antes de consultar o cache ou o provedor. Cada segmento precisa de `information` como lista; cada item precisa de `info` não vazio e `context` string UTF-8. Contexto vazio é permitido, e somente `info` é enviado ao provedor e influencia os scores. `information: []` produz `set_different_information: []` e `canonical_information: []`, sem inferência; a ausência da lista é um erro explícito. `tokens`, `syntagm` e campos adicionais permanecem integralmente no histórico original e podem ser recuperados pelos índices da saída canônica:

```json
"set_different_information": [
    [
        {"info": "A casa é verde.", "context": "Qual é a cor da casa?", "avg_similarity": 1.0},
        {"info": "A casa é verde.", "context": "Como a casa foi descrita?", "avg_similarity": 1.0}
    ],
    [{"info": "A casa não tem porta.", "context": "A casa tem porta?", "avg_similarity": 1.0}]
],
"canonical_information": [
    {"info": "A casa é verde.", "context": "Como a casa foi descrita?", "representative_index": 1, "member_indices": [0, 1], "avg_similarity": 1.0},
    {"info": "A casa não tem porta.", "context": "A casa tem porta?", "representative_index": 2, "member_indices": [2], "avg_similarity": 1.0}
]
```

Os grupos locais formam uma partição: cada ocorrência original aparece exatamente uma vez, inclusive strings idênticas com contextos iguais ou diferentes. A primeira etapa mantém os segmentos separados. O cache pode compartilhar o vetor de um texto idêntico, mas a matriz de cossenos compara **todos os pares de ocorrências dentro de cada segmento**. Essa matriz é calculada uma vez e guardada em memória para as fusões, validação, descrição dos grupos e escolha dos representantes. `avg_similarity` é a média dos cossenos do membro com os demais membros de seu grupo, excluindo ele mesmo. Para singleton, `1.0` é uma convenção explícita, pois não há pares.

A estratégia padrão é o **agrupamento hierárquico aglomerativo por complete linkage**. Cada ocorrência começa em seu próprio singleton. A similaridade entre dois grupos é o **menor cosseno entre todos os seus pares cruzados**; em cada etapa, funde-se o par de grupos com a maior dessas similaridades, desde que ela seja maior ou igual ao limiar. Assim, todo par dentro de um grupo final precisa atingir `0.90` na configuração padrão. Uma média alta nunca permite incluir um par abaixo do limiar, e `avg_similarity` não controla a participação no grupo.

O processo termina quando nenhuma fusão elegível resta. Cada fusão reduz o número de grupos em um; com `n` ocorrências, há no máximo `n - 1` fusões. Empates usam as tuplas ordenadas de índices originais; grupos e membros são apresentados em ordem de primeira ocorrência. Um validador independente verifica a partição exata, todos os pares internos e a ausência de fusões complete linkage elegíveis entre os grupos finais. A sequência hierárquica é determinística para uma mesma matriz e limiar, sem prometer o menor número possível de grupos.

### Canonicalização após o agrupamento

Cada item de `canonical_information` representa exatamente um grupo já validado, preservando o formato e todas as variantes de `set_different_information`. `info` e `context` são copiados de uma ocorrência original. `representative_index` é o índice dessa ocorrência em `information`; `member_indices` contém os índices de todas as ocorrências do grupo, em ordem crescente. Ambos são índices locais ao segmento, começando em zero. Assim, o consumidor usa `canonical_information[*].info` como forma unificada e consulta `information[representative_index]` para obter `tokens`, `syntagm` e metadados; os contextos alternativos continuam acessíveis em `information[member_indices[*]]`. A lista canônica segue a ordem dos grupos, mas os índices identificam sua proveniência sem depender apenas desse alinhamento.

A escolha é inspirada na seção 6 do [artigo CESI](https://malllabiisc.github.io/publications/papers/cesi_www18.pdf), que usa complete linkage e escolhe uma expressão existente próxima da média dos embeddings ponderada pela frequência de ocorrência. Aqui adotamos normalização L2 dos embeddings, distância cosseno e frequências das ocorrências em `information`; são adaptações explícitas, pois o artigo não fixa a normalização nem a métrica dessa escolha final. O experimento não reproduz o treinamento HolE nem as informações auxiliares do CESI. O artigo seleciona o limiar em validação; `0.90` é a configuração deste projeto.

Para cada grupo, vence a ocorrência de maior `avg_similarity`, ou seja, o medoide pela soma das distâncias `1 - cosseno`. Essa seleção equivale à variante mais próxima por cosseno do centroide dos vetores normalizados, ponderado pelas frequências: para `n > 1`, vetor unitário `u_i` e centroide não nulo `c`, `cos(u_i, c) = [1 + (n - 1) * avg_similarity_i] / [n * ||c||]`. O denominador é comum a todos os membros. Repetições contam como ocorrências, mesmo quando o cache compartilha um único vetor; a magnitude original dos embeddings não adiciona peso. A implementação reutiliza as médias e a matriz existentes, sem novas chamadas à API nem novo agrupamento.

Empates de score exato usam primeiro `info` em ordem lexical Unicode, depois `context` e, por último, o menor índice original. Não há arredondamento nem normalização dos textos. O contexto só desempata a proveniência de textos idênticos; `tokens`, `syntagm` e demais metadados não participam da seleção. Assim, um empate entre as mesmas variantes escolhe o mesmo texto e contexto mesmo se sua ordem mudar; a própria formação dos grupos continua sujeita ao desempate por índices descrito acima. Singleton representa a si mesmo. Se uma configuração com limiar baixo permitir um centroide nulo, o cosseno com o centroide é indefinido, mas a soma de distâncias entre membros continua definida e a mesma regra de seleção e desempate se aplica.

A forma canônica é um representante local, sem síntese ou correção factual. Similaridade alta pode reunir afirmações com detalhes diferentes; as variantes e suas evidências permanecem disponíveis para análise. A canonicalização não altera a participação nos grupos.

Em 17/11/2026, a CLI foi verificada com uma cópia temporária do primeiro segmento de `bebe_real_sources.json`: **4 ocorrências e 4 textos únicos**, uma requisição real a `text-embedding-3-small`, embeddings de 1536 dimensões, dois grupos de dois membros com todos os pares acima de `0.90` e dois representantes canônicos. A escolha foi conferida contra o cálculo direto do centroide normalizado. A repetição com cache fez zero novas requisições HTTP e produziu bytes JSON idênticos. A entrada original e seus metadados foram preservados; o perfil de cache exclusivo do teste foi removido ao final.

### Catálogo global após a canonicalização local

O arquivo `*.global.json` tem `information` como lista ordenada de candidatos: uma entrada para cada item de `canonical_information` local, percorrendo fontes, segmentos e grupos na ordem do dataset. Apenas seus textos `info` são comparados. Os vetores são relidos do mesmo cache float32; esta etapa não gera textos nem chama novamente o provedor de embeddings. Representantes de segmentos da mesma fonte e de fontes distintas podem compartilhar um grupo global.

Na raiz desse arquivo, `set_different_information` contém os grupos globais e `canonical_information` contém uma variante observada por grupo, escolhida pela mesma maior média cosseno e pelo mesmo desempate lexical local. Seus `representative_index` e `member_indices` são índices, a partir de zero, na lista **global** `information`. Cada candidato local contribui uma ocorrência global, independentemente de quantas variantes originais representa. Portanto, a frequência de folhas de um único segmento não dá peso adicional ao seu representante nessa segunda etapa.

Cada candidato e representante global carrega `provenance`:

```json
{
    "info": "A casa é verde.",
    "context": "Como a casa foi descrita?",
    "provenance": {
        "source_index": 0,
        "segment_index": 1,
        "canonical_index": 0,
        "representative_index": 1,
        "member_indices": [0, 1]
    }
}
```

Esses índices de proveniência são **locais**: acesse o segmento em `dataset[source_index].segments[segment_index]`, o representante em `canonical_information[canonical_index]` e as evidências originais em `information[member_indices[*]]`. Assim, todas as variantes, contextos, `tokens`, `syntagm` e demais campos permanecem rastreáveis no dataset principal. `sources` registra os índices e metadados de identidade das fontes e segmentos; `metadata.dataset_file` aponta para o dataset agrupado, normalmente por caminho relativo ao catálogo. `metadata` também registra versão do schema, perfil de embedding, limiar e contagens de fontes, segmentos, ocorrências originais, representantes locais e representantes globais. O total global conta grupos semânticos de representantes, sem afirmar unicidade factual absoluta.

O complete linkage global exige cosseno `>= 0.90` entre **todos os representantes locais** de cada grupo. Uma cadeia A–B e B–C acima do limiar não permite reunir A, B e C se A–C falha. Essa garantia não se estende automaticamente a todos os pares de variantes originais descartadas pela canonicalização local: elas não entram na segunda matriz. A matriz global usa memória quadrática no número de representantes locais. Nenhuma informação produz um catálogo vazio, sem credenciais ou banco; singleton representa a si mesmo. A repetição com os mesmos dados, ordem, vetores persistidos e configuração gera os mesmos dois JSONs.

Para consumir ambos os artefatos em Python, use `group_dataset_with_catalog(...)` ou `run_grouping_with_catalog(...)`; o retorno `GroupingResult` expõe `.dataset` e `.global_catalog`. `build_global_catalog(grouped, vector_store=..., profile=..., threshold=0.90)` aceita um dataset já canonicalizado e verifica os índices antes de acessar o cache. `group_dataset(...)`, `grouping_pipeline(...)` e `SemanticGroupingExperiment` continuam sendo as APIs locais; `run_grouping(...)` mantém o retorno `list[source]`, salvando também o catálogo irmão. Todos os helpers aceitam a estratégia injetada e o limiar configurado.

### Cache e outros provedores

O schema `semantic_grouping` armazena perfis em `embedding_profiles` e vetores em `embeddings`. A identidade inclui provider, model, endpoint, dimensão solicitada, `encoding_format="float"`, `cache_revision` e texto exato. Metadados e dimensão efetiva são verificados na leitura e na escrita. `vector` sem dimensão fixa permite modelos com dimensões diferentes; um perfil mantém uma única dimensão efetiva. Upserts são transacionais e idempotentes; não há índice ANN, pois a comparação exaustiva acontece em Python, por segmento e depois entre os representantes locais.

pgvector armazena componentes em float32. A aplicação persiste e **relê todos os vetores solicitados antes de calcular similaridades**, inclusive na primeira execução. Assim, a repetição com cache usa a mesma representação numérica; valores no limiar não são arredondados para forçar agrupamentos. Vetores nulos, não finitos, malformados ou com dimensão incorreta são rejeitados.

Para um servidor local com `/v1/embeddings`, copie o TOML e altere:

```toml
provider = "openai-compatible"
model = "modelo-local-de-embeddings"
base_url = "http://127.0.0.1:1234/v1"
api_key_env = "LOCAL_AI_API_KEY"
threshold = 0.90
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
    threshold=0.90,
)
```

Um `PostgresVectorStore` construído diretamente deve receber `initialize()` antes do primeiro uso. `grouping_pipeline(...)` compõe somente essa etapa; `SemanticGroupingExperiment(pipeline, repository=...)` permite trocar o pipeline inteiro. `run_grouping(..., config=..., provider=..., vector_store=..., strategy=..., repository=...)` oferece a composição com arquivos. Dependências injetadas não exigem credenciais da OpenAI nem um banco real nos testes offline.

Fontes oficiais: [Embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create), [guia de embeddings](https://developers.openai.com/api/docs/guides/embeddings), [pgvector e Docker](https://github.com/pgvector/pgvector#docker), [instalação do Psycopg](https://www.psycopg.org/install/).

Em 25/10/2026, uma única requisição real de embeddings processou os **3 textos únicos das 6 ocorrências** do dataset atual: vetores de 1536 dimensões persistidos e grupos com 4 e 2 membros. A igualdade profunda dos campos originais foi verificada. Após reiniciar somente o serviço desse experimento, a repetição usou zero chamadas ao provedor e produziu os mesmos bytes JSON. O banco ficou com os 3 vetores reais; as integrações removem somente seus perfis temporários exclusivos.

Na verificação histórica com limiar `0.85`, complete linkage manteve **6 ocorrências em grupos de 4 e 2**: o cosseno entre os textos A e B era `0.8853809907660802`, acima de `0.85`; A–C e B–C eram `0.7220780303650856` e `0.6829254158505015`. Com o novo padrão `0.90`, A e B deixam de atingir o limiar; use uma configuração explícita com `threshold = 0.85` para reproduzir aquele agrupamento.

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
│       ├── application/           # Agrupamento local e catálogo global, com releitura do cache
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
    "data/transcriptions/bebe_real_sources.json",
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

Os testes do agrupamento verificam limiar inclusivo, singletons, duplicatas, todos os pares calculados uma vez, prioridade pelo maior mínimo entre grupos, empates, maximalidade da partição, isolamento por fonte/segmento, preservação dos campos, rejeição de pares abaixo do limiar mesmo quando todas as médias passam, cache/invalidação, float32, batching/índices/erros do provedor e atomicidade do JSON. A canonicalização é comparada ao cálculo direto do centroide normalizado e cobre frequência de ocorrências, magnitudes distintas, empates sob permutações, centroide nulo, índices de proveniência, invalidação após extração e repetição da CLI com bytes idênticos. As integrações usam perfis com UUID e removem exclusivamente seus registros no final; verificam pgvector real, persistência, idempotência, dimensões e a CLI com servidor HTTP local.

O catálogo global acrescenta verificações de duplicatas entre fontes e entre segmentos da mesma fonte, entrada restrita aos representantes locais, ausência de reponderação pelas folhas originais, cadeia não transitiva no limiar inclusivo `0.90`, exclusão dos metadados dos scores, representantes determinísticos, proveniência de todas as ocorrências, vazios e singleton. A persistência cobre os dois arquivos, repetição com cache, destino global padrão e explícito, aliases/capitalização de caminhos e restauração após falha na segunda substituição.

Os testes são offline e não requerem chave de API. Cobrem o exemplo de oito palavras, múltiplas fontes e segmentos, substituição de mapas, reinicialização de `count`, pontuação, caixa, acentos, Unicode, repetições, vazios, preservação dos dados, reuso por identidade em memória e pipeline genérico. A extração cobre concorrência realmente simultânea com conclusão fora de ordem, limite de tarefas, parada independente por segmento, contagem por informação, concatenação de duplicatas, substituição de extrações anteriores e suas contagens, novas chamadas para segmentos previamente cobertos, imutabilidade dos dados, mapas inválidos antes de qualquer chamada, substituição de históricos antigos ou inválidos, mapas vazios e de um token, feedback numérico e ordinal por geração, revisão das ocorrências 17/23, snapshots no limite com chamadas em voo e segmentos não iniciados, reinício após resultados parciais, invalidação de agrupamentos anteriores, validação estrita dos IDs/respostas, separação sistema/dados/schema, escapes, propagação da configuração, respostas recusadas/truncadas/inválidas, erros HTTP sanitizados sem retries, provedores/pipelines/repositórios alternativos e proteção atômica de arquivos. Um servidor HTTP local simulado verifica sucesso completo e salvamento parcial com saída nãozero no fluxo da CLI.
