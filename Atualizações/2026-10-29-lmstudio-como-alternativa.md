# LM Studio como alternativa ao Ollama na Fase 2

**Timestamp:** 2026-10-29T17:40
**Branch:** `feature/lmstudio` (sem commit)

## Contexto

Origem dos pedidos:

- **Passado pelo usuário (2026-10-29):**
  - "inclua a possibilidade de utilizar o LMStudio";
  - deixar o LM Studio carregar modelos sozinho, mas analisar a memória antes e alertar se houver problema;
  - "deve ser opcional ao usuário escolher entre Ollama e LM Studio";
  - criar a branch `feature/lmstudio`;
  - autorizar testes curtos com o Qwen 3.5 9B, que já estava carregado.
- **Proposto pelo Claude:** prefixo `lmstudio:` no nome do modelo, a API nativa `/api/v1` do LM Studio, a margem de 20% + 2 GB na conta de memória e a checagem de contexto antes e depois de cada pedido.

## Diagnóstico

- Todas as chamadas de LLM da Fase 2 (resumo, revisor adversarial e detecção de título/autor/data) passavam por `common/ollama_client.py`, ligado fixo no Ollama.
- O LM Studio oferece duas APIs. A compatível com a OpenAI (`/v1`) não informa se o modelo está carregado nem com que contexto, e não permite desligar o raciocínio por pedido. A nativa (`/api/v1/models` e `/api/v1/chat`) faz as duas coisas: `loaded_instances[].config.context_length` e `reasoning: "off"`.
- O LM Studio carrega um modelo sob demanda no primeiro pedido. Num Mac de 24 GB, um modelo de 16 GB carregado assim leva a máquina ao swap.

## Alterações realizadas

- **Novo `common/lmstudio_client.py`:**
  - `is_available`, `list_models`, `describe_model`, `check_memory` e `generate`, este com o mesmo contrato do `ollama_client.generate`.
  - Memória: o tamanho do modelo × 1,2 mais 2 GB de folga, comparado com a memória livre (`psutil`).
  - Contexto: um prompt maior que o contexto do modelo já carregado é recusado antes do envio; uma resposta que ocupa o contexto inteiro gera erro, porque pode ter sido cortada.
  - `num_ctx` vira `context_length` só para um modelo ainda não carregado, para não arriscar uma segunda cópia na memória.
- **Novo `common/llm_client.py`:**
  - escolhe o servidor pelo nome do modelo: `lmstudio:<id>` vai para o LM Studio e qualquer outro nome vai para o Ollama, como antes;
  - `SERVERS = ("ollama", "lmstudio")`, `qualify_model`, `server_of`, e `LLMError` como tupla dos dois tipos de erro.
- **`phase2/condense.py`:**
  - as três chamadas passam por `llm_client.generate`;
  - novas funções `choose_llm_server` (Enter = Ollama), `print_available_models` (lista só os modelos do servidor escolhido) e `confirm_lmstudio_load` (pergunta s/n quando o modelo pode não caber na memória e registra `lmstudio_memory_warning`);
  - `run_condensation_setup` ganhou o parâmetro `server=`.
- **`run_pipeline.py`:**
  - novas opções `--llm-server {ollama,lmstudio}` e `--model`, este como apelido de `--ollama-model`;
  - no modo por opções, o servidor padrão é o Ollama, ou o LM Studio se o nome já vier com o prefixo;
  - o aviso de memória pergunta mesmo no modo por opções, e "n" encerra antes de carregar qualquer modelo.
- **Interface web:**
  - campo "LLM server" (Ollama por padrão; LM Studio quando o Ollama está desligado e o LM Studio não);
  - os modelos do LM Studio aparecem em dois grupos: "Loaded" primeiro, depois "Not loaded", com o tamanho e o aviso de memória;
  - a detecção automática de título e autor nunca carrega um modelo sozinha (botão novo "detect with AI");
  - a tela pede confirmação ao escolher ou enviar um modelo com aviso de memória, e cada confirmação vai para o log;
  - nova rota `GET /api/lmstudio/models`.
- **Logs de decisão:** novo tipo `llm_server_choice`. O tipo `ollama_model_choice` foi mantido para não quebrar logs antigos; só o texto da pergunta deixou de citar o Ollama.
- **Dependências:** `psutil` entrou no `requirements.txt`.
- **Documentação:** `README.md`, `RUN_PIPELINE_GUIDE.md` (opções, exemplos e solução de problemas) e `FUNCTIONS.md` (seções novas para os dois módulos).

## Resultados e validação

- **Cliente, 17/17:** listagem; geração real com o Qwen (`reasoning: off`, `store: false`, sem `context_length`); prompt grande demais recusado antes do envio; modelo inexistente; carregamento sob demanda e estouro de contexto simulados; nomes sem prefixo continuam indo para o Ollama.
- **Escolha de servidor, 11/11:** Enter escolhe o Ollama; o prefixo é aplicado; aviso de memória com resposta "n" pergunta de novo; servidor inválido pergunta de novo; modo por opções nas variantes Ollama, LM Studio e prefixo implícito; recusa encerra a execução.
- **Teste antigo** `data/peel-local_adversarial_test/.../test_classify.py`: 12/12 com `PYTHONPATH` na raiz. Ele já esperava a variável `PEEL_REPO` e não rodava sem ela, independentemente desta mudança.
- **Navegador, numa cópia descartável de um texto de exemplo, já apagada:**
  - com o Ollama desligado, a tela escolheu o LM Studio e o Qwen carregado;
  - a detecção encontrou o título e a autora corretos;
  - o `lbs-27b` (16,2 GB) mostrou o aviso, a detecção foi pulada e o envio bloqueado;
  - nenhum erro de JavaScript.
- Nenhum modelo foi carregado ou descarregado nos testes. O LM Studio terminou só com o Qwen carregado, no mesmo contexto de 32 000 tokens.
- Depois dos testes, o usuário rodou a Fase 2 pela interface com o LM Studio (Qwen nos dois papéis, 10%) no texto `Boisseau-qwen9b-25cond-10informativesents`. Os registros foram gravados com `llm_server_choice = lmstudio`.

## Correção depois do primeiro uso real (2026-10-29T18:15)

- **Erro relatado pelo usuário:** a regeneração de 5% pela interface falhou com `LMStudioError: 'qwen/qwen3.5-9b' used 4451 tokens, the whole 4096-token context it is loaded with`.
- **Causa:**
  - o Qwen, carregado antes com 32 000 tokens, expirou pelo TTL do LM Studio;
  - na regeneração, o PEEL-Local mandou carregá-lo de novo passando como `context_length` a estimativa justa do Ollama (`num_ctx` = 4096);
  - a estimativa de 1,4 token por palavra ficou abaixo do uso real, de 4 451 tokens.
- **O que o caso confirmou sobre o LM Studio:**
  - ele **respeita** o `context_length` no carregamento sob demanda;
  - ele **não recusa** um pedido que estoura o contexto e segue gerando;
  - a checagem posterior funcionou e descartou o texto possivelmente cortado.
- **Correção:**
  - nova constante `JIT_MIN_CONTEXT = 32768` em `common/lmstudio_client.py`;
  - num carregamento disparado pelo PEEL-Local, o contexto pedido é `max(32768, 2 × num_ctx)`, limitado ao máximo do modelo, e passa a valer também para as chamadas sem `num_ctx` (detecção de metadados e revisor);
  - as mensagens de erro trazem os comandos `lms unload`/`lms load --context-length` com o valor sugerido.
- **Testes:** 4 casos novos de carregamento simulado (19/20 no total). A única falha vem do ambiente: um teste antigo espera o Qwen carregado com 32 000 tokens, e agora ele está com 4 096. A escolha de servidor seguiu em 11/11.
- **Recarregamento:** com o aval do usuário, o Qwen foi recarregado com 32 768 tokens (`lms unload qwen/qwen3.5-9b` e depois `lms load qwen/qwen3.5-9b --context-length 32768`; carregou em 5,8 s, 5,57 GiB). Carregado assim, pelo `lms load`, ele fica sem TTL e só sai da memória se for descarregado à mão. O servidor web foi reiniciado com a correção, e a sessão, que já estava parada no erro, voltou ao estado ocioso.

## Um modelo do LM Studio por vez (2026-10-29T18:35)

- **Pedido do usuário:** "quando tiver que utilizar 2 modelos no LMStudio, primeiro use 1, descarregue e depois carregue o outro e utilize", por exemplo na análise adversarial.
- **Alteração em `common/lmstudio_client.py`:**
  - `_used_keys` guarda os modelos que o processo já usou;
  - antes de cada pedido, `_unload_other_used_models` descarrega os outros modelos dessa lista que estejam carregados, usando a nova `unload_model`, que chama `POST /api/v1/models/unload`;
  - um modelo que o PEEL-Local não usou não é tocado;
  - um nome de modelo inexistente falha antes de qualquer descarregamento, e uma falha ao descarregar vira erro, sem carregar o outro por cima.
- **Aviso de memória da configuração:** o do revisor passou a contar como liberada a memória do modelo do resumo, que sai antes:
  - `check_memory(..., unload_first=)` e `describe_model(..., unload_first=)` recebem esse modelo;
  - `confirm_lmstudio_load(..., unload_first=)` repassa o modelo, no terminal e no `run_pipeline.py`;
  - na tela web, `memoryWarning`/`confirmModelLoad` recebem `unloadFirst`, e o servidor passou a informar `headroom_gb`.
- **Testes:** 13/13 no revezamento simulado (A → B → A sem os dois carregados juntos, modelo alheio preservado, mesmo modelo sem troca, nome inexistente, falha ao descarregar, crédito de memória). Cliente 20/20 e escolha de servidor 11/11.
- **Documentação:** `README.md`, `RUN_PIPELINE_GUIDE.md` e `FUNCTIONS.md` atualizados.
- **Servidor web:** reiniciado com o aval do usuário, que estava com uma sessão aberta na Fase 2 e a retoma depois.
- **Teste real, com o aval do usuário (2026-10-29T18:40):**
  - sequência Qwen 3.5 9B → Gemma 4 12B → Qwen pelo `llm_client.generate`, com pedidos curtos;
  - o Qwen foi descarregado antes do Gemma (carregado em 8,7 s) e o Gemma antes de o Qwen voltar (7,0 s), todos com 32 768 tokens;
  - os dois nunca ficaram juntos na memória e o swap ficou em 0,4 GB o tempo todo; resultado PASS;
  - no fim, só o Qwen está carregado, pelo PEEL-Local, com o TTL de 20 min do LM Studio.
- **Defeito revelado pelo teste:** os avisos de memória impressos logo depois de cada descarregamento eram alarmes falsos. O macOS ainda não tinha devolvido a memória à leitura do `psutil` (6,8 GB "livres" para um Gemma de 8,1 GB, que carregou sem swap).
  - **Correção:** `_unload_other_used_models` devolve a memória livre a usar, o maior valor entre a leitura atual e a leitura anterior ao descarregamento somada ao tamanho dos modelos descarregados; `check_memory(..., available=)` aceita essa leitura.
  - **Teste novo:** com a leitura parada, o aviso não dispara (14/14).
- **Observação:** o `memory_pressure` (`kern.memorystatus_level`) não serve para essa conta, porque trata os pesos do modelo carregado como cache de arquivo reaproveitável e marcava 74% livre com o Qwen na memória. O `psutil` continua sendo a fonte, mais conservadora.

## Fork no GitHub (2026-10-29)

- A pedido do usuário, foi criado o fork `leolimalab/peel-local` (https://github.com/leolimalab/peel-local), público como o original `PUC-Behring-AI/peel-local`, com `gh repo fork --clone=false --remote=false`.
- No primeiro momento, o fork tinha só o que já estava no GitHub (`main` do original).
- Depois, a pedido do usuário ("adicione o fork como remote e envie a branch para lá. Não abra PR para o repo original"):
  - o fork virou o remote `fork` (`https://github.com/leolimalab/peel-local.git`); o `origin` continua sendo o `PUC-Behring-AI`;
  - a `feature/lmstudio` foi enviada com `git push -u fork feature/lmstudio`, em dois commits: `bf4c895` (LM Studio) e `4649483` (`Atualizações/` e `.gitignore`);
  - **nenhum PR foi aberto**.
- **Repositório local desvinculado do original**, a pedido do usuário ("desvincule com PUC-Behring-AI, coloque o origin para o meu repo tb"):
  - `origin` passou a ser `https://github.com/leolimalab/peel-local.git`, e o remote `fork` foi removido por ficar duplicado;
  - `main` e `feature/lmstudio` acompanham `origin/main` e `origin/feature/lmstudio`;
  - nenhum remote ou configuração do git aponta mais para o `PUC-Behring-AI`.
  - No GitHub, o `leolimalab/peel-local` continua marcado como fork de `PUC-Behring-AI/peel-local`. Desfazer esse vínculo de rede é coisa do GitHub (Suporte) e não foi feito.
- **Ficou fora dos commits:** as alterações em `data/` feitas pelas execuções de teste do usuário nos textos de exemplo `Boisseau-qwen9b-25cond-10informativesents` e `-20informativesents`. Elas sobrescrevem saídas de exemplo versionadas (o `informative_sentences.json` do `-20informativesents` perdeu 632 linhas), então continuam só na máquina local, sem commit e sem descarte.

## Issues remanescentes

- **Não testados de verdade:** o carregamento sob demanda (envio de `context_length` e leitura do contexto obtido) e o comportamento do LM Studio ao estourar o contexto (corta ou dá erro). Ambos foram só simulados, porque testá-los exigiria carregar outro modelo.
- **Conta de memória aproximada:** ela não considera o tamanho real do KV cache para o contexto pedido.
- **Ollama sem checagem de memória:** a detecção automática ao abrir a tela da Fase 2 carrega o primeiro modelo do Ollama (`clef-flash`, 12,4 GB nesta máquina). É comportamento anterior a esta mudança.
- **Dados de exemplo alterados:** a execução do usuário grava em `data/Boisseau-qwen9b-25cond-10informativesents/`, que é versionado, e aparece no `git status`.

## Arquivos tocados

- `common/lmstudio_client.py` (novo)
- `common/llm_client.py` (novo)
- `phase2/condense.py`
- `run_pipeline.py`
- `webapp/app.py`, `webapp/pipeline_session.py`, `webapp/static/app.js`, `webapp/static/index.html`
- `requirements.txt`
- `README.md`, `RUN_PIPELINE_GUIDE.md`, `FUNCTIONS.md`
- `Atualizações/README.md`
