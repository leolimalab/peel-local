# Ambiente local no Mac e interface web no ar

**Timestamp:** 2026-10-29T17:14

## Contexto

O usuário pediu para "iniciar o projeto" e, em seguida, uma explicação simplificada do que ele faz. Esta sessão montou o ambiente Python neste Mac (Apple M4 Pro, 24 GB, sem GPU NVIDIA) e subiu a interface web.

## Diagnóstico

- `requirements.txt` puxa o `torch` de um índice da NVIDIA com CUDA 13.0 (`--extra-index-url https://download.pytorch.org/whl/cu130`). No Mac não existe CUDA.
- `webapp/app.py` fixa a porta 5000, que no macOS já é usada pelo Receptor AirPlay (processo `ControlCenter`).
- O download do WordNet pelo NLTK falhou com `CERTIFICATE_VERIFY_FAILED`: o Python 3.12 do python.org não vem com certificados configurados.
- A escolha de dispositivo (`run_pipeline.py:199`, `webapp/pipeline_session.py:354`) só considera `cuda` ou `cpu`. No Mac a Fase 1 roda na CPU, embora `torch.backends.mps.is_available()` seja `True`.
- O Ollama está instalado (`clef-flash`, 12,4 GB; `laya`, 0,8 GB), mas o servidor estava desligado.

## Alterações realizadas

- Criado `.venv/` com Python 3.12.8 via `uv`. As dependências foram instaladas a partir de uma cópia do `requirements.txt` sem a linha `--extra-index-url`; o arquivo original não foi alterado.
- Instalados o modelo `en_core_web_sm` 3.8.0 do spaCy e os dados `wordnet` e `omw-1.4` do NLTK (baixados com `SSL_CERT_FILE` apontando para o `certifi`).
- `.gitignore`: acrescentado `.venv/`.
- Nenhuma alteração em código.

Comando usado para subir a interface web na porta 5001 sem mexer no código:

```bash
SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi;print(certifi.where())')" \
  .venv/bin/python -c "from webapp.app import app; app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)"
```

## Resultados e validação

- Versões: torch 2.15.0, transformers 5.22.0, sentence-transformers 6.2.0, spaCy 3.8.18, Flask 3.1.3.
- `GET /` → 200 (`<title>PEEL-Local</title>`); `GET /api/status` → `"status":"idle"`.
- `GET /api/corpora` lista os quatro textos de exemplo em `data/`, todos aptos a retomar nas fases 1, 2 e 3.
- `GET /api/ollama/models` → primeiro `{"available":false}`, com o Ollama desligado. Com o aval do usuário, `ollama serve` (versão 0.40.3) foi ligado e a rota passou a listar `clef-flash:latest` e `laya:latest`. `ollama ps` confirmou que nenhum modelo foi carregado.
- O pipeline completo **não** foi executado nesta sessão.

## Issues remanescentes

- Porta 5000 fixa no `app.py`: em Macs com o AirPlay ligado, `python webapp/app.py` falha ou colide. Uma opção é aceitar a porta por variável de ambiente.
- Fase 1 sem aceleração no Mac: incluir `mps` na escolha de dispositivo poderia acelerar o GlossBERT. É mudança de código e ainda precisa de aval.
- O README orienta `pip install -r requirements.txt` sem avisar que, no Mac, a linha do índice CUDA deve sair.
- O primeiro uso da Fase 1 baixa o GlossBERT (~420 MB) do Hugging Face.
- A Fase 2 depende do Ollama. O `clef-flash` (12,4 GB) é pesado para esta máquina de 24 GB se rodar junto com o LM Studio.

## Arquivos tocados

- `.gitignore`
- `Atualizações/2026-10-29-ambiente-local-mac.md` (novo)
- `Atualizações/README.md` (novo)
- `.venv/` (novo, ignorado pelo git)
