"""Routes Phase 2's LLM calls to whichever local server serves the chosen
model: a name starting with "lmstudio:" (e.g. "lmstudio:qwen/qwen3.5-9b")
goes to LM Studio (common/lmstudio_client.py); every other name goes to
Ollama (common/ollama_client.py) exactly as before, so existing Ollama
model names, CLI flags, and decision logs keep working unchanged.

The prefix travels with the model name everywhere it's stored -- CLI
flags, web form fields, decision logs -- so a log always says which
server produced a given text.
"""

from common import lmstudio_client, ollama_client

# Either server's error type, usable directly in an `except` clause.
LLMError = (ollama_client.OllamaError, lmstudio_client.LMStudioError)

# The researcher's choice of server, Ollama first (the default).
SERVERS = ("ollama", "lmstudio")


def is_lmstudio_model(model: str) -> bool:
    return model.startswith(lmstudio_client.MODEL_PREFIX)


def qualify_model(model: str, server: str) -> str:
    """The model name as stored and routed: an LM Studio model gets the
    "lmstudio:" prefix if it doesn't have it yet; an Ollama name is left
    as is. Empty stays empty (the reviewer's "reuse the generation model")."""
    if server == "lmstudio" and model and not is_lmstudio_model(model):
        return lmstudio_client.MODEL_PREFIX + model
    return model


def server_of(model: str) -> str:
    return "lmstudio" if is_lmstudio_model(model) else "ollama"


def generate(model: str, prompt: str, host: str = ollama_client.DEFAULT_HOST, timeout: float = 600,
             options: dict | None = None, think: bool | None = None) -> str:
    """Same signature and contract as ollama_client.generate. `host` is
    Ollama's -- an LM Studio model always goes to
    lmstudio_client.DEFAULT_HOST."""
    if is_lmstudio_model(model):
        return lmstudio_client.generate(model, prompt, timeout=timeout, options=options, think=think)
    return ollama_client.generate(model, prompt, host=host, timeout=timeout, options=options, think=think)
