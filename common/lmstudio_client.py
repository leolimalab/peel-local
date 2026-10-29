"""Minimal client for a locally running LM Studio server
(https://lmstudio.ai, default http://localhost:1234) -- the alternative
to Ollama for Phase 2's LLM calls.

Talks to LM Studio's native REST API (/api/v1) directly via `requests`,
same as ollama_client, rather than its OpenAI-compatible /v1 endpoints:
/api/v1/models reports which models are loaded and with what context
length, and /api/v1/chat takes a per-request "reasoning" switch -- the
equivalent of Ollama's "think" flag. Neither is available through /v1.

Model names arrive here with the "lmstudio:" prefix that tells
common/llm_client.py to route them to LM Studio (e.g.
"lmstudio:qwen/qwen3.5-9b"); the prefix is stripped before anything is
sent to the server.

LM Studio loads a model that isn't loaded yet on its first request (JIT
loading). Loading a model that doesn't fit in free memory can push the
machine into swap, so check_memory compares the model's size against the
memory actually free right now: list_models attaches that check to every
not-yet-loaded model (for the setup screens to warn before a model is
chosen), and generate prints it to the log right before a request that
would trigger a load.

One LM Studio model at a time: when the pipeline uses two different LM
Studio models (e.g. a generation model and a different adversarial
reviewer), generate unloads the one used before it ahead of a request to
the other, so they are used one after the other -- use, unload, load,
use -- and never sit in memory together. Only models this process has
sent requests to are ever unloaded; anything else loaded in LM Studio is
left alone.
"""

import psutil
import requests

DEFAULT_HOST = "http://localhost:1234"
MODEL_PREFIX = "lmstudio:"

# Memory a model is assumed to need once loaded: its weights plus ~20% for
# the KV cache and runtime buffers. A rough heuristic, not LM Studio's own
# estimate -- the real figure depends on the context length it's loaded with.
LOAD_OVERHEAD = 1.2
# Memory that must stay free for the OS and other apps after a load.
HEADROOM_BYTES = 2e9
# Same words-to-tokens factor phase2/condense.py's estimate_num_ctx uses.
TOKENS_PER_WORD = 1.4
# Smallest context a model gets when a request here makes LM Studio load it.
# Unlike Ollama's per-request num_ctx, the context a model is loaded with is
# fixed for every later call until it's unloaded -- adversarial review,
# regenerations, the full-source-text escalation -- and LM Studio doesn't
# refuse a request that overflows it (it keeps generating), so loading with
# just the current call's estimate is too tight (seen: a 4096 load for a
# request that needed 4451 tokens).
JIT_MIN_CONTEXT = 32768

# Keys of every model this process has sent a request to -- the only ones
# _unload_other_used_models will unload.
_used_keys = set()


class LMStudioError(RuntimeError):
    pass


def model_key(model: str) -> str:
    """"lmstudio:qwen/qwen3.5-9b" -> "qwen/qwen3.5-9b" (LM Studio's own id)."""
    return model[len(MODEL_PREFIX):] if model.startswith(MODEL_PREFIX) else model


def _gb(n_bytes) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


def _fetch_models(host: str, timeout: float) -> list:
    """Every downloaded LLM (embedding models excluded) as LM Studio
    describes it in GET /api/v1/models."""
    try:
        resp = requests.get(f"{host}/api/v1/models", timeout=timeout)
    except requests.RequestException as e:
        raise LMStudioError(f"Could not reach LM Studio at {host}: {e}") from e
    if resp.status_code != 200:
        raise LMStudioError(f"LM Studio returned {resp.status_code}: {resp.text[:300]}")
    try:
        models = resp.json()["models"]
    except (ValueError, KeyError) as e:
        raise LMStudioError(f"Unexpected response from LM Studio: {resp.text[:300]}") from e
    return [m for m in models if m.get("type") == "llm"]


def _find_model(models: list, key: str) -> dict:
    for m in models:
        if m.get("key") == key:
            return m
    raise LMStudioError(
        f"LM Studio has no model '{key}'. Downloaded models: "
        + ", ".join(m.get("key", "?") for m in models)
    )


def _loaded_context(entry: dict, instance_id: str | None = None):
    """Context length (tokens) the model is loaded with, or None if it
    isn't loaded. With several loaded instances, the smallest one counts
    (LM Studio picks the instance), unless `instance_id` names one."""
    contexts = [
        inst.get("config", {}).get("context_length")
        for inst in entry.get("loaded_instances") or []
        if instance_id is None or inst.get("id") == instance_id
    ]
    contexts = [c for c in contexts if c]
    return min(contexts) if contexts else None


def check_memory(entry: dict, models: list, unload_first=(), available: float | None = None) -> dict:
    """Whether loading `entry` should fit in the memory free right now --
    plus, for each model key in `unload_first` that is loaded, the memory
    unloading it frees (the other model of a generation/reviewer pair,
    which generate unloads before this one loads). `available` (bytes)
    overrides psutil's free-memory reading. Returns {"ok", "message",
    "required_gb", "available_gb", "headroom_gb"}; a model that is
    already loaded is always ok (nothing new gets loaded)."""
    if _loaded_context(entry) is not None:
        return {"ok": True, "message": f"loaded (context {_loaded_context(entry)} tokens)",
                "required_gb": 0.0, "available_gb": None}

    required = entry.get("size_bytes", 0) * LOAD_OVERHEAD
    freed_models = [m for m in models if m.get("key") in unload_first and m is not entry
                    and _loaded_context(m) is not None]
    available = ((available if available is not None else psutil.virtual_memory().available)
                 + sum(m.get("size_bytes", 0) for m in freed_models))
    ok = required + HEADROOM_BYTES <= available
    message = (
        f"not loaded -- LM Studio will load it on first use: ~{_gb(required)} with overhead "
        f"({_gb(entry.get('size_bytes', 0))} of weights), {_gb(available)} of memory free"
        + (f" once {', '.join(m['key'] for m in freed_models)} is unloaded first" if freed_models else " right now")
    )
    if not ok:
        message += (
            f". That leaves less than {_gb(HEADROOM_BYTES)} for the system, so loading it may push "
            "the machine into swap and slow everything down"
        )
        others = [m for m in models if m is not entry and m not in freed_models and _loaded_context(m) is not None]
        if others:
            message += (
                ". Also loaded in LM Studio (unloading frees memory): "
                + ", ".join(f"{m.get('key')} ({_gb(m.get('size_bytes', 0))})" for m in others)
            )
    return {"ok": ok, "message": message, "required_gb": round(required / 1e9, 1),
            "available_gb": round(available / 1e9, 1), "headroom_gb": round(HEADROOM_BYTES / 1e9, 1)}


def _summary(entry: dict, models: list) -> dict:
    return {
        "name": MODEL_PREFIX + entry.get("key", ""),
        "display_name": entry.get("display_name") or entry.get("key", ""),
        "size_gb": round(entry.get("size_bytes", 0) / 1e9, 1),
        "loaded": _loaded_context(entry) is not None,
        "context_length": _loaded_context(entry),
        "memory": check_memory(entry, models),
    }


def is_available(host: str = DEFAULT_HOST, timeout: float = 2) -> bool:
    try:
        return requests.get(f"{host}/api/v1/models", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def list_models(host: str = DEFAULT_HOST, timeout: float = 5) -> list:
    """Downloaded LLMs as dicts -- "name" (prefixed, ready to pass to
    llm_client.generate), "display_name", "size_gb", "loaded",
    "context_length" (None when not loaded), and "memory" (check_memory's
    result) -- loaded models first."""
    models = _fetch_models(host, timeout)
    summaries = [_summary(m, models) for m in models]
    return sorted(summaries, key=lambda s: not s["loaded"])


def describe_model(model: str, host: str = DEFAULT_HOST, timeout: float = 5, unload_first=()) -> dict:
    """list_models' dict for one model, prefixed or not. `unload_first`
    (models, prefixed or not) is passed to check_memory."""
    models = _fetch_models(host, timeout)
    entry = _find_model(models, model_key(model))
    summary = _summary(entry, models)
    if unload_first:
        summary["memory"] = check_memory(entry, models, unload_first={model_key(m) for m in unload_first})
    return summary


def unload_model(key: str, host: str = DEFAULT_HOST, timeout: float = 60, models: list | None = None) -> list:
    """Unloads every loaded instance of model `key` (prefixed or not) via
    POST /api/v1/models/unload. Returns the unloaded instance ids."""
    key = model_key(key)
    models = models if models is not None else _fetch_models(host, timeout=5)
    instance_ids = [inst.get("id") for inst in _find_model(models, key).get("loaded_instances") or []]
    for instance_id in instance_ids:
        try:
            resp = requests.post(f"{host}/api/v1/models/unload", json={"instance_id": instance_id}, timeout=timeout)
        except requests.RequestException as e:
            raise LMStudioError(f"Could not reach LM Studio at {host}: {e}") from e
        if resp.status_code != 200:
            raise LMStudioError(f"LM Studio could not unload '{instance_id}' ({resp.status_code}): {resp.text[:300]}")
    return instance_ids


def _unload_other_used_models(key: str, models: list, host: str):
    """One LM Studio model at a time: before a request to `key`, unloads
    every other model this process has used that is still loaded. Returns
    (models, available): the model list, refetched if anything was
    unloaded, and the free memory to judge the next load by -- None when
    nothing was unloaded (read it fresh), else the larger of a fresh
    reading and the pre-unload reading plus the unloaded weights, since
    macOS takes a moment to report memory an unload freed (seen: a
    reading taken right after unloading a 6 GB model still showed it as
    used)."""
    others = [m for m in models if m.get("key") in _used_keys and m.get("key") != key
              and _loaded_context(m) is not None]
    if not others:
        return models, None
    before = psutil.virtual_memory().available
    for m in others:
        print(f"[LM Studio] Unloading '{m['key']}' before using '{key}' -- one LM Studio model at a time.")
        unload_model(m["key"], host=host, models=models)
    freed = sum(m.get("size_bytes", 0) for m in others)
    return _fetch_models(host, timeout=5), max(psutil.virtual_memory().available, before + freed)


def generate(model: str, prompt: str, host: str = DEFAULT_HOST, timeout: float = 600,
             options: dict | None = None, think: bool | None = None) -> str:
    """Blocking, non-streaming call to POST /api/v1/chat, with the same
    contract as ollama_client.generate.

    `options` takes Ollama's option names. When the model isn't loaded
    yet, the request carries a "context_length" for LM Studio to load it
    with: twice "num_ctx" (when given), never less than JIT_MIN_CONTEXT,
    capped at the model's own maximum. An already-loaded model keeps the
    context it was loaded with rather than risk LM Studio loading a second
    copy. "repeat_penalty", "temperature", "top_p", "top_k", and "min_p"
    pass through unchanged; anything else is ignored.

    `think` maps to LM Studio's "reasoning" setting ("off"/"on"), sent
    only when the model supports that value -- LM Studio rejects the
    request otherwise.

    The context length is checked on both sides of the call: a prompt
    estimated to be longer than an already-loaded model's context is
    refused before sending, and a response whose input + output tokens
    reach the context length raises instead of being returned, since the
    prompt or the output may have been cut short.

    Before anything else, any other model this process used before that
    is still loaded gets unloaded (_unload_other_used_models), so the
    memory check and the load below see the memory it freed.

    Raises LMStudioError on connection failure, an unknown model, a
    failed unload, a non-200 response, a response with no message text,
    or a context overflow.
    """
    options = options or {}
    key = model_key(model)
    models = _fetch_models(host, timeout=5)
    _find_model(models, key)  # an unknown model fails here, before anything gets unloaded
    models, available = _unload_other_used_models(key, models, host)
    _used_keys.add(key)
    entry = _find_model(models, key)
    loaded_context = _loaded_context(entry)

    payload = {"model": key, "input": prompt, "stream": False, "store": False}
    for name in ("repeat_penalty", "temperature", "top_p", "top_k", "min_p"):
        if name in options:
            payload[name] = options[name]

    wanted_context = max(JIT_MIN_CONTEXT, 2 * options.get("num_ctx", 0))
    if entry.get("max_context_length"):
        wanted_context = min(wanted_context, entry["max_context_length"])
    reload_hint = (f"Reload it with a larger context: `lms unload {key}` then "
                   f"`lms load {key} --context-length {wanted_context}`.")

    if loaded_context is None:
        memory = check_memory(entry, models, available=available)
        print(f"[LM Studio] {'' if memory['ok'] else 'WARNING: '}'{key}' {memory['message']}; "
              f"loading it with a {wanted_context}-token context.")
        payload["context_length"] = wanted_context
    else:
        estimated_tokens = len(prompt.split()) * TOKENS_PER_WORD
        if estimated_tokens >= loaded_context:
            raise LMStudioError(
                f"The prompt (~{int(estimated_tokens)} tokens) does not fit in the {loaded_context}-token "
                f"context '{key}' is loaded with in LM Studio. {reload_hint}"
            )

    if think is not None:
        wanted = "on" if think else "off"
        allowed = ((entry.get("capabilities") or {}).get("reasoning") or {}).get("allowed_options", [])
        if wanted in allowed:
            payload["reasoning"] = wanted

    try:
        resp = requests.post(f"{host}/api/v1/chat", json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise LMStudioError(f"Could not reach LM Studio at {host}: {e}") from e

    if resp.status_code != 200:
        raise LMStudioError(f"LM Studio returned {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        messages = [item["content"] for item in data["output"] if item.get("type") == "message"]
    except (ValueError, KeyError, TypeError) as e:
        raise LMStudioError(f"Unexpected response from LM Studio: {resp.text[:300]}") from e
    if not messages:
        raise LMStudioError(f"LM Studio returned no message text: {resp.text[:300]}")

    if loaded_context is None:
        # JIT-loaded just now: look up the context it actually got.
        loaded_context = _loaded_context(
            _find_model(_fetch_models(host, timeout=5), key), data.get("model_instance_id"),
        )
    stats = data.get("stats") or {}
    used_tokens = stats.get("input_tokens", 0) + stats.get("total_output_tokens", 0)
    if loaded_context and used_tokens >= loaded_context:
        raise LMStudioError(
            f"'{key}' used {used_tokens} tokens, the whole {loaded_context}-token context it is loaded "
            f"with, so the prompt or the output may have been cut short. {reload_hint}"
        )

    return "".join(messages)
