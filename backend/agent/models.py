"""
Model manager with automatic fallback chain.
Primary → Groq → OpenRouter
Never crashes because a model is unavailable.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_model(tier: str = "primary", streaming: bool = True) -> ChatOpenAI:
    """
    tier: "primary" | "fallback1" | "fallback2" | "sub"
    Returns a ChatOpenAI-compatible model.
    All providers expose OpenAI-compatible APIs.
    """
    configs = {
        # Primary: OpenCode Zen — MiniMax M2.5 Free
        "primary": {
            "base_url": os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1"),
            "api_key": os.getenv("OPENCODE_API_KEY", ""),
            "model": os.getenv("PRIMARY_MODEL", "minimax-m2.5-free"),
            "temperature": 0.7,
        },
        # Fallback 1: Groq — openai/gpt-oss-120b
        "fallback1": {
            "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "temperature": 0.7,
        },
        # Fallback 2: Groq — qwen/qwen3-32b
        "fallback2": {
            "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "model": os.getenv("GROQ2_MODEL", "qwen/qwen3-32b"),
            "temperature": 0.6,
        },
        # Sub-agents — lighter, faster model for rewriting / reflection calls
        "sub": {
            "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.4,
        },
    }

    cfg = configs.get(tier, configs["primary"])
    return ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        streaming=streaming,
    )


def get_model_with_fallback(
    tier: str = "primary", streaming: bool = True
) -> ChatOpenAI:
    """
    Tries the requested tier first. Falls back along the chain if the API key
    is missing or if the provider raises an error at init time.
    """
    fallback_chain = ["primary", "fallback1", "fallback2"]

    if tier not in fallback_chain:
        return get_model(tier, streaming)  # e.g. "sub" — no fallback needed

    start = fallback_chain.index(tier)
    for t in fallback_chain[start:]:
        try:
            key_env = {
                "primary": "OPENCODE_API_KEY",
                "fallback1": "GROQ_API_KEY",
                "fallback2": "GROQ_API_KEY",
            }.get(t)
            if key_env and not os.getenv(key_env):
                print(f"[models] {t} key missing — skipping")
                continue
            model = get_model(t, streaming)
            print(f"[models] Using: {t} ({model.model_name})")
            return model
        except Exception as e:
            print(f"[models] {t} init failed: {e} — trying next")
            continue

    raise RuntimeError("All model providers failed. Check your API keys in .env")


def invoke_with_fallback(messages, tier="primary", streaming=False):
    """
    Invoke a model with automatic fallback on 429/500 errors.
    Returns (response, model_name) tuple.
    """
    fallback_chain = ["primary", "fallback1", "fallback2"]
    start = fallback_chain.index(tier) if tier in fallback_chain else 0

    last_error = None
    for t in fallback_chain[start:]:
        try:
            key_env = {
                "primary": "OPENCODE_API_KEY",
                "fallback1": "GROQ_API_KEY",
                "fallback2": "GROQ_API_KEY",
            }.get(t)
            if key_env and not os.getenv(key_env):
                print(f"[models] {t}: no API key — skipping")
                continue
            model = get_model(t, streaming)
            print(f"[models] Trying: {t} ({model.model_name})")
            resp = model.invoke(messages)
            print(f"[models] ✓ {t} ({model.model_name}) succeeded")
            return resp, f"{t}/{model.model_name}"
        except Exception as e:
            err_str = str(e)
            last_error = e
            if "429" in err_str or "rate" in err_str.lower():
                print(f"[models] ✗ {t}: RATE LIMITED — trying next tier")
            else:
                print(f"[models] ✗ {t}: {err_str[:100]} — trying next tier")
            continue

    raise last_error or RuntimeError("All model providers failed")
