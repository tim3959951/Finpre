"""Pick an LLM per agent from settings (global provider/model + optional per-agent overrides)."""
from __future__ import annotations

import logging
import os

from ..config import Settings, get_settings
from .base import LLMClient, NullLLM

log = logging.getLogger(__name__)
PROVIDERS = ("anthropic", "openai", "ollama", "none")


def make_llm(provider: str, model: str | None = None, settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    provider = (provider or "none").lower()
    model = model or s.get_path(f"llm.defaults.{provider}", "")
    temp = float(s.get_path("llm.temperature", 0.2))
    max_tokens = int(s.get_path("llm.max_tokens", 2000))
    try:
        if provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return NullLLM("ANTHROPIC_API_KEY 未設定")
            from .providers import AnthropicClient
            try:
                return AnthropicClient(model or None, temp, max_tokens)
            except Exception as e:
                return NullLLM(f"Anthropic 初始化失敗: {e}")
        if provider == "openai":
            if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")):
                return NullLLM("OPENAI_API_KEY / OPENAI_BASE_URL 未設定")
            from .providers import OpenAICompatClient
            return OpenAICompatClient(model, temp, max_tokens)
        if provider == "ollama":
            from .providers import OllamaClient
            client = OllamaClient(model, temp, max_tokens, think=bool(s.get_path("llm.ollama.think", False)),
                                  num_ctx=int(s.get_path("llm.ollama.num_ctx", 16384)))
            if client.enabled:
                return client
            installed = client.installed_models()
            if not installed:
                return NullLLM(f"Ollama 未啟動或沒有任何模型（ollama serve；ollama pull {model}）")
            log.warning("Ollama model %s not installed, falling back to %s", model, installed[0])
            client.model = installed[0]
            return client
    except ImportError as e:
        return NullLLM(f"缺少套件: {e}")
    return NullLLM("provider=none (rule-only mode)")


def get_llm(agent: str | None = None, settings: Settings | None = None,
            provider: str | None = None, model: str | None = None) -> LLMClient:
    s = settings or get_settings()
    override = (s.get_path(f"llm.per_agent.{agent}") or {}) if agent else {}
    prov = provider or override.get("provider") or s.get_path("llm.provider", "none")
    mdl = model or override.get("model") or (s.get_path("llm.model") or None)
    if provider and not model:
        mdl = None  # switching provider at runtime -> use that provider's default model
    return make_llm(prov, mdl, s)
