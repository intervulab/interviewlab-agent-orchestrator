"""
Centralized LLM Factory
Supports:
1. Groq (Fast LPU inference: gsk_... keys at https://api.groq.com/openai/v1)
2. Grok / xAI (xai-... keys at https://api.x.ai/v1)
3. OpenAI (sk-... keys at https://api.openai.com/v1)
"""

import logging
from langchain_openai import ChatOpenAI
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def build_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Construct a ChatOpenAI client with intelligent auto-detection:
    - If key starts with 'gsk_', it's Groq -> route to https://api.groq.com/openai/v1 with openai/gpt-oss-120b
    - If key starts with 'xai_', it's xAI Grok -> route to https://api.x.ai/v1 with grok-2
    - If key starts with 'sk-', it's OpenAI -> route to https://api.openai.com/v1 with gpt-4o-mini
    """
    settings = get_settings()

    # Look for any key provided in env
    api_key = (
        getattr(settings, "GROQ_API_KEY", "").strip()
        or settings.GROK_API_KEY.strip()
        or settings.XAI_API_KEY.strip()
        or settings.OPENAI_API_KEY.strip()
    )

    # 1. Groq detection (gsk_ prefix or provider == 'groq')
    if api_key.startswith("gsk_") or settings.LLM_PROVIDER.lower() == "groq":
        base_url = "https://api.groq.com/openai/v1"
        # Preferred models on Groq: openai/gpt-oss-120b or qwen/qwen3.8-27b
        configured_model = getattr(settings, "GROQ_MODEL", "") or settings.GROK_MODEL
        if configured_model and configured_model not in ("grok-2", "grok-beta", "gpt-4o-mini"):
            model = configured_model
        else:
            model = "openai/gpt-oss-120b"
        logger.info(f"[LLM Factory] Auto-detected Groq LPU provider with model={model}")

    # 2. Grok / xAI detection (xai- prefix or provider == 'grok'/'xai')
    elif api_key.startswith("xai-") or settings.LLM_PROVIDER.lower() in ("grok", "xai"):
        base_url = settings.GROK_BASE_URL.strip() or "https://api.x.ai/v1"
        model = settings.GROK_MODEL.strip() or "grok-2"
        logger.info(f"[LLM Factory] Initializing Grok (xAI) with model={model}")

    # 3. Standard OpenAI
    else:
        base_url = settings.OPENAI_BASE_URL.strip() or "https://api.openai.com/v1"
        model = settings.OPENAI_MODEL.strip() or "gpt-4o-mini"
        logger.info(f"[LLM Factory] Initializing OpenAI with model={model}")

    effective_key = api_key if api_key else "placeholder_key"

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=effective_key,
        base_url=base_url,
    )
