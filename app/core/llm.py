"""
Centralized LLM Factory
Supports Grok (xAI) as primary provider, with automatic fallback / support for OpenAI.
"""

import logging
from langchain_openai import ChatOpenAI
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def build_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Construct a ChatOpenAI client configured for either Grok (xAI) or OpenAI.

    Detection logic:
    1. If GROK_API_KEY or XAI_API_KEY is provided, use Grok.
    2. If OPENAI_API_KEY starts with 'xai-', it's a Grok key entered in the OpenAI field -> use Grok!
    3. If LLM_PROVIDER is 'grok', use Grok.
    4. Otherwise, use OpenAI.
    """
    settings = get_settings()

    grok_key = (settings.GROK_API_KEY.strip() or settings.XAI_API_KEY.strip())
    openai_key = settings.OPENAI_API_KEY.strip()

    # Determine if Grok should be active
    is_grok = (
        bool(grok_key)
        or openai_key.startswith("xai-")
        or settings.LLM_PROVIDER.lower() in ("grok", "xai")
    )

    if is_grok:
        api_key = grok_key or openai_key
        base_url = settings.GROK_BASE_URL.strip() or "https://api.x.ai/v1"
        model = settings.GROK_MODEL.strip() or "grok-2"
        logger.info(f"[LLM Factory] Initializing Grok (xAI) with model={model}")
    else:
        api_key = openai_key
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
