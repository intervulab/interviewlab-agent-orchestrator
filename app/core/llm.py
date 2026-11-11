"""
Centralized LLM Factory
Powered exclusively by Groq LPU inference (https://api.groq.com/openai/v1).
"""

import logging
from langchain_openai import ChatOpenAI
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def build_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Construct a ChatOpenAI client connecting directly to Groq's API.
    """
    settings = get_settings()

    api_key = settings.GROQ_API_KEY.strip()
    if not api_key:
        api_key = "placeholder_groq_key"
        logger.warning("[LLM Factory] No GROQ_API_KEY found in environment.")

    return ChatOpenAI(
        model=settings.GROQ_MODEL,
        temperature=temperature,
        api_key=api_key,
        base_url=settings.GROQ_BASE_URL,
    )
