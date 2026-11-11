"""
Core configuration loaded from environment variables.
Supports Groq (LPU), Grok (xAI), and OpenAI providers seamlessly.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- LLM Provider Selection ---
    # Options: "groq", "grok", or "openai"
    LLM_PROVIDER: str = "groq"

    # --- Groq (console.groq.com) Configuration ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # --- Grok / xAI Configuration ---
    GROK_API_KEY: str = ""
    XAI_API_KEY: str = ""                       # Standard alias used by xAI
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-2"                  # grok-2, grok-beta, etc.

    # --- OpenAI Configuration (Alternative / Fallback) ---
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # --- App ---
    APP_ENV: str = "development"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # --- CORS ---
    # Comma-separated list of allowed origins
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://localhost:4000"

    # --- Interview defaults ---
    MAX_DEBATE_ROUNDS: int = 2        # how many times Devil's Advocate challenges
    MAX_QUESTIONS_PER_ROUND: int = 5  # max questions the interviewer generates

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
