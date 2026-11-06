"""
Core configuration loaded from environment variables.
Uses pydantic-settings so every value is validated on startup.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- OpenAI / LLM ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"          # cheap & fast; swap to gpt-4o for prod

    # --- App ---
    APP_ENV: str = "development"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # --- CORS ---
    # Comma-separated list of allowed origins
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    # --- Interview defaults ---
    MAX_DEBATE_ROUNDS: int = 2        # how many times Devil's Advocate challenges
    MAX_QUESTIONS_PER_ROUND: int = 5  # max questions the interviewer generates

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
