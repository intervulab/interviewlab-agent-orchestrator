"""
Core configuration loaded from environment variables.
Powered exclusively by Groq (https://console.groq.com) for high-speed inference.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Groq API Configuration ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # --- App Configuration ---
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
