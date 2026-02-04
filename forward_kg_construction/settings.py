# forward_kg_construction/settings.py

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Neo4j configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687", description="Neo4j connection URI"
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")

    # API Keys
    groq_api_key: str = Field(..., description="Groq API key for LLM inference")
    ss_api_key: Optional[str] = Field(
        default=None, description="Semantic Scholar API key"
    )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
