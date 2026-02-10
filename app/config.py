"""
Application Settings — All via environment variables with sensible defaults.
"""
import os
from typing import Optional


class Settings:
    # ── Server ──
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Database ──
    # Default: SQLite (zero config). Production: set DATABASE_URL env var.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ml_agent.db")

    # ── CORS ──
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://localhost:3000,*")

    # ── LLM Reasoner (optional — engine works in rules_only mode without this) ──
    AGENT_LLM_PROVIDER: str = os.getenv("AGENT_LLM_PROVIDER", "rules_only")
    AGENT_LLM_API_KEY: str = os.getenv("AGENT_LLM_API_KEY", "")
    AGENT_LLM_MODEL: str = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")
    AGENT_LLM_BASE_URL: Optional[str] = os.getenv("AGENT_LLM_BASE_URL")

    # ── Feature Flags ──
    ENABLE_DOMAIN_PROFILES: bool = os.getenv("ENABLE_DOMAIN_PROFILES", "true").lower() == "true"
    ENABLE_SEMANTIC_INTENT: bool = os.getenv("ENABLE_SEMANTIC_INTENT", "true").lower() == "true"
    ENABLE_STATISTICAL_TESTS: bool = os.getenv("ENABLE_STATISTICAL_TESTS", "true").lower() == "true"
    ENABLE_FEEDBACK_TRACKER: bool = os.getenv("ENABLE_FEEDBACK_TRACKER", "true").lower() == "true"
    ENABLE_DOMAIN_COSTS: bool = os.getenv("ENABLE_DOMAIN_COSTS", "true").lower() == "true"


settings = Settings()
