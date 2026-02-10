"""
ML Expert Agent Engine — FastAPI Server (Port 8001)
======================================================
Complete ML guidance system: 200+ rules, 12 domain profiles, semantic Q&A,
real statistical tests, closed-loop feedback, domain-specific business impact.

Run:
  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
  # or
  python main.py

Merge into existing app: See MERGE_GUIDE.md
"""

import logging
import os
from contextlib import asynccontextmanager

# Load .env file BEFORE anything reads os.getenv()
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars only

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ──
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml_agent")


# ── Lifespan: create tables + warm up ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.database import engine, Base

    # Import ALL models so they register with Base.metadata
    import app.models.platform       # noqa: F401
    import app.models.agent_memory   # noqa: F401

    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")

    # Adapt models to match REAL database schema
    # (removes declared columns that don't exist in the actual DB)
    try:
        from app.models.platform import adapt_models_to_schema
        adapt_models_to_schema(engine)
    except Exception as e:
        logger.warning(f"Schema adaptation failed (non-fatal): {e}")

    # Warm up engine
    try:
        from app.core.agent.rule_engine import RuleEngine
        from app.core.agent.semantic_intent import SemanticIntentClassifier
        from app.core.agent.domain_profiles import DomainProfileManager
        _re = RuleEngine()
        _sc = SemanticIntentClassifier()
        _dp = DomainProfileManager.list_domains()
        logger.info(
            f"Engine warmed up: RuleEngine ready, "
            f"semantic classifier ready, {len(_dp)} domain profiles"
        )
    except Exception as e:
        logger.warning(f"Engine warmup partial: {e}")

    yield
    logger.info("Shutting down ML Agent Engine")


# ── Create FastAPI app ──
app = FastAPI(
    title="ML Expert Agent Engine",
    description=(
        "Context-aware ML guidance system: 200+ deterministic rules, "
        "12 domain-adaptive profiles, semantic intent classification, "
        "real statistical tests (PSI/KS/chi²/Cohen's d), "
        "closed-loop feedback tracking, domain-specific business impact. "
        "23 API endpoints."
    ),
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow your existing app on port 8000 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8000,http://localhost:3000,*"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount all API routes ──
from app.api.router import api_router  # noqa: E402
app.include_router(api_router, prefix="/api/v1")


# ── Root ──
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "ML Expert Agent Engine",
        "version": "4.0.0",
        "docs": "/docs",
        "endpoints": {
            "core": "/api/v1/agent/ (7 endpoints)",
            "enhanced": "/api/v1/agent/enhanced/ (16 endpoints)",
        },
        "health": "/api/v1/agent/health",
    }


# ── Direct run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level="info",
    )