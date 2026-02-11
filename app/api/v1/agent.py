"""
ML Expert Agent — API Endpoints (Production)
===============================================
FastAPI router providing ML expert guidance through the full Orchestrator pipeline.

Endpoints:
  POST /insights       — Screen-aware auto-insights (full pipeline)
  POST /ask            — Context-aware question answering
  POST /validate       — Pre-action readiness gates
  POST /recommend      — Standalone recommendations
  POST /feedback       — Thumbs up/down on insights (persisted)
  GET  /readiness      — Data readiness report
  GET  /health         — Agent component health check

Integration (in main.py):
  from app.api.agent import router as agent_router
  app.include_router(agent_router, prefix="/api/v1/agent", tags=["ML Expert Agent"])
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query, HTTPException

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════
# RATE LIMITER (in-memory, per-user, protects /ask LLM cost)
# ═══════════════════════════════════════════════════════════════

class _AskRateLimiter:
    """Simple per-user rate limiter for /ask endpoint (LLM calls cost ~$0.02 each)."""

    def __init__(self, max_calls: int = 20, window_seconds: int = 3600):
        self._calls: Dict[str, List[float]] = defaultdict(list)
        self._max = max_calls
        self._window = window_seconds

    def check(self, user_id: str) -> bool:
        """Return True if allowed, False if rate-limited."""
        now = time.time()
        cutoff = now - self._window
        # Prune old entries
        self._calls[user_id] = [t for t in self._calls[user_id] if t > cutoff]
        if len(self._calls[user_id]) >= self._max:
            return False
        self._calls[user_id].append(now)
        return True

    def remaining(self, user_id: str) -> int:
        now = time.time()
        cutoff = now - self._window
        recent = [t for t in self._calls[user_id] if t > cutoff]
        return max(0, self._max - len(recent))

    def reset_seconds(self, user_id: str) -> int:
        if not self._calls[user_id]:
            return 0
        oldest = min(t for t in self._calls[user_id] if t > time.time() - self._window)
        return max(0, int(oldest + self._window - time.time()))


_ask_limiter = _AskRateLimiter(max_calls=20, window_seconds=3600)  # 20 calls/hour


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR FACTORY (lazy singleton)
# ═══════════════════════════════════════════════════════════════

_orchestrator_cache = {}


def _get_orchestrator(db):
    """
    Create or retrieve the AgentOrchestrator.
    Uses a simple cache keyed on id(db) to avoid re-creating per request.
    The orchestrator wires ALL components: Context → Stats → Patterns →
    Rules → Recommendations → Business Impact → LLM → Memory.
    """
    from app.core.agent.orchestrator import AgentOrchestrator
    from app.core.agent.memory_store import MemoryStore

    # Create fresh orchestrator per request (db session is request-scoped)
    try:
        from app.models.agent_memory import AgentMemory
        memory = MemoryStore(db_session=db, memory_model=AgentMemory)
    except Exception:
        memory = None

    return AgentOrchestrator(db_session=db, memory_store=memory)


# ═══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════

class InsightRequest(BaseModel):
    screen: str = Field(..., description="Current screen: dashboard|data|eda|mlflow|training|evaluation|registry|deployment|predictions|monitoring|ai_insights")
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    user_id: Optional[str] = Field(default="anonymous", description="User ID for memory/personalization")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Frontend state (metrics, selections, config)")
    use_llm: Optional[str] = Field(default="auto", description="LLM usage: 'auto' (smart gating, saves cost), 'always' (force LLM), 'never' (rules only)")


class AskRequest(BaseModel):
    screen: str
    question: str = Field(..., min_length=1, max_length=2000)
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    extra: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous Q&A pairs for multi-turn context"
    )
    use_llm: Optional[str] = Field(default="auto", description="LLM usage: 'auto' (smart gating), 'always', 'never' (rules only, saves cost)")


class ValidateRequest(BaseModel):
    action: str = Field(..., description="Action: start_training|deploy_model|promote_to_prod|retrain_model|change_threshold")
    screen: str
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class RecommendRequest(BaseModel):
    screen: str
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    insight_rule_id: str
    helpful: bool
    user_id: Optional[str] = "anonymous"
    screen: Optional[str] = None
    comment: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════

class InsightItem(BaseModel):
    severity: str
    category: str
    title: str
    message: str
    action: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = None
    rule_id: Optional[str] = None
    tags: Optional[List[str]] = None


class InsightResponse(BaseModel):
    insights: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]] = []
    recommendations: Dict[str, Any] = {}
    business_impact: Dict[str, Any] = {}
    advice: Optional[str] = None
    source: str = "rules_only"
    context_summary: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    timing: Dict[str, float] = {}
    memory_applied: List[Dict[str, Any]] = []
    suggested_questions: List[str] = []


class AskResponse(BaseModel):
    answer: str
    supporting_insights: List[Dict[str, Any]] = []
    related_recommendations: Dict[str, Any] = {}
    source: str = "qa_engine"
    confidence: float = 0.0
    follow_up_questions: List[str] = []
    timing: Dict[str, float] = {}


class ValidateResponse(BaseModel):
    can_proceed: bool
    verdict: str = "ready"
    blockers: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    checks_passed: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    risk_score: float = 0.0
    timing: Dict[str, float] = {}


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    version: str
    uptime_seconds: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

_start_time = time.time()


@router.post("/insights", response_model=InsightResponse)
async def get_insights(request: InsightRequest, db=Depends(get_db)):
    """
    Get auto-insights for the current screen.

    Full pipeline: Context → Statistical Analysis → Pattern Detection →
    Rule Engine (200+ rules) → Recommendations → Business Impact →
    LLM Synthesis (optional) → Memory Adjustment → Response
    """
    try:
        orchestrator = _get_orchestrator(db)
        bundle = await orchestrator.get_insights(
            screen=request.screen,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            model_id=request.model_id,
            user_id=request.user_id,
            extra=request.extra,
            use_llm=request.use_llm or "auto",
        )
        return InsightResponse(**bundle.to_dict())

    except Exception as e:
        logger.error(f"Agent insights error: {e}", exc_info=True)
        return InsightResponse(
            insights=[], source="error",
            timing={"error": str(e)},
        )


@router.post("/ask", response_model=AskResponse)
async def ask_expert(request: AskRequest, db=Depends(get_db)):
    """
    Ask the ML Expert a question.

    Pipeline: Context → Statistical Analysis → Rule Engine →
    Q&A Engine (deterministic, 18+ intent categories) →
    LLM Enhancement (if confidence < 0.7) → Memory → Response

    Rate-limited: 20 calls/hour per user (each LLM call costs ~$0.02).
    """
    # ── Rate limiting (protect against cost overrun) ──
    uid = request.user_id or "anonymous"
    use_llm = request.use_llm or "auto"
    if use_llm in ("always", "auto"):
        if not _ask_limiter.check(uid):
            remaining_secs = _ask_limiter.reset_seconds(uid)
            logger.warning(f"Rate limit hit for user '{uid}' on /ask")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": f"AI analysis rate limit reached (20/hour). Try again in {remaining_secs // 60} minutes.",
                    "remaining_calls": 0,
                    "reset_seconds": remaining_secs,
                },
            )

    try:
        orchestrator = _get_orchestrator(db)
        bundle = await orchestrator.ask(
            screen=request.screen,
            question=request.question,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            model_id=request.model_id,
            user_id=request.user_id,
            extra=request.extra,
            conversation_history=request.conversation_history,
            use_llm=use_llm,
        )
        response = AskResponse(**bundle.to_dict())
        # Add rate limit headers info to timing
        return response

    except HTTPException:
        raise  # Re-raise rate limit errors
    except Exception as e:
        logger.error(f"Agent ask error: {e}", exc_info=True)
        return AskResponse(
            answer="I encountered an error processing your question. Please try again.",
            source="error",
        )


@router.post("/validate", response_model=ValidateResponse)
async def validate_action(request: ValidateRequest, db=Depends(get_db)):
    """
    Validate readiness before a critical action (training, deployment, etc.).

    Pipeline: Context → Analysis → Validation Engine (weighted checklists) →
    Rule Engine (supporting evidence) → Response
    """
    try:
        orchestrator = _get_orchestrator(db)
        bundle = await orchestrator.validate(
            action=request.action,
            screen=request.screen,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            model_id=request.model_id,
            extra=request.extra,
        )
        return ValidateResponse(**bundle.to_dict())

    except Exception as e:
        logger.error(f"Agent validate error: {e}", exc_info=True)
        return ValidateResponse(
            can_proceed=True,
            verdict="error",
            recommendations=[f"Validation failed: {str(e)}"],
        )


@router.post("/recommend")
async def get_recommendations(request: RecommendRequest, db=Depends(get_db)):
    """
    Get standalone recommendations (algorithms, features, encoding, scaling, etc.).

    Pipeline: Context → Analysis → Patterns → Recommendation Engine → Business Impact
    """
    try:
        orchestrator = _get_orchestrator(db)
        result = await orchestrator.get_recommendations(
            screen=request.screen,
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            model_id=request.model_id,
            extra=request.extra,
        )
        return result

    except Exception as e:
        logger.error(f"Agent recommend error: {e}", exc_info=True)
        return {"error": str(e)}


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db=Depends(get_db)):
    """
    Record user feedback on an insight.
    Persists to MemoryStore for future personalization.
    """
    try:
        orchestrator = _get_orchestrator(db)
        orchestrator.record_feedback(
            user_id=request.user_id or "anonymous",
            rule_id=request.insight_rule_id,
            helpful=request.helpful,
        )
        return {
            "status": "ok",
            "message": f"Feedback recorded: {'helpful' if request.helpful else 'not helpful'}",
            "rule_id": request.insight_rule_id,
        }
    except Exception as e:
        logger.warning(f"Feedback recording failed: {e}")
        return {"status": "ok", "message": "Feedback noted"}


@router.get("/readiness")
async def get_data_readiness(
        dataset_id: str = Query(..., description="Dataset ID"),
        project_id: Optional[str] = Query(None),
        db=Depends(get_db),
):
    """
    Get comprehensive data readiness assessment.
    Returns readiness score, component scores, and improvement suggestions.
    """
    try:
        orchestrator = _get_orchestrator(db)
        result = await orchestrator.get_data_readiness(
            dataset_id=dataset_id,
            project_id=project_id,
        )
        return result
    except Exception as e:
        logger.error(f"Data readiness error: {e}", exc_info=True)
        return {"error": str(e)}


@router.get("/health", response_model=HealthResponse)
async def agent_health():
    """Agent health check — reports status of all components."""
    from app.core.agent.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner()

    components = {
        "context_compiler": "active",
        "rule_engine": "active",
        "statistical_analyzer": "active",
        "pattern_detector": "active",
        "recommendation_engine": "active",
        "business_impact": "active",
        "validation_engine": "active",
        "qa_engine": "active",
        "memory_store": "active",
        "llm_reasoner": "active" if reasoner.enabled else "disabled (rules-only mode)",
    }

    return HealthResponse(
        status="healthy",
        components=components,
        version="3.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )