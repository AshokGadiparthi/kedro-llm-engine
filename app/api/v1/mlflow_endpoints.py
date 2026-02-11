"""
ML Flow AI Endpoints — World-Class Post-Training Intelligence
================================================================
PHASE 1 (existing 4 endpoints):
  POST /mlflow/smart-config     — Pre-training: history-aware configuration
  POST /mlflow/compare          — Post-training: full leaderboard analysis
  POST /mlflow/explain          — Post-training: why the best model won
  POST /mlflow/head-to-head     — Post-training: compare 2 specific models

PHASE 2 (NEW 4 endpoints):
  POST /mlflow/production-readiness  — 10-point deployment score card
  POST /mlflow/next-steps            — Prioritized action roadmap
  POST /mlflow/registry-analysis     — Registry intelligence + version trends
  POST /mlflow/deep-analysis         — Full analysis (confusion matrix, features, history)

UTILITY (NEW):
  GET  /mlflow/registry/models       — List registered models for UI dropdown

Usage Flow:
  Pre-training:
    1. GET  /mlflow/registry/models → populate registered models dropdown
    2. POST /mlflow/smart-config    → AI-recommended algorithms (history-aware)

  Post-training (call all at once or on-demand):
    3. POST /mlflow/compare              → rankings, overfitting, spread
    4. POST /mlflow/production-readiness → deployment score card
    5. POST /mlflow/next-steps           → what to do next
    6. POST /mlflow/deep-analysis        → confusion matrix, features, history

  On-demand:
    7. POST /mlflow/explain         → "Why did X win?"
    8. POST /mlflow/head-to-head    → "Compare X vs Y"
    9. POST /mlflow/registry-analysis → registry alerts, version trends
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mlflow", tags=["ML Flow AI"])


# ═══════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════

# ── Shared Models ─────────────────────────────────────────────

class ModelResult(BaseModel):
    """Single model's training result."""
    algorithm: str = Field(..., description="Algorithm name (e.g., 'GradientBoostingClassifier')")
    train_score: Optional[float] = Field(None, description="Training accuracy (0-1 or 0-100)")
    test_score: Optional[float] = Field(None, description="Test accuracy (0-1 or 0-100)")
    accuracy: Optional[float] = Field(None, description="Alias for test_score")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    training_time: Optional[float] = Field(None, description="Training time in seconds")
    status: Optional[str] = Field("success", description="'success' or 'failed'")
    error_message: Optional[str] = None


# ── Phase 1: Smart Config ────────────────────────────────────

class SmartConfigRequest(BaseModel):
    """Request for pre-training smart configuration."""
    dataset_id: str = Field(..., description="Dataset ID")
    target_column: Optional[str] = Field(None, description="Target column name")
    problem_type: Optional[str] = Field("classification", description="'classification' or 'regression'")
    project_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"


class SmartConfigResponse(BaseModel):
    """Pre-training smart configuration recommendation."""
    problem_type: str
    target_column: Optional[str] = None
    recommended_algorithms: List[Dict[str, Any]] = []
    scaling_method: str = "standard"
    encoding_strategy: Any = None
    cv_folds: int = 5
    test_size: float = 0.2
    stratified: bool = True
    preprocessing: Dict[str, bool] = {}
    estimated_training_time: str = "~2-5 minutes"
    rationale: List[str] = []
    history_insights: List[str] = []
    suggested_questions: List[str] = []
    timing: Dict[str, float] = {}


# ── Phase 1: Compare ─────────────────────────────────────────

class CompareRequest(BaseModel):
    """Request for post-training model comparison."""
    model_results: List[ModelResult] = Field(..., min_length=1, description="All trained model results")
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"


class CompareResponse(BaseModel):
    """Full post-training analysis."""
    total_models_trained: int = 0
    total_models_failed: int = 0
    failed_algorithms: List[str] = []
    winner: Dict[str, Any] = {}
    rankings: List[Dict[str, Any]] = []
    overfitting_analysis: Dict[str, Any] = {}
    performance_spread: Dict[str, Any] = {}
    category_analysis: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []
    training_issues: List[Dict[str, Any]] = []
    insights: List[str] = []
    suggested_questions: List[str] = []
    timing: Dict[str, float] = {}


# ── Phase 1: Explain ─────────────────────────────────────────

class ExplainRequest(BaseModel):
    """Request for winner explanation."""
    model_results: List[ModelResult] = Field(..., min_length=1)
    dataset_id: Optional[str] = None


class ExplainResponse(BaseModel):
    """Why the best model won."""
    winner: str = ""
    score: float = 0
    score_pct: str = ""
    category: str = ""
    reasons: List[str] = []
    explanation: str = ""
    timing: Dict[str, float] = {}


# ── Phase 1: Head-to-Head ────────────────────────────────────

class HeadToHeadRequest(BaseModel):
    """Compare two specific models."""
    model_a: ModelResult
    model_b: ModelResult


class HeadToHeadResponse(BaseModel):
    """Head-to-head comparison result."""
    model_a: str = ""
    model_b: str = ""
    winner: str = ""
    score_difference: float = 0
    score_difference_pct: str = ""
    statistically_significant: bool = False
    metrics_comparison: Dict[str, Any] = {}
    advantages: Dict[str, List[str]] = {}
    overfitting: Dict[str, Any] = {}
    recommendation: str = ""
    timing: Dict[str, float] = {}


# ── Phase 2: Production Readiness ────────────────────────────

class ProductionReadinessRequest(BaseModel):
    """Request for production readiness assessment."""
    model_results: List[ModelResult] = Field(..., min_length=1)
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None
    model_id: Optional[str] = Field(None, description="Registered model ID for richer metrics from DB")


class ProductionReadinessResponse(BaseModel):
    """10-point deployment score card."""
    score: int = 0
    max_score: int = 10
    percentage: float = 0
    verdict: str = ""  # ready, ready_with_caveats, not_ready
    message: str = ""
    checks: List[Dict[str, Any]] = []
    blockers: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    unknowns: List[Dict[str, Any]] = []
    winner_algorithm: str = ""
    winner_score: str = ""
    timing: Dict[str, float] = {}


# ── Phase 2: Next Steps ──────────────────────────────────────

class NextStepsRequest(BaseModel):
    """Request for next steps roadmap."""
    model_results: List[ModelResult] = Field(..., min_length=1)
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None


class NextStepsResponse(BaseModel):
    """Prioritized action roadmap."""
    steps: List[Dict[str, Any]] = []
    total_steps: int = 0
    high_priority_count: int = 0
    timing: Dict[str, float] = {}


# ── Phase 2: Registry Analysis ───────────────────────────────

class RegistryAnalysisRequest(BaseModel):
    """Request for registry intelligence."""
    project_id: Optional[str] = None
    model_id: Optional[str] = None
    current_best_score: Optional[float] = Field(None, description="Best score from current training for comparison")


class RegistryAnalysisResponse(BaseModel):
    """Registry intelligence with alerts and trends."""
    total_registered: int = 0
    total_deployed: int = 0
    total_versions: int = 0
    models: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []
    deployment_recommendation: Optional[Dict[str, Any]] = None
    version_trend: Optional[Dict[str, Any]] = None
    timing: Dict[str, float] = {}


# ── Phase 2: Deep Analysis ───────────────────────────────────

class DeepAnalysisRequest(BaseModel):
    """Request for deep post-training analysis (confusion matrix, features, history)."""
    model_results: List[ModelResult] = Field(..., min_length=1)
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None
    model_id: Optional[str] = Field(None, description="Registered model ID for confusion matrix + feature importances")
    class_names: Optional[List[str]] = Field(None, description="Class labels e.g. ['No Churn', 'Churn']")


class DeepAnalysisResponse(BaseModel):
    """Deep analysis combining confusion matrix, features, history."""
    production_readiness: Dict[str, Any] = {}
    confusion_matrix_analysis: Dict[str, Any] = {}
    feature_importance_analysis: Dict[str, Any] = {}
    training_history_analysis: Dict[str, Any] = {}
    registry_analysis: Dict[str, Any] = {}
    next_steps: List[Dict[str, Any]] = []
    timing: Dict[str, float] = {}


# ── Utility: Registry Models List ─────────────────────────────

class RegisteredModelInfo(BaseModel):
    """Single registered model for UI dropdown."""
    id: str
    name: str
    best_algorithm: Optional[str] = None
    best_accuracy: Optional[float] = None
    best_accuracy_pct: str = ""
    problem_type: Optional[str] = None
    is_deployed: bool = False
    total_versions: int = 0
    current_version: Optional[str] = None
    status: Optional[str] = None
    source_dataset_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RegistryModelsResponse(BaseModel):
    """List of registered models for UI dropdown."""
    total: int = 0
    models: List[RegisteredModelInfo] = []
    deployed_count: int = 0


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_orchestrator(db):
    """Get or create orchestrator instance (matches agent.py pattern)."""
    from app.core.agent.orchestrator import AgentOrchestrator
    from app.core.agent.memory_store import MemoryStore

    try:
        from app.models.agent_memory import AgentMemory
        memory = MemoryStore(db_session=db, memory_model=AgentMemory)
    except Exception:
        memory = None

    return AgentOrchestrator(db_session=db, memory_store=memory)


async def _get_context_data(db, project_id: str = None, dataset_id: str = None, model_id: str = None):
    """
    Fetch training_history, registry_info, model_versions from DB via context_compiler.
    Returns a dict with all three.
    """
    orchestrator = _get_orchestrator(db)
    cc = orchestrator.context_compiler

    training_history = {}
    registry_info = {}
    model_versions = {}

    try:
        training_history = await cc._get_training_history(project_id or "")
    except Exception as e:
        logger.warning(f"Could not fetch training history: {e}")

    try:
        registry_info = await cc._get_registry_info(project_id or "", model_id)
    except Exception as e:
        logger.warning(f"Could not fetch registry info: {e}")

    try:
        model_versions = await cc._get_model_versions(project_id or "", model_id)
    except Exception as e:
        logger.warning(f"Could not fetch model versions: {e}")

    return {
        "orchestrator": orchestrator,
        "training_history": training_history,
        "registry_info": registry_info,
        "model_versions": model_versions,
    }


def _enrich_model_results_from_db(model_dicts: List[Dict], model_versions: Dict) -> List[Dict]:
    """
    Enrich frontend model_results with full metrics from DB model_versions.

    The frontend often sends only {algorithm, accuracy/test_score} when loading
    from the registered models screen. The DB has the full picture:
    train_score, precision, recall, f1_score, roc_auc, training_time.

    This function matches by algorithm name and fills in missing metrics.
    """
    if not model_versions or not model_versions.get("versions"):
        logger.info(f"[Enrichment] No model_versions data to enrich from (model_versions={type(model_versions)})")
        return model_dicts

    # Build a lookup: algorithm_name → best version metrics
    db_lookup = {}
    for v in model_versions.get("versions", []):
        algo = (v.get("algorithm") or "").lower().strip()
        metrics = v.get("metrics", {})
        if algo and algo not in db_lookup:
            db_lookup[algo] = {
                "train_score": metrics.get("train_score"),
                "test_score": metrics.get("test_score"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "roc_auc": metrics.get("roc_auc"),
                "training_time": v.get("training_time_seconds"),
            }

    logger.info(f"[Enrichment] DB lookup has {len(db_lookup)} algorithms: {list(db_lookup.keys())}")
    train_scores = {k: v.get("train_score") for k, v in db_lookup.items()}
    logger.info(f"[Enrichment] DB train_scores: {train_scores}")

    enriched = []
    for m in model_dicts:
        m = dict(m)  # copy
        algo_key = (m.get("algorithm", "")).lower().strip()
        db_data = db_lookup.get(algo_key, {})

        if not db_data:
            logger.info(f"[Enrichment] No DB match for '{algo_key}' (available: {list(db_lookup.keys())})")

        # Fill in any missing metrics from DB
        enriched_fields = []
        for field in ["train_score", "test_score", "precision", "recall", "f1_score", "roc_auc", "training_time"]:
            current = m.get(field)
            db_val = db_data.get(field)
            if (current is None or current == 0) and db_val is not None and db_val != 0:
                m[field] = db_val
                enriched_fields.append(f"{field}={db_val}")

        if enriched_fields:
            logger.info(f"[Enrichment] {algo_key} enriched: {', '.join(enriched_fields)}")

        # Also fill accuracy ↔ test_score mapping
        if (m.get("test_score") is None or m.get("test_score") == 0) and m.get("accuracy"):
            m["test_score"] = m["accuracy"]
        if (m.get("accuracy") is None or m.get("accuracy") == 0) and m.get("test_score"):
            m["accuracy"] = m["test_score"]

        enriched.append(m)

    return enriched


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 1: SMART CONFIG — Pre-training (History-Aware)
# ═══════════════════════════════════════════════════════════════

@router.post("/smart-config", response_model=SmartConfigResponse)
async def get_smart_config(request: SmartConfigRequest, db=Depends(get_db)):
    """
    AI-recommended training configuration — now history-aware.

    Returns optimal algorithms, scaling, CV config.
    Learns from past training: skips failed algos, uses historical timing,
    shows deployed model benchmark, detects accuracy plateaus.
    """
    t0 = time.time()
    timings = {}

    try:
        # Get orchestrator and DB context
        ctx_data = await _get_context_data(db, request.project_id, request.dataset_id)
        orchestrator = ctx_data["orchestrator"]

        # Build context
        t1 = time.time()
        context = await orchestrator.context_compiler.compile(
            screen="mlflow",
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            extra={"target_column": request.target_column, "problem_type": request.problem_type},
        )
        timings["context_compile"] = round(time.time() - t1, 3)

        # Statistical analysis
        t1 = time.time()
        analysis = orchestrator._get_cached_analysis(context, request.dataset_id)
        timings["statistical_analysis"] = round(time.time() - t1, 3)

        # Recommendations
        t1 = time.time()
        recommendations = orchestrator._get_screen_recommendations("mlflow", context, analysis)
        timings["recommendations"] = round(time.time() - t1, 3)

        # Smart config (history-aware)
        t1 = time.time()
        profile = context.get("dataset_profile", {})
        profile["target_column"] = request.target_column
        profile["problem_type"] = request.problem_type

        config = orchestrator.mlflow_analyzer.smart_config(
            dataset_profile=profile,
            recommendations=recommendations,
            training_history=ctx_data["training_history"],
            registry_info=ctx_data["registry_info"],
        )
        timings["smart_config"] = round(time.time() - t1, 3)

        config["suggested_questions"] = orchestrator.mlflow_analyzer.generate_pre_training_questions(profile)
        config["timing"] = timings
        return SmartConfigResponse(**config)

    except Exception as e:
        logger.error(f"Smart config error: {e}", exc_info=True)
        return SmartConfigResponse(
            problem_type=request.problem_type or "classification",
            rationale=[f"Error generating config: {str(e)}"],
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 2: COMPARE — Post-training leaderboard
# ═══════════════════════════════════════════════════════════════

@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest, db=Depends(get_db)):
    """
    Full leaderboard analysis with rankings, overfitting, spread,
    category analysis, training issues, and recommendations.
    """
    t0 = time.time()
    timings = {}

    try:
        orchestrator = _get_orchestrator(db)
        model_dicts = [m.model_dump() for m in request.model_results]

        # Enrich with full DB metrics if project_id/dataset_id available
        if request.project_id or request.dataset_id:
            try:
                ctx_data = await _get_context_data(db, request.project_id, request.dataset_id)
                model_dicts = _enrich_model_results_from_db(model_dicts, ctx_data.get("model_versions", {}))
            except Exception:
                pass  # Non-critical: proceed with frontend data

        t1 = time.time()
        analysis = orchestrator.mlflow_analyzer.analyze_leaderboard(
            model_results=model_dicts,
        )
        timings["leaderboard_analysis"] = round(time.time() - t1, 3)

        t1 = time.time()
        issues = orchestrator.mlflow_analyzer.detect_training_issues(
            model_results=model_dicts,
        )
        timings["issue_detection"] = round(time.time() - t1, 3)

        analysis["training_issues"] = issues
        analysis["timing"] = timings

        return CompareResponse(**analysis)

    except Exception as e:
        logger.error(f"Model comparison error: {e}", exc_info=True)
        return CompareResponse(
            training_issues=[{"type": "error", "severity": "warning",
                              "title": "Analysis Error", "description": str(e)}],
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 3: EXPLAIN — Why did the winner win?
# ═══════════════════════════════════════════════════════════════

@router.post("/explain", response_model=ExplainResponse)
async def explain_winner(request: ExplainRequest, db=Depends(get_db)):
    """
    Human-readable explanation of why the best model won.
    Data-driven reasons with supporting evidence.
    """
    t0 = time.time()

    try:
        orchestrator = _get_orchestrator(db)
        model_dicts = [m.model_dump() for m in request.model_results]

        explanation = orchestrator.mlflow_analyzer.explain_winner(model_dicts)
        explanation["timing"] = {"explain": round(time.time() - t0, 3)}

        return ExplainResponse(**explanation)

    except Exception as e:
        logger.error(f"Explain winner error: {e}", exc_info=True)
        return ExplainResponse(
            explanation=f"Error generating explanation: {str(e)}",
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 4: HEAD-TO-HEAD — Compare two models
# ═══════════════════════════════════════════════════════════════

@router.post("/head-to-head", response_model=HeadToHeadResponse)
async def head_to_head_compare(request: HeadToHeadRequest, db=Depends(get_db)):
    """
    Detailed comparison of two specific models.
    Per-metric comparison, overfitting, advantages, recommendation.
    """
    t0 = time.time()

    try:
        orchestrator = _get_orchestrator(db)

        comparison = orchestrator.mlflow_analyzer.compare_models(
            model_a=request.model_a.model_dump(),
            model_b=request.model_b.model_dump(),
        )
        comparison["timing"] = {"compare": round(time.time() - t0, 3)}

        return HeadToHeadResponse(**comparison)

    except Exception as e:
        logger.error(f"Head-to-head error: {e}", exc_info=True)
        return HeadToHeadResponse(
            recommendation=f"Error: {str(e)}",
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 5: PRODUCTION READINESS — Deployment score card (NEW)
# ═══════════════════════════════════════════════════════════════

@router.post("/production-readiness", response_model=ProductionReadinessResponse)
async def assess_production_readiness(request: ProductionReadinessRequest, db=Depends(get_db)):
    """
    10-point deployment readiness score card.

    Checks: accuracy, overfitting, precision, recall, F1, ROC-AUC,
    statistical significance, algorithm diversity, data size.

    If model_id is provided, enriches with DB metrics (confusion matrix, etc.)
    for a more complete assessment.
    """
    t0 = time.time()
    timings = {}

    try:
        # Get DB data if model_id provided
        model_version_data = None
        context_profile = None
        ctx_data = None

        if request.model_id or request.project_id:
            t1 = time.time()
            ctx_data = await _get_context_data(db, request.project_id, request.dataset_id, request.model_id)
            timings["db_lookup"] = round(time.time() - t1, 3)

            # Get best version metrics for enrichment
            mv = ctx_data["model_versions"]
            if mv and mv.get("best_version"):
                model_version_data = mv["best_version"]

        if request.dataset_id:
            try:
                orchestrator = _get_orchestrator(db)
                context = await orchestrator.context_compiler.compile(
                    screen="mlflow", dataset_id=request.dataset_id,
                    project_id=request.project_id, extra={},
                )
                context_profile = context.get("dataset_profile")
            except Exception:
                pass

        orchestrator = _get_orchestrator(db)
        model_dicts = [m.model_dump() for m in request.model_results]

        # Enrich with full metrics from DB if available
        if ctx_data and ctx_data.get("model_versions"):
            model_dicts = _enrich_model_results_from_db(model_dicts, ctx_data["model_versions"])

        t1 = time.time()
        readiness = orchestrator.mlflow_analyzer.assess_production_readiness(
            model_results=model_dicts,
            model_version_data=model_version_data,
            context=context_profile,
        )
        timings["readiness_assessment"] = round(time.time() - t1, 3)

        readiness["timing"] = timings
        return ProductionReadinessResponse(**readiness)

    except Exception as e:
        logger.error(f"Production readiness error: {e}", exc_info=True)
        return ProductionReadinessResponse(
            verdict="error", message=f"Error: {str(e)}",
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 6: NEXT STEPS — Prioritized action roadmap (NEW)
# ═══════════════════════════════════════════════════════════════

@router.post("/next-steps", response_model=NextStepsResponse)
async def get_next_steps(request: NextStepsRequest, db=Depends(get_db)):
    """
    Prioritized, actionable next steps roadmap.

    Considers model performance, registry state, training history,
    production readiness. Returns ordered steps with priority, effort, impact.
    """
    t0 = time.time()
    timings = {}

    try:
        # Fetch all DB context
        t1 = time.time()
        ctx_data = await _get_context_data(db, request.project_id, request.dataset_id)
        orchestrator = ctx_data["orchestrator"]
        timings["db_context"] = round(time.time() - t1, 3)

        model_dicts = [m.model_dump() for m in request.model_results]
        # Enrich with full metrics from DB
        model_dicts = _enrich_model_results_from_db(model_dicts, ctx_data["model_versions"])

        # Get production readiness first (needed for next steps logic)
        t1 = time.time()
        production_readiness = orchestrator.mlflow_analyzer.assess_production_readiness(
            model_results=model_dicts,
        )
        timings["readiness_check"] = round(time.time() - t1, 3)

        # Get dataset context
        context_profile = None
        try:
            if request.dataset_id:
                context = await orchestrator.context_compiler.compile(
                    screen="mlflow", dataset_id=request.dataset_id,
                    project_id=request.project_id, extra={},
                )
                context_profile = context.get("dataset_profile")
        except Exception:
            pass

        # Generate next steps
        t1 = time.time()
        steps = orchestrator.mlflow_analyzer.generate_next_steps(
            model_results=model_dicts,
            registry_info=ctx_data["registry_info"],
            training_history=ctx_data["training_history"],
            production_readiness=production_readiness,
            context=context_profile,
        )
        timings["next_steps"] = round(time.time() - t1, 3)

        high_priority = sum(1 for s in steps if s.get("priority") in ("high", "critical"))

        return NextStepsResponse(
            steps=steps,
            total_steps=len(steps),
            high_priority_count=high_priority,
            timing=timings,
        )

    except Exception as e:
        logger.error(f"Next steps error: {e}", exc_info=True)
        return NextStepsResponse(
            steps=[{"step": 1, "priority": "critical", "title": "Error",
                    "description": str(e), "category": "debug"}],
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 7: REGISTRY ANALYSIS — Registry intelligence (NEW)
# ═══════════════════════════════════════════════════════════════

@router.post("/registry-analysis", response_model=RegistryAnalysisResponse)
async def analyze_registry(request: RegistryAnalysisRequest, db=Depends(get_db)):
    """
    Model registry intelligence: portfolio summary, version trends,
    staleness alerts, upgrade recommendations.

    Pass current_best_score to compare against deployed model.
    """
    t0 = time.time()
    timings = {}

    try:
        t1 = time.time()
        ctx_data = await _get_context_data(db, request.project_id, model_id=request.model_id)
        orchestrator = ctx_data["orchestrator"]
        timings["db_context"] = round(time.time() - t1, 3)

        t1 = time.time()
        analysis = orchestrator.mlflow_analyzer.analyze_registry(
            registry_info=ctx_data["registry_info"],
            model_versions=ctx_data["model_versions"],
            current_training_score=request.current_best_score,
        )
        timings["registry_analysis"] = round(time.time() - t1, 3)

        analysis["timing"] = timings
        return RegistryAnalysisResponse(**analysis)

    except Exception as e:
        logger.error(f"Registry analysis error: {e}", exc_info=True)
        return RegistryAnalysisResponse(
            alerts=[{"type": "error", "severity": "warning",
                     "title": "Analysis Error", "message": str(e)}],
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# ENDPOINT 8: DEEP ANALYSIS — Full post-training intelligence (NEW)
# ═══════════════════════════════════════════════════════════════

@router.post("/deep-analysis", response_model=DeepAnalysisResponse)
async def deep_analysis(request: DeepAnalysisRequest, db=Depends(get_db)):
    """
    Comprehensive post-training analysis combining ALL intelligence:

    1. Production readiness score card
    2. Confusion matrix analysis (where the model fails)
    3. Feature importance analysis (what drives predictions)
    4. Training history analysis (learn from past runs)
    5. Registry intelligence (version trends, staleness)
    6. Prioritized next steps

    This is the "one call to rule them all" endpoint.
    Call this when training completes for the richest possible analysis.

    model_id: Required for confusion matrix and feature importance (from DB).
              Without it, only readiness + history + next steps are available.
    class_names: e.g. ["No Churn", "Churn"] for confusion matrix labels.
    """
    t0 = time.time()
    timings = {}

    try:
        # Fetch all DB context
        t1 = time.time()
        ctx_data = await _get_context_data(db, request.project_id, request.dataset_id, request.model_id)
        orchestrator = ctx_data["orchestrator"]

        # Also fetch ALL project versions (not just selected model) for enrichment
        all_project_versions = ctx_data["model_versions"]
        if request.model_id:
            try:
                cc = orchestrator.context_compiler
                all_project_versions = await cc._get_model_versions(request.project_id or "", None)
            except Exception:
                pass  # Fall back to model-specific versions

        timings["db_context"] = round(time.time() - t1, 3)

        model_dicts = [m.model_dump() for m in request.model_results]
        # Enrich with full metrics from DB (train_score, precision, recall, etc.)
        model_dicts = _enrich_model_results_from_db(model_dicts, all_project_versions)

        # Get dataset context
        context_profile = None
        try:
            if request.dataset_id:
                context = await orchestrator.context_compiler.compile(
                    screen="mlflow", dataset_id=request.dataset_id,
                    project_id=request.project_id, extra={},
                )
                context_profile = context.get("dataset_profile")
        except Exception:
            pass

        result = {}

        # 1. Production Readiness
        t1 = time.time()
        best_version = (ctx_data["model_versions"] or {}).get("best_version")
        result["production_readiness"] = orchestrator.mlflow_analyzer.assess_production_readiness(
            model_results=model_dicts,
            model_version_data=best_version,
            context=context_profile,
        )
        timings["production_readiness"] = round(time.time() - t1, 3)

        # 2. Confusion Matrix (from registered model in DB)
        t1 = time.time()
        confusion_matrix_data = None
        if best_version and best_version.get("confusion_matrix"):
            confusion_matrix_data = best_version["confusion_matrix"]
        result["confusion_matrix_analysis"] = orchestrator.mlflow_analyzer.analyze_confusion_matrix(
            confusion_matrix=confusion_matrix_data,
            class_names=request.class_names,
        )
        timings["confusion_matrix"] = round(time.time() - t1, 3)

        # 3. Feature Importance (from registered model in DB)
        t1 = time.time()
        feature_importances = None
        feature_names = None
        if best_version:
            feature_importances = best_version.get("feature_importances")
            feature_names = best_version.get("feature_names")
        result["feature_importance_analysis"] = orchestrator.mlflow_analyzer.analyze_feature_importance(
            feature_importances=feature_importances,
            feature_names=feature_names if isinstance(feature_names, list) else None,
            context=context_profile,
        )
        timings["feature_importance"] = round(time.time() - t1, 3)

        # 4. Training History
        t1 = time.time()
        result["training_history_analysis"] = orchestrator.mlflow_analyzer.analyze_training_history(
            training_history=ctx_data["training_history"],
            registry_info=ctx_data["registry_info"],
        )
        timings["training_history"] = round(time.time() - t1, 3)

        # 5. Registry Intelligence
        t1 = time.time()
        best_score = max(
            (_safe_float(m.get("test_score") or m.get("accuracy"), 0) for m in model_dicts),
            default=0,
        )
        result["registry_analysis"] = orchestrator.mlflow_analyzer.analyze_registry(
            registry_info=ctx_data["registry_info"],
            model_versions=ctx_data["model_versions"],
            current_training_score=best_score if best_score > 0 else None,
        )
        timings["registry_analysis"] = round(time.time() - t1, 3)

        # 6. Next Steps
        t1 = time.time()
        result["next_steps"] = orchestrator.mlflow_analyzer.generate_next_steps(
            model_results=model_dicts,
            registry_info=ctx_data["registry_info"],
            training_history=ctx_data["training_history"],
            production_readiness=result["production_readiness"],
            context=context_profile,
        )
        timings["next_steps"] = round(time.time() - t1, 3)

        result["timing"] = timings
        return DeepAnalysisResponse(**result)

    except Exception as e:
        logger.error(f"Deep analysis error: {e}", exc_info=True)
        return DeepAnalysisResponse(
            production_readiness={"verdict": "error", "message": str(e)},
            timing={"error": round(time.time() - t0, 3)},
        )


# ═══════════════════════════════════════════════════════════════
# UTILITY: LIST REGISTERED MODELS (for UI dropdown)
# ═══════════════════════════════════════════════════════════════

@router.get("/registry/models", response_model=RegistryModelsResponse)
async def list_registered_models(
        project_id: Optional[str] = None,
        db=Depends(get_db),
):
    """
    List all registered models for the UI dropdown.

    Returns model ID, name, best algorithm, accuracy, deployment status,
    version count — everything the frontend needs for a model selector.

    Use this to populate a "Registered Models" dropdown alongside datasets.
    """
    try:
        from app.models.models import RegisteredModel

        query = db.query(RegisteredModel)
        if project_id:
            query = query.filter(RegisteredModel.project_id == project_id)

        # Order by most recently updated first
        try:
            query = query.order_by(RegisteredModel.updated_at.desc())
        except Exception:
            try:
                query = query.order_by(RegisteredModel.created_at.desc())
            except Exception:
                pass

        models = query.all()

        result_models = []
        deployed_count = 0

        for m in models:
            acc = _safe_float(getattr(m, "best_accuracy", None), 0)
            is_deployed = bool(getattr(m, "is_deployed", False))
            if is_deployed:
                deployed_count += 1

            created = getattr(m, "created_at", None)
            updated = getattr(m, "updated_at", None)

            result_models.append(RegisteredModelInfo(
                id=m.id,
                name=getattr(m, "name", "Unnamed") or "Unnamed",
                best_algorithm=getattr(m, "best_algorithm", None),
                best_accuracy=acc if acc > 0 else None,
                best_accuracy_pct=f"{acc * 100:.2f}%" if acc > 0 else "",
                problem_type=getattr(m, "problem_type", None),
                is_deployed=is_deployed,
                total_versions=getattr(m, "total_versions", 0) or 0,
                current_version=getattr(m, "current_version", None),
                status=getattr(m, "status", None),
                source_dataset_name=getattr(m, "source_dataset_name", None),
                created_at=created.isoformat() if created else None,
                updated_at=updated.isoformat() if updated else None,
            ))

        return RegistryModelsResponse(
            total=len(result_models),
            models=result_models,
            deployed_count=deployed_count,
        )

    except Exception as e:
        logger.error(f"List registry models error: {e}", exc_info=True)
        return RegistryModelsResponse()


def _safe_float(val, default=None):
    """Safely convert to float."""
    if val is None:
        return default
    try:
        import math
        v = float(val)
        return default if math.isnan(v) else v
    except (ValueError, TypeError):
        return default