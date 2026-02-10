"""
Enhanced ML Agent — API Endpoints (v4.0)
==========================================
16 new endpoints layered on top of the original 7.

Domain Profiles:
  GET  /enhanced/domains           — List 12+ domain profiles
  POST /enhanced/detect-domain     — Auto-detect domain from columns
  POST /enhanced/insights          — Domain-aware auto-insights

Semantic Intent:
  POST /classify-intent            — Classify question intent
  POST /enhanced/ask               — Semantic intent Q&A

Statistical Tests:
  POST /enhanced/statistical-tests — Individual stat tests (PSI, KS, chi², etc.)
  POST /enhanced/drift-suite       — Full multi-test drift analysis

Feedback Loop:
  POST /feedback/record-action     — Log user action on recommendation
  POST /feedback/record-outcome    — Log outcome after action
  POST /feedback/auto-detect       — Auto-detect pending outcomes
  GET  /feedback/effectiveness     — Recommendation effectiveness report
  GET  /feedback/benchmarking      — A/B: followed vs ignored

Cost Matrices:
  GET  /cost-matrices              — List domain cost matrices
  POST /cost-matrices/impact       — Compute business impact

Health:
  GET  /health/enhanced            — All component status
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════════════════════════
# COMPONENT SINGLETONS (lazy init, thread-safe enough for sync)
# ═══════════════════════════════════════════════════════════════

_components: Dict[str, Any] = {}


def _get_enhanced_orchestrator(db):
    """Build an EnhancedOrchestrator wrapping the original."""
    from app.core.agent.integration_wiring import EnhancedOrchestrator
    from app.core.agent.memory_store import MemoryStore

    try:
        from app.models.agent_memory import AgentMemory
        memory = MemoryStore(db_session=db, memory_model=AgentMemory)
    except Exception:
        memory = None

    return EnhancedOrchestrator(db_session=db, memory_store=memory)


def _get_domain_manager():
    if "domain" not in _components:
        from app.core.agent.domain_profiles import DomainProfileManager
        _components["domain"] = DomainProfileManager()
    return _components["domain"]


def _get_semantic_classifier():
    if "semantic" not in _components:
        from app.core.agent.semantic_intent import SemanticIntentClassifier
        _components["semantic"] = SemanticIntentClassifier()
    return _components["semantic"]


def _get_stat_engine():
    if "stats" not in _components:
        from app.core.agent.statistical_tests import StatisticalTestEngine
        _components["stats"] = StatisticalTestEngine()
    return _components["stats"]


def _get_feedback_tracker():
    if "feedback" not in _components:
        from app.core.agent.feedback_tracker import FeedbackTracker
        _components["feedback"] = FeedbackTracker()
    return _components["feedback"]


def _get_cost_manager():
    if "costs" not in _components:
        from app.core.agent.domain_cost_matrices import DomainCostMatrixManager
        _components["costs"] = DomainCostMatrixManager
    return _components["costs"]


# ═══════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════════

class EnhancedInsightRequest(BaseModel):
    screen: str = Field(..., description="Current screen: dashboard|data|eda|mlflow|training|evaluation|registry|deployment|predictions|monitoring")
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    extra: Optional[Dict[str, Any]] = None
    user_domain: Optional[str] = Field(None, description="Force domain: healthcare|finance|retail|manufacturing|cybersecurity|insurance|telecom|hr|marketing|nlp|energy|education")
    threshold_overrides: Optional[Dict[str, float]] = Field(None, description="Custom threshold overrides, e.g. {\"missing_pct_critical\": 40.0}")


class EnhancedAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    screen: Optional[str] = "general"
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    extra: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class ClassifyIntentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    screen: Optional[str] = None
    top_k: int = Field(5, ge=1, le=18)


class StatTestRequest(BaseModel):
    test_type: str = Field(..., description="psi|ks|chi_square|js_divergence|cohens_d|normality|bootstrap_ci|model_comparison")
    reference_histogram: Optional[List[float]] = None
    current_histogram: Optional[List[float]] = None
    reference_percentiles: Optional[Dict[str, float]] = None
    current_percentiles: Optional[Dict[str, float]] = None
    reference_distribution: Optional[Dict[str, float]] = None
    current_distribution: Optional[Dict[str, float]] = None
    ref_mean: Optional[float] = None
    ref_std: Optional[float] = None
    cur_mean: Optional[float] = None
    cur_std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    n_samples: Optional[int] = None
    metric_value: Optional[float] = None
    metric_name: Optional[str] = "accuracy"
    scores_a: Optional[List[float]] = None
    scores_b: Optional[List[float]] = None


class DriftSuiteRequest(BaseModel):
    feature: str
    ref_stats: Dict[str, Any] = Field(..., description="Reference: {mean, std, ...}")
    cur_stats: Dict[str, Any] = Field(..., description="Current: {mean, std, ...}")
    ref_histogram: Optional[List[float]] = None
    cur_histogram: Optional[List[float]] = None
    ref_percentiles: Optional[Dict[str, float]] = None
    cur_percentiles: Optional[Dict[str, float]] = None
    ref_distribution: Optional[Dict[str, float]] = None
    cur_distribution: Optional[Dict[str, float]] = None


class BusinessImpactRequest(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    domain: Optional[str] = None
    use_case: Optional[str] = None
    n_predictions: int = Field(10000, ge=1)


class DetectDomainRequest(BaseModel):
    column_names: List[str] = Field(..., description="Column names from dataset")
    column_types: Optional[Dict[str, str]] = None


class FeedbackActionRequest(BaseModel):
    user_id: str
    recommendation_id: str
    action: str = Field(..., description="followed|ignored|modified|deferred")
    details: Optional[Dict[str, Any]] = None


class FeedbackOutcomeRequest(BaseModel):
    user_id: str
    recommendation_id: str
    metrics_after: Dict[str, float]
    metrics_before: Optional[Dict[str, float]] = None


class AutoDetectOutcomesRequest(BaseModel):
    user_id: str
    project_id: str
    current_metrics: Dict[str, float]


# ═══════════════════════════════════════════════════════════════
# DOMAIN-AWARE INSIGHTS
# ═══════════════════════════════════════════════════════════════

@router.post("/enhanced/insights")
async def get_enhanced_insights(request: EnhancedInsightRequest, db=Depends(get_db)):
    """Domain-aware auto-insights with adaptive thresholds + feedback adjustment."""
    try:
        orchestrator = _get_enhanced_orchestrator(db)
        result = orchestrator.get_insights(
            user_id=request.user_id or "anonymous",
            screen=request.screen,
            frontend_state=request.extra,
            extra=request.extra,
            user_domain=request.user_domain,
            user_threshold_overrides=request.threshold_overrides,
        )
        # Handle async coroutine if base orchestrator returns one
        import asyncio
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            result = await result
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result
    except Exception as e:
        logger.error(f"Enhanced insights error: {e}", exc_info=True)
        return {"insights": [], "error": str(e)}


@router.post("/enhanced/ask")
async def enhanced_ask(request: EnhancedAskRequest, db=Depends(get_db)):
    """Semantic intent Q&A — handles paraphrases, slang, and screen context."""
    try:
        orchestrator = _get_enhanced_orchestrator(db)
        result = orchestrator.ask(
            user_id=request.user_id or "anonymous",
            question=request.question,
            screen=request.screen or "general",
            frontend_state=request.extra,
            conversation_history=request.conversation_history,
            extra=request.extra,
        )
        # Handle async coroutine if base orchestrator returns one
        import asyncio
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            result = await result
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return result
    except Exception as e:
        logger.error(f"Enhanced ask error: {e}", exc_info=True)
        return {"answer": f"Error: {str(e)}", "source": "error"}


# ═══════════════════════════════════════════════════════════════
# DOMAIN PROFILES
# ═══════════════════════════════════════════════════════════════

@router.get("/enhanced/domains")
async def list_domains():
    """List all available domain profiles with calibrated thresholds."""
    try:
        mgr = _get_domain_manager()
        domains = mgr.list_domains()
        result = {}
        for d in domains:
            profile = mgr.detect({}, user_domain=d)
            result[d] = {
                "description": profile.description,
                "thresholds": profile.thresholds.to_dict(),
            }
        return {"domains": result, "count": len(result)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/enhanced/detect-domain")
async def detect_domain(request: DetectDomainRequest):
    """Auto-detect industry domain from column names."""
    try:
        mgr = _get_domain_manager()
        ctx = {
            "dataset_profile": {
                "dtypes": request.column_types or {c: "unknown" for c in request.column_names},
                "column_names": request.column_names,
            },
            "feature_stats": {"numeric_stats": {}},
        }
        profile = mgr.detect(ctx)
        return {
            "domain": profile.domain,
            "confidence": profile.detection_confidence,
            "description": profile.description,
            "thresholds": profile.thresholds.to_dict(),
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# SEMANTIC INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

@router.post("/classify-intent")
async def classify_intent(request: ClassifyIntentRequest):
    """Classify a question into ML intents using regex + TF-IDF + keyword expansion."""
    try:
        clf = _get_semantic_classifier()
        intent, confidence, method = clf.classify(request.question, screen=request.screen)
        all_scores = clf.get_all_scores(request.question, screen=request.screen)
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "method": method,
            "top_intents": all_scores[:request.top_k],
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════

@router.post("/enhanced/statistical-tests")
async def run_statistical_test(request: StatTestRequest):
    """Run individual statistical tests (PSI, KS, chi², JS, Cohen's d, normality, bootstrap, model comparison)."""
    try:
        eng = _get_stat_engine()
        t = request.test_type.lower()

        if t == "psi":
            r = eng.compute_psi(request.reference_histogram, request.current_histogram)
        elif t == "ks":
            r = eng.compute_ks_from_percentiles(request.reference_percentiles, request.current_percentiles)
        elif t == "chi_square":
            r = eng.compute_chi_square(request.reference_distribution, request.current_distribution)
        elif t == "js_divergence":
            r = eng.compute_js_divergence(request.reference_histogram, request.current_histogram)
        elif t == "cohens_d":
            r = eng.compute_cohens_d(request.ref_mean, request.ref_std, request.cur_mean, request.cur_std)
        elif t == "normality":
            r = eng.assess_normality(
                request.ref_mean or 0, request.ref_std or 1,
                request.skewness or 0, request.kurtosis or 3,
                request.n_samples or 100,
            )
            return {
                "is_normal": r.is_normal,
                "tests": [
                    {"test": tr.test_name, "statistic": tr.statistic, "p_value": tr.p_value, "significant": tr.is_significant}
                    for tr in r.individual_tests
                ] if hasattr(r, "individual_tests") else [],
                "suggested_transform": r.suggested_transform,
            }
        elif t == "bootstrap_ci":
            return eng.bootstrap_metric_ci(request.metric_value, request.n_samples or 100, request.metric_name)
        elif t == "model_comparison":
            r = eng.compare_models_significance(request.scores_a, request.scores_b)
        else:
            return {"error": f"Unknown test: {t}. Available: psi, ks, chi_square, js_divergence, cohens_d, normality, bootstrap_ci, model_comparison"}

        return {
            "test": r.test_name,
            "statistic": round(r.statistic, 6),
            "p_value": round(r.p_value, 6) if r.p_value is not None else None,
            "is_significant": r.is_significant,
            "interpretation": r.interpretation,
            "severity": r.severity,
            "thresholds": r.thresholds,
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/enhanced/drift-suite")
async def run_drift_suite(request: DriftSuiteRequest):
    """Run comprehensive multi-test drift analysis on a single feature."""
    try:
        eng = _get_stat_engine()
        suite = eng.run_drift_suite(
            feature=request.feature,
            ref_stats=request.ref_stats, cur_stats=request.cur_stats,
            ref_histogram=request.ref_histogram, cur_histogram=request.cur_histogram,
            ref_percentiles=request.ref_percentiles, cur_percentiles=request.cur_percentiles,
            ref_distribution=request.ref_distribution, cur_distribution=request.cur_distribution,
        )
        return suite.to_dict()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════

@router.post("/feedback/record-action")
async def record_feedback_action(request: FeedbackActionRequest):
    """Log whether user followed/ignored/modified a recommendation."""
    try:
        tracker = _get_feedback_tracker()
        success = tracker.log_user_action(
            request.user_id, request.recommendation_id,
            request.action, request.details,
        )
        return {"success": success, "recommendation_id": request.recommendation_id}
    except Exception as e:
        return {"error": str(e)}


@router.post("/feedback/record-outcome")
async def record_feedback_outcome(request: FeedbackOutcomeRequest):
    """Log outcome metrics after user acted on a recommendation."""
    try:
        tracker = _get_feedback_tracker()
        verdict = tracker.log_outcome(
            request.user_id, request.recommendation_id,
            request.metrics_after, request.metrics_before,
        )
        return {"verdict": verdict, "recommendation_id": request.recommendation_id}
    except Exception as e:
        return {"error": str(e)}


@router.post("/feedback/auto-detect")
async def auto_detect_outcomes(request: AutoDetectOutcomesRequest):
    """Auto-detect outcomes for pending recommendations by comparing metrics."""
    try:
        tracker = _get_feedback_tracker()
        resolved = tracker.auto_detect_outcomes(
            request.user_id, request.project_id, request.current_metrics,
        )
        return {"resolved": resolved, "count": len(resolved)}
    except Exception as e:
        return {"error": str(e)}


@router.get("/feedback/effectiveness")
async def get_effectiveness(
    category: Optional[str] = Query(None, description="Filter by category: algorithm|data_quality|feature_engineering|..."),
    user_id: Optional[str] = Query(None),
):
    """Get recommendation effectiveness metrics (success rate, follow rate, avg improvement)."""
    try:
        tracker = _get_feedback_tracker()
        return tracker.get_recommendation_effectiveness(category, user_id)
    except Exception as e:
        return {"error": str(e)}


@router.get("/feedback/benchmarking")
async def get_benchmarking(user_id: Optional[str] = Query(None)):
    """A/B benchmarking: compare outcomes of followed vs ignored recommendations."""
    try:
        tracker = _get_feedback_tracker()
        return tracker.get_benchmarking_report(user_id)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# DOMAIN COST MATRICES
# ═══════════════════════════════════════════════════════════════

@router.get("/cost-matrices")
async def list_cost_matrices():
    """List all available domain cost matrices (healthcare, finance, retail, etc.)."""
    try:
        mgr = _get_cost_manager()
        return {"matrices": mgr.list_available()}
    except Exception as e:
        return {"error": str(e)}


@router.post("/cost-matrices/impact")
async def compute_business_impact(request: BusinessImpactRequest):
    """Compute domain-calibrated business impact with ROI and capacity planning."""
    try:
        mgr = _get_cost_manager()
        matrix = None
        if request.domain and request.use_case:
            matrix = mgr.get(request.domain, request.use_case)
        if not matrix and request.domain:
            matrix = mgr.get_default_for_domain(request.domain)
        if not matrix:
            matrix = mgr.get_default_for_domain("general")
        if not matrix:
            return {"error": f"No cost matrix found for domain={request.domain}, use_case={request.use_case}"}
        return matrix.compute_impact(
            precision=request.precision,
            recall=request.recall,
            n_predictions=request.n_predictions,
        )
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# ENHANCED HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

_start_time = time.time()


@router.get("/health/enhanced")
async def enhanced_health():
    """Health check for ALL components including v4.0 enhancements."""
    components = {}

    try:
        from app.core.agent.rule_engine import RuleEngine
        re = RuleEngine()
        components["rule_engine"] = {"status": "active", "rules": len(re._rules)}
    except Exception as e:
        components["rule_engine"] = {"status": "error", "detail": str(e)}

    for name, getter in [
        ("semantic_intent", _get_semantic_classifier),
        ("domain_profiles", _get_domain_manager),
        ("statistical_tests", _get_stat_engine),
        ("feedback_tracker", _get_feedback_tracker),
    ]:
        try:
            obj = getter()
            info = {"status": "active"}
            if hasattr(obj, "_exemplars"):
                info["intents"] = len(obj._exemplars)
            if hasattr(obj, "list_domains"):
                info["domains"] = len(obj.list_domains())
            components[name] = info
        except Exception as e:
            components[name] = {"status": "error", "detail": str(e)}

    try:
        mgr = _get_cost_manager()
        components["cost_matrices"] = {"status": "active", "matrices": len(mgr.list_available())}
    except Exception as e:
        components["cost_matrices"] = {"status": "error", "detail": str(e)}

    all_active = all(c.get("status") == "active" for c in components.values())
    return {
        "status": "healthy" if all_active else "degraded",
        "version": "4.0.0",
        "components": components,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
