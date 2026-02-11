"""
ML Flow AI Endpoints
=====================
Dedicated endpoints for ML Flow training intelligence:

  POST /mlflow/smart-config   — Pre-training: optimal configuration for dataset
  POST /mlflow/compare        — Post-training: compare all models, explain winner
  POST /mlflow/explain        — Post-training: why did the best model win?

These complement the existing /agent/insights endpoint (which handles
both pre-training and post-training analysis via screen="mlflow").

Usage:
  Pre-training flow:
    1. User opens ML Flow → frontend calls /agent/insights {screen:"mlflow"}
    2. TC/AS rules fire with pre-training advice
    3. Frontend calls /mlflow/smart-config for algorithm recommendations

  Post-training flow:
    1. Training completes → frontend calls /agent/insights {screen:"mlflow", extra:{training_completed:true, model_results:[...]}}
    2. MF rules fire with post-training analysis
    3. Frontend calls /mlflow/compare for detailed comparison
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
    suggested_questions: List[str] = []
    timing: Dict[str, float] = {}


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


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_orchestrator(db):
    """Get or create orchestrator instance."""
    from app.core.agent.orchestrator import Orchestrator
    return Orchestrator(db)


# ═══════════════════════════════════════════════════════════════
# 1. SMART CONFIG — Pre-training configuration
# ═══════════════════════════════════════════════════════════════

@router.post("/smart-config", response_model=SmartConfigResponse)
async def get_smart_config(request: SmartConfigRequest, db=Depends(get_db)):
    """
    Get AI-recommended training configuration for a dataset.

    Analyzes the dataset and returns optimal:
    - Algorithm selection (ranked by suitability)
    - Scaling method
    - Encoding strategy
    - CV folds and test size
    - Preprocessing options

    Call this BEFORE starting ML Flow training.
    """
    t0 = time.time()
    timings = {}

    try:
        orchestrator = _get_orchestrator(db)

        # Build context
        t1 = time.time()
        context = await orchestrator.context_compiler.compile(
            screen="mlflow",
            project_id=request.project_id,
            dataset_id=request.dataset_id,
            extra={"target_column": request.target_column, "problem_type": request.problem_type},
        )
        timings["context_compile"] = round(time.time() - t1, 3)

        # Get statistical analysis
        t1 = time.time()
        analysis = orchestrator._get_cached_analysis(context, request.dataset_id)
        timings["statistical_analysis"] = round(time.time() - t1, 3)

        # Get recommendations from all engines
        t1 = time.time()
        recommendations = orchestrator._get_screen_recommendations("mlflow", context, analysis)
        timings["recommendations"] = round(time.time() - t1, 3)

        # Wrap into smart config
        t1 = time.time()
        profile = context.get("dataset_profile", {})
        profile["target_column"] = request.target_column
        profile["problem_type"] = request.problem_type

        config = orchestrator.mlflow_analyzer.smart_config(profile, recommendations)
        timings["smart_config"] = round(time.time() - t1, 3)

        # Add suggested questions
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
# 2. COMPARE — Post-training model comparison
# ═══════════════════════════════════════════════════════════════

@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest, db=Depends(get_db)):
    """
    Analyze and compare all trained models.

    Returns comprehensive analysis:
    - Winner summary with explanation
    - Full rankings with generalization grades
    - Overfitting analysis across all models
    - Performance spread (tight vs wide)
    - Category analysis (linear vs tree vs boosting)
    - Actionable recommendations
    - Training issues detection

    Call this AFTER ML Flow training completes.
    """
    t0 = time.time()
    timings = {}

    try:
        orchestrator = _get_orchestrator(db)

        # Convert Pydantic models to dicts
        model_dicts = [m.model_dump() for m in request.model_results]

        # Run leaderboard analysis
        t1 = time.time()
        analysis = orchestrator.mlflow_analyzer.analyze_leaderboard(
            model_results=model_dicts,
        )
        timings["leaderboard_analysis"] = round(time.time() - t1, 3)

        # Detect training issues
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
# 3. EXPLAIN — Why did the winner win?
# ═══════════════════════════════════════════════════════════════

@router.post("/explain", response_model=ExplainResponse)
async def explain_winner(request: ExplainRequest, db=Depends(get_db)):
    """
    Generate a human-readable explanation of why the best model won.

    Returns:
    - Winner name and score
    - List of reasons (data-driven)
    - Full narrative explanation

    Useful for the AI chat: "Why did GradientBoosting win?"
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
# 4. HEAD-TO-HEAD — Compare two specific models
# ═══════════════════════════════════════════════════════════════

@router.post("/head-to-head", response_model=HeadToHeadResponse)
async def head_to_head_compare(request: HeadToHeadRequest, db=Depends(get_db)):
    """
    Detailed comparison of two specific models.

    Returns:
    - Winner and margin
    - Per-metric comparison
    - Overfitting comparison
    - Advantages of each model
    - Recommendation

    Useful for: "Compare XGBoost vs RandomForest"
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