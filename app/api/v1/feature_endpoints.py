"""
Feature Engineering AI Endpoints — World-Class Intelligence
================================================================
POST /features/deep-analysis        — Full FE intelligence (all analyses combined)
POST /features/transformation-audit — Audit every FE transformation decision
POST /features/selection-explanation — SHAP-like: why each feature was selected/dropped
POST /features/quality-score        — 10-point feature pipeline quality scorecard
POST /features/error-patterns       — Data leakage, multicollinearity, info loss detection
POST /features/smart-config         — AI-recommended FE settings for this data
POST /features/next-steps           — Prioritized improvement roadmap
POST /features/compare-configs      — Compare two FE configurations side-by-side

Data Flow:
  Frontend sends:
    - feature_config: {scaling_method, handle_missing, handle_outliers, encode_categories, ...}
    - feature_results: {original_columns, selected_features, dropped_features, column_types, ...}
    - project_id/dataset_id (for DB enrichment)

  Backend enriches from:
    - eda_results table (statistics, correlations, quality)
    - model_versions table (feature_names, feature_importances)
    - jobs table (pipeline history, timing)

  Returns:
    - Rich analysis JSON for frontend rendering
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/features", tags=["Feature Engineering AI"])


# ═══════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════

class FeatureConfig(BaseModel):
    """Feature engineering configuration sent by frontend."""
    scaling_method: Optional[str] = Field("standard", description="standard|robust|minmax|none")
    handle_missing_values: Optional[bool] = Field(True)
    handle_outliers: Optional[bool] = Field(True)
    encode_categories: Optional[bool] = Field(True)
    create_polynomial_features: Optional[bool] = Field(False)
    create_interactions: Optional[bool] = Field(False)
    variance_threshold: Optional[float] = Field(0.01)
    n_features_to_select: Optional[int] = Field(None)
    imputation_strategy: Optional[str] = Field("automatic")
    outlier_method: Optional[str] = Field("IQR-based detection")


class FeatureResults(BaseModel):
    """Feature engineering pipeline results from Kedro."""
    original_columns: Optional[List[str]] = Field(None, description="All input columns before FE")
    selected_features: Optional[List[str]] = Field(None, description="Final selected features")
    numeric_features: Optional[List[str]] = Field(None, description="Numeric columns detected")
    categorical_features: Optional[List[str]] = Field(None, description="Categorical columns detected")
    id_columns_detected: Optional[List[str]] = Field(None, description="ID columns removed")
    variance_removed: Optional[List[str]] = Field(None, description="Features removed by variance filter")
    encoding_strategies: Optional[Dict[str, str]] = Field(None, description="Encoding strategy per column")
    column_cardinalities: Optional[Dict[str, int]] = Field(None, description="Unique value count per column")
    original_shape: Optional[List[int]] = Field(None, description="[n_rows, n_cols] before FE")
    train_shape: Optional[List[int]] = Field(None, description="[n_rows, n_cols] after FE (train)")
    test_shape: Optional[List[int]] = Field(None, description="[n_rows, n_cols] after FE (test)")
    n_rows: Optional[int] = Field(None, description="Number of training rows")
    features_before_variance: Optional[int] = Field(None, description="Feature count before variance filter")
    features_after_variance: Optional[int] = Field(None, description="Feature count after variance filter")
    n_features_requested: Optional[int] = Field(None, description="How many features were requested for selection")
    rare_categories_grouped: Optional[List[str]] = Field(None, description="Columns where rare categories were grouped")
    execution_time_seconds: Optional[float] = Field(None, description="FE pipeline execution time")


# ── Endpoint Request Models ────────────────────────────────

class FeatureDeepAnalysisRequest(BaseModel):
    """Request for full feature engineering analysis."""
    feature_config: FeatureConfig = Field(..., description="FE configuration used")
    feature_results: FeatureResults = Field(..., description="FE pipeline results")
    project_id: Optional[str] = Field(None, description="Project ID for DB enrichment")
    dataset_id: Optional[str] = Field(None, description="Dataset ID for EDA data")
    model_id: Optional[str] = Field(None, description="Model ID for feature importance")


class FeatureQualityRequest(BaseModel):
    """Request for feature quality scorecard."""
    feature_config: FeatureConfig = Field(..., description="FE configuration used")
    feature_results: FeatureResults = Field(..., description="FE pipeline results")
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None


class FeatureSmartConfigRequest(BaseModel):
    """Request for AI-recommended FE settings."""
    feature_results: FeatureResults = Field(..., description="Current FE results (or basic data stats)")
    current_config: Optional[FeatureConfig] = Field(None, description="Current FE settings")
    project_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None


class FeatureCompareRequest(BaseModel):
    """Request to compare two FE configurations."""
    config_a: FeatureConfig
    results_a: FeatureResults
    config_b: FeatureConfig
    results_b: FeatureResults
    dataset_id: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# DB ENRICHMENT HELPERS
# ═══════════════════════════════════════════════════════════════

async def _get_eda_data(db, dataset_id: Optional[str]) -> Dict[str, Any]:
    """Fetch EDA results from database for enrichment."""
    if not dataset_id or not db:
        return {}

    try:
        from app.models.models import EdaResult
        eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
        if not eda:
            return {}

        result = {}
        for field in ["summary", "statistics", "quality", "correlations"]:
            raw = getattr(eda, field, None)
            if raw:
                try:
                    result[field] = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    result[field] = {}

        logger.info(f"[FE Enrichment] Loaded EDA data for dataset {dataset_id}: "
                    f"statistics={len(result.get('statistics', {}))}, "
                    f"correlations={'yes' if result.get('correlations') else 'no'}")
        return result

    except Exception as e:
        logger.warning(f"[FE Enrichment] Failed to load EDA data: {e}")
        return {}


async def _get_model_versions(db, project_id: Optional[str], model_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch model versions for feature importance data."""
    if not db or (not project_id and not model_id):
        return {}

    try:
        from app.models.models import RegisteredModel, ModelVersion

        query = db.query(ModelVersion)

        if model_id:
            query = query.filter(ModelVersion.model_id == model_id)
        elif project_id:
            # Get all models for this project
            model_ids = db.query(RegisteredModel.id).filter(
                RegisteredModel.project_id == project_id
            ).all()
            if model_ids:
                query = query.filter(ModelVersion.model_id.in_([m[0] for m in model_ids]))
            else:
                return {"versions": []}

        versions = query.order_by(ModelVersion.created_at.desc()).limit(20).all()

        version_list = []
        for v in versions:
            vd = {
                "algorithm": getattr(v, "algorithm", None),
                "accuracy": getattr(v, "accuracy", None),
                "test_score": getattr(v, "test_score", None),
                "train_score": getattr(v, "train_score", None),
                "feature_names": getattr(v, "feature_names", None),
                "feature_importances": getattr(v, "feature_importances", None),
                "confusion_matrix": getattr(v, "confusion_matrix", None),
                "hyperparameters": getattr(v, "hyperparameters", None),
            }

            # Parse JSON fields
            for field in ["feature_names", "feature_importances", "confusion_matrix", "hyperparameters"]:
                raw = vd.get(field)
                if isinstance(raw, str):
                    try:
                        vd[field] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass

            version_list.append(vd)

        has_importance = any(v.get("feature_importances") for v in version_list)
        logger.info(f"[FE Enrichment] Loaded {len(version_list)} model versions, "
                    f"feature_importances={'yes' if has_importance else 'no'}")
        return {"versions": version_list}

    except Exception as e:
        logger.warning(f"[FE Enrichment] Failed to load model versions: {e}")
        return {"versions": []}


async def _get_pipeline_state(db, project_id: Optional[str]) -> Dict[str, Any]:
    """Fetch pipeline execution state from jobs table."""
    if not db or not project_id:
        return {}

    try:
        from app.models.models import Job

        jobs = db.query(Job).filter(
            Job.project_id == project_id
        ).order_by(Job.created_at.desc()).limit(50).all()

        phases_completed = []
        phases_failed = []
        last_fe_job = None

        for job in jobs:
            pl = (getattr(job, "pipeline_name", None) or
                  getattr(job, "job_type", None) or "").lower()
            status = (getattr(job, "status", None) or "").lower()

            if "feature" in pl or "phase2" in pl:
                if status == "completed" and not last_fe_job:
                    last_fe_job = {
                        "status": status,
                        "duration": getattr(job, "duration_seconds", None),
                        "created_at": str(getattr(job, "created_at", "")),
                    }

            if status == "completed":
                phases_completed.append(pl)
            elif status == "failed":
                phases_failed.append(pl)

        return {
            "phases_completed": list(set(phases_completed)),
            "phases_failed": list(set(phases_failed)),
            "last_fe_job": last_fe_job,
        }

    except Exception as e:
        logger.warning(f"[FE Enrichment] Failed to load pipeline state: {e}")
        return {}


def _enrich_results_from_eda(results_dict: Dict, eda_data: Dict) -> Dict:
    """
    Enrich feature_results with EDA data that the frontend might not have sent.

    Frontend sends basic results. EDA table has column statistics, correlations, etc.
    """
    stats = eda_data.get("statistics") or {}
    quality = eda_data.get("quality") or {}
    summary = eda_data.get("summary") or {}

    # Enrich numeric features list from EDA if not provided
    if not results_dict.get("numeric_features") and stats:
        numeric = []
        categorical = []
        for col, cs in stats.items():
            if isinstance(cs, dict):
                dtype = cs.get("dtype", "").lower()
                if any(t in dtype for t in ["int", "float", "numeric"]):
                    numeric.append(col)
                elif any(t in dtype for t in ["object", "category", "string", "bool"]):
                    categorical.append(col)
        if numeric:
            results_dict["numeric_features"] = numeric
            logger.info(f"[FE Enrichment] Inferred {len(numeric)} numeric features from EDA")
        if categorical and not results_dict.get("categorical_features"):
            results_dict["categorical_features"] = categorical
            logger.info(f"[FE Enrichment] Inferred {len(categorical)} categorical features from EDA")

    # Enrich original_columns from EDA if not provided
    if not results_dict.get("original_columns") and stats:
        results_dict["original_columns"] = list(stats.keys())
        logger.info(f"[FE Enrichment] Inferred {len(stats)} original columns from EDA")

    # Enrich n_rows from summary if not provided
    if not results_dict.get("n_rows") and summary:
        n_rows = summary.get("total_rows") or summary.get("n_rows") or summary.get("row_count")
        if n_rows:
            results_dict["n_rows"] = int(n_rows)

    return results_dict


# ═══════════════════════════════════════════════════════════════
# ANALYZER INSTANCE
# ═══════════════════════════════════════════════════════════════

def _get_analyzer():
    """Lazy import and instantiate the FeatureAnalyzer."""
    from app.core.agent.feature_analyzer import FeatureAnalyzer
    return FeatureAnalyzer()


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════


@router.post("/deep-analysis")
async def feature_deep_analysis(request: FeatureDeepAnalysisRequest, db=Depends(get_db)):
    """
    Full feature engineering intelligence — combines all analyses.

    Returns:
      - transformation_audit: Grade A-F on every FE step
      - selection_explanation: SHAP-like feature selection reasons
      - error_patterns: Data leakage, multicollinearity, info loss
      - quality_score: 10-point scorecard with pass/fail
      - next_steps: Prioritized improvement roadmap
      - feature_interactions: Correlation clusters, interaction candidates
    """
    t0 = time.time()
    logger.info(f"[FE AI] deep-analysis called — project={request.project_id}, dataset={request.dataset_id}")

    try:
        # Fetch enrichment data from DB
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)
        pipeline_state = await _get_pipeline_state(db, request.project_id)

        # Convert request models to dicts
        config_dict = request.feature_config.model_dump()
        results_dict = request.feature_results.model_dump()

        # Enrich results from EDA
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        # Run deep analysis
        analyzer = _get_analyzer()
        result = analyzer.deep_analysis(
            feature_config=config_dict,
            feature_results=results_dict,
            eda_data=eda_data,
            model_versions=model_versions,
            pipeline_state=pipeline_state,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] deep-analysis completed in {elapsed}s — "
                    f"quality={result.get('quality_score', {}).get('percentage', '?')}%, "
                    f"issues={result.get('error_patterns', {}).get('total_issues', '?')}")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] deep-analysis error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/transformation-audit")
async def feature_transformation_audit(request: FeatureDeepAnalysisRequest, db=Depends(get_db)):
    """
    Audit every transformation decision — scaling, encoding, variance filter, etc.

    Returns per-step audit with score, findings, and recommendations.
    """
    t0 = time.time()
    logger.info(f"[FE AI] transformation-audit called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)

        config_dict = request.feature_config.model_dump()
        results_dict = request.feature_results.model_dump()
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        analyzer = _get_analyzer()
        result = analyzer.analyze_transformations(
            feature_config=config_dict,
            feature_results=results_dict,
            eda_data=eda_data,
            model_versions=model_versions,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] transformation-audit: grade={result.get('grade', '?')}, "
                    f"score={result.get('score', '?')}/{result.get('max_score', '?')}")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] transformation-audit error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/selection-explanation")
async def feature_selection_explanation(request: FeatureDeepAnalysisRequest, db=Depends(get_db)):
    """
    SHAP-like explanation of feature selection decisions.

    Returns:
      - selected_features: [{name, importance, rank, reason}]
      - dropped_features: [{name, drop_reason, importance_lost}]
      - information_retention_score
      - chart_data for visualization
    """
    t0 = time.time()
    logger.info(f"[FE AI] selection-explanation called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)

        results_dict = request.feature_results.model_dump()
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        analyzer = _get_analyzer()
        result = analyzer.explain_feature_selection(
            feature_results=results_dict,
            eda_data=eda_data,
            model_versions=model_versions,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] selection-explanation: {result.get('features_selected', '?')} selected, "
                    f"{result.get('features_dropped', '?')} dropped, "
                    f"retention={result.get('information_retention_score', '?')}%")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] selection-explanation error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/quality-score")
async def feature_quality_score(request: FeatureQualityRequest, db=Depends(get_db)):
    """
    10-point feature pipeline quality scorecard.

    Returns:
      - checks: [{id, name, passed, detail, weight, category}]
      - score/max_score/percentage
      - verdict: excellent|good|needs_improvement|poor
      - blockers and warnings
    """
    t0 = time.time()
    logger.info(f"[FE AI] quality-score called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)

        config_dict = request.feature_config.model_dump()
        results_dict = request.feature_results.model_dump()
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        analyzer = _get_analyzer()
        result = analyzer.assess_quality(
            feature_config=config_dict,
            feature_results=results_dict,
            eda_data=eda_data,
            model_versions=model_versions,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] quality-score: {result.get('score', '?')}/{result.get('max_score', '?')} "
                    f"({result.get('percentage', '?')}%) — verdict={result.get('verdict', '?')}")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] quality-score error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/error-patterns")
async def feature_error_patterns(request: FeatureDeepAnalysisRequest, db=Depends(get_db)):
    """
    Detect common feature engineering mistakes.

    Checks for:
      - Data leakage (target correlation, future data, ID columns)
      - Multicollinearity (highly correlated feature pairs)
      - Information loss (aggressive filtering)
      - Curse of dimensionality
      - Encoding issues
      - Train-test consistency
      - Scaling-algorithm mismatch
    """
    t0 = time.time()
    logger.info(f"[FE AI] error-patterns called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)

        config_dict = request.feature_config.model_dump()
        results_dict = request.feature_results.model_dump()
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        analyzer = _get_analyzer()
        result = analyzer.detect_error_patterns(
            feature_config=config_dict,
            feature_results=results_dict,
            eda_data=eda_data,
            model_versions=model_versions,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] error-patterns: {result.get('total_issues', 0)} issues, "
                    f"health={result.get('health', '?')}")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] error-patterns error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/smart-config")
async def feature_smart_config(request: FeatureSmartConfigRequest, db=Depends(get_db)):
    """
    AI-recommended feature engineering settings.

    Analyzes data characteristics and recommends optimal:
      - Scaling method
      - Encoding strategy
      - Missing value handling
      - Outlier handling
      - Feature selection count
      - Polynomial features (yes/no)
      - Interactions (yes/no)
    """
    t0 = time.time()
    logger.info(f"[FE AI] smart-config called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)

        results_dict = request.feature_results.model_dump()
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        current_dict = request.current_config.model_dump() if request.current_config else {}

        analyzer = _get_analyzer()
        result = analyzer.generate_smart_config(
            feature_results=results_dict,
            eda_data=eda_data,
            current_config=current_dict,
            model_versions=model_versions,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] smart-config: {result.get('n_changes', 0)} changes recommended")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] smart-config error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/next-steps")
async def feature_next_steps(request: FeatureDeepAnalysisRequest, db=Depends(get_db)):
    """
    Prioritized next steps for feature engineering improvement.

    Returns ordered list with priority (critical/high/medium/low),
    effort, and expected impact.
    """
    t0 = time.time()
    logger.info(f"[FE AI] next-steps called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id, request.model_id)
        pipeline_state = await _get_pipeline_state(db, request.project_id)

        config_dict = request.feature_config.model_dump()
        results_dict = request.feature_results.model_dump()
        results_dict = _enrich_results_from_eda(results_dict, eda_data)

        analyzer = _get_analyzer()
        result = analyzer.generate_next_steps(
            feature_config=config_dict,
            feature_results=results_dict,
            eda_data=eda_data,
            model_versions=model_versions,
            pipeline_state=pipeline_state,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] next-steps: {result.get('total_steps', 0)} steps, "
                    f"{result.get('critical_steps', 0)} critical")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] next-steps error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}


@router.post("/compare-configs")
async def feature_compare_configs(request: FeatureCompareRequest, db=Depends(get_db)):
    """
    Compare two feature engineering configurations side-by-side.

    Returns quality scores for both configs, differences, and a winner recommendation.
    """
    t0 = time.time()
    logger.info(f"[FE AI] compare-configs called")

    try:
        eda_data = await _get_eda_data(db, request.dataset_id)

        analyzer = _get_analyzer()
        result = analyzer.compare_configs(
            config_a=request.config_a.model_dump(),
            results_a=request.results_a.model_dump(),
            config_b=request.config_b.model_dump(),
            results_b=request.results_b.model_dump(),
            eda_data=eda_data,
        )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"[FE AI] compare-configs: winner={result.get('winner', '?')}")

        return {"status": "success", "data": result, "elapsed_seconds": elapsed}

    except Exception as e:
        logger.error(f"[FE AI] compare-configs error: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "data": None}