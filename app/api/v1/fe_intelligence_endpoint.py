"""
Feature Engineering Pipeline Intelligence API Endpoint
=======================================================
NEW endpoint: /api/v1/features/pipeline-intelligence

Provides the world-class FE pipeline analysis that catches every issue
a senior ML scientist would find in a pipeline review.

Adds to the existing 7 endpoints:
  POST /features/deep-analysis
  POST /features/transformation-audit
  POST /features/selection-explanation
  POST /features/quality-score
  POST /features/error-patterns
  POST /features/smart-config
  POST /features/next-steps
  POST /features/compare-configs

NEW:
  POST /features/pipeline-intelligence   ← THE BIG ONE
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/features", tags=["Feature Intelligence"])


# ── Request Model ──

class PipelineIntelligenceRequest(BaseModel):
    """
    Request model for the pipeline intelligence endpoint.

    The frontend sends everything available from the FE pipeline execution.
    Any field can be null — the analyzer handles missing data gracefully.
    """
    # ── Pipeline Configuration (what the user configured) ──
    scaling_method: str = "standard"
    handle_missing_values: bool = True
    handle_outliers: bool = True
    encode_categories: bool = True
    create_polynomial_features: bool = False
    create_interactions: bool = False
    variance_threshold: float = 0.01
    n_features_to_select: Optional[int] = None

    # ── Pipeline Results (what actually happened) ──
    original_columns: List[str] = []
    selected_features: List[str] = []
    final_columns: List[str] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    variance_removed: List[str] = []
    id_columns_detected: List[str] = []
    original_shape: Optional[List[int]] = None      # [n_rows, n_cols]
    final_shape: Optional[List[int]] = None          # [n_rows, n_final_features]
    train_shape: Optional[List[int]] = None
    test_shape: Optional[List[int]] = None
    n_rows: Optional[int] = None
    n_numeric: Optional[int] = None
    n_categorical: Optional[int] = None
    execution_time_seconds: Optional[float] = None
    target_column: Optional[str] = None

    # ── Encoding Details (per-column) ──
    encoding_details: Dict[str, Any] = {}
    # Example: {"TotalCharges": {"unique_values": 6531, "rare_grouped": 5276, "strategy": "one_hot"}}

    # ── Feature Selection Details ──
    features_before_variance: Optional[int] = None
    features_after_variance: Optional[int] = None
    features_input_to_selection: Optional[int] = None
    n_selected: Optional[int] = None

    # ── DB References (for enrichment) ──
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None
    model_id: Optional[str] = None
    job_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "scaling_method": "standard",
                "handle_missing_values": True,
                "handle_outliers": True,
                "encode_categories": True,
                "create_polynomial_features": False,
                "create_interactions": False,
                "original_columns": [
                    "customerID", "gender", "SeniorCitizen", "Partner",
                    "Dependents", "tenure", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies", "Contract", "PaperlessBilling",
                    "PaymentMethod", "MonthlyCharges", "TotalCharges"
                ],
                "categorical_features": [
                    "customerID", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaymentMethod", "TotalCharges"
                ],
                "numeric_features": [
                    "gender", "SeniorCitizen", "Partner", "Dependents",
                    "tenure", "PhoneService", "PaperlessBilling", "MonthlyCharges"
                ],
                "variance_removed": [
                    "gender_scaled", "Partner_scaled", "Dependents_scaled",
                    "PhoneService_scaled", "PaperlessBilling_scaled"
                ],
                "id_columns_detected": [],
                "original_shape": [5634, 20],
                "final_shape": [5634, 10],
                "train_shape": [5634, 10],
                "test_shape": [1409, 10],
                "encoding_details": {
                    "customerID": {
                        "unique_values": 5634,
                        "rare_grouped": 5634,
                        "strategy": "one_hot (1 categories)"
                    },
                    "TotalCharges": {
                        "unique_values": 6531,
                        "rare_grouped": 5276,
                        "strategy": "one_hot (1 categories)"
                    },
                    "MultipleLines": {
                        "unique_values": 3,
                        "rare_grouped": 0,
                        "strategy": "one_hot (3 categories)"
                    },
                    "Contract": {
                        "unique_values": 3,
                        "rare_grouped": 0,
                        "strategy": "one_hot (3 categories)"
                    }
                },
                "execution_time_seconds": 5.13,
                "features_before_variance": 29,
                "features_after_variance": 24,
                "n_selected": 10
            }
        }


# ── Helper: Get DB data ──

async def _get_eda_data(db, dataset_id: Optional[str]) -> Dict[str, Any]:
    """Fetch EDA data from database for enrichment."""
    if not dataset_id or not db:
        return {}
    try:
        from sqlalchemy import text
        result = db.execute(text(
            "SELECT statistics, correlations, data_quality "
            "FROM eda_results WHERE dataset_id = :did "
            "ORDER BY created_at DESC LIMIT 1"
        ), {"did": dataset_id})
        row = result.fetchone()
        if row:
            import json
            return {
                "statistics": json.loads(row[0]) if row[0] else {},
                "correlations": json.loads(row[1]) if row[1] else {},
                "data_quality": json.loads(row[2]) if row[2] else {},
            }
    except Exception as e:
        logger.warning(f"EDA data fetch error: {e}")
    return {}


async def _get_model_versions(db, project_id: Optional[str]) -> Dict[str, Any]:
    """Fetch model versions for feature importance cross-reference."""
    if not project_id or not db:
        return {}
    try:
        from sqlalchemy import text
        result = db.execute(text(
            "SELECT feature_names, feature_importances "
            "FROM model_versions WHERE project_id = :pid "
            "ORDER BY created_at DESC LIMIT 5"
        ), {"pid": project_id})
        rows = result.fetchall()
        if rows:
            import json
            return {
                "versions": [
                    {
                        "feature_names": json.loads(r[0]) if r[0] else [],
                        "feature_importances": json.loads(r[1]) if r[1] else {},
                    }
                    for r in rows
                ]
            }
    except Exception as e:
        logger.warning(f"Model versions fetch error: {e}")
    return {}


async def _get_job_history(db, project_id: Optional[str]) -> List[Dict]:
    """Fetch previous FE pipeline jobs for comparison."""
    if not project_id or not db:
        return []
    try:
        from sqlalchemy import text
        result = db.execute(text(
            "SELECT config, metrics, duration_seconds, status, created_at "
            "FROM jobs WHERE project_id = :pid AND pipeline = 'feature_engineering' "
            "ORDER BY created_at DESC LIMIT 10"
        ), {"pid": project_id})
        rows = result.fetchall()
        if rows:
            import json
            return [
                {
                    "config": json.loads(r[0]) if r[0] else {},
                    "metrics": json.loads(r[1]) if r[1] else {},
                    "duration_seconds": r[2],
                    "status": r[3],
                    "created_at": str(r[4]),
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"Job history fetch error: {e}")
    return []


# ── Dependency to get DB session ──

def get_db():
    """Get database session — must be provided by the app."""
    try:
        from app.database import get_db as _get_db
        return next(_get_db())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# THE MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════

@router.post("/pipeline-intelligence")
async def feature_pipeline_intelligence(
        request: PipelineIntelligenceRequest,
        db=Depends(get_db),
):
    """
    🧠 World-Class Feature Engineering Pipeline Intelligence

    Analyzes the complete FE pipeline execution with the depth and rigor
    of a senior Staff ML Scientist. Returns:

    - **findings**: Expert-level findings with severity, explanation, fix, and ML theory
    - **quality_score**: 0-100 pipeline quality score across 10 dimensions
    - **quality_grade**: A+ to F letter grade
    - **critical_blockers**: Issues that MUST be fixed before training
    - **quick_wins**: Easy improvements with high impact
    - **optimal_config**: AI-recommended configuration changes
    - **domain_detected**: Auto-detected ML domain (churn, fraud, credit, etc.)
    - **executive_summary**: One-paragraph summary for stakeholders

    Send ALL available pipeline execution data — the more data, the deeper
    the analysis. Every field is optional; the analyzer adapts gracefully.
    """
    try:
        # ── Build config dict ──
        config = {
            "scaling_method": request.scaling_method,
            "handle_missing_values": request.handle_missing_values,
            "handle_outliers": request.handle_outliers,
            "encode_categories": request.encode_categories,
            "create_polynomial_features": request.create_polynomial_features,
            "create_interactions": request.create_interactions,
            "variance_threshold": request.variance_threshold,
            "n_features_to_select": request.n_features_to_select,
        }

        # ── Build results dict ──
        results = {
            "original_columns": request.original_columns,
            "selected_features": request.selected_features,
            "final_columns": request.final_columns,
            "numeric_features": request.numeric_features,
            "categorical_features": request.categorical_features,
            "variance_removed": request.variance_removed,
            "id_columns_detected": request.id_columns_detected,
            "original_shape": request.original_shape,
            "final_shape": request.final_shape,
            "train_shape": request.train_shape,
            "test_shape": request.test_shape,
            "n_rows": request.n_rows,
            "n_numeric": request.n_numeric,
            "n_categorical": request.n_categorical,
            "execution_time_seconds": request.execution_time_seconds,
            "target_column": request.target_column,
            "encoding_details": request.encoding_details,
            "features_before_variance": request.features_before_variance,
            "features_after_variance": request.features_after_variance,
            "features_input_to_selection": request.features_input_to_selection,
            "n_selected": request.n_selected,
            "train_rows": request.n_rows or (request.train_shape[0] if request.train_shape else 0),
        }

        # ── Fetch enrichment data from DB ──
        eda_data = await _get_eda_data(db, request.dataset_id)
        model_versions = await _get_model_versions(db, request.project_id)
        job_history = await _get_job_history(db, request.project_id)

        # ── Run the Intelligence Engine ──
        from app.core.agent.fe_pipeline_intelligence import FEPipelineIntelligence

        engine = FEPipelineIntelligence()
        analysis = engine.analyze_pipeline(
            config=config,
            results=results,
            eda_data=eda_data,
            model_versions=model_versions,
            job_history=job_history,
        )

        return {
            "status": "success",
            "analysis": analysis,
        }

    except Exception as e:
        logger.error(f"Pipeline intelligence error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "analysis": None,
        }


# ═══════════════════════════════════════════════════════════════
# QUICK ANALYSIS ENDPOINT (lightweight version)
# ═══════════════════════════════════════════════════════════════

class QuickAnalysisRequest(BaseModel):
    """Lightweight request for quick pipeline health check."""
    original_columns: List[str] = []
    categorical_features: List[str] = []
    variance_removed: List[str] = []
    id_columns_detected: List[str] = []
    encoding_details: Dict[str, Any] = {}
    final_shape: Optional[List[int]] = None
    scaling_method: str = "standard"


@router.post("/quick-health")
async def feature_quick_health(request: QuickAnalysisRequest):
    """
    Quick pipeline health check — returns only critical/warning findings.
    Faster than full pipeline-intelligence, suitable for real-time UI feedback.
    """
    try:
        from app.core.agent.fe_pipeline_intelligence import FEPipelineIntelligence

        config = {"scaling_method": request.scaling_method}
        results = {
            "original_columns": request.original_columns,
            "categorical_features": request.categorical_features,
            "variance_removed": request.variance_removed,
            "id_columns_detected": request.id_columns_detected,
            "encoding_details": request.encoding_details,
            "final_shape": request.final_shape,
        }

        engine = FEPipelineIntelligence()

        # Run only the fast analyzers
        findings = []
        findings.extend(engine.detect_type_misclassification(config, results, {}))
        findings.extend(engine.audit_id_detection(config, results, {}))
        findings.extend(engine.audit_variance_filter(config, results, {}))

        # Filter to critical and warning only
        urgent = [f for f in findings if f.get("severity") in ("critical", "warning")]

        return {
            "status": "success",
            "health": "healthy" if not urgent else "needs_attention",
            "critical_count": sum(1 for f in urgent if f["severity"] == "critical"),
            "warning_count": sum(1 for f in urgent if f["severity"] == "warning"),
            "urgent_findings": urgent,
        }

    except Exception as e:
        logger.error(f"Quick health error: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}