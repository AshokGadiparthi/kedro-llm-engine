"""
EDA Screen Enhancement Endpoints
==================================
New endpoints specifically for making the EDA screen world-class:

  GET  /eda/sample          — Sample rows from dataset (df.head())
  GET  /eda/distributions   — Per-feature distribution data for charts
  GET  /eda/correlations    — Full correlation matrix for heatmap
  POST /eda/suggestions     — Dynamic suggested questions based on insights
  GET  /eda/target-analysis — Auto-detect and analyze target column
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eda", tags=["EDA Enhancements"])


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _safe_json(val, default=None):
    if default is None:
        default = {}
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════
# 1. SAMPLE DATA — df.head() for the UI
# ═══════════════════════════════════════════════════════════════

class SampleDataResponse(BaseModel):
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    total_rows: int = 0
    total_columns: int = 0
    dtypes: Dict[str, str] = {}


@router.get("/sample", response_model=SampleDataResponse)
async def get_sample_data(
        dataset_id: str = Query(..., description="Dataset ID"),
        n: int = Query(5, ge=1, le=20, description="Number of sample rows"),
        db=Depends(get_db),
):
    """
    Return first N rows of the dataset for preview.
    Reads from the actual file_path stored in the datasets table.
    """
    try:
        from app.models.models import Dataset, EdaResult
        import os

        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not ds:
            return SampleDataResponse()

        # Try to read from file
        file_path = getattr(ds, "file_path", None)
        if file_path and os.path.exists(file_path):
            import pandas as pd
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, nrows=n)
                elif file_path.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(file_path, nrows=n)
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                    df = df.head(n)
                else:
                    df = pd.read_csv(file_path, nrows=n)

                # Get total row count from EDA or dataset
                eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
                total_rows = 0
                if eda and eda.summary:
                    summary = _safe_json(eda.summary)
                    shape = summary.get("shape", [0, 0])
                    if isinstance(shape, list) and len(shape) >= 2:
                        total_rows = int(shape[0])

                return SampleDataResponse(
                    columns=list(df.columns),
                    rows=df.head(n).to_dict(orient="records"),
                    total_rows=total_rows or len(df),
                    total_columns=len(df.columns),
                    dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                )
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")

        # Fallback: construct sample from EDA statistics
        eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
        if eda:
            summary = _safe_json(eda.summary)
            stats = _safe_json(eda.statistics)
            columns = summary.get("columns", [])
            shape = summary.get("shape", [0, 0])
            dtypes = summary.get("dtypes", {})
            return SampleDataResponse(
                columns=columns,
                rows=[],  # Can't reconstruct rows without file
                total_rows=int(shape[0]) if isinstance(shape, list) and shape else 0,
                total_columns=len(columns),
                dtypes=dtypes,
            )

        return SampleDataResponse()

    except Exception as e:
        logger.error(f"Sample data error: {e}", exc_info=True)
        return SampleDataResponse()


# ═══════════════════════════════════════════════════════════════
# 2. DISTRIBUTION DATA — For histograms and bar charts
# ═══════════════════════════════════════════════════════════════

class DistributionItem(BaseModel):
    column: str
    dtype: str  # "numeric" or "categorical"
    # For numeric: histogram bins
    histogram_bins: Optional[List[float]] = None
    histogram_counts: Optional[List[int]] = None
    # For categorical: value counts
    value_counts: Optional[Dict[str, int]] = None
    # Stats
    stats: Dict[str, Any] = {}


class DistributionsResponse(BaseModel):
    distributions: List[DistributionItem] = []
    total_features: int = 0


@router.get("/distributions", response_model=DistributionsResponse)
async def get_distributions(
        dataset_id: str = Query(..., description="Dataset ID"),
        max_features: int = Query(20, ge=1, le=50, description="Max features to return"),
        db=Depends(get_db),
):
    """
    Return distribution data for each feature.
    For numeric: histogram bins + counts.
    For categorical: top value counts.
    """
    try:
        from app.models.models import Dataset, EdaResult
        import os

        distributions = []

        # Try to read from file for real histograms
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = getattr(ds, "file_path", None) if ds else None

        if file_path and os.path.exists(file_path):
            import pandas as pd
            import numpy as np

            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(file_path)
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)

                for col in df.columns[:max_features]:
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            clean = df[col].dropna()
                            if len(clean) == 0:
                                continue
                            counts, bin_edges = np.histogram(clean, bins=min(30, len(clean.unique())))
                            distributions.append(DistributionItem(
                                column=col,
                                dtype="numeric",
                                histogram_bins=[round(float(b), 4) for b in bin_edges],
                                histogram_counts=[int(c) for c in counts],
                                stats={
                                    "mean": round(float(clean.mean()), 4),
                                    "median": round(float(clean.median()), 4),
                                    "std": round(float(clean.std()), 4),
                                    "min": round(float(clean.min()), 4),
                                    "max": round(float(clean.max()), 4),
                                    "skew": round(float(clean.skew()), 4) if len(clean) > 2 else 0,
                                    "missing": int(df[col].isna().sum()),
                                    "unique": int(clean.nunique()),
                                },
                            ))
                        else:
                            vc = df[col].value_counts().head(15).to_dict()
                            distributions.append(DistributionItem(
                                column=col,
                                dtype="categorical",
                                value_counts={str(k): int(v) for k, v in vc.items()},
                                stats={
                                    "unique": int(df[col].nunique()),
                                    "top": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                                    "missing": int(df[col].isna().sum()),
                                    "total": len(df),
                                },
                            ))
                    except Exception:
                        continue

                return DistributionsResponse(
                    distributions=distributions,
                    total_features=len(df.columns),
                )

            except Exception as e:
                logger.warning(f"Failed to compute distributions from file: {e}")

        # Fallback: construct from EDA statistics
        eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
        if eda:
            stats = _safe_json(eda.statistics)
            for col, cs in stats.get("numeric_statistics", {}).items():
                if not isinstance(cs, dict):
                    continue
                distributions.append(DistributionItem(
                    column=col,
                    dtype="numeric",
                    stats={
                        "mean": cs.get("mean"),
                        "median": cs.get("median", cs.get("50%")),
                        "std": cs.get("std"),
                        "min": cs.get("min"),
                        "max": cs.get("max"),
                        "q25": cs.get("q25", cs.get("25%")),
                        "q75": cs.get("q75", cs.get("75%")),
                    },
                ))
            for col, cs in stats.get("categorical_statistics", {}).items():
                if not isinstance(cs, dict):
                    continue
                distributions.append(DistributionItem(
                    column=col,
                    dtype="categorical",
                    stats={
                        "unique": cs.get("unique"),
                        "top": cs.get("top"),
                        "missing": cs.get("missing", 0),
                    },
                ))

        return DistributionsResponse(
            distributions=distributions[:max_features],
            total_features=len(distributions),
        )

    except Exception as e:
        logger.error(f"Distributions error: {e}", exc_info=True)
        return DistributionsResponse()


# ═══════════════════════════════════════════════════════════════
# 3. CORRELATION MATRIX — For heatmap visualization
# ═══════════════════════════════════════════════════════════════

class CorrelationMatrixResponse(BaseModel):
    columns: List[str] = []
    matrix: List[List[float]] = []  # 2D array
    high_pairs: List[Dict[str, Any]] = []
    method: str = "pearson"


@router.get("/correlations", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix(
        dataset_id: str = Query(..., description="Dataset ID"),
        db=Depends(get_db),
):
    """Return full correlation matrix for heatmap visualization."""
    try:
        from app.models.models import Dataset, EdaResult
        import os

        # Try to compute from file
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = getattr(ds, "file_path", None) if ds else None

        if file_path and os.path.exists(file_path):
            import pandas as pd
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(file_path)
                elif file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)

                numeric_df = df.select_dtypes(include=["number"])
                if numeric_df.shape[1] < 2:
                    return CorrelationMatrixResponse()

                # Limit to 20 columns for readability
                if numeric_df.shape[1] > 20:
                    numeric_df = numeric_df.iloc[:, :20]

                corr = numeric_df.corr(method="pearson")
                columns = list(corr.columns)
                matrix = [[round(float(v), 4) for v in row] for row in corr.values]

                # Find high pairs
                high_pairs = []
                for i in range(len(columns)):
                    for j in range(i + 1, len(columns)):
                        val = abs(corr.iloc[i, j])
                        if val >= 0.5:
                            high_pairs.append({
                                "feature1": columns[i],
                                "feature2": columns[j],
                                "correlation": round(float(corr.iloc[i, j]), 4),
                                "abs_correlation": round(float(val), 4),
                            })
                high_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

                return CorrelationMatrixResponse(
                    columns=columns,
                    matrix=matrix,
                    high_pairs=high_pairs[:20],
                    method="pearson",
                )
            except Exception as e:
                logger.warning(f"Failed to compute correlations from file: {e}")

        # Fallback: from EDA results
        eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
        if eda and eda.correlations:
            corr_data = _safe_json(eda.correlations)
            if isinstance(corr_data, dict):
                matrix_data = corr_data.get("matrix", {})
                if isinstance(matrix_data, dict):
                    columns = list(matrix_data.keys())
                    matrix = []
                    for col in columns:
                        row = matrix_data.get(col, {})
                        matrix.append([round(float(row.get(c, 0)), 4) for c in columns])
                    high_pairs = corr_data.get("high_pairs", [])
                    return CorrelationMatrixResponse(
                        columns=columns,
                        matrix=matrix,
                        high_pairs=high_pairs,
                    )

        return CorrelationMatrixResponse()

    except Exception as e:
        logger.error(f"Correlation matrix error: {e}", exc_info=True)
        return CorrelationMatrixResponse()


# ═══════════════════════════════════════════════════════════════
# 4. DYNAMIC SUGGESTIONS — Smart questions based on insights
# ═══════════════════════════════════════════════════════════════

class SuggestionsRequest(BaseModel):
    dataset_id: str
    insights: List[Dict[str, Any]] = Field(default=[], description="Pass insights from /insights response")
    screen: str = "eda"


class SuggestionsResponse(BaseModel):
    suggestions: List[Dict[str, str]] = []  # {text, context}


@router.post("/suggestions", response_model=SuggestionsResponse)
async def get_dynamic_suggestions(request: SuggestionsRequest, db=Depends(get_db)):
    """
    Generate smart suggested questions based on the actual insights fired.
    Instead of hardcoded "Focus on customer churn", this returns dataset-specific questions.
    """
    suggestions = []
    rule_ids = {i.get("rule_id", "") for i in request.insights}
    severities = {i.get("severity", "") for i in request.insights}
    categories = {i.get("category", "") for i in request.insights}

    # Priority-based suggestions
    if "critical" in severities:
        critical = [i for i in request.insights if i.get("severity") == "critical"]
        suggestions.append({
            "text": f"How do I fix the {len(critical)} critical issues first?",
            "context": "critical_issues",
        })

    # Data leakage questions
    if any(r.startswith("DL-") for r in rule_ids):
        leakage_cols = [
            i.get("evidence", "")
            for i in request.insights
            if i.get("rule_id", "").startswith("DL-")
        ]
        suggestions.append({
            "text": "Explain the data leakage risk and how to fix it",
            "context": "data_leakage",
        })

    # Feature engineering questions
    fe_count = sum(1 for r in rule_ids if r.startswith("FE-"))
    if fe_count > 0:
        suggestions.append({
            "text": f"Which of the {fe_count} feature engineering suggestions will have the most impact?",
            "context": "feature_engineering",
        })

    # High cardinality questions
    if "FH-004" in rule_ids:
        suggestions.append({
            "text": "What's the best encoding strategy for high-cardinality features?",
            "context": "encoding",
        })

    # Categorical-heavy dataset
    if "FH-006" in rule_ids:
        suggestions.append({
            "text": "Which algorithm handles categorical features best for my data?",
            "context": "algorithm_selection",
        })

    # Data quality issues
    if any(r.startswith("DQ-") and r not in ("DQ-003", "DQ-009") for r in rule_ids):
        suggestions.append({
            "text": "What data quality issues should I fix before training?",
            "context": "data_quality",
        })

    # Target variable questions
    if any(r.startswith("TV-") for r in rule_ids):
        suggestions.append({
            "text": "Is my target variable suitable for the prediction task?",
            "context": "target",
        })

    # Sample size questions
    if any(r.startswith("SS-") for r in rule_ids):
        suggestions.append({
            "text": "Is my dataset large enough? What techniques work for small data?",
            "context": "sample_size",
        })

    # Multicollinearity
    if any(r.startswith("MC-") for r in rule_ids):
        suggestions.append({
            "text": "How should I handle the multicollinearity in my features?",
            "context": "multicollinearity",
        })

    # Generic fallbacks if few suggestions
    if len(suggestions) < 3:
        if "Feature Engineering" in categories:
            suggestions.append({
                "text": "What's the fastest path from EDA to a production model?",
                "context": "roadmap",
            })
        suggestions.append({
            "text": "What accuracy should I expect with this data?",
            "context": "performance_estimate",
        })
        suggestions.append({
            "text": "What hyperparameters should I tune first?",
            "context": "hyperparameters",
        })

    return SuggestionsResponse(suggestions=suggestions[:5])


# ═══════════════════════════════════════════════════════════════
# 5. TARGET ANALYSIS — Auto-detect and analyze target column
# ═══════════════════════════════════════════════════════════════

class TargetAnalysisResponse(BaseModel):
    detected_target: Optional[str] = None
    detection_method: Optional[str] = None  # "name_pattern", "binary_column", "user_specified"
    target_type: Optional[str] = None  # "binary", "multiclass", "regression"
    class_distribution: Optional[Dict[str, int]] = None
    class_balance_ratio: Optional[float] = None  # min/max class ratio
    is_imbalanced: bool = False
    candidates: List[Dict[str, Any]] = []  # Other possible targets


@router.get("/target-analysis", response_model=TargetAnalysisResponse)
async def get_target_analysis(
        dataset_id: str = Query(..., description="Dataset ID"),
        target_column: Optional[str] = Query(None, description="Override target column"),
        db=Depends(get_db),
):
    """
    Auto-detect the target column and analyze its distribution.
    This fills the 'gap' that the rules engine identified.
    """
    try:
        from app.models.models import Dataset, EdaResult
        import os

        result = TargetAnalysisResponse()

        # First check EDA results for potential targets
        eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
        if not eda:
            return result

        summary = _safe_json(eda.summary)
        stats = _safe_json(eda.statistics)
        columns = summary.get("columns", [])
        col_types = summary.get("column_types", {})
        numeric_cols = col_types.get("numeric", [])
        cat_cols = col_types.get("categorical", [])

        # Auto-detect target candidates
        target_patterns = ["target", "label", "class", "churn", "fraud", "outcome",
                           "default", "survived", "y", "is_", "has_", "attrition",
                           "response", "result", "status", "diagnosis"]
        candidates = []

        for col in columns:
            cl = col.lower().strip()
            score = 0
            method = None

            # Name pattern matching
            if any(p in cl for p in target_patterns):
                score += 3
                method = "name_pattern"

            # Binary columns
            cat_stats = stats.get("categorical_statistics", {}).get(col, {})
            num_stats = stats.get("numeric_statistics", {}).get(col, {})

            if isinstance(cat_stats, dict) and cat_stats.get("unique") == 2:
                score += 2
                method = method or "binary_categorical"
            elif isinstance(num_stats, dict):
                mn = num_stats.get("min", -1)
                mx = num_stats.get("max", -1)
                if mn == 0 and mx == 1:
                    score += 2
                    method = method or "binary_numeric"

            # Last column heuristic (common in ML datasets)
            if col == columns[-1] and score == 0:
                score += 1
                method = "last_column"

            if score > 0:
                candidates.append({
                    "column": col,
                    "score": score,
                    "method": method,
                    "dtype": "numeric" if col in numeric_cols else "categorical",
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        result.candidates = candidates[:5]

        # Use specified target or best candidate
        target = target_column
        if not target and candidates:
            target = candidates[0]["column"]
            result.detection_method = candidates[0]["method"]

        if not target:
            return result

        result.detected_target = target

        # Analyze target distribution from file if possible
        ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        file_path = getattr(ds, "file_path", None) if ds else None

        if file_path and os.path.exists(file_path):
            import pandas as pd
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, usecols=[target])
                elif file_path.endswith((".xls", ".xlsx")):
                    df = pd.read_excel(file_path, usecols=[target])
                else:
                    df = pd.read_csv(file_path, usecols=[target])

                if target in df.columns:
                    vc = df[target].value_counts()
                    n_unique = len(vc)

                    if n_unique <= 20:
                        result.class_distribution = {str(k): int(v) for k, v in vc.items()}
                        result.target_type = "binary" if n_unique == 2 else "multiclass"

                        min_count = int(vc.min())
                        max_count = int(vc.max())
                        result.class_balance_ratio = round(min_count / max_count, 3) if max_count > 0 else 0
                        result.is_imbalanced = result.class_balance_ratio < 0.4
                    else:
                        result.target_type = "regression"
                        result.class_distribution = {
                            "min": round(float(df[target].min()), 4),
                            "max": round(float(df[target].max()), 4),
                            "mean": round(float(df[target].mean()), 4),
                            "std": round(float(df[target].std()), 4),
                        }
            except Exception as e:
                logger.warning(f"Target analysis from file failed: {e}")

        return result

    except Exception as e:
        logger.error(f"Target analysis error: {e}", exc_info=True)
        return TargetAnalysisResponse()