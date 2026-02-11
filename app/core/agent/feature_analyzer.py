"""
Feature Engineering Analyzer — World-Class AI Intelligence
===================================================================
Deep analysis of feature transformations, selection decisions,
data quality, error patterns, and optimization recommendations.

Methods (all pure-Python, no ML libraries required):
  • analyze_transformations()     — Audit every FE decision (scaling, encoding, variance)
  • explain_feature_selection()   — SHAP-like: why each feature was selected/dropped
  • detect_error_patterns()       — Data leakage, multicollinearity, information loss
  • assess_quality()              — 10-point feature pipeline quality scorecard
  • generate_smart_config()       — AI-recommended FE settings for this data
  • generate_next_steps()         — Prioritized improvement roadmap
  • analyze_feature_interactions()— Correlation clusters, interaction candidates
  • compare_configs()             — Side-by-side FE configuration comparison

Data Sources:
  Frontend:  feature_config, feature_results, column_stats
  DB:        eda_results (statistics, correlations, quality)
             model_versions (feature_names, feature_importances)
             jobs (FE pipeline history)
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS & DOMAIN KNOWLEDGE
# ═══════════════════════════════════════════════════════════════

# Scaling method recommendations by data characteristic
SCALER_RECOMMENDATIONS = {
    "has_outliers": {
        "best": "robust",
        "reason": "RobustScaler uses median/IQR — resistant to outliers",
        "avoid": "standard",
        "avoid_reason": "StandardScaler uses mean/std — sensitive to outliers",
    },
    "gaussian_like": {
        "best": "standard",
        "reason": "StandardScaler works best with normally distributed data",
        "avoid": "minmax",
        "avoid_reason": "MinMax compresses the range but doesn't normalize the distribution",
    },
    "bounded_range": {
        "best": "minmax",
        "reason": "MinMaxScaler preserves the bounded structure [0,1]",
        "avoid": "standard",
        "avoid_reason": "StandardScaler can produce values outside original bounds",
    },
    "skewed": {
        "best": "robust",
        "reason": "RobustScaler handles skewness better via median centering",
        "avoid": "standard",
        "avoid_reason": "StandardScaler is distorted by the skewed tail",
    },
    "default": {
        "best": "standard",
        "reason": "StandardScaler is a safe default for most datasets",
    },
}

# Encoding recommendations by cardinality
ENCODING_THRESHOLDS = {
    "binary": 2,           # 2 values → binary/label
    "low_card": 10,        # ≤10 → one-hot
    "medium_card": 50,     # 11-50 → target or ordinal
    "high_card": 200,      # 51-200 → target encoding
    "very_high": 201,      # 201+ → hashing or embedding
}

# Variance threshold guidance
VARIANCE_THRESHOLDS = {
    "aggressive": 0.05,    # Removes many features
    "balanced": 0.01,      # Default — good balance
    "conservative": 0.001, # Keeps most features
}

# Feature selection guidance
SELECTION_RATIOS = {
    "very_small": (0, 20),       # <20 cols: keep 60-80%
    "small": (20, 50),           # 20-50 cols: keep 40-60%
    "medium": (50, 200),         # 50-200 cols: keep 20-40%
    "large": (200, 1000),        # 200-1000 cols: keep 10-20%
    "very_large": (1000, 99999), # 1000+ cols: keep 5-10%
}

# Common data leakage patterns
LEAKAGE_PATTERNS = [
    {"pattern": "id", "fields": ["id", "_id", "key", "index", "row_num", "record"],
     "severity": "critical", "message": "ID columns should never be features — they have no predictive power and cause overfitting"},
    {"pattern": "target_derivative", "fields": ["target", "label", "y_", "outcome", "result", "status"],
     "severity": "critical", "message": "This column name suggests it may be derived from the target variable"},
    {"pattern": "future_data", "fields": ["next_", "future_", "will_", "predicted_", "expected_"],
     "severity": "critical", "message": "Column name suggests future information not available at prediction time"},
    {"pattern": "timestamp_exact", "fields": ["created_at", "updated_at", "timestamp", "datetime"],
     "severity": "warning", "message": "Exact timestamps often cause overfitting — extract components (hour, day, month) instead"},
    {"pattern": "aggregation", "fields": ["total_", "sum_", "count_", "avg_", "mean_", "cumulative_"],
     "severity": "info", "message": "Aggregated columns may contain leakage if computed using the target or future data"},
]

# Quality scorecard weights
QUALITY_CHECKS = {
    "scaling_appropriate":      {"weight": 2, "category": "Transformation Quality"},
    "encoding_appropriate":     {"weight": 2, "category": "Transformation Quality"},
    "missing_values_handled":   {"weight": 1, "category": "Data Completeness"},
    "outliers_handled":         {"weight": 1, "category": "Data Completeness"},
    "no_data_leakage":          {"weight": 3, "category": "Data Integrity"},
    "no_high_multicollinearity":{"weight": 2, "category": "Feature Independence"},
    "feature_selection_done":   {"weight": 1, "category": "Dimensionality"},
    "variance_filter_applied":  {"weight": 1, "category": "Dimensionality"},
    "information_retention":    {"weight": 2, "category": "Information Preservation"},
    "production_reproducible":  {"weight": 1, "category": "Production Readiness"},
}


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _safe_float(val, default=None):
    """Convert value to float, return default if not possible."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _pct(value, total, decimals=1):
    """Safe percentage calculation."""
    if not total:
        return 0
    return round(value / total * 100, decimals)


def _severity_rank(s):
    return {"critical": 0, "warning": 1, "info": 2, "tip": 3}.get(s, 9)


def _classify_cardinality(unique_count):
    """Classify column cardinality for encoding strategy."""
    if unique_count <= ENCODING_THRESHOLDS["binary"]:
        return "binary"
    elif unique_count <= ENCODING_THRESHOLDS["low_card"]:
        return "low_cardinality"
    elif unique_count <= ENCODING_THRESHOLDS["medium_card"]:
        return "medium_cardinality"
    elif unique_count <= ENCODING_THRESHOLDS["high_card"]:
        return "high_cardinality"
    else:
        return "very_high_cardinality"


def _ideal_encoding(cardinality_class, n_rows):
    """Recommend ideal encoding for a cardinality class."""
    strategies = {
        "binary": ("label_encoding", "Simple 0/1 mapping — no dimensionality increase"),
        "low_cardinality": ("one_hot", "One-hot encoding — creates interpretable binary columns"),
        "medium_cardinality": ("target_encoding", "Target encoding — captures target relationship without dimension explosion"),
        "high_cardinality": ("target_encoding", "Target encoding or frequency encoding — one-hot would create too many columns"),
        "very_high_cardinality": ("hashing_or_embedding", "Feature hashing or learned embeddings — too many categories for traditional encoding"),
    }
    return strategies.get(cardinality_class, ("one_hot", "Default"))


def _ideal_scaler(column_stats):
    """Recommend ideal scaler based on column statistics."""
    if not column_stats:
        return SCALER_RECOMMENDATIONS["default"]

    has_outliers = column_stats.get("has_outliers", False)
    skewness = _safe_float(column_stats.get("skewness"), 0)
    is_bounded = column_stats.get("is_bounded", False)

    if has_outliers:
        return SCALER_RECOMMENDATIONS["has_outliers"]
    elif is_bounded:
        return SCALER_RECOMMENDATIONS["bounded_range"]
    elif abs(skewness) > 1.0:
        return SCALER_RECOMMENDATIONS["skewed"]
    else:
        return SCALER_RECOMMENDATIONS["gaussian_like"]


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════

class FeatureAnalyzer:
    """
    World-class Feature Engineering intelligence engine.

    All methods are pure-Python, stateless, and deterministic.
    No ML libraries required — works with metadata only.
    """

    # ──────────────────────────────────────────────────────────
    # 1. TRANSFORMATION AUDIT
    # ──────────────────────────────────────────────────────────

    def analyze_transformations(
            self,
            feature_config: Dict[str, Any],
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Audit every transformation decision.

        Returns per-step analysis:
          - scaling_audit: Was the scaler appropriate for this data?
          - encoding_audit: Was each column encoded optimally?
          - variance_audit: Was the variance threshold right?
          - missing_values_audit: Was imputation handled well?
          - outlier_audit: Were outliers handled appropriately?
          - id_detection_audit: Were ID columns correctly identified?
          - overall_grade: A-F letter grade
        """
        config = feature_config or {}
        results = feature_results or {}
        eda = eda_data or {}

        audits = []
        total_score = 0
        max_score = 0

        # ── Scaling Audit ─────────────────────────────
        scaling_audit = self._audit_scaling(config, results, eda)
        audits.append(scaling_audit)
        total_score += scaling_audit["score"]
        max_score += scaling_audit["max_score"]

        # ── Encoding Audit ────────────────────────────
        encoding_audit = self._audit_encoding(config, results, eda)
        audits.append(encoding_audit)
        total_score += encoding_audit["score"]
        max_score += encoding_audit["max_score"]

        # ── Variance Filter Audit ─────────────────────
        variance_audit = self._audit_variance_filter(config, results)
        audits.append(variance_audit)
        total_score += variance_audit["score"]
        max_score += variance_audit["max_score"]

        # ── Missing Values Audit ──────────────────────
        missing_audit = self._audit_missing_values(config, results, eda)
        audits.append(missing_audit)
        total_score += missing_audit["score"]
        max_score += missing_audit["max_score"]

        # ── Outlier Handling Audit ────────────────────
        outlier_audit = self._audit_outlier_handling(config, results, eda)
        audits.append(outlier_audit)
        total_score += outlier_audit["score"]
        max_score += outlier_audit["max_score"]

        # ── ID Column Detection Audit ─────────────────
        id_audit = self._audit_id_detection(results, eda)
        audits.append(id_audit)
        total_score += id_audit["score"]
        max_score += id_audit["max_score"]

        # ── Feature Selection Audit ───────────────────
        selection_audit = self._audit_feature_selection(results, eda, model_versions)
        audits.append(selection_audit)
        total_score += selection_audit["score"]
        max_score += selection_audit["max_score"]

        # ── Overall Grade ─────────────────────────────
        pct = _pct(total_score, max_score) if max_score > 0 else 0
        grade = (
            "A+" if pct >= 95 else "A" if pct >= 90 else "A-" if pct >= 85
            else "B+" if pct >= 80 else "B" if pct >= 75 else "B-" if pct >= 70
            else "C+" if pct >= 65 else "C" if pct >= 60 else "C-" if pct >= 55
            else "D" if pct >= 50 else "F"
        )

        return {
            "audits": audits,
            "score": total_score,
            "max_score": max_score,
            "percentage": pct,
            "grade": grade,
            "summary": self._generate_transformation_summary(audits, grade, pct),
        }

    def _audit_scaling(self, config, results, eda):
        """Audit scaling method choice."""
        method = (config.get("scaling_method") or "standard").lower()
        numeric_cols = results.get("numeric_features") or []
        col_stats = eda.get("statistics") or {}

        findings = []
        score = 0
        max_score = 3

        if not numeric_cols:
            return {"step": "scaling", "title": "Numeric Scaling", "score": max_score, "max_score": max_score,
                    "status": "skipped", "findings": [{"severity": "info", "message": "No numeric features to scale"}]}

        # Check if data has outliers
        has_outlier_cols = []
        has_skewed_cols = []
        for col in numeric_cols:
            cs = col_stats.get(col, {})
            if isinstance(cs, dict):
                skew = _safe_float(cs.get("skewness"), 0)
                iqr_outliers = _safe_float(cs.get("outlier_count") or cs.get("outliers"), 0)
                if iqr_outliers and iqr_outliers > 0:
                    has_outlier_cols.append(col)
                if abs(skew) > 1.0:
                    has_skewed_cols.append(col)

        # Evaluate choice
        if has_outlier_cols and method == "standard":
            findings.append({
                "severity": "warning",
                "message": f"StandardScaler used but {len(has_outlier_cols)} columns have outliers — "
                           f"RobustScaler would be more appropriate",
                "affected_columns": has_outlier_cols[:5],
                "recommendation": "Switch to RobustScaler (uses median/IQR instead of mean/std)",
            })
            score = 1
        elif has_skewed_cols and method == "minmax":
            findings.append({
                "severity": "info",
                "message": f"MinMaxScaler used but {len(has_skewed_cols)} columns are skewed — "
                           f"consider RobustScaler or log-transform first",
                "affected_columns": has_skewed_cols[:5],
            })
            score = 2
        elif method in ("standard", "robust", "minmax"):
            score = 3
            findings.append({
                "severity": "pass",
                "message": f"{method.title()}Scaler is appropriate for this data — "
                           f"scaled {len(numeric_cols)} numeric features",
            })
        else:
            score = 2
            findings.append({
                "severity": "info",
                "message": f"Scaling method '{method}' applied to {len(numeric_cols)} features",
            })

        # Additional check: are binary columns being scaled?
        binary_scaled = []
        for col in numeric_cols:
            cs = col_stats.get(col, {})
            if isinstance(cs, dict):
                unique = _safe_int(cs.get("unique_count") or cs.get("distinct"), 0)
                if unique == 2:
                    binary_scaled.append(col)

        if binary_scaled:
            findings.append({
                "severity": "info",
                "message": f"{len(binary_scaled)} binary columns are being scaled — "
                           f"this is unnecessary but harmless for most algorithms",
                "affected_columns": binary_scaled[:5],
            })

        status = "optimal" if score == max_score else "acceptable" if score >= 2 else "needs_improvement"
        return {
            "step": "scaling", "title": "Numeric Scaling",
            "score": score, "max_score": max_score,
            "status": status,
            "method_used": method,
            "columns_scaled": len(numeric_cols),
            "findings": findings,
        }

    def _audit_encoding(self, config, results, eda):
        """Audit categorical encoding decisions."""
        encode_enabled = config.get("encode_categories", True)
        categorical_cols = results.get("categorical_features") or []
        encoding_strategies = results.get("encoding_strategies") or {}
        col_stats = eda.get("statistics") or {}
        n_rows = _safe_int(results.get("n_rows") or results.get("original_shape", [0])[0], 1000)

        findings = []
        score = 0
        max_score = 3

        if not encode_enabled:
            return {"step": "encoding", "title": "Categorical Encoding", "score": 0, "max_score": max_score,
                    "status": "skipped",
                    "findings": [{"severity": "warning", "message": "Categorical encoding disabled — most algorithms require numeric input"}]}

        if not categorical_cols:
            return {"step": "encoding", "title": "Categorical Encoding", "score": max_score, "max_score": max_score,
                    "status": "skipped",
                    "findings": [{"severity": "info", "message": "No categorical features found in the data"}]}

        # Analyze each column's encoding choice
        suboptimal_encodings = []
        good_encodings = 0
        dimension_explosion = []

        for col in categorical_cols:
            cs = col_stats.get(col, {})
            unique = _safe_int(cs.get("unique_count") or cs.get("distinct"), 0) if isinstance(cs, dict) else 0
            if unique == 0:
                # Try from results
                unique = _safe_int((results.get("column_cardinalities") or {}).get(col), 0)

            card_class = _classify_cardinality(unique)
            ideal_enc, ideal_reason = _ideal_encoding(card_class, n_rows)
            actual_enc = encoding_strategies.get(col, "one_hot").lower().replace("-", "_").replace(" ", "_")

            # Check dimension explosion
            if unique > 10 and "one_hot" in actual_enc:
                dimension_explosion.append({"column": col, "categories": unique,
                                            "columns_created": unique - 1})

            # Check suboptimal
            if card_class in ("high_cardinality", "very_high_cardinality") and "one_hot" in actual_enc:
                suboptimal_encodings.append({
                    "column": col, "unique": unique, "used": actual_enc,
                    "recommended": ideal_enc, "reason": ideal_reason,
                })
            elif card_class == "binary" and "one_hot" in actual_enc:
                suboptimal_encodings.append({
                    "column": col, "unique": unique, "used": actual_enc,
                    "recommended": "label_encoding",
                    "reason": "Binary columns only need 0/1 — one-hot wastes a column",
                })
            else:
                good_encodings += 1

        if dimension_explosion:
            total_new_cols = sum(d["columns_created"] for d in dimension_explosion)
            findings.append({
                "severity": "warning" if total_new_cols > 50 else "info",
                "message": f"One-hot encoding created {total_new_cols} new columns from "
                           f"{len(dimension_explosion)} high-cardinality features",
                "details": dimension_explosion[:5],
                "recommendation": "Consider target encoding for columns with >10 categories",
            })

        if suboptimal_encodings:
            findings.append({
                "severity": "warning",
                "message": f"{len(suboptimal_encodings)} columns could use a better encoding strategy",
                "details": suboptimal_encodings[:5],
            })
            score = 1
        elif dimension_explosion:
            score = 2
        else:
            score = 3
            findings.append({
                "severity": "pass",
                "message": f"All {len(categorical_cols)} categorical columns encoded appropriately",
            })

        status = "optimal" if score == max_score else "acceptable" if score >= 2 else "needs_improvement"
        return {
            "step": "encoding", "title": "Categorical Encoding",
            "score": score, "max_score": max_score,
            "status": status,
            "columns_encoded": len(categorical_cols),
            "encoding_strategies_used": list(set(encoding_strategies.values())) if encoding_strategies else ["one_hot"],
            "findings": findings,
        }

    def _audit_variance_filter(self, config, results):
        """Audit variance-based feature removal."""
        variance_removed = results.get("variance_removed") or results.get("low_variance_removed") or []
        features_before = _safe_int(results.get("features_before_variance"), 0)
        features_after = _safe_int(results.get("features_after_variance"), 0)
        threshold = _safe_float(config.get("variance_threshold"), 0.01)

        findings = []
        score = 0
        max_score = 2

        if not variance_removed:
            score = 2
            findings.append({
                "severity": "pass",
                "message": "No features removed by variance filter — all features carry sufficient information",
            })
        else:
            n_removed = len(variance_removed)
            pct_removed = _pct(n_removed, features_before) if features_before else 0

            if pct_removed > 30:
                findings.append({
                    "severity": "warning",
                    "message": f"Variance filter removed {n_removed} features ({pct_removed}%) — "
                               f"this is aggressive. Consider lowering the threshold.",
                    "removed_features": variance_removed[:10],
                    "threshold_used": threshold,
                })
                score = 1
            elif pct_removed > 0:
                score = 2
                findings.append({
                    "severity": "pass",
                    "message": f"Removed {n_removed} low-variance features ({pct_removed}%) — "
                               f"good balance between dimensionality reduction and information retention",
                    "removed_features": variance_removed[:10],
                })

            # Check if important-sounding features were removed
            important_removed = [f for f in variance_removed
                                 if any(kw in f.lower() for kw in
                                        ["target", "price", "amount", "revenue", "score", "rating"])]
            if important_removed:
                findings.append({
                    "severity": "warning",
                    "message": f"Potentially important features removed by variance filter: "
                               f"{', '.join(important_removed)}",
                    "recommendation": "Review these features manually — low variance doesn't always mean low importance",
                })
                score = min(score, 1)

        status = "optimal" if score == max_score else "acceptable" if score >= 1 else "needs_improvement"
        return {
            "step": "variance_filter", "title": "Variance Filter",
            "score": score, "max_score": max_score,
            "status": status,
            "features_removed": len(variance_removed),
            "threshold": threshold,
            "findings": findings,
        }

    def _audit_missing_values(self, config, results, eda):
        """Audit missing value handling."""
        handle_missing = config.get("handle_missing_values", True)
        quality = eda.get("quality") or {}
        missing_cols = quality.get("missing_columns") or []
        total_missing_pct = _safe_float(quality.get("total_missing_pct") or quality.get("missing_percentage"), 0)

        findings = []
        score = 0
        max_score = 2

        if not handle_missing and missing_cols:
            findings.append({
                "severity": "critical",
                "message": f"Missing value handling DISABLED but {len(missing_cols)} columns have missing data — "
                           f"this will cause errors in most algorithms",
                "recommendation": "Enable missing value handling or use algorithms that handle NaN natively (XGBoost, LightGBM)",
            })
            score = 0
        elif not missing_cols and total_missing_pct == 0:
            score = 2
            findings.append({
                "severity": "pass",
                "message": "No missing values in the dataset — imputation not needed",
            })
        elif handle_missing:
            strategy = config.get("imputation_strategy", "automatic")
            score = 2
            findings.append({
                "severity": "pass",
                "message": f"Missing values handled via {strategy} imputation",
            })

            # Check for high missing rate columns
            high_missing = [c for c in missing_cols
                            if isinstance(c, dict) and _safe_float(c.get("missing_pct"), 0) > 50]
            if high_missing:
                findings.append({
                    "severity": "warning",
                    "message": f"{len(high_missing)} columns have >50% missing values — "
                               f"imputation may introduce significant bias",
                    "recommendation": "Consider dropping columns with >50% missing, or create binary 'is_missing' indicators",
                    "affected_columns": [c.get("name", c) for c in high_missing[:5]],
                })
                score = 1
        else:
            score = 1
            findings.append({
                "severity": "info",
                "message": "Missing value handling disabled — ensure your algorithm handles NaN natively",
            })

        status = "optimal" if score == max_score else "acceptable" if score >= 1 else "needs_improvement"
        return {
            "step": "missing_values", "title": "Missing Value Handling",
            "score": score, "max_score": max_score,
            "status": status,
            "missing_columns_count": len(missing_cols) if isinstance(missing_cols, list) else 0,
            "findings": findings,
        }

    def _audit_outlier_handling(self, config, results, eda):
        """Audit outlier handling."""
        handle_outliers = config.get("handle_outliers", True)
        quality = eda.get("quality") or {}
        stats = eda.get("statistics") or {}

        findings = []
        score = 0
        max_score = 2

        # Count columns with outliers
        outlier_cols = []
        for col, cs in stats.items():
            if isinstance(cs, dict):
                outlier_count = _safe_float(cs.get("outlier_count") or cs.get("outliers"), 0)
                if outlier_count and outlier_count > 0:
                    outlier_cols.append({"column": col, "outliers": int(outlier_count)})

        if not outlier_cols:
            score = 2
            findings.append({
                "severity": "pass",
                "message": "No significant outliers detected in the data",
            })
        elif handle_outliers:
            score = 2
            method = config.get("outlier_method", "IQR-based detection")
            findings.append({
                "severity": "pass",
                "message": f"Outliers handled via {method} for {len(outlier_cols)} columns",
            })

            # Check for extreme outlier situations
            extreme = [c for c in outlier_cols if c["outliers"] > 100]
            if extreme:
                findings.append({
                    "severity": "info",
                    "message": f"{len(extreme)} columns have many outliers (>100 each) — "
                               f"verify these are true outliers, not natural variation",
                    "affected_columns": [c["column"] for c in extreme[:5]],
                })
        else:
            if len(outlier_cols) > 3:
                findings.append({
                    "severity": "warning",
                    "message": f"Outlier handling DISABLED but {len(outlier_cols)} columns have outliers — "
                               f"this may distort model training",
                    "recommendation": "Enable outlier handling or use tree-based algorithms (resistant to outliers)",
                })
                score = 0
            else:
                score = 1
                findings.append({
                    "severity": "info",
                    "message": f"Outlier handling disabled — {len(outlier_cols)} columns have minor outliers",
                })

        status = "optimal" if score == max_score else "acceptable" if score >= 1 else "needs_improvement"
        return {
            "step": "outliers", "title": "Outlier Handling",
            "score": score, "max_score": max_score,
            "status": status,
            "columns_with_outliers": len(outlier_cols),
            "findings": findings,
        }

    def _audit_id_detection(self, results, eda):
        """Audit ID column detection."""
        id_cols = results.get("id_columns_detected") or results.get("id_columns") or []
        original_columns = results.get("original_columns") or []
        selected_features = results.get("selected_features") or []

        findings = []
        score = 0
        max_score = 2

        # Check if ID-like columns are in the selected features
        id_keywords = ["id", "_id", "key", "index", "row_num", "customerid", "user_id",
                       "record_id", "transaction_id", "order_id", "account_id"]
        leaked_ids = [f for f in selected_features
                      if any(kw in f.lower() for kw in id_keywords)]

        if leaked_ids:
            findings.append({
                "severity": "critical",
                "message": f"ID-like columns in selected features: {', '.join(leaked_ids)} — "
                           f"these have no predictive power and cause severe overfitting",
                "recommendation": "Remove these columns before training",
            })
            score = 0
        elif id_cols:
            score = 2
            findings.append({
                "severity": "pass",
                "message": f"Correctly detected and removed {len(id_cols)} ID columns: "
                           f"{', '.join(id_cols[:5])}",
            })
        else:
            # Check if there might be undetected IDs
            suspicious = [c for c in original_columns
                          if any(kw in c.lower() for kw in id_keywords)]
            if suspicious:
                findings.append({
                    "severity": "info",
                    "message": f"Columns with ID-like names found: {', '.join(suspicious[:5])} — "
                               f"verify these were handled correctly",
                })
                score = 1
            else:
                score = 2
                findings.append({
                    "severity": "pass",
                    "message": "No ID columns detected — data appears clean",
                })

        status = "optimal" if score == max_score else "acceptable" if score >= 1 else "needs_improvement"
        return {
            "step": "id_detection", "title": "ID Column Detection",
            "score": score, "max_score": max_score,
            "status": status,
            "id_columns": id_cols,
            "findings": findings,
        }

    def _audit_feature_selection(self, results, eda, model_versions):
        """Audit feature selection decisions."""
        original_cols = len(results.get("original_columns") or [])
        selected = results.get("selected_features") or []
        n_selected = len(selected)
        n_requested = _safe_int(results.get("n_features_requested"), n_selected)

        findings = []
        score = 0
        max_score = 3

        if not selected:
            return {"step": "feature_selection", "title": "Feature Selection",
                    "score": 0, "max_score": max_score, "status": "not_done",
                    "findings": [{"severity": "info", "message": "Feature selection not performed — all features retained"}]}

        # Check selection ratio
        if original_cols > 0:
            retention_pct = _pct(n_selected, original_cols)

            if retention_pct > 90:
                findings.append({
                    "severity": "info",
                    "message": f"Selected {n_selected}/{original_cols} features ({retention_pct}%) — "
                               f"very conservative selection, most features retained",
                })
                score = 2
            elif retention_pct < 20 and original_cols < 50:
                findings.append({
                    "severity": "warning",
                    "message": f"Selected only {n_selected}/{original_cols} features ({retention_pct}%) — "
                               f"aggressive selection may lose important information",
                    "recommendation": "Try selecting more features and compare model performance",
                })
                score = 1
            else:
                score = 3
                findings.append({
                    "severity": "pass",
                    "message": f"Selected {n_selected}/{original_cols} features ({retention_pct}%) — "
                               f"good balance of dimensionality reduction and information retention",
                })

        # Check if feature importances from model are available
        if model_versions:
            versions = model_versions.get("versions") or []
            for v in versions:
                fi = v.get("feature_importances")
                fn = v.get("feature_names")
                if fi and fn:
                    try:
                        if isinstance(fi, str):
                            import json
                            fi = json.loads(fi)
                        if isinstance(fn, str):
                            fn = json.loads(fn)

                        if isinstance(fi, list) and isinstance(fn, list) and len(fi) == len(fn):
                            # Check if selected features match top important features
                            paired = sorted(zip(fn, fi), key=lambda x: abs(x[1]), reverse=True)
                            top_important = [p[0] for p in paired[:n_selected]]
                            overlap = set(selected) & set(top_important)
                            overlap_pct = _pct(len(overlap), n_selected)

                            findings.append({
                                "severity": "pass" if overlap_pct > 70 else "warning",
                                "message": f"{overlap_pct}% of selected features match model's top-{n_selected} "
                                           f"most important features ({len(overlap)}/{n_selected})",
                                "top_model_features": top_important[:5],
                            })
                    except Exception:
                        pass
                    break  # Only check first version with data

        status = "optimal" if score == max_score else "acceptable" if score >= 2 else "needs_improvement"
        return {
            "step": "feature_selection", "title": "Feature Selection",
            "score": score, "max_score": max_score,
            "status": status,
            "selected_count": n_selected,
            "original_count": original_cols,
            "selected_features": selected[:20],
            "findings": findings,
        }

    def _generate_transformation_summary(self, audits, grade, pct):
        """Generate human-readable summary of transformation audit."""
        optimal = [a for a in audits if a.get("status") == "optimal"]
        needs_work = [a for a in audits if a.get("status") == "needs_improvement"]
        acceptable = [a for a in audits if a.get("status") == "acceptable"]

        parts = []
        if grade in ("A+", "A", "A-"):
            parts.append(f"Excellent feature engineering pipeline (Grade {grade}, {pct}%).")
        elif grade.startswith("B"):
            parts.append(f"Good feature engineering pipeline with minor improvements possible (Grade {grade}, {pct}%).")
        elif grade.startswith("C"):
            parts.append(f"Acceptable feature engineering but several areas need attention (Grade {grade}, {pct}%).")
        else:
            parts.append(f"Feature engineering pipeline needs significant improvement (Grade {grade}, {pct}%).")

        if optimal:
            parts.append(f"{len(optimal)} steps are optimal: {', '.join(a['title'] for a in optimal)}.")
        if needs_work:
            parts.append(f"{len(needs_work)} steps need improvement: {', '.join(a['title'] for a in needs_work)}.")

        return " ".join(parts)

    # ──────────────────────────────────────────────────────────
    # 2. FEATURE SELECTION EXPLANATION (SHAP-like)
    # ──────────────────────────────────────────────────────────

    def explain_feature_selection(
            self,
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        SHAP-like explanation of why each feature was selected or dropped.

        Returns:
          - selected_features: [{name, importance, rank, reason, category}]
          - dropped_features: [{name, drop_reason, importance_lost, recoverable}]
          - information_retention_score: 0-100
          - feature_importance_chart: data for bar chart visualization
        """
        results = feature_results or {}
        eda = eda_data or {}
        stats = eda.get("statistics") or {}
        correlations = eda.get("correlations") or {}

        selected = results.get("selected_features") or []
        all_original = results.get("original_columns") or []
        dropped_variance = results.get("variance_removed") or []
        id_cols = results.get("id_columns_detected") or results.get("id_columns") or []

        # Build feature importance from model if available
        model_importance = self._extract_model_importance(model_versions)

        # Build feature analysis for selected features
        selected_analysis = []
        for rank, feat in enumerate(selected, 1):
            importance = model_importance.get(feat, {})
            imp_score = importance.get("score", 0)
            reason = importance.get("reason", "Selected by feature selection algorithm")

            # Enrich with EDA stats
            feat_stats = stats.get(feat, {})
            correlation_with_target = None
            if isinstance(correlations, dict):
                target_corr = correlations.get("target_correlations") or {}
                correlation_with_target = _safe_float(target_corr.get(feat))

            category = self._classify_feature(feat, feat_stats)

            selected_analysis.append({
                "name": feat,
                "rank": rank,
                "importance_score": round(imp_score, 4) if imp_score else None,
                "correlation_with_target": round(correlation_with_target, 4) if correlation_with_target else None,
                "reason": reason,
                "category": category,
                "stats": self._summarize_feature_stats(feat_stats),
            })

        # Sort by importance if available
        if any(f["importance_score"] for f in selected_analysis):
            selected_analysis.sort(key=lambda x: abs(x["importance_score"] or 0), reverse=True)
            for i, f in enumerate(selected_analysis, 1):
                f["rank"] = i

        # Analyze dropped features
        all_dropped = set(all_original) - set(selected) - set(id_cols)
        dropped_analysis = []
        for feat in sorted(all_dropped):
            drop_reasons = []
            recoverable = True

            if feat in dropped_variance:
                drop_reasons.append("Low variance — feature has nearly constant values")
                recoverable = False
            elif feat in id_cols:
                drop_reasons.append("Identified as ID column — no predictive value")
                recoverable = False
            else:
                # Was it encoding or selection?
                imp = model_importance.get(feat, {}).get("score", 0)
                if imp == 0:
                    drop_reasons.append("Low importance in feature selection ranking")
                else:
                    drop_reasons.append(f"Below selection threshold (importance: {imp:.4f})")

            correlation_with_target = None
            if isinstance(correlations, dict):
                target_corr = correlations.get("target_correlations") or {}
                correlation_with_target = _safe_float(target_corr.get(feat))

            dropped_analysis.append({
                "name": feat,
                "drop_reasons": drop_reasons,
                "importance_lost": round(model_importance.get(feat, {}).get("score", 0), 4),
                "correlation_with_target": round(correlation_with_target, 4) if correlation_with_target else None,
                "recoverable": recoverable,
                "recommendation": "Consider re-including if model performance is low" if recoverable else "Safe to exclude",
            })

        # Sort dropped by importance lost (highest first)
        dropped_analysis.sort(key=lambda x: abs(x.get("importance_lost") or 0), reverse=True)

        # Calculate information retention
        total_importance = sum(abs(v.get("score", 0)) for v in model_importance.values()) if model_importance else 0
        selected_importance = sum(abs(model_importance.get(f, {}).get("score", 0)) for f in selected) if model_importance else 0
        retention_score = _pct(selected_importance, total_importance) if total_importance else None

        # Build chart data
        chart_data = []
        for fa in selected_analysis[:15]:
            chart_data.append({
                "feature": fa["name"],
                "importance": fa["importance_score"] or 0,
                "selected": True,
            })
        for da in dropped_analysis[:5]:
            if da["importance_lost"] > 0:
                chart_data.append({
                    "feature": da["name"],
                    "importance": da["importance_lost"],
                    "selected": False,
                })

        return {
            "selected_features": selected_analysis,
            "dropped_features": dropped_analysis,
            "information_retention_score": round(retention_score, 1) if retention_score else None,
            "total_features_analyzed": len(all_original),
            "features_selected": len(selected),
            "features_dropped": len(dropped_analysis),
            "chart_data": chart_data,
            "summary": self._generate_selection_summary(selected_analysis, dropped_analysis, retention_score),
        }

    def _extract_model_importance(self, model_versions):
        """Extract feature importance from model_versions DB data."""
        if not model_versions:
            return {}

        importance_map = {}
        versions = model_versions.get("versions") or []

        for v in versions:
            fi = v.get("feature_importances")
            fn = v.get("feature_names")
            if fi and fn:
                try:
                    import json
                    if isinstance(fi, str):
                        fi = json.loads(fi)
                    if isinstance(fn, str):
                        fn = json.loads(fn)

                    if isinstance(fi, list) and isinstance(fn, list):
                        for name, score in zip(fn, fi):
                            imp = _safe_float(score, 0)
                            importance_map[name] = {
                                "score": imp,
                                "reason": f"Model-derived importance: {imp:.4f}",
                                "source": v.get("algorithm", "unknown"),
                            }
                except Exception:
                    pass
                break  # Use first version with data

        return importance_map

    def _classify_feature(self, name, stats):
        """Classify a feature by type based on name and stats."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["_scaled", "scaled_"]):
            return "numeric_scaled"
        elif any(kw in name_lower for kw in ["_yes", "_no", "is_", "has_"]):
            return "binary_encoded"
        elif any(kw in name_lower for kw in ["_encoded", "_label"]):
            return "categorical_encoded"
        elif "_" in name and name.split("_")[-1] in ("0", "1", "True", "False"):
            return "one_hot_encoded"
        else:
            return "original"

    def _summarize_feature_stats(self, stats):
        """Create a compact feature stats summary."""
        if not stats or not isinstance(stats, dict):
            return None
        return {
            "mean": _safe_float(stats.get("mean")),
            "std": _safe_float(stats.get("std")),
            "min": _safe_float(stats.get("min")),
            "max": _safe_float(stats.get("max")),
            "unique": _safe_int(stats.get("unique_count") or stats.get("distinct")),
            "missing_pct": _safe_float(stats.get("missing_pct") or stats.get("null_percentage"), 0),
        }

    def _generate_selection_summary(self, selected, dropped, retention):
        """Generate human-readable feature selection summary."""
        parts = [f"Selected {len(selected)} features, dropped {len(dropped)}."]

        if retention is not None:
            if retention > 90:
                parts.append(f"Retained {retention}% of model-derived importance — excellent information preservation.")
            elif retention > 70:
                parts.append(f"Retained {retention}% of model-derived importance — good balance.")
            else:
                parts.append(f"Retained only {retention}% of importance — consider selecting more features.")

        # Highlight top features
        top = [f["name"] for f in selected[:3]]
        if top:
            parts.append(f"Top features: {', '.join(top)}.")

        # Highlight risky drops
        risky = [d for d in dropped if d.get("importance_lost", 0) > 0.05]
        if risky:
            parts.append(f"{len(risky)} dropped features had moderate importance — may be worth re-evaluating.")

        return " ".join(parts)

    # ──────────────────────────────────────────────────────────
    # 3. ERROR PATTERN DETECTION
    # ──────────────────────────────────────────────────────────

    def detect_error_patterns(
            self,
            feature_config: Dict[str, Any],
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect common feature engineering mistakes.

        Checks:
          - Data leakage (target-correlated features, future data)
          - Multicollinearity (highly correlated feature pairs)
          - Information loss (aggressive filtering, bad encoding)
          - Curse of dimensionality (too many features vs samples)
          - Target encoding leakage
          - Scaling-algorithm mismatch
          - Train-test inconsistency
        """
        config = feature_config or {}
        results = feature_results or {}
        eda = eda_data or {}
        stats = eda.get("statistics") or {}
        correlations = eda.get("correlations") or {}

        patterns = []

        # 1. Data Leakage Detection
        patterns.extend(self._check_data_leakage(results, eda))

        # 2. Multicollinearity
        patterns.extend(self._check_multicollinearity(results, correlations))

        # 3. Information Loss
        patterns.extend(self._check_information_loss(results, eda))

        # 4. Curse of Dimensionality
        patterns.extend(self._check_dimensionality(results))

        # 5. Encoding Issues
        patterns.extend(self._check_encoding_issues(config, results, stats))

        # 6. Train-Test Consistency
        patterns.extend(self._check_train_test_consistency(results))

        # 7. Scaling-Algorithm Mismatch
        patterns.extend(self._check_scaling_algorithm_mismatch(config, model_versions))

        # Sort by severity
        patterns.sort(key=lambda p: _severity_rank(p.get("severity", "info")))

        # Count by severity
        severity_counts = {}
        for p in patterns:
            s = p.get("severity", "info")
            severity_counts[s] = severity_counts.get(s, 0) + 1

        critical_count = severity_counts.get("critical", 0)
        warning_count = severity_counts.get("warning", 0)

        health = (
            "critical" if critical_count > 0
            else "warning" if warning_count > 2
            else "good" if warning_count > 0
            else "excellent"
        )

        return {
            "patterns": patterns,
            "total_issues": len(patterns),
            "severity_counts": severity_counts,
            "health": health,
            "summary": self._generate_error_summary(patterns, health),
        }

    def _check_data_leakage(self, results, eda):
        """Check for data leakage patterns."""
        patterns = []
        selected = results.get("selected_features") or []
        all_cols = results.get("original_columns") or []
        correlations = eda.get("correlations") or {}
        target_corrs = correlations.get("target_correlations") or {}

        # Check column names for leakage patterns
        for leakage_type in LEAKAGE_PATTERNS:
            for col in selected:
                col_lower = col.lower().replace(" ", "_")
                if any(kw in col_lower for kw in leakage_type["fields"]):
                    patterns.append({
                        "type": "data_leakage",
                        "severity": leakage_type["severity"],
                        "feature": col,
                        "pattern": leakage_type["pattern"],
                        "message": f"'{col}' — {leakage_type['message']}",
                        "recommendation": "Remove this feature and retrain to verify model still performs well",
                    })

        # Check for suspiciously high correlations with target
        for col, corr in target_corrs.items():
            corr_val = _safe_float(corr, 0)
            if corr_val is not None and abs(corr_val) > 0.95 and col in selected:
                patterns.append({
                    "type": "data_leakage",
                    "severity": "critical",
                    "feature": col,
                    "pattern": "high_target_correlation",
                    "message": f"'{col}' has {abs(corr_val)*100:.1f}% correlation with target — "
                               f"this is suspiciously high and may indicate data leakage",
                    "recommendation": "Investigate whether this feature is derived from the target "
                                      "or contains information not available at prediction time",
                    "correlation": round(corr_val, 4),
                })

        return patterns

    def _check_multicollinearity(self, results, correlations):
        """Check for highly correlated feature pairs."""
        patterns = []
        selected = set(results.get("selected_features") or [])
        corr_matrix = correlations.get("matrix") or correlations.get("correlation_matrix") or {}

        if not corr_matrix:
            return patterns

        # Find highly correlated pairs among selected features
        high_corr_pairs = []
        checked = set()

        if isinstance(corr_matrix, dict):
            for col1, row in corr_matrix.items():
                if col1 not in selected or not isinstance(row, dict):
                    continue
                for col2, corr_val in row.items():
                    if col2 not in selected or col1 == col2:
                        continue
                    pair_key = tuple(sorted([col1, col2]))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    cv = _safe_float(corr_val, 0)
                    if cv is not None and abs(cv) > 0.85:
                        high_corr_pairs.append({
                            "feature_a": col1, "feature_b": col2,
                            "correlation": round(cv, 4),
                        })

        if high_corr_pairs:
            high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            patterns.append({
                "type": "multicollinearity",
                "severity": "warning" if len(high_corr_pairs) <= 3 else "critical",
                "message": f"{len(high_corr_pairs)} pairs of selected features are highly correlated (>0.85) — "
                           f"this causes instability in linear models and wastes model capacity",
                "pairs": high_corr_pairs[:10],
                "recommendation": "Drop one feature from each correlated pair, or use PCA to combine them. "
                                  "Tree-based models (RF, XGBoost) are more robust to multicollinearity.",
            })

        return patterns

    def _check_information_loss(self, results, eda):
        """Check for excessive information loss."""
        patterns = []
        original_cols = len(results.get("original_columns") or [])
        selected = len(results.get("selected_features") or [])
        dropped_variance = len(results.get("variance_removed") or [])

        if original_cols > 0 and selected > 0:
            retention_ratio = selected / original_cols

            if retention_ratio < 0.2 and original_cols < 50:
                patterns.append({
                    "type": "information_loss",
                    "severity": "warning",
                    "message": f"Only {selected}/{original_cols} features retained ({retention_ratio*100:.0f}%) — "
                               f"aggressive feature selection may lose important information",
                    "recommendation": "Try training with more features and compare performance. "
                                      "Some models (XGBoost, Neural Networks) can handle high dimensionality.",
                })

            if dropped_variance > original_cols * 0.3:
                patterns.append({
                    "type": "information_loss",
                    "severity": "warning",
                    "message": f"Variance filter removed {dropped_variance} features ({_pct(dropped_variance, original_cols)}%) — "
                               f"consider lowering the variance threshold",
                    "recommendation": "Lower variance threshold from 0.01 to 0.001 and compare model performance",
                })

        return patterns

    def _check_dimensionality(self, results):
        """Check for curse of dimensionality."""
        patterns = []
        n_features = len(results.get("selected_features") or [])
        n_rows = _safe_int(results.get("n_rows") or (results.get("original_shape") or [0])[0])

        if n_rows > 0 and n_features > 0:
            ratio = n_rows / n_features

            if ratio < 5:
                patterns.append({
                    "type": "curse_of_dimensionality",
                    "severity": "critical",
                    "message": f"Only {ratio:.1f}x more samples ({n_rows}) than features ({n_features}) — "
                               f"high risk of overfitting (rule of thumb: need at least 10x)",
                    "recommendation": "Reduce features (PCA, stronger selection) or collect more data. "
                                      "Use strong regularization (L1/L2) if training with these dimensions.",
                    "samples": n_rows,
                    "features": n_features,
                    "ratio": round(ratio, 1),
                })
            elif ratio < 10:
                patterns.append({
                    "type": "curse_of_dimensionality",
                    "severity": "warning",
                    "message": f"{ratio:.1f}x sample-to-feature ratio ({n_rows} samples, {n_features} features) — "
                               f"borderline for some algorithms",
                    "recommendation": "Use regularization and cross-validation to prevent overfitting",
                    "samples": n_rows,
                    "features": n_features,
                    "ratio": round(ratio, 1),
                })

        return patterns

    def _check_encoding_issues(self, config, results, stats):
        """Check for encoding-related issues."""
        patterns = []
        encoding_strategies = results.get("encoding_strategies") or {}
        categorical_cols = results.get("categorical_features") or []

        # Check high-cardinality one-hot
        for col in categorical_cols:
            cs = stats.get(col, {})
            unique = _safe_int(cs.get("unique_count") or cs.get("distinct"), 0) if isinstance(cs, dict) else 0
            strategy = encoding_strategies.get(col, "one_hot")

            if unique > 50 and "one_hot" in strategy.lower():
                patterns.append({
                    "type": "encoding_issue",
                    "severity": "warning",
                    "feature": col,
                    "message": f"'{col}' has {unique} categories but uses one-hot encoding — "
                               f"creates {unique-1} sparse columns",
                    "recommendation": "Switch to target encoding or frequency encoding to avoid "
                                      "dimensionality explosion",
                })

        # Check if rare categories exist
        rare_cat_cols = results.get("rare_categories_grouped") or []
        if rare_cat_cols:
            patterns.append({
                "type": "encoding_issue",
                "severity": "info",
                "message": f"{len(rare_cat_cols)} columns had rare categories grouped into 'Other' — "
                           f"verify this doesn't lose important minority patterns",
                "affected_columns": rare_cat_cols[:5] if isinstance(rare_cat_cols, list) else [],
            })

        return patterns

    def _check_train_test_consistency(self, results):
        """Check train-test data consistency."""
        patterns = []
        train_shape = results.get("train_shape") or results.get("final_train_shape")
        test_shape = results.get("test_shape") or results.get("final_test_shape")

        if train_shape and test_shape:
            train_cols = train_shape[1] if isinstance(train_shape, (list, tuple)) and len(train_shape) > 1 else 0
            test_cols = test_shape[1] if isinstance(test_shape, (list, tuple)) and len(test_shape) > 1 else 0

            if train_cols != test_cols:
                patterns.append({
                    "type": "train_test_mismatch",
                    "severity": "critical",
                    "message": f"Train ({train_cols} columns) and test ({test_cols} columns) have different shapes — "
                               f"this will cause prediction errors in production",
                    "recommendation": "Ensure encoding uses fit_transform on train and transform on test (never fit on test)",
                })

        return patterns

    def _check_scaling_algorithm_mismatch(self, config, model_versions):
        """Check if scaling choice conflicts with algorithm choice."""
        patterns = []
        scaling = (config.get("scaling_method") or "standard").lower()

        if not model_versions:
            return patterns

        versions = model_versions.get("versions") or []
        algorithms = [v.get("algorithm", "").lower() for v in versions if v.get("algorithm")]

        # Tree-based algorithms don't need scaling
        tree_algos = [a for a in algorithms if any(t in a for t in
                                                   ["tree", "forest", "xgboost", "lightgbm", "catboost", "gradient", "adaboost", "bagging"])]

        # Distance-based algorithms need scaling
        distance_algos = [a for a in algorithms if any(t in a for t in
                                                       ["knn", "svm", "svc", "svr", "kmeans", "dbscan"])]

        if tree_algos and not distance_algos:
            patterns.append({
                "type": "scaling_mismatch",
                "severity": "info",
                "message": f"Scaling applied but all algorithms are tree-based ({', '.join(list(set(tree_algos))[:3])}) — "
                           f"trees don't benefit from feature scaling",
                "recommendation": "Scaling doesn't hurt tree-based models but adds unnecessary complexity. "
                                  "If only using trees, you can disable scaling.",
            })

        if distance_algos and scaling == "none":
            patterns.append({
                "type": "scaling_mismatch",
                "severity": "warning",
                "message": f"Distance-based algorithms ({', '.join(list(set(distance_algos))[:3])}) used without scaling — "
                           f"features with larger ranges will dominate distance calculations",
                "recommendation": "Enable scaling (StandardScaler or MinMaxScaler) for distance-based algorithms",
            })

        return patterns

    def _generate_error_summary(self, patterns, health):
        """Generate human-readable error pattern summary."""
        if health == "excellent":
            return "No significant issues detected in your feature engineering pipeline. Clean data, good transformations."
        elif health == "good":
            n_warnings = sum(1 for p in patterns if p.get("severity") == "warning")
            return f"Feature pipeline is healthy with {n_warnings} minor warning(s) to review."
        elif health == "warning":
            return f"Found {len(patterns)} issues in feature engineering — review warnings to improve model quality."
        else:
            critical = sum(1 for p in patterns if p.get("severity") == "critical")
            return f"Critical: {critical} serious issue(s) detected that may significantly hurt model performance."

    # ──────────────────────────────────────────────────────────
    # 4. QUALITY SCORECARD
    # ──────────────────────────────────────────────────────────

    def assess_quality(
            self,
            feature_config: Dict[str, Any],
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        10-point feature pipeline quality scorecard.

        Returns:
          - checks: [{id, name, passed, detail, weight, category}]
          - score/max_score/percentage
          - verdict: "excellent" | "good" | "needs_improvement" | "poor"
          - blockers: critical issues
          - warnings: non-critical issues
        """
        config = feature_config or {}
        results = feature_results or {}
        eda = eda_data or {}

        checks = []
        score = 0

        # 1. Scaling Appropriate
        scaling_ok = self._check_scaling_quality(config, results, eda)
        checks.append(scaling_ok)
        if scaling_ok["passed"]:
            score += scaling_ok["weight"]

        # 2. Encoding Appropriate
        encoding_ok = self._check_encoding_quality(config, results, eda)
        checks.append(encoding_ok)
        if encoding_ok["passed"]:
            score += encoding_ok["weight"]

        # 3. Missing Values Handled
        missing_ok = self._check_missing_quality(config, results, eda)
        checks.append(missing_ok)
        if missing_ok["passed"]:
            score += missing_ok["weight"]

        # 4. Outliers Handled
        outlier_ok = self._check_outlier_quality(config, results, eda)
        checks.append(outlier_ok)
        if outlier_ok["passed"]:
            score += outlier_ok["weight"]

        # 5. No Data Leakage
        leakage_ok = self._check_leakage_quality(results, eda)
        checks.append(leakage_ok)
        if leakage_ok["passed"]:
            score += leakage_ok["weight"]

        # 6. No High Multicollinearity
        multicollin_ok = self._check_multicollinearity_quality(results, eda)
        checks.append(multicollin_ok)
        if multicollin_ok["passed"]:
            score += multicollin_ok["weight"]

        # 7. Feature Selection Done
        selection_ok = self._check_selection_quality(results)
        checks.append(selection_ok)
        if selection_ok["passed"]:
            score += selection_ok["weight"]

        # 8. Variance Filter Applied
        variance_ok = self._check_variance_quality(results)
        checks.append(variance_ok)
        if variance_ok["passed"]:
            score += variance_ok["weight"]

        # 9. Information Retention
        retention_ok = self._check_retention_quality(results, model_versions)
        checks.append(retention_ok)
        if retention_ok["passed"]:
            score += retention_ok["weight"]

        # 10. Production Reproducible
        prod_ok = self._check_production_quality(config, results)
        checks.append(prod_ok)
        if prod_ok["passed"]:
            score += prod_ok["weight"]

        # Calculate totals
        evaluated = [c for c in checks if c.get("passed") is not None]
        max_score = sum(c["weight"] for c in evaluated)
        pct = _pct(score, max_score) if max_score > 0 else 0

        verdict = (
            "excellent" if pct >= 85
            else "good" if pct >= 70
            else "needs_improvement" if pct >= 50
            else "poor"
        )

        blockers = [c for c in checks if c.get("passed") is False and c["weight"] >= 2]
        warnings = [c for c in checks if c.get("passed") is False and c["weight"] < 2]

        return {
            "checks": checks,
            "score": score,
            "max_score": max_score,
            "percentage": round(pct, 1),
            "verdict": verdict,
            "blockers": [{"name": b["name"], "detail": b["detail"]} for b in blockers],
            "warnings": [{"name": w["name"], "detail": w["detail"]} for w in warnings],
            "summary": self._generate_quality_summary(score, max_score, pct, verdict, blockers),
        }

    def _check_scaling_quality(self, config, results, eda):
        method = config.get("scaling_method", "standard")
        numeric = results.get("numeric_features") or []
        passed = bool(method and numeric) or not numeric
        return {
            "id": "scaling", "name": "Scaling Method Applied",
            "passed": passed, "weight": 2, "category": "Transformation Quality",
            "detail": f"{method.title()}Scaler on {len(numeric)} features" if passed else "No scaling applied to numeric features",
            "value": method if numeric else "N/A",
        }

    def _check_encoding_quality(self, config, results, eda):
        categorical = results.get("categorical_features") or []
        encode = config.get("encode_categories", True)
        passed = encode or not categorical
        return {
            "id": "encoding", "name": "Categorical Encoding Applied",
            "passed": passed, "weight": 2, "category": "Transformation Quality",
            "detail": f"Encoded {len(categorical)} categorical features" if passed else "Categorical features not encoded — most algorithms require numeric input",
            "value": f"{len(categorical)} encoded" if categorical else "N/A",
        }

    def _check_missing_quality(self, config, results, eda):
        quality = eda.get("quality") or {}
        missing = quality.get("missing_columns") or []
        handle = config.get("handle_missing_values", True)
        passed = handle or not missing
        return {
            "id": "missing", "name": "Missing Values Handled",
            "passed": passed, "weight": 1, "category": "Data Completeness",
            "detail": "All missing values imputed" if passed else f"{len(missing)} columns have unhandled missing values",
            "value": "✓ Clean" if passed else f"{len(missing)} columns",
        }

    def _check_outlier_quality(self, config, results, eda):
        handle = config.get("handle_outliers", True)
        passed = handle
        return {
            "id": "outliers", "name": "Outlier Detection Applied",
            "passed": passed, "weight": 1, "category": "Data Completeness",
            "detail": "IQR-based outlier detection applied" if passed else "Outlier handling disabled",
            "value": "✓ Applied" if passed else "Disabled",
        }

    def _check_leakage_quality(self, results, eda):
        selected = results.get("selected_features") or []
        correlations = eda.get("correlations") or {}
        target_corrs = correlations.get("target_correlations") or {}

        # Check for suspiciously high correlations
        leaky = [col for col in selected
                 if abs(_safe_float(target_corrs.get(col), 0)) > 0.95]

        # Check for ID-like features
        id_kw = ["id", "_id", "key", "index", "row_num"]
        id_leaks = [f for f in selected if any(kw in f.lower() for kw in id_kw)]

        all_leaks = list(set(leaky + id_leaks))
        passed = len(all_leaks) == 0

        return {
            "id": "leakage", "name": "No Data Leakage Detected",
            "passed": passed, "weight": 3, "category": "Data Integrity",
            "detail": "No suspicious features detected" if passed else f"Potential leakage: {', '.join(all_leaks[:3])}",
            "value": "✓ Clean" if passed else f"{len(all_leaks)} suspicious",
        }

    def _check_multicollinearity_quality(self, results, eda):
        correlations = eda.get("correlations") or {}
        corr_matrix = correlations.get("matrix") or correlations.get("correlation_matrix") or {}
        selected = set(results.get("selected_features") or [])

        high_corr_count = 0
        if isinstance(corr_matrix, dict):
            checked = set()
            for c1, row in corr_matrix.items():
                if c1 not in selected or not isinstance(row, dict):
                    continue
                for c2, val in row.items():
                    if c2 not in selected or c1 == c2:
                        continue
                    pk = tuple(sorted([c1, c2]))
                    if pk in checked:
                        continue
                    checked.add(pk)
                    if abs(_safe_float(val, 0)) > 0.90:
                        high_corr_count += 1

        passed = high_corr_count <= 2
        return {
            "id": "multicollinearity", "name": "Low Multicollinearity",
            "passed": passed, "weight": 2, "category": "Feature Independence",
            "detail": f"No highly correlated feature pairs" if high_corr_count == 0
            else f"{high_corr_count} highly correlated pairs (>0.90)",
            "value": f"{high_corr_count} pairs" if high_corr_count > 0 else "✓ Independent",
        }

    def _check_selection_quality(self, results):
        selected = results.get("selected_features") or []
        passed = len(selected) > 0
        return {
            "id": "selection", "name": "Feature Selection Applied",
            "passed": passed, "weight": 1, "category": "Dimensionality",
            "detail": f"Selected {len(selected)} features" if passed else "No feature selection performed",
            "value": f"{len(selected)} features" if passed else "All features",
        }

    def _check_variance_quality(self, results):
        removed = results.get("variance_removed") or []
        passed = True  # Variance filter being applied or not needed
        return {
            "id": "variance", "name": "Low-Variance Features Filtered",
            "passed": passed, "weight": 1, "category": "Dimensionality",
            "detail": f"Removed {len(removed)} constant/near-constant features" if removed else "All features have sufficient variance",
            "value": f"{len(removed)} removed" if removed else "✓ All varied",
        }

    def _check_retention_quality(self, results, model_versions):
        original = len(results.get("original_columns") or [])
        selected = len(results.get("selected_features") or [])
        if original == 0:
            return {"id": "retention", "name": "Information Retention",
                    "passed": None, "weight": 2, "category": "Information Preservation",
                    "detail": "Cannot assess — original column count unknown", "value": "N/A"}

        ratio = selected / original if original > 0 else 0
        passed = ratio >= 0.15 or original < 10  # At least 15% retained or small dataset
        return {
            "id": "retention", "name": "Information Retention ≥ 15%",
            "passed": passed, "weight": 2, "category": "Information Preservation",
            "detail": f"Retained {selected}/{original} features ({ratio*100:.0f}%)" if passed
            else f"Only {selected}/{original} features retained ({ratio*100:.0f}%) — may lose critical information",
            "value": f"{ratio*100:.0f}%",
        }

    def _check_production_quality(self, config, results):
        # Check that the pipeline is deterministic and reproducible
        scaling = config.get("scaling_method")
        encoding = config.get("encode_categories", True)
        train_shape = results.get("train_shape") or results.get("final_train_shape")
        test_shape = results.get("test_shape") or results.get("final_test_shape")

        issues = []
        if train_shape and test_shape:
            tc = train_shape[1] if isinstance(train_shape, (list, tuple)) and len(train_shape) > 1 else -1
            ec = test_shape[1] if isinstance(test_shape, (list, tuple)) and len(test_shape) > 1 else -2
            if tc != ec:
                issues.append("Train-test column mismatch")

        passed = len(issues) == 0
        return {
            "id": "production", "name": "Production Reproducible",
            "passed": passed, "weight": 1, "category": "Production Readiness",
            "detail": "Pipeline produces consistent train-test outputs" if passed
            else f"Issues: {', '.join(issues)}",
            "value": "✓ Consistent" if passed else "Mismatch detected",
        }

    def _generate_quality_summary(self, score, max_score, pct, verdict, blockers):
        v_map = {
            "excellent": f"Excellent feature engineering quality ({pct}%) — pipeline is production-ready.",
            "good": f"Good feature engineering quality ({pct}%) — minor improvements possible.",
            "needs_improvement": f"Feature engineering needs improvement ({pct}%) — address the issues below.",
            "poor": f"Feature engineering has significant issues ({pct}%) — critical problems need fixing.",
        }
        text = v_map.get(verdict, f"Score: {pct}%")
        if blockers:
            text += f" {len(blockers)} critical blocker(s) found."
        return text

    # ──────────────────────────────────────────────────────────
    # 5. SMART CONFIG RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────

    def generate_smart_config(
            self,
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            current_config: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        AI-recommended feature engineering settings based on data characteristics.

        Returns optimal settings with explanations for each choice.
        """
        results = feature_results or {}
        eda = eda_data or {}
        current = current_config or {}
        stats = eda.get("statistics") or {}
        quality = eda.get("quality") or {}

        recommendations = {}

        # 1. Scaling Method
        rec_scaling = self._recommend_scaling(results, stats, quality)
        recommendations["scaling_method"] = rec_scaling

        # 2. Encoding Strategy
        rec_encoding = self._recommend_encoding(results, stats)
        recommendations["encoding_strategy"] = rec_encoding

        # 3. Missing Value Handling
        rec_missing = self._recommend_missing_handling(results, quality)
        recommendations["handle_missing_values"] = rec_missing

        # 4. Outlier Handling
        rec_outliers = self._recommend_outlier_handling(results, stats, quality)
        recommendations["handle_outliers"] = rec_outliers

        # 5. Feature Selection Count
        rec_selection = self._recommend_selection_count(results, eda)
        recommendations["feature_selection"] = rec_selection

        # 6. Polynomial Features
        rec_poly = self._recommend_polynomial(results, eda, model_versions)
        recommendations["polynomial_features"] = rec_poly

        # 7. Feature Interactions
        rec_interactions = self._recommend_interactions(results, eda)
        recommendations["feature_interactions"] = rec_interactions

        # Generate the recommended config
        recommended_config = {
            "scaling_method": rec_scaling["value"],
            "handle_missing_values": rec_missing["value"],
            "handle_outliers": rec_outliers["value"],
            "encode_categories": True,
            "create_polynomial_features": rec_poly["value"],
            "create_interactions": rec_interactions["value"],
            "n_features_to_select": rec_selection["value"],
        }

        # Compare with current
        changes = []
        for key, new_val in recommended_config.items():
            old_val = current.get(key)
            if old_val is not None and str(old_val).lower() != str(new_val).lower():
                changes.append({
                    "setting": key,
                    "current": old_val,
                    "recommended": new_val,
                    "reason": recommendations.get(key, {}).get("reason", ""),
                })

        return {
            "recommended_config": recommended_config,
            "recommendations": recommendations,
            "changes_from_current": changes,
            "n_changes": len(changes),
            "summary": self._generate_config_summary(changes, current),
        }

    def _recommend_scaling(self, results, stats, quality):
        """Recommend optimal scaling method."""
        numeric_cols = results.get("numeric_features") or []
        has_outliers = False
        has_skew = False

        for col in numeric_cols:
            cs = stats.get(col, {})
            if isinstance(cs, dict):
                if _safe_float(cs.get("outlier_count") or cs.get("outliers"), 0) > 0:
                    has_outliers = True
                if abs(_safe_float(cs.get("skewness"), 0)) > 1.0:
                    has_skew = True

        if has_outliers:
            return {"value": "robust", "confidence": "high",
                    "reason": "Data has outliers — RobustScaler uses median/IQR, resistant to extreme values"}
        elif has_skew:
            return {"value": "robust", "confidence": "medium",
                    "reason": "Data has skewed features — RobustScaler handles non-symmetric distributions better"}
        else:
            return {"value": "standard", "confidence": "high",
                    "reason": "Data is well-behaved — StandardScaler is the most widely compatible choice"}

    def _recommend_encoding(self, results, stats):
        """Recommend optimal encoding strategy."""
        categorical_cols = results.get("categorical_features") or []
        high_card = []

        for col in categorical_cols:
            cs = stats.get(col, {})
            unique = _safe_int(cs.get("unique_count") or cs.get("distinct"), 0) if isinstance(cs, dict) else 0
            if unique > 15:
                high_card.append(col)

        if high_card:
            return {"value": "mixed", "confidence": "high",
                    "reason": f"{len(high_card)} columns have high cardinality — use one-hot for low-card, "
                              f"target/frequency encoding for high-card columns",
                    "high_cardinality_columns": high_card[:5]}
        else:
            return {"value": "one_hot", "confidence": "high",
                    "reason": "All categorical columns have low cardinality — one-hot encoding is optimal"}

    def _recommend_missing_handling(self, results, quality):
        """Recommend missing value strategy."""
        missing_cols = quality.get("missing_columns") or []
        total_missing = _safe_float(quality.get("total_missing_pct"), 0)

        if not missing_cols and total_missing == 0:
            return {"value": True, "confidence": "high",
                    "reason": "No missing values detected, but keeping enabled as a safety net"}
        elif total_missing > 20:
            return {"value": True, "confidence": "high",
                    "reason": f"High missing rate ({total_missing}%) — imputation essential. Consider median for numeric, mode for categorical"}
        else:
            return {"value": True, "confidence": "high",
                    "reason": "Missing values present — automatic imputation recommended"}

    def _recommend_outlier_handling(self, results, stats, quality):
        """Recommend outlier handling."""
        numeric_cols = results.get("numeric_features") or []
        outlier_count = 0
        for col in numeric_cols:
            cs = stats.get(col, {})
            if isinstance(cs, dict):
                outlier_count += _safe_int(cs.get("outlier_count") or cs.get("outliers"), 0)

        if outlier_count > 0:
            return {"value": True, "confidence": "high",
                    "reason": f"{outlier_count} outliers detected across numeric features — IQR-based handling recommended"}
        else:
            return {"value": False, "confidence": "medium",
                    "reason": "No significant outliers detected — can skip to reduce processing time"}

    def _recommend_selection_count(self, results, eda):
        """Recommend number of features to select."""
        original = len(results.get("original_columns") or [])
        n_rows = _safe_int(results.get("n_rows") or (results.get("original_shape") or [0])[0])

        if original <= 10:
            recommended = original
            reason = f"Small feature set ({original}) — keep all features"
        elif original <= 30:
            recommended = min(original, max(10, int(original * 0.5)))
            reason = f"Medium feature set — select ~50% ({recommended}) for good balance"
        elif original <= 100:
            recommended = min(original, max(15, int(original * 0.3)))
            reason = f"Large feature set — select ~30% ({recommended}) to reduce dimensionality"
        else:
            recommended = min(original, max(20, int(original * 0.15)))
            reason = f"Very large feature set — select ~15% ({recommended}) to avoid curse of dimensionality"

        # Adjust based on sample-to-feature ratio
        if n_rows > 0 and n_rows / recommended < 10:
            recommended = max(5, n_rows // 10)
            reason += f". Adjusted down to maintain 10:1 sample-to-feature ratio"

        return {"value": recommended, "confidence": "medium", "reason": reason}

    def _recommend_polynomial(self, results, eda, model_versions):
        """Recommend polynomial features."""
        n_features = len(results.get("selected_features") or results.get("original_columns") or [])
        n_rows = _safe_int(results.get("n_rows") or (results.get("original_shape") or [0])[0])

        # Check if model performance is already high
        best_score = 0
        if model_versions:
            for v in (model_versions.get("versions") or []):
                s = _safe_float(v.get("accuracy") or v.get("test_score"), 0)
                if s > best_score:
                    best_score = s

        if n_features > 15:
            return {"value": False, "confidence": "high",
                    "reason": f"Too many features ({n_features}) — polynomial would explode dimensionality"}
        elif best_score > 0.90:
            return {"value": False, "confidence": "medium",
                    "reason": f"Model already at {best_score*100:.0f}% accuracy — polynomial unlikely to help significantly"}
        elif n_rows > 1000 and n_features <= 10:
            return {"value": True, "confidence": "medium",
                    "reason": "Enough data and moderate features — polynomial terms may capture non-linear relationships"}
        else:
            return {"value": False, "confidence": "low",
                    "reason": "Default: skip polynomial features unless model performance is unsatisfactory"}

    def _recommend_interactions(self, results, eda):
        """Recommend feature interactions."""
        correlations = eda.get("correlations") or {}
        n_features = len(results.get("selected_features") or results.get("original_columns") or [])

        if n_features > 20:
            return {"value": False, "confidence": "high",
                    "reason": f"Too many features ({n_features}) — interaction terms would create "
                              f"{n_features*(n_features-1)//2} new columns"}
        elif n_features <= 10:
            return {"value": True, "confidence": "low",
                    "reason": "Moderate feature count — interaction terms may capture combined effects"}
        else:
            return {"value": False, "confidence": "low",
                    "reason": "Default: skip interactions unless domain knowledge suggests specific pairs"}

    def _generate_config_summary(self, changes, current):
        if not changes:
            return "Current configuration is already optimal — no changes recommended."
        return f"{len(changes)} setting change(s) recommended for improved feature engineering quality."

    # ──────────────────────────────────────────────────────────
    # 6. NEXT STEPS
    # ──────────────────────────────────────────────────────────

    def generate_next_steps(
            self,
            feature_config: Dict[str, Any],
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
            pipeline_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate prioritized next steps for feature engineering.

        Returns ordered list of actions with priority, effort, and impact.
        """
        config = feature_config or {}
        results = feature_results or {}
        eda = eda_data or {}
        state = pipeline_state or {}

        steps = []

        # 1. Check if model training should be next
        phases_completed = state.get("phases_completed") or []
        fe_done = any("feature" in str(p).lower() or "phase 2" in str(p).lower() or "phase2" in str(p).lower()
                      for p in phases_completed)
        training_done = any("training" in str(p).lower() or "phase 3" in str(p).lower() or "phase3" in str(p).lower()
                            for p in phases_completed)

        if fe_done and not training_done:
            n_features = len(results.get("selected_features") or [])
            steps.append({
                "id": "proceed_training",
                "title": "Proceed to Model Training",
                "description": f"Feature engineering is complete with {n_features} features selected. "
                               f"Run Phase 3 (Model Training) to train your first models.",
                "priority": "high",
                "effort": "low",
                "impact": "high",
                "action": "Navigate to Model Selection → Configure algorithms → Start training",
                "category": "Pipeline Progress",
            })

        # 2. Fix critical issues
        error_patterns = self.detect_error_patterns(config, results, eda, model_versions)
        critical = [p for p in error_patterns.get("patterns", []) if p.get("severity") == "critical"]
        if critical:
            for cp in critical[:2]:
                steps.append({
                    "id": f"fix_{cp.get('type', 'issue')}",
                    "title": f"Fix: {cp.get('message', 'Critical issue')[:80]}",
                    "description": cp.get("recommendation", "Address this critical issue before training"),
                    "priority": "critical",
                    "effort": "medium",
                    "impact": "high",
                    "action": cp.get("recommendation", "Review and fix"),
                    "category": "Critical Fix",
                })

        # 3. Configuration improvements
        if model_versions:
            smart_config = self.generate_smart_config(results, eda, config, model_versions)
            changes = smart_config.get("changes_from_current") or []
            if changes:
                change_list = ", ".join(f"{c['setting']}: {c['current']}→{c['recommended']}" for c in changes[:3])
                steps.append({
                    "id": "optimize_config",
                    "title": f"Optimize Feature Engineering Configuration ({len(changes)} changes)",
                    "description": f"Recommended changes: {change_list}",
                    "priority": "medium",
                    "effort": "low",
                    "impact": "medium",
                    "action": "Re-run feature engineering with recommended settings",
                    "category": "Optimization",
                })

        # 4. Try alternative approaches
        encoding_strategies = results.get("encoding_strategies") or {}
        all_one_hot = all("one_hot" in str(s).lower() for s in encoding_strategies.values()) if encoding_strategies else True
        if all_one_hot and len(results.get("categorical_features") or []) > 5:
            steps.append({
                "id": "try_target_encoding",
                "title": "Try Target Encoding for High-Cardinality Features",
                "description": "All categorical columns use one-hot encoding. For columns with many categories, "
                               "target encoding can capture target relationships without dimensionality explosion.",
                "priority": "medium",
                "effort": "medium",
                "impact": "medium",
                "action": "Enable target encoding for columns with >10 categories",
                "category": "Experimentation",
            })

        # 5. Check for missing techniques
        poly_enabled = config.get("create_polynomial_features", False)
        interactions_enabled = config.get("create_interactions", False)
        n_features = len(results.get("selected_features") or [])

        if not poly_enabled and n_features <= 10:
            steps.append({
                "id": "try_polynomial",
                "title": "Experiment with Polynomial Features",
                "description": f"With only {n_features} features, polynomial terms (degree 2) may capture "
                               f"non-linear relationships that improve model accuracy.",
                "priority": "low",
                "effort": "low",
                "impact": "medium",
                "action": "Enable 'Create Polynomial Features (Degree 2)' and compare performance",
                "category": "Experimentation",
            })

        if not interactions_enabled and n_features <= 15:
            steps.append({
                "id": "try_interactions",
                "title": "Experiment with Feature Interactions",
                "description": "Feature interaction terms (A×B) can reveal combined effects that individual "
                               "features can't capture alone.",
                "priority": "low",
                "effort": "low",
                "impact": "low",
                "action": "Enable 'Create Interactions' and compare performance",
                "category": "Experimentation",
            })

        # 6. Feature engineering depth
        if training_done:
            best_score = 0
            if model_versions:
                for v in (model_versions.get("versions") or []):
                    s = _safe_float(v.get("accuracy") or v.get("test_score"), 0)
                    if s > best_score:
                        best_score = s

            if best_score < 0.80:
                steps.append({
                    "id": "domain_features",
                    "title": "Create Domain-Specific Features",
                    "description": f"Current best accuracy is {best_score*100:.1f}%. Consider creating "
                                   f"domain-specific features based on business knowledge — e.g., ratios, "
                                   f"aggregations, time-based features.",
                    "priority": "high",
                    "effort": "high",
                    "impact": "high",
                    "action": "Analyze which features correlate most with the target and create derived features",
                    "category": "Advanced Feature Engineering",
                })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        steps.sort(key=lambda s: priority_order.get(s.get("priority", "low"), 9))

        return {
            "steps": steps,
            "total_steps": len(steps),
            "critical_steps": len([s for s in steps if s["priority"] == "critical"]),
            "summary": f"{len(steps)} recommended next steps — "
                       f"{len([s for s in steps if s['priority'] in ('critical', 'high')])} high-priority.",
        }

    # ──────────────────────────────────────────────────────────
    # 7. FEATURE INTERACTIONS ANALYSIS
    # ──────────────────────────────────────────────────────────

    def analyze_feature_interactions(
            self,
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze feature relationships: correlations, clusters, interaction candidates.
        """
        results = feature_results or {}
        eda = eda_data or {}
        correlations = eda.get("correlations") or {}
        selected = results.get("selected_features") or []

        corr_matrix = correlations.get("matrix") or correlations.get("correlation_matrix") or {}
        target_corrs = correlations.get("target_correlations") or {}

        # Build correlation pairs
        pairs = []
        checked = set()
        if isinstance(corr_matrix, dict):
            for c1, row in corr_matrix.items():
                if c1 not in selected or not isinstance(row, dict):
                    continue
                for c2, val in row.items():
                    if c2 not in selected or c1 == c2:
                        continue
                    pk = tuple(sorted([c1, c2]))
                    if pk in checked:
                        continue
                    checked.add(pk)
                    cv = _safe_float(val, 0)
                    if cv is not None and abs(cv) > 0.3:
                        pairs.append({
                            "feature_a": c1, "feature_b": c2,
                            "correlation": round(cv, 4),
                            "strength": "strong" if abs(cv) > 0.7 else "moderate" if abs(cv) > 0.5 else "weak",
                        })

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Identify feature clusters (groups of correlated features)
        clusters = self._find_feature_clusters(pairs, selected)

        # Interaction candidates (moderately correlated with target but not with each other)
        interaction_candidates = []
        for i, f1 in enumerate(selected):
            for f2 in selected[i+1:]:
                t1 = abs(_safe_float(target_corrs.get(f1), 0))
                t2 = abs(_safe_float(target_corrs.get(f2), 0))

                # Both correlated with target
                if t1 > 0.1 and t2 > 0.1:
                    # But not too correlated with each other
                    mutual = 0
                    if isinstance(corr_matrix, dict):
                        row = corr_matrix.get(f1, {})
                        if isinstance(row, dict):
                            mutual = abs(_safe_float(row.get(f2), 0))

                    if mutual < 0.5:
                        interaction_candidates.append({
                            "feature_a": f1, "feature_b": f2,
                            "target_corr_a": round(t1, 4),
                            "target_corr_b": round(t2, 4),
                            "mutual_corr": round(mutual, 4),
                            "potential": "high" if t1 > 0.3 and t2 > 0.3 else "medium",
                        })

        interaction_candidates.sort(key=lambda x: (x["target_corr_a"] + x["target_corr_b"]), reverse=True)

        return {
            "correlation_pairs": pairs[:20],
            "feature_clusters": clusters,
            "interaction_candidates": interaction_candidates[:10],
            "target_correlations": {f: round(_safe_float(target_corrs.get(f), 0), 4) for f in selected},
            "summary": f"{len(pairs)} correlated pairs found, {len(clusters)} feature clusters, "
                       f"{len(interaction_candidates)} interaction candidates.",
        }

    def _find_feature_clusters(self, pairs, selected):
        """Simple clustering of correlated features."""
        # Union-find for clustering
        parent = {f: f for f in selected}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for p in pairs:
            if p["feature_a"] in parent and p["feature_b"] in parent:
                if abs(p["correlation"]) > 0.5:
                    union(p["feature_a"], p["feature_b"])

        clusters_map = {}
        for f in selected:
            root = find(f)
            if root not in clusters_map:
                clusters_map[root] = []
            clusters_map[root].append(f)

        # Only return clusters with 2+ features
        return [{"features": members, "size": len(members)}
                for members in clusters_map.values() if len(members) > 1]

    # ──────────────────────────────────────────────────────────
    # 8. DEEP ANALYSIS (combines all)
    # ──────────────────────────────────────────────────────────

    def deep_analysis(
            self,
            feature_config: Dict[str, Any],
            feature_results: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
            model_versions: Optional[Dict[str, Any]] = None,
            pipeline_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full feature engineering intelligence — combines all analyses.

        Returns everything the frontend needs in a single call.
        """
        # Run all analyses
        transformation_audit = self.analyze_transformations(
            feature_config, feature_results, eda_data, model_versions)

        selection_explanation = self.explain_feature_selection(
            feature_results, eda_data, model_versions)

        error_patterns = self.detect_error_patterns(
            feature_config, feature_results, eda_data, model_versions)

        quality_score = self.assess_quality(
            feature_config, feature_results, eda_data, model_versions)

        next_steps = self.generate_next_steps(
            feature_config, feature_results, eda_data, model_versions, pipeline_state)

        interactions = self.analyze_feature_interactions(
            feature_results, eda_data)

        return {
            "transformation_audit": transformation_audit,
            "selection_explanation": selection_explanation,
            "error_patterns": error_patterns,
            "quality_score": quality_score,
            "next_steps": next_steps,
            "feature_interactions": interactions,
            "pipeline_summary": self._generate_pipeline_summary(
                feature_config, feature_results, quality_score, error_patterns),
        }

    def _generate_pipeline_summary(self, config, results, quality, errors):
        """Generate a concise pipeline overview."""
        original = len(results.get("original_columns") or [])
        selected = len(results.get("selected_features") or [])
        numeric = len(results.get("numeric_features") or [])
        categorical = len(results.get("categorical_features") or [])
        n_rows = _safe_int(results.get("n_rows") or (results.get("original_shape") or [0])[0])
        scaling = config.get("scaling_method", "standard")

        return {
            "data_shape": f"{n_rows:,} rows × {original} columns → {selected} features",
            "column_types": f"{numeric} numeric, {categorical} categorical",
            "scaling": scaling,
            "quality_grade": quality.get("grade") if "grade" in quality else quality.get("verdict", "unknown"),
            "quality_score": f"{quality.get('score', 0)}/{quality.get('max_score', 0)} ({quality.get('percentage', 0)}%)",
            "issues_found": errors.get("total_issues", 0),
            "health": errors.get("health", "unknown"),
        }

    # ──────────────────────────────────────────────────────────
    # 9. COMPARE CONFIGS
    # ──────────────────────────────────────────────────────────

    def compare_configs(
            self,
            config_a: Dict[str, Any],
            results_a: Dict[str, Any],
            config_b: Dict[str, Any],
            results_b: Dict[str, Any],
            eda_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compare two feature engineering configurations side-by-side.
        """
        quality_a = self.assess_quality(config_a, results_a, eda_data)
        quality_b = self.assess_quality(config_b, results_b, eda_data)

        differences = []
        all_keys = set(list(config_a.keys()) + list(config_b.keys()))
        for key in sorted(all_keys):
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if str(val_a) != str(val_b):
                differences.append({
                    "setting": key,
                    "config_a": val_a,
                    "config_b": val_b,
                })

        # Compare results
        features_a = len(results_a.get("selected_features") or [])
        features_b = len(results_b.get("selected_features") or [])

        winner = "A" if quality_a["percentage"] > quality_b["percentage"] else \
            "B" if quality_b["percentage"] > quality_a["percentage"] else "tie"

        return {
            "config_a_quality": quality_a,
            "config_b_quality": quality_b,
            "differences": differences,
            "features_a": features_a,
            "features_b": features_b,
            "winner": winner,
            "recommendation": f"Configuration {'A' if winner == 'A' else 'B'} scores higher "
                              f"({max(quality_a['percentage'], quality_b['percentage'])}% vs "
                              f"{min(quality_a['percentage'], quality_b['percentage'])}%)" if winner != "tie"
            else "Both configurations score equally",
        }