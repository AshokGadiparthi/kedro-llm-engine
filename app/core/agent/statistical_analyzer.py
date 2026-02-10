"""
Statistical Analyzer — Deep Metadata-Driven Statistical Engine
================================================================
Computes derived statistical measures from EdaResult metadata.
NEVER touches raw data. Works entirely from pre-computed statistics
stored in EdaResult.summary, EdaResult.statistics, EdaResult.quality.

Capabilities:
  1. Distribution Profiling   — Skewness classification, kurtosis, modality estimation
  2. Outlier Scoring          — IQR-based outlier fraction estimation from percentiles
  3. Normality Assessment     — Heuristic normality scoring from skew + kurtosis
  4. Information Measures     — Estimated entropy & mutual information from cardinality
  5. Cardinality Analysis     — High/low cardinality detection, encoding recommendations
  6. Feature-Target Scoring   — Correlation-to-target ranking, predictive power estimation
  7. Data Shape Analysis      — Wide vs tall, sparse vs dense, balanced vs imbalanced
  8. Temporal Pattern Detection— Detect monotonic trends, date columns, time-series signals
  9. Missing Data Mechanism   — MCAR/MAR/MNAR likelihood estimation from patterns
  10. Composite Scores        — Overall data readiness, feature quality, modeling difficulty
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DistributionProfile:
    """Profile of a single feature's distribution."""
    column: str
    skewness: float = 0.0
    kurtosis: float = 0.0
    skew_class: str = "symmetric"         # symmetric | moderate_skew | high_skew | extreme_skew
    tail_class: str = "mesokurtic"        # platykurtic | mesokurtic | leptokurtic
    normality_score: float = 0.5          # 0=clearly non-normal, 1=approximately normal
    outlier_fraction: float = 0.0         # estimated % of outliers via IQR
    is_zero_inflated: bool = False
    is_bimodal_candidate: bool = False
    suggested_transform: Optional[str] = None  # log | sqrt | box_cox | yeo_johnson | none

@dataclass
class CardinalityProfile:
    """Cardinality analysis for a categorical feature."""
    column: str
    unique_count: int = 0
    unique_ratio: float = 0.0             # unique_count / total_rows
    cardinality_class: str = "low"        # low | medium | high | very_high | likely_id
    recommended_encoding: str = "onehot"  # onehot | ordinal | target | hash | drop
    is_binary: bool = False
    is_constant: bool = False

@dataclass
class FeatureTargetScore:
    """Estimated predictive power of a feature w.r.t. the target."""
    column: str
    correlation_abs: float = 0.0          # absolute correlation with target
    estimated_mi: float = 0.0             # estimated mutual information
    predictive_rank: int = 0              # rank among all features
    predictive_class: str = "unknown"     # strong | moderate | weak | noise | leakage_suspect

@dataclass
class DataReadinessReport:
    """Overall assessment of data readiness for modeling."""
    overall_score: float = 0.0            # 0-100
    completeness_score: float = 0.0
    quality_score: float = 0.0
    feature_score: float = 0.0
    target_score: float = 0.0
    complexity_score: float = 0.0         # higher = more difficult to model
    estimated_difficulty: str = "medium"  # easy | medium | hard | very_hard
    bottlenecks: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


class StatisticalAnalyzer:
    """
    Computes deep statistical analysis from compiled context metadata.
    All methods are pure functions on metadata — no database access.
    """

    # ──────────────────────────────────────────────────────────
    # 1. DISTRIBUTION PROFILING
    # ──────────────────────────────────────────────────────────

    def profile_distributions(self, context: Dict[str, Any]) -> List[DistributionProfile]:
        """
        Profile all numeric feature distributions from EDA statistics.
        Uses skewness, kurtosis, percentiles to classify distributions.
        """
        profiles = []
        feature_stats = context.get("feature_stats", {})
        numeric_stats = feature_stats.get("numeric_stats", {})

        for col, stats in numeric_stats.items():
            if not isinstance(stats, dict):
                continue

            skew = stats.get("skewness", 0.0) or 0.0
            kurt = stats.get("kurtosis", 0.0) or 0.0
            mean = stats.get("mean", 0.0) or 0.0
            std = stats.get("std", 1.0) or 1.0
            min_val = stats.get("min", 0.0)
            max_val = stats.get("max", 0.0)
            q1 = stats.get("25%", stats.get("q1", 0.0)) or 0.0
            q3 = stats.get("75%", stats.get("q3", 0.0)) or 0.0
            median = stats.get("50%", stats.get("median", mean)) or mean
            zero_count = stats.get("zero_count", 0)
            count = stats.get("count", 1) or 1

            profile = DistributionProfile(column=col, skewness=skew, kurtosis=kurt)

            # Classify skewness
            abs_skew = abs(skew)
            if abs_skew < 0.5:
                profile.skew_class = "symmetric"
            elif abs_skew < 1.0:
                profile.skew_class = "moderate_skew"
            elif abs_skew < 2.0:
                profile.skew_class = "high_skew"
            else:
                profile.skew_class = "extreme_skew"

            # Classify kurtosis (excess kurtosis: normal = 0)
            if kurt < -1:
                profile.tail_class = "platykurtic"   # light tails, flat top
            elif kurt > 3:
                profile.tail_class = "leptokurtic"   # heavy tails, sharp peak
            else:
                profile.tail_class = "mesokurtic"     # approximately normal

            # Normality score (heuristic: penalize skew and excess kurtosis)
            skew_penalty = min(abs_skew / 2.0, 1.0)
            kurt_penalty = min(abs(kurt) / 6.0, 1.0)
            profile.normality_score = round(max(0, 1.0 - 0.6 * skew_penalty - 0.4 * kurt_penalty), 3)

            # Outlier fraction estimation via IQR
            iqr = q3 - q1
            if iqr > 0:
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                # Estimate: what fraction of the range is outside fences
                total_range = max_val - min_val
                if total_range > 0:
                    below = max(0, lower_fence - min_val) / total_range
                    above = max(0, max_val - upper_fence) / total_range
                    profile.outlier_fraction = round(min(below + above, 0.5), 4)

            # Zero-inflation detection
            if count > 0 and zero_count:
                zero_pct = zero_count / count
                if zero_pct > 0.3 and min_val >= 0:
                    profile.is_zero_inflated = True

            # Bimodality heuristic: if mean is far from median relative to std
            if std > 0:
                mean_median_gap = abs(mean - median) / std
                if mean_median_gap > 0.3 and profile.skew_class in ("moderate_skew", "high_skew"):
                    profile.is_bimodal_candidate = True

            # Transform suggestion
            if profile.skew_class == "extreme_skew":
                if min_val > 0:
                    profile.suggested_transform = "log"
                elif min_val >= 0:
                    profile.suggested_transform = "sqrt"
                else:
                    profile.suggested_transform = "yeo_johnson"
            elif profile.skew_class == "high_skew":
                if min_val > 0:
                    profile.suggested_transform = "log"
                else:
                    profile.suggested_transform = "yeo_johnson"
            elif profile.skew_class == "moderate_skew":
                profile.suggested_transform = "yeo_johnson"
            else:
                profile.suggested_transform = "none"

            profiles.append(profile)

        return profiles

    # ──────────────────────────────────────────────────────────
    # 2. CARDINALITY ANALYSIS
    # ──────────────────────────────────────────────────────────

    def analyze_cardinality(self, context: Dict[str, Any]) -> List[CardinalityProfile]:
        """
        Classify categorical features by cardinality and recommend encoding.
        """
        profiles = []
        feature_stats = context.get("feature_stats", {})
        dataset_profile = context.get("dataset_profile", {})
        rows = dataset_profile.get("rows", 1) or 1
        cat_stats = feature_stats.get("categorical_stats", {})
        col_types = dataset_profile.get("column_types", {})
        cat_cols = col_types.get("categorical", [])

        for col in cat_cols:
            stats = cat_stats.get(col, {})
            if not isinstance(stats, dict):
                continue

            unique = stats.get("unique", stats.get("nunique", 0)) or 0
            cp = CardinalityProfile(
                column=col,
                unique_count=unique,
                unique_ratio=round(unique / rows, 4) if rows > 0 else 0,
            )

            # Classify cardinality
            if unique <= 1:
                cp.cardinality_class = "constant"
                cp.is_constant = True
                cp.recommended_encoding = "drop"
            elif unique == 2:
                cp.cardinality_class = "binary"
                cp.is_binary = True
                cp.recommended_encoding = "onehot"
            elif unique <= 10:
                cp.cardinality_class = "low"
                cp.recommended_encoding = "onehot"
            elif unique <= 50:
                cp.cardinality_class = "medium"
                cp.recommended_encoding = "ordinal"
            elif unique <= 200:
                cp.cardinality_class = "high"
                cp.recommended_encoding = "target"
            elif cp.unique_ratio > 0.8:
                cp.cardinality_class = "likely_id"
                cp.recommended_encoding = "drop"
            else:
                cp.cardinality_class = "very_high"
                cp.recommended_encoding = "hash"

            profiles.append(cp)

        return profiles

    # ──────────────────────────────────────────────────────────
    # 3. FEATURE-TARGET SCORING
    # ──────────────────────────────────────────────────────────

    def score_feature_target(self, context: Dict[str, Any]) -> List[FeatureTargetScore]:
        """
        Score features by estimated predictive power w.r.t. target.
        Uses correlations and heuristic mutual information estimates.
        """
        scores = []
        correlations = context.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])
        all_correlations = correlations.get("all_pairs", {})

        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")

        if not target:
            return scores

        # Build target correlation map
        target_corrs = {}
        if isinstance(all_correlations, dict):
            for key, val in all_correlations.items():
                if target in str(key):
                    # Parse the feature name from correlation key
                    parts = str(key).replace("(", "").replace(")", "").replace("'", "").split(",")
                    for part in parts:
                        part = part.strip()
                        if part != target and part:
                            target_corrs[part] = abs(float(val)) if val else 0

        # Also check high_pairs for target correlations
        for pair in high_pairs:
            f1, f2 = pair.get("feature1", ""), pair.get("feature2", "")
            corr = abs(pair.get("correlation", 0))
            if f1 == target:
                target_corrs[f2] = max(target_corrs.get(f2, 0), corr)
            elif f2 == target:
                target_corrs[f1] = max(target_corrs.get(f1, 0), corr)

        # Score all features
        all_features = list(set(
            list(target_corrs.keys()) +
            context.get("dataset_profile", {}).get("column_names", [])
        ))

        for feat in all_features:
            if feat == target:
                continue

            corr = target_corrs.get(feat, 0)

            # Estimate mutual information from correlation
            # MI ≈ -0.5 * ln(1 - r²) for bivariate normal
            if corr < 1.0:
                estimated_mi = -0.5 * math.log(max(1 - corr ** 2, 1e-10))
            else:
                estimated_mi = 5.0  # Capped for perfect correlation

            fts = FeatureTargetScore(
                column=feat,
                correlation_abs=round(corr, 4),
                estimated_mi=round(estimated_mi, 4),
            )

            # Classify predictive power
            if corr >= 0.95:
                fts.predictive_class = "leakage_suspect"
            elif corr >= 0.3:
                fts.predictive_class = "strong"
            elif corr >= 0.15:
                fts.predictive_class = "moderate"
            elif corr >= 0.05:
                fts.predictive_class = "weak"
            else:
                fts.predictive_class = "noise"

            scores.append(fts)

        # Rank by correlation
        scores.sort(key=lambda s: s.correlation_abs, reverse=True)
        for i, s in enumerate(scores):
            s.predictive_rank = i + 1

        return scores

    # ──────────────────────────────────────────────────────────
    # 4. MISSING DATA MECHANISM ESTIMATION
    # ──────────────────────────────────────────────────────────

    def estimate_missing_mechanism(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate whether missing data is MCAR, MAR, or MNAR.
        
        Heuristic approach (without access to raw data):
        - MCAR: Missing pattern is random (similar % across all columns)
        - MAR: Missingness correlates with observed variables (clustered missingness)
        - MNAR: Missingness depends on the missing value itself (can't detect from metadata alone)
        """
        quality = context.get("data_quality", {})
        missing_by_col = quality.get("missing_by_column", {})

        if not missing_by_col:
            return {
                "mechanism": "complete",
                "confidence": 1.0,
                "explanation": "No missing values detected.",
                "imputation_strategy": "none_needed",
            }

        # Collect missing percentages
        pcts = []
        for col, info in missing_by_col.items():
            pct = info.get("percent", 0) if isinstance(info, dict) else 0
            if pct > 0:
                pcts.append(pct)

        if not pcts:
            return {
                "mechanism": "complete",
                "confidence": 1.0,
                "explanation": "No missing values detected.",
                "imputation_strategy": "none_needed",
            }

        avg_pct = sum(pcts) / len(pcts)
        std_pct = (sum((p - avg_pct) ** 2 for p in pcts) / max(len(pcts) - 1, 1)) ** 0.5
        cv = std_pct / avg_pct if avg_pct > 0 else 0  # Coefficient of variation

        # MCAR heuristic: low variation in missing % across columns
        if cv < 0.5 and avg_pct < 20:
            return {
                "mechanism": "likely_MCAR",
                "confidence": round(max(0.5, 1.0 - cv), 2),
                "explanation": (
                    f"Missing values are relatively uniformly distributed across columns "
                    f"(avg: {avg_pct:.1f}%, std: {std_pct:.1f}%). This suggests Missing Completely "
                    f"At Random (MCAR), meaning missingness is independent of the data."
                ),
                "imputation_strategy": "simple",
                "recommended_methods": [
                    "Median imputation for numeric columns",
                    "Mode imputation for categorical columns",
                    "Simple random imputation preserves distribution",
                ],
                "columns_affected": len(pcts),
                "avg_missing_pct": round(avg_pct, 1),
            }

        # MAR heuristic: high variation suggests missingness is structured
        if cv > 1.0 or max(pcts) > 3 * min(pcts):
            return {
                "mechanism": "likely_MAR",
                "confidence": round(min(0.8, cv / 3), 2),
                "explanation": (
                    f"Missing values vary significantly across columns (range: "
                    f"{min(pcts):.1f}% — {max(pcts):.1f}%). This structured pattern suggests "
                    f"Missing At Random (MAR), where missingness depends on other observed variables."
                ),
                "imputation_strategy": "multivariate",
                "recommended_methods": [
                    "IterativeImputer (sklearn) — models each feature as a function of others",
                    "KNNImputer — uses K nearest neighbors in feature space",
                    "MICE (Multiple Imputation by Chained Equations) for statistical analysis",
                ],
                "columns_affected": len(pcts),
                "avg_missing_pct": round(avg_pct, 1),
                "high_missing_columns": [
                    col for col, info in missing_by_col.items()
                    if (info.get("percent", 0) if isinstance(info, dict) else 0) > 20
                ],
            }

        # Default: inconclusive
        return {
            "mechanism": "inconclusive",
            "confidence": 0.4,
            "explanation": (
                f"Missing data pattern is inconclusive from metadata alone. "
                f"Average missing: {avg_pct:.1f}%, variation: {cv:.2f}. "
                f"Consider running Little's MCAR test on the raw data."
            ),
            "imputation_strategy": "cautious_multivariate",
            "recommended_methods": [
                "IterativeImputer as a safe default",
                "Add binary missing indicators for columns with >5% missing",
                "Compare model performance with and without imputation",
            ],
            "columns_affected": len(pcts),
            "avg_missing_pct": round(avg_pct, 1),
        }

    # ──────────────────────────────────────────────────────────
    # 5. DATA SHAPE ANALYSIS
    # ──────────────────────────────────────────────────────────

    def analyze_data_shape(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive shape analysis: wide vs tall, density, complexity.
        """
        profile = context.get("dataset_profile", {})
        quality = context.get("data_quality", {})
        rows = profile.get("rows", 0) or 0
        cols = profile.get("columns", 0) or 0
        numeric = profile.get("numeric_count", 0)
        categorical = profile.get("categorical_count", 0)
        datetime_count = profile.get("datetime_count", 0)
        completeness = quality.get("completeness", 100)
        dup_pct = quality.get("duplicate_pct", 0)

        if rows == 0 or cols == 0:
            return {"error": "No data shape available"}

        # Effective samples (after dedup and missing)
        effective_rows = int(rows * (1 - dup_pct / 100) * (completeness / 100))
        ratio = rows / cols if cols > 0 else float('inf')

        # Shape classification
        if cols > rows:
            shape_class = "ultra_wide"
            shape_risk = "critical"
        elif ratio < 5:
            shape_class = "wide"
            shape_risk = "high"
        elif ratio < 20:
            shape_class = "moderately_wide"
            shape_risk = "medium"
        elif ratio > 1000:
            shape_class = "very_tall"
            shape_risk = "low"
        else:
            shape_class = "balanced"
            shape_risk = "low"

        # Density (non-missing fraction)
        density = completeness / 100

        # Feature composition
        total_features = numeric + categorical + datetime_count
        composition = {
            "numeric_pct": round(numeric / total_features * 100, 1) if total_features > 0 else 0,
            "categorical_pct": round(categorical / total_features * 100, 1) if total_features > 0 else 0,
            "datetime_pct": round(datetime_count / total_features * 100, 1) if total_features > 0 else 0,
            "is_mixed_type": numeric > 0 and categorical > 0,
            "dominant_type": "numeric" if numeric > categorical else "categorical" if categorical > numeric else "mixed",
        }

        # Sample size adequacy
        features_per_sample = cols / rows if rows > 0 else float('inf')
        if features_per_sample > 1:
            size_adequacy = "grossly_insufficient"
        elif features_per_sample > 0.1:
            size_adequacy = "insufficient"
        elif features_per_sample > 0.02:
            size_adequacy = "marginal"
        elif features_per_sample > 0.005:
            size_adequacy = "adequate"
        else:
            size_adequacy = "abundant"

        return {
            "shape_class": shape_class,
            "shape_risk": shape_risk,
            "rows": rows,
            "columns": cols,
            "effective_rows": effective_rows,
            "row_col_ratio": round(ratio, 1),
            "density": round(density, 3),
            "composition": composition,
            "size_adequacy": size_adequacy,
            "features_per_sample": round(features_per_sample, 4),
            "has_temporal": datetime_count > 0,
            "max_safe_features": int(effective_rows / 10),
            "max_safe_polynomial": int(effective_rows / 20),
        }

    # ──────────────────────────────────────────────────────────
    # 6. TEMPORAL PATTERN DETECTION
    # ──────────────────────────────────────────────────────────

    def detect_temporal_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if the dataset has time-series characteristics.
        Checks for: datetime columns, monotonic features, sequential IDs.
        """
        profile = context.get("dataset_profile", {})
        feature_stats = context.get("feature_stats", {})
        col_types = profile.get("column_types", {})

        datetime_cols = col_types.get("datetime", [])
        numeric_stats = feature_stats.get("numeric_stats", {})
        column_names = profile.get("column_names", [])

        signals = {
            "has_datetime_columns": len(datetime_cols) > 0,
            "datetime_columns": datetime_cols,
            "temporal_name_patterns": [],
            "monotonic_candidates": [],
            "is_likely_timeseries": False,
            "recommendation": "",
        }

        # Check column names for temporal patterns
        temporal_keywords = [
            "date", "time", "timestamp", "year", "month", "day", "week",
            "hour", "minute", "created", "updated", "period", "quarter",
            "epoch", "datetime", "ts", "_at", "_on",
        ]
        for col in column_names:
            col_lower = col.lower()
            for keyword in temporal_keywords:
                if keyword in col_lower:
                    signals["temporal_name_patterns"].append(col)
                    break

        # Check for monotonic numeric features (potential index/time proxy)
        for col, stats in numeric_stats.items():
            if not isinstance(stats, dict):
                continue
            min_val = stats.get("min", 0) or 0
            max_val = stats.get("max", 0) or 0
            count = stats.get("count", 0) or 0
            unique = stats.get("unique", count)

            if count > 0 and unique == count and max_val > min_val:
                # Perfectly unique + sequential range suggests monotonic
                expected_range = count - 1
                actual_range = max_val - min_val
                if actual_range > 0 and abs(expected_range / actual_range - 1) < 0.1:
                    signals["monotonic_candidates"].append(col)

        # Overall assessment
        temporal_signals = (
                len(datetime_cols) +
                len(signals["temporal_name_patterns"]) +
                len(signals["monotonic_candidates"])
        )
        signals["is_likely_timeseries"] = temporal_signals >= 2

        if signals["is_likely_timeseries"]:
            signals["recommendation"] = (
                "This dataset appears to have temporal structure. Consider: "
                "(1) Time-based train/test split instead of random split. "
                "(2) Extract temporal features (day_of_week, month, lag features). "
                "(3) Check for temporal leakage (future data in features). "
                "(4) Use time-series cross-validation (TimeSeriesSplit)."
            )
        elif datetime_cols:
            signals["recommendation"] = (
                "Datetime columns detected. Extract temporal features and consider "
                "whether the prediction task is time-dependent."
            )

        return signals

    # ──────────────────────────────────────────────────────────
    # 7. COMPOSITE DATA READINESS SCORE
    # ──────────────────────────────────────────────────────────

    def compute_data_readiness(self, context: Dict[str, Any]) -> DataReadinessReport:
        """
        Compute a comprehensive data readiness score (0-100).
        Factors: completeness, quality, feature health, target health, complexity.
        """
        report = DataReadinessReport()

        quality = context.get("data_quality", {})
        profile = context.get("dataset_profile", {})
        correlations = context.get("correlations", {})
        feature_stats = context.get("feature_stats", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        # ── Completeness Score (0-100) ──
        completeness = quality.get("completeness", 100)
        high_missing = quality.get("columns_with_high_missing", [])
        report.completeness_score = max(0, completeness - len(high_missing) * 5)
        if completeness >= 95:
            report.strengths.append("Excellent data completeness")
        elif completeness < 70:
            report.bottlenecks.append(f"Low completeness ({completeness:.0f}%)")

        # ── Quality Score (0-100) ──
        dup_pct = quality.get("duplicate_pct", 0)
        quality_raw = quality.get("overall_quality_score", 80)
        dup_penalty = min(dup_pct * 2, 30)  # Cap at 30pt penalty
        report.quality_score = max(0, quality_raw - dup_penalty)
        if dup_pct > 10:
            report.bottlenecks.append(f"High duplicate rate ({dup_pct:.1f}%)")
        if quality_raw >= 80:
            report.strengths.append("Good overall data quality")

        # ── Feature Score (0-100) ──
        feature_score = 70  # Base score
        # Bonus for mixed types (good for tree models)
        numeric = profile.get("numeric_count", 0)
        categorical = profile.get("categorical_count", 0)
        if numeric > 0 and categorical > 0:
            feature_score += 10
        # Penalty for too many features relative to samples
        if rows > 0 and cols > 0:
            ratio = rows / cols
            if ratio < 5:
                feature_score -= 30
                report.bottlenecks.append(f"Low sample-to-feature ratio ({ratio:.1f}:1)")
            elif ratio < 20:
                feature_score -= 10
        # Penalty for high multicollinearity
        high_pairs = correlations.get("high_pairs", [])
        high_corr = [p for p in high_pairs if p.get("abs_correlation", 0) >= 0.8]
        if len(high_corr) > 5:
            feature_score -= 15
            report.bottlenecks.append(f"{len(high_corr)} highly correlated feature pairs")
        # Bonus for low ID column contamination
        id_cols = feature_stats.get("potential_id_columns", [])
        if not id_cols:
            feature_score += 5
            report.strengths.append("No ID column contamination detected")
        else:
            feature_score -= 10 * len(id_cols)
            report.bottlenecks.append(f"{len(id_cols)} potential ID columns in features")

        report.feature_score = max(0, min(100, feature_score))

        # ── Target Score (0-100) ──
        target_score = 60  # Unknown target = assume some risk, not "decent"
        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}

        # Bridge: also check target_variable directly
        target_info = context.get("target_variable", {})
        target = screen_ctx.get("target_column") or frontend.get("target_column")
        if not target and isinstance(target_info, dict):
            target = target_info.get("name")

        if target:
            numeric_stats = feature_stats.get("numeric_stats", {})
            cat_stats = feature_stats.get("categorical_stats", {})

            if target in numeric_stats:
                stats = numeric_stats[target]
                if isinstance(stats, dict):
                    mean_val = stats.get("mean", 0.5)
                    if 0 <= mean_val <= 1:
                        minority_pct = min(mean_val, 1 - mean_val) * 100
                        if minority_pct < 5:
                            target_score = 30
                            report.bottlenecks.append(f"Severe class imbalance ({minority_pct:.1f}% minority)")
                        elif minority_pct < 15:
                            target_score = 55
                            report.bottlenecks.append(f"Moderate class imbalance ({minority_pct:.1f}% minority)")
                        elif minority_pct >= 30:
                            target_score = 90
                            report.strengths.append("Well-balanced target distribution")
                        else:
                            target_score = 75

            elif target in cat_stats:
                # Categorical target — check unique count
                cs = cat_stats[target]
                if isinstance(cs, dict):
                    unique = cs.get("unique", 0)
                    if unique == 2:
                        target_score = 80
                        report.strengths.append(f"Binary target detected: '{target}'")
                    elif unique <= 10:
                        target_score = 70
                        report.strengths.append(f"Multi-class target detected: '{target}' ({unique} classes)")
                    elif unique <= 20:
                        target_score = 55
                        report.bottlenecks.append(f"Many target classes ({unique}) — may need grouping")
                    else:
                        target_score = 35
                        report.bottlenecks.append(f"Target '{target}' has {unique} classes — unlikely classification target")
            else:
                target_score = 70  # Target found but no stats available
                report.strengths.append(f"Target column identified: '{target}'")
        else:
            report.bottlenecks.append("Target column not specified — cannot assess class balance")

        report.target_score = target_score

        # ── Complexity Score (0-100, higher = harder) ──
        complexity = 40  # Base difficulty
        if rows < 500:
            complexity += 25
        elif rows < 2000:
            complexity += 10
        if cols > 50:
            complexity += 15
        if len(high_corr) > 3:
            complexity += 10
        if completeness < 80:
            complexity += 15
        report.complexity_score = min(100, complexity)

        if complexity >= 75:
            report.estimated_difficulty = "very_hard"
        elif complexity >= 55:
            report.estimated_difficulty = "hard"
        elif complexity >= 35:
            report.estimated_difficulty = "medium"
        else:
            report.estimated_difficulty = "easy"

        # ── Overall Score ──
        weights = {
            "completeness": 0.20,
            "quality": 0.20,
            "feature": 0.25,
            "target": 0.20,
            "complexity": 0.15,  # inverted: low complexity = high score
        }
        report.overall_score = round(
            weights["completeness"] * report.completeness_score +
            weights["quality"] * report.quality_score +
            weights["feature"] * report.feature_score +
            weights["target"] * report.target_score +
            weights["complexity"] * (100 - report.complexity_score),
            1
        )

        return report

    # ──────────────────────────────────────────────────────────
    # 8. MULTICOLLINEARITY CLUSTER DETECTION
    # ──────────────────────────────────────────────────────────

    def detect_correlation_clusters(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find clusters of inter-correlated features (not just pairs).
        Uses Union-Find on high-correlation pairs to group related features.
        """
        correlations = context.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])
        threshold = 0.7

        # Filter relevant pairs
        relevant = [p for p in high_pairs if p.get("abs_correlation", 0) >= threshold]
        if not relevant:
            return []

        # Union-Find
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for pair in relevant:
            union(pair["feature1"], pair["feature2"])

        # Group clusters
        clusters = {}
        for feat in parent:
            root = find(feat)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(feat)

        result = []
        for root, members in clusters.items():
            if len(members) >= 2:
                # Find the pair correlations within this cluster
                cluster_pairs = [
                    p for p in relevant
                    if p["feature1"] in members and p["feature2"] in members
                ]
                avg_corr = sum(p.get("abs_correlation", 0) for p in cluster_pairs) / max(len(cluster_pairs), 1)

                result.append({
                    "cluster_id": root,
                    "features": sorted(members),
                    "size": len(members),
                    "avg_internal_correlation": round(avg_corr, 3),
                    "pairs": [
                        {"f1": p["feature1"], "f2": p["feature2"], "r": round(p["correlation"], 3)}
                        for p in cluster_pairs
                    ],
                    "recommendation": (
                        f"These {len(members)} features are inter-correlated. "
                        f"Keep the one with highest target correlation, or use PCA to "
                        f"combine them into 1-2 components."
                    ),
                })

        result.sort(key=lambda c: c["size"], reverse=True)
        return result

    # ──────────────────────────────────────────────────────────
    # 9. FEATURE TYPE MISMATCH DETECTION
    # ──────────────────────────────────────────────────────────

    def detect_type_mismatches(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect features that may be mistyped:
        - Numeric columns that are really categorical (e.g., zip codes, IDs)
        - Categorical columns that are really numeric (e.g., "5.0", "100")
        - Boolean columns stored as strings
        """
        mismatches = []
        feature_stats = context.get("feature_stats", {})
        profile = context.get("dataset_profile", {})
        numeric_stats = feature_stats.get("numeric_stats", {})
        col_types = profile.get("column_types", {})
        rows = profile.get("rows", 1) or 1

        # Check numeric columns that might be categorical
        numeric_cols = col_types.get("numeric", [])
        for col in numeric_cols:
            stats = numeric_stats.get(col, {})
            if not isinstance(stats, dict):
                continue

            unique = stats.get("unique", rows)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)

            # Low unique count relative to rows → likely categorical
            if unique <= 10 and rows > 100:
                mismatches.append({
                    "column": col,
                    "current_type": "numeric",
                    "suggested_type": "categorical",
                    "reason": f"Only {unique} unique values in {rows} rows",
                    "confidence": 0.8 if unique <= 5 else 0.6,
                    "action": f"Convert '{col}' to categorical before encoding. "
                              f"This changes how the model interprets the values — "
                              f"from continuous magnitude to discrete groups.",
                })

            # Integer-only with small range → might be ordinal
            if (min_val == int(min_val) and max_val == int(max_val) and
                    (max_val - min_val) < 10 and unique <= max_val - min_val + 1):
                if col.lower() not in ("age", "year", "count", "total", "amount", "price"):
                    mismatches.append({
                        "column": col,
                        "current_type": "numeric",
                        "suggested_type": "ordinal",
                        "reason": f"Integer values in range [{int(min_val)}, {int(max_val)}] with {unique} unique",
                        "confidence": 0.5,
                        "action": f"Verify if '{col}' represents ordered categories (e.g., ratings, levels). "
                                  f"If so, treat as ordinal rather than continuous.",
                    })

        return mismatches

    # ──────────────────────────────────────────────────────────
    # 10. AGGREGATE ANALYSIS
    # ──────────────────────────────────────────────────────────

    def full_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ALL statistical analyses and return a comprehensive report.
        This is the main entry point for the orchestrator.
        """
        # Bridge target_variable → screen_context for all sub-analyses
        target_info = context.get("target_variable", {})
        if isinstance(target_info, dict) and target_info.get("name"):
            sc = context.get("screen_context")
            if sc is None:
                sc = {}
                context["screen_context"] = sc
            if not sc.get("target_column"):
                sc["target_column"] = target_info["name"]
            fs = context.get("frontend_state")
            if fs is None:
                fs = {}
                context["frontend_state"] = fs
            if not fs.get("target_column"):
                fs["target_column"] = target_info["name"]

        return {
            "distributions": [vars(d) for d in self.profile_distributions(context)],
            "cardinality": [vars(c) for c in self.analyze_cardinality(context)],
            "feature_target_scores": [vars(s) for s in self.score_feature_target(context)],
            "missing_mechanism": self.estimate_missing_mechanism(context),
            "data_shape": self.analyze_data_shape(context),
            "temporal_patterns": self.detect_temporal_patterns(context),
            "data_readiness": vars(self.compute_data_readiness(context)),
            "correlation_clusters": self.detect_correlation_clusters(context),
            "type_mismatches": self.detect_type_mismatches(context),
        }
