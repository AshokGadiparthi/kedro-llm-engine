"""
Rule Engine — 120+ Deterministic ML Expert Rules
=================================================
The brain that NEVER hallucinates. Every rule is grounded in statistics,
ML theory, and production engineering best practices.

Architecture:
  - Rules are organized into 15 categories
  - Each rule has: severity, impact score, confidence, category, evidence
  - Rules reference specific metrics/thresholds from the compiled context
  - Output is a prioritized list of Insight objects

Categories:
   1. Data Quality          — Completeness, validity, consistency
   2. Feature Health        — Distributions, cardinality, variance
   3. Feature Engineering   — Transformations, encodings, scaling
   4. Target Variable       — Imbalance, leakage, type detection
   5. Data Leakage          — Future data, proxy variables, ID columns
   6. Multicollinearity     — VIF, correlation clusters, redundancy
   7. Sample Size           — Power analysis, feature-to-sample ratio
   8. Training Config       — Split strategy, CV folds, scaling choices
   9. Algorithm Selection   — Algorithm-data fit, complexity budget
  10. Evaluation Metrics    — Metric traps, threshold analysis, calibration
  11. Model Comparison      — Significance testing, Occam's razor
  12. Production Readiness  — Latency, robustness, monitoring hooks
  13. Deployment Safety     — Shadow testing, canary, rollback
  14. Monitoring & Drift    — Data drift, concept drift, performance decay
  15. Pipeline Lifecycle    — Phase ordering, reproducibility, versioning

Severity Levels:
  critical  — Must fix before proceeding. Will cause model failure.
  warning   — Should address. Significant risk to model quality.
  info      — Good to know. Optimization opportunity.
  tip       — Expert suggestion. Advanced technique.
  success   — Something done well. Positive reinforcement.

Design Principles:
  - ZERO hallucination: Every insight is derived from deterministic logic
  - Evidence-based: Each rule includes the metric that triggered it
  - Actionable: Every non-success insight has a concrete next step
  - Contextual: Rules activate only when their screen/context applies
  - Non-obvious: Rules encode expert knowledge most users wouldn't know
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Late import to avoid circular dependency
def _get_extended_mixin():
    try:
        from .rules_extended import ExtendedRulesMixin
        return ExtendedRulesMixin
    except ImportError:
        return object


# ═══════════════════════════════════════════════════════════════════
# INSIGHT DATA CLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Insight:
    severity: str              # critical | warning | info | tip | success
    category: str              # One of 15 categories
    title: str                 # Short headline
    message: str               # Detailed explanation
    action: Optional[str]      # What to do about it
    evidence: Optional[str] = None    # The data/metric that triggered this
    metric_key: Optional[str] = None
    metric_value: Optional[float] = None
    impact: Optional[str] = None      # high | medium | low
    confidence: float = 1.0           # 0-1, how certain the rule is
    tags: List[str] = field(default_factory=list)
    rule_id: Optional[str] = None     # Unique rule identifier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "action": self.action,
            "evidence": self.evidence,
            "metric_key": self.metric_key,
            "metric_value": self.metric_value,
            "impact": self.impact,
            "confidence": self.confidence,
            "tags": self.tags,
            "rule_id": self.rule_id,
        }


# ═══════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════════

class RuleEngine(_get_extended_mixin()):
    """
    Evaluates 150+ deterministic rules against compiled platform context.
    Returns prioritized, actionable insights.
    Inherits 90+ extended rules from ExtendedRulesMixin.
    """

    def evaluate(self, context: Dict[str, Any]) -> List[Insight]:
        insights: List[Insight] = []
        screen = context.get("screen", "")

        # ── Universal rules (run on every screen) ──
        self._rules_data_quality(context, insights)
        self._rules_feature_health(context, insights)
        self._rules_data_leakage(context, insights)
        self._rules_target_variable(context, insights)
        self._rules_sample_size(context, insights)
        self._rules_multicollinearity(context, insights)

        # ── Extended universal rules ──
        if hasattr(self, '_rules_data_quality_extended'):
            self._rules_data_quality_extended(context, insights)
        if hasattr(self, '_rules_data_leakage_extended'):
            self._rules_data_leakage_extended(context, insights)
        if hasattr(self, '_rules_target_variable_extended'):
            self._rules_target_variable_extended(context, insights)

        # ── Screen-specific rules ──
        if screen in ("eda",):
            self._rules_feature_engineering(context, insights)
            if hasattr(self, '_rules_feature_engineering_extended'):
                self._rules_feature_engineering_extended(context, insights)

        if screen in ("mlflow", "training"):
            self._rules_training_config(context, insights)
            self._rules_algorithm_selection(context, insights)
            if hasattr(self, '_rules_training_config_extended'):
                self._rules_training_config_extended(context, insights)
            if hasattr(self, '_rules_algorithm_selection_extended'):
                self._rules_algorithm_selection_extended(context, insights)

        if screen in ("evaluation",):
            self._rules_evaluation_metrics(context, insights)
            self._rules_model_comparison(context, insights)
            if hasattr(self, '_rules_evaluation_metrics_extended'):
                self._rules_evaluation_metrics_extended(context, insights)

        if screen in ("registry", "deployment"):
            self._rules_production_readiness(context, insights)
            self._rules_deployment_safety(context, insights)
            if hasattr(self, '_rules_deployment_extended'):
                self._rules_deployment_extended(context, insights)

        if screen in ("monitoring", "predictions"):
            self._rules_monitoring_drift(context, insights)
            if hasattr(self, '_rules_monitoring_extended'):
                self._rules_monitoring_extended(context, insights)

        # ── Always: pipeline lifecycle ──
        self._rules_pipeline_lifecycle(context, insights)
        if hasattr(self, '_rules_pipeline_lifecycle_extended'):
            self._rules_pipeline_lifecycle_extended(context, insights)

        # ── Sort by severity priority ──
        order = {"critical": 0, "warning": 1, "info": 2, "tip": 3, "success": 4}
        insights.sort(key=lambda x: (order.get(x.severity, 99), -(x.confidence or 0)))

        return insights

    # ══════════════════════════════════════════════════════════════
    # 1. DATA QUALITY RULES (DQ-001 → DQ-015)
    # ══════════════════════════════════════════════════════════════

    def _rules_data_quality(self, ctx: Dict, out: List[Insight]):
        quality = ctx.get("data_quality", {})
        profile = ctx.get("dataset_profile", {})
        if not quality and not profile:
            return

        completeness = quality.get("completeness", 100)
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        # DQ-001: Severe incompleteness
        if completeness < 70:
            out.append(Insight(
                severity="critical", category="Data Quality",
                title="Severe Data Incompleteness",
                message=(
                    f"Dataset is only {completeness:.1f}% complete — over {100 - completeness:.0f}% "
                    f"of cells contain missing values. Models trained on data this incomplete "
                    f"typically suffer from biased parameter estimates and inflated variance. "
                    f"Most imputation methods break down when missingness exceeds 30%."
                ),
                action=(
                    "First determine the missing data mechanism: (1) MCAR — values missing "
                    "completely at random — impute with median/mode. (2) MAR — missingness "
                    "depends on observed variables — use iterative imputation (IterativeImputer). "
                    "(3) MNAR — missingness depends on the missing value itself — consider "
                    "indicator variables or domain-specific strategies. Drop columns with >60% missing."
                ),
                evidence=f"Overall completeness: {completeness:.1f}%",
                metric_key="completeness", metric_value=completeness,
                impact="high", confidence=1.0,
                tags=["missing_data", "data_quality"],
                rule_id="DQ-001",
            ))
        elif completeness < 85:
            out.append(Insight(
                severity="warning", category="Data Quality",
                title="Notable Missing Data",
                message=(
                    f"Dataset completeness is {completeness:.1f}%. Missing data at this level "
                    f"can introduce bias in parameter estimates. Tree-based models handle "
                    f"missingness natively, but linear models and neural networks require imputation."
                ),
                action=(
                    "Configure imputation in Feature Engineering: median for numeric columns, "
                    "mode (most frequent) for categorical. Consider adding binary indicator "
                    "columns for features with >5% missing — the missingness pattern itself "
                    "can be predictive."
                ),
                evidence=f"Overall completeness: {completeness:.1f}%",
                metric_key="completeness", metric_value=completeness,
                impact="medium", confidence=1.0,
                tags=["missing_data"], rule_id="DQ-002",
            ))
        elif completeness >= 98:
            out.append(Insight(
                severity="success", category="Data Quality",
                title="Excellent Data Completeness",
                message=f"Dataset is {completeness:.1f}% complete. Minimal imputation needed.",
                action=None,
                evidence=f"Completeness: {completeness:.1f}%",
                metric_key="completeness", metric_value=completeness,
                impact="low", tags=["data_quality"], rule_id="DQ-003",
            ))

        # DQ-004: Per-column high missing
        high_missing = quality.get("columns_with_high_missing", [])
        if high_missing:
            severe = [c for c in high_missing if c.get("missing_pct", 0) > 50]
            moderate = [c for c in high_missing if 20 < c.get("missing_pct", 0) <= 50]

            if severe:
                col_list = ", ".join(
                    f"**{c['column']}** ({c['missing_pct']:.0f}%)" for c in severe[:5]
                )
                out.append(Insight(
                    severity="critical", category="Data Quality",
                    title=f"{len(severe)} Column(s) Are Mostly Empty (>50% Missing)",
                    message=(
                        f"Columns with majority missing: {col_list}. "
                        f"Imputing >50% of values essentially fabricates data — the imputed "
                        f"distribution will dominate the real signal. These columns add noise, "
                        f"not information."
                    ),
                    action=(
                        "Drop these columns unless domain knowledge confirms they carry critical "
                        "signal. If dropping, verify no downstream features depend on them."
                    ),
                    evidence=f"{len(severe)} columns with >50% missing",
                    impact="high", confidence=1.0,
                    tags=["missing_data", "column_drop"], rule_id="DQ-004",
                ))

            if moderate:
                col_list = ", ".join(
                    f"{c['column']} ({c['missing_pct']:.0f}%)" for c in moderate[:5]
                )
                out.append(Insight(
                    severity="warning", category="Data Quality",
                    title=f"{len(moderate)} Column(s) Have 20–50% Missing",
                    message=f"Moderate missingness in: {col_list}.",
                    action=(
                        "Use IterativeImputer (multivariate) for these columns — it models "
                        "each feature as a function of other features, producing more realistic "
                        "imputed values than simple median fill."
                    ),
                    evidence=f"{len(moderate)} columns with 20-50% missing",
                    impact="medium", tags=["missing_data"], rule_id="DQ-005",
                ))

        # DQ-006: Duplicate rows
        dup_rows = quality.get("duplicate_rows", 0)
        dup_pct = quality.get("duplicate_pct", 0)
        if dup_pct > 10:
            out.append(Insight(
                severity="critical", category="Data Quality",
                title=f"Excessive Duplicates: {dup_rows:,} Rows ({dup_pct:.1f}%)",
                message=(
                    f"{dup_pct:.1f}% of rows are exact duplicates. This artificially inflates "
                    f"the effective training set size, biases cross-validation scores upward "
                    f"(duplicates can appear in both train and validation folds), and causes "
                    f"the model to overweight repeated patterns."
                ),
                action=(
                    "Remove duplicates before any train/test split. Use pandas "
                    "df.drop_duplicates(). If duplicates are legitimate (e.g., repeated "
                    "transactions), add a frequency count column instead."
                ),
                evidence=f"{dup_rows:,} duplicate rows ({dup_pct:.1f}%)",
                metric_key="duplicate_pct", metric_value=dup_pct,
                impact="high", tags=["duplicates"], rule_id="DQ-006",
            ))
        elif dup_pct > 3:
            out.append(Insight(
                severity="warning", category="Data Quality",
                title=f"Duplicate Rows Detected: {dup_rows:,} ({dup_pct:.1f}%)",
                message=(
                    f"Found {dup_rows:,} duplicate rows. Verify whether these represent "
                    f"true repeated observations or data collection errors."
                ),
                action="Review a sample of duplicates. If errors, remove them. If legitimate, document why.",
                evidence=f"{dup_rows:,} duplicates",
                metric_key="duplicate_pct", metric_value=dup_pct,
                impact="medium", tags=["duplicates"], rule_id="DQ-007",
            ))

        # DQ-008: Wide dataset (more columns than rows)
        if rows > 0 and cols > 0 and cols > rows:
            out.append(Insight(
                severity="critical", category="Data Quality",
                title=f"More Features Than Samples (p={cols} > n={rows})",
                message=(
                    f"Your dataset has {cols} columns but only {rows} rows. "
                    f"In this p >> n regime, most models will overfit catastrophically. "
                    f"Ordinary least squares is undefined, tree models will memorize, and "
                    f"even regularized models need careful tuning."
                ),
                action=(
                    "Apply aggressive feature selection: (1) Remove zero-variance columns. "
                    "(2) Use mutual information or chi-squared tests. (3) Apply L1 (Lasso) "
                    "regularization. (4) Consider PCA to reduce dimensionality below n/5. "
                    "Target a feature-to-sample ratio of at least 1:10."
                ),
                evidence=f"p/n ratio: {cols}/{rows} = {cols/rows:.1f}",
                impact="high", confidence=1.0,
                tags=["high_dimensionality", "overfitting"], rule_id="DQ-008",
            ))

        # DQ-009: Extreme class imbalance detection from quality data
        quality_score = quality.get("overall_quality_score", 0)
        if quality_score > 0 and quality_score >= 90:
            out.append(Insight(
                severity="success", category="Data Quality",
                title=f"Strong Overall Data Quality ({quality_score:.0f}/100)",
                message=f"Data quality score of {quality_score:.0f}% indicates clean, well-structured data.",
                action=None,
                evidence=f"Quality score: {quality_score:.0f}",
                metric_key="quality_score", metric_value=quality_score,
                impact="low", tags=["data_quality"], rule_id="DQ-009",
            ))

        # DQ-010: Memory usage warning
        memory_mb = profile.get("memory_mb", 0)
        if memory_mb > 500:
            out.append(Insight(
                severity="info", category="Data Quality",
                title=f"Large Dataset in Memory ({memory_mb:.0f} MB)",
                message=(
                    f"Dataset occupies {memory_mb:.0f} MB. Large datasets can cause "
                    f"out-of-memory errors during cross-validation (which creates multiple copies) "
                    f"and hyperparameter search (which trains many models in parallel)."
                ),
                action=(
                    "Consider: (1) Downcast numeric types (float64→float32). "
                    "(2) Convert low-cardinality string columns to categories. "
                    "(3) Use incremental learning algorithms for very large data."
                ),
                evidence=f"Memory: {memory_mb:.0f} MB",
                metric_key="memory_mb", metric_value=memory_mb,
                impact="low", tags=["performance"], rule_id="DQ-010",
            ))

    # ══════════════════════════════════════════════════════════════
    # 2. FEATURE HEALTH RULES (FH-001 → FH-015)
    # ══════════════════════════════════════════════════════════════

    def _rules_feature_health(self, ctx: Dict, out: List[Insight]):
        features = ctx.get("feature_stats", {})
        profile = ctx.get("dataset_profile", {})
        if not features:
            return

        rows = profile.get("rows", 0)

        # FH-001: Skewed features
        skewed = features.get("skewed_features", [])
        if len(skewed) >= 3:
            names = ", ".join(s["column"] for s in skewed[:5])
            out.append(Insight(
                severity="info", category="Feature Health",
                title=f"{len(skewed)} Features Have Skewed Distributions",
                message=(
                    f"Features with significant skew: {names}. "
                    f"Skewed features compress most data points into a narrow range, "
                    f"reducing the model's ability to discriminate. Linear models, SVMs, "
                    f"and KNN are particularly sensitive to skew."
                ),
                action=(
                    "Apply log transformation (for right-skewed, strictly positive data), "
                    "square root transformation (for moderate skew with zeros), or "
                    "Box-Cox/Yeo-Johnson (automated optimal power transformation). "
                    "Note: tree-based models are inherently invariant to monotonic transformations."
                ),
                evidence=f"{len(skewed)} features with |skew indicator| > 0.5",
                impact="medium", tags=["distribution", "transformation"], rule_id="FH-001",
            ))
        elif len(skewed) == 1 or len(skewed) == 2:
            names = ", ".join(s["column"] for s in skewed)
            out.append(Insight(
                severity="tip", category="Feature Health",
                title=f"Skewed Feature(s): {names}",
                message=f"Light skew detected. Consider log-transform if using linear models.",
                action="Apply Yeo-Johnson transform for automatic optimal normalization.",
                evidence=f"{len(skewed)} skewed features",
                impact="low", tags=["distribution"], rule_id="FH-002",
            ))

        # FH-003: Constant columns (zero variance)
        constant = features.get("constant_columns", [])
        if constant:
            out.append(Insight(
                severity="warning", category="Feature Health",
                title=f"{len(constant)} Zero-Variance Column(s) Detected",
                message=(
                    f"Columns with identical values in every row: {', '.join(constant[:5])}. "
                    f"These carry zero predictive information and waste computation."
                ),
                action="Remove these columns before training. They add nothing and can cause errors in some algorithms.",
                evidence=f"Constant columns: {', '.join(constant[:5])}",
                impact="medium", tags=["feature_selection"], rule_id="FH-003",
            ))

        # FH-004: High cardinality categoricals
        high_card = features.get("high_cardinality_categoricals", [])
        if high_card:
            for col_info in high_card[:3]:
                col = col_info["column"]
                unique = col_info.get("unique_count", 0)
                ratio = col_info.get("unique_ratio", 0)

                if ratio > 0.8:
                    out.append(Insight(
                        severity="warning", category="Feature Health",
                        title=f"Near-Unique Categorical: {col}",
                        message=(
                            f"'{col}' has {unique} unique values across {rows} rows "
                            f"({ratio*100:.0f}% unique). This is likely an identifier, not a "
                            f"feature. One-hot encoding would create {unique} sparse columns."
                        ),
                        action=(
                            f"If '{col}' is an ID/name field, exclude it from features. "
                            f"If it's a legitimate high-cardinality feature (e.g., zip code), "
                            f"use target encoding or hash encoding instead of one-hot."
                        ),
                        evidence=f"{col}: {unique} unique values ({ratio*100:.0f}% of rows)",
                        impact="medium", tags=["cardinality", "encoding"], rule_id="FH-004",
                    ))
                elif unique > 50:
                    out.append(Insight(
                        severity="info", category="Feature Health",
                        title=f"High-Cardinality Feature: {col} ({unique} values)",
                        message=(
                            f"One-hot encoding '{col}' would add {unique} binary columns. "
                            f"This may increase dimensionality significantly."
                        ),
                        action=(
                            "Consider target encoding, frequency encoding, or grouping rare "
                            "categories into an 'Other' bucket (combine categories with <1% frequency)."
                        ),
                        evidence=f"{col}: {unique} unique categories",
                        impact="low", tags=["cardinality"], rule_id="FH-005",
                    ))

        # FH-006: Feature type balance
        num_count = profile.get("numeric_count", 0)
        cat_count = profile.get("categorical_count", 0)
        total_feats = num_count + cat_count
        if total_feats > 0 and cat_count > 0:
            cat_ratio = cat_count / total_feats
            if cat_ratio > 0.7:
                out.append(Insight(
                    severity="info", category="Feature Health",
                    title=f"Predominantly Categorical Dataset ({cat_count}/{total_feats} features)",
                    message=(
                        f"{cat_ratio*100:.0f}% of features are categorical. "
                        f"Tree-based models (Random Forest, XGBoost, LightGBM) handle mixed "
                        f"types well. Avoid Logistic Regression and SVM without proper encoding."
                    ),
                    action=(
                        "Use ordinal encoding for tree-based models, target encoding for "
                        "high-cardinality features. LightGBM has native categorical support."
                    ),
                    evidence=f"{cat_count} categorical / {total_feats} total features",
                    impact="medium", tags=["data_types", "encoding"], rule_id="FH-006",
                ))

    # ══════════════════════════════════════════════════════════════
    # 3. FEATURE ENGINEERING RULES (FE-001 → FE-010)
    # ══════════════════════════════════════════════════════════════

    def _rules_feature_engineering(self, ctx: Dict, out: List[Insight]):
        features = ctx.get("feature_stats", {})
        profile = ctx.get("dataset_profile", {})
        col_types = profile.get("column_types", {})
        screen_ctx = ctx.get("screen_context", {})

        datetime_cols = col_types.get("datetime", [])
        numeric_cols = col_types.get("numeric", [])
        cat_cols = col_types.get("categorical", [])

        # FE-001: Datetime feature extraction opportunity
        if datetime_cols:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title=f"Datetime Feature Extraction Opportunity ({len(datetime_cols)} columns)",
                message=(
                    f"Datetime columns detected: {', '.join(datetime_cols[:3])}. "
                    f"Raw datetime values are not useful for models, but derived features "
                    f"(hour, day of week, month, is_weekend, time_since) can be highly predictive."
                ),
                action=(
                    "Extract: year, month, day_of_week, hour, is_weekend, is_holiday, "
                    "quarter, days_since_epoch. For event data, compute time_between_events. "
                    "Cyclical encoding (sin/cos) preserves periodicity for hour and month."
                ),
                evidence=f"Datetime columns: {', '.join(datetime_cols[:3])}",
                impact="medium", tags=["datetime", "feature_creation"], rule_id="FE-001",
            ))

        # FE-002: Interaction features for small feature sets
        if 2 <= len(numeric_cols) <= 10:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title="Consider Polynomial / Interaction Features",
                message=(
                    f"With only {len(numeric_cols)} numeric features, the feature space is small "
                    f"enough to benefit from interaction terms (feature_A × feature_B) and "
                    f"polynomial features (feature²). These capture non-linear relationships "
                    f"that linear models would otherwise miss."
                ),
                action=(
                    "Use sklearn's PolynomialFeatures(degree=2, interaction_only=True) to "
                    "generate pairwise interactions. For {len(numeric_cols)} features this "
                    f"produces ~{len(numeric_cols) * (len(numeric_cols)-1) // 2} interaction terms."
                ),
                evidence=f"{len(numeric_cols)} numeric features",
                impact="medium", tags=["interactions", "polynomial"], rule_id="FE-002",
            ))

        # FE-003: Ratio features for numeric pairs
        if len(numeric_cols) >= 4:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title="Ratio Features May Add Predictive Power",
                message=(
                    "When you have related numeric features (e.g., income & expenses, "
                    "height & weight, revenue & cost), their ratio often captures "
                    "signal that individual features miss. Ratio features are scale-invariant "
                    "and can reveal proportional relationships."
                ),
                action=(
                    "Identify semantically related feature pairs and create ratios: "
                    "feature_A / (feature_B + epsilon). Add epsilon to avoid division by zero."
                ),
                evidence=f"{len(numeric_cols)} numeric features available for ratio creation",
                impact="low", tags=["ratios", "feature_creation"], rule_id="FE-003",
            ))

    # ══════════════════════════════════════════════════════════════
    # 4. TARGET VARIABLE RULES (TV-001 → TV-008)
    # ══════════════════════════════════════════════════════════════

    def _rules_target_variable(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")

        if not target:
            return

        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        dtypes = profile.get("dtypes", {})
        features = ctx.get("feature_stats", {})
        numeric_stats = features.get("numeric_stats", {})
        col_types = profile.get("column_types", {})

        # TV-001: Target in ID columns
        id_cols = features.get("potential_id_columns", [])
        id_names = [c["column"] for c in id_cols]
        if target in id_names:
            out.append(Insight(
                severity="critical", category="Target Variable",
                title=f"Target '{target}' Appears to Be an ID Column",
                message=(
                    f"'{target}' was detected as a potential identifier (high uniqueness). "
                    f"Training a model to predict an ID will produce meaningless results — "
                    f"the model will memorize rather than learn generalizable patterns."
                ),
                action=f"Select a different target variable. '{target}' should be excluded from features entirely.",
                evidence=f"'{target}' flagged as potential ID column",
                impact="high", confidence=0.9,
                tags=["target", "id_column"], rule_id="TV-001",
            ))

        # TV-002: Binary target imbalance
        if target in numeric_stats:
            stats = numeric_stats[target]
            if isinstance(stats, dict):
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 1)
                mean_val = stats.get("mean", 0.5)

                # Detect binary classification target
                if min_val == 0 and max_val == 1:
                    minority_pct = min(mean_val, 1 - mean_val) * 100
                    if minority_pct < 5:
                        out.append(Insight(
                            severity="critical", category="Target Variable",
                            title=f"Severe Class Imbalance ({minority_pct:.1f}% minority)",
                            message=(
                                f"The positive class represents only {mean_val*100:.1f}% of samples. "
                                f"At this imbalance level, a model that predicts all negatives "
                                f"achieves {max(mean_val, 1-mean_val)*100:.1f}% accuracy — but "
                                f"catches zero positive cases. Standard accuracy is completely misleading."
                            ),
                            action=(
                                "Use these techniques in combination: "
                                "(1) SMOTE oversampling of minority class. "
                                "(2) class_weight='balanced' in the algorithm. "
                                "(3) Evaluate with F1, Precision-Recall AUC, and Matthews Correlation Coefficient — "
                                "NEVER accuracy alone. "
                                "(4) Stratified cross-validation to maintain ratio in each fold."
                            ),
                            evidence=f"Target '{target}' mean={mean_val:.3f} → {minority_pct:.1f}% minority",
                            metric_key="class_imbalance", metric_value=minority_pct,
                            impact="high", confidence=0.95,
                            tags=["imbalance", "binary_classification"], rule_id="TV-002",
                        ))
                    elif minority_pct < 20:
                        out.append(Insight(
                            severity="warning", category="Target Variable",
                            title=f"Moderate Class Imbalance ({minority_pct:.1f}% minority)",
                            message=(
                                f"Minority class is {minority_pct:.1f}% of the data. "
                                f"Models may bias toward the majority class."
                            ),
                            action=(
                                "Use class_weight='balanced' and evaluate with F1 score. "
                                "Consider stratified K-fold cross-validation."
                            ),
                            evidence=f"Target '{target}' minority ratio: {minority_pct:.1f}%",
                            metric_key="class_imbalance", metric_value=minority_pct,
                            impact="medium", tags=["imbalance"], rule_id="TV-003",
                        ))

    # ══════════════════════════════════════════════════════════════
    # 5. DATA LEAKAGE RULES (DL-001 → DL-010)
    # ══════════════════════════════════════════════════════════════

    def _rules_data_leakage(self, ctx: Dict, out: List[Insight]):
        features = ctx.get("feature_stats", {})
        correlations = ctx.get("correlations", {})

        # DL-001: ID columns in features
        id_cols = features.get("potential_id_columns", [])
        if id_cols:
            name_detected = [c for c in id_cols if c.get("detected_by") == "name_pattern"]
            stat_detected = [c for c in id_cols if c.get("detected_by") != "name_pattern"]

            for col_info in id_cols[:5]:
                col = col_info["column"]
                reason = "name matches ID pattern" if col_info.get("detected_by") == "name_pattern" else \
                         f"{col_info.get('unique_ratio', 0)*100:.0f}% unique values"
                out.append(Insight(
                    severity="warning", category="Data Leakage",
                    title=f"Potential ID Column: '{col}'",
                    message=(
                        f"'{col}' appears to be an identifier ({reason}). "
                        f"ID columns create data leakage — the model memorizes row-level "
                        f"identifiers instead of learning generalizable patterns. This produces "
                        f"artificially high training scores that collapse on new data."
                    ),
                    action=(
                        f"Exclude '{col}' from the feature set. If it's needed for joining "
                        f"tables, remove it after the join."
                    ),
                    evidence=f"Column '{col}': {reason}",
                    impact="high", confidence=0.85,
                    tags=["leakage", "id_column"], rule_id="DL-001",
                ))

        # DL-002: Perfect or near-perfect correlation (leakage signal)
        high_pairs = correlations.get("high_pairs", [])
        perfect = [p for p in high_pairs if p.get("abs_correlation", 0) >= 0.98]
        if perfect:
            for pair in perfect[:3]:
                out.append(Insight(
                    severity="critical", category="Data Leakage",
                    title=f"Near-Perfect Correlation: {pair['feature1']} ↔ {pair['feature2']}",
                    message=(
                        f"Correlation of {pair['correlation']:.3f} between '{pair['feature1']}' "
                        f"and '{pair['feature2']}'. This level of correlation almost always "
                        f"indicates: (1) derived/duplicate features, (2) data leakage (one "
                        f"feature is a proxy for the target), or (3) encoding artifacts."
                    ),
                    action=(
                        "Investigate this pair. If one is derived from the other, drop it. "
                        "If one is a post-hoc feature (created after the outcome), it's "
                        "target leakage and MUST be removed."
                    ),
                    evidence=f"Correlation: {pair['correlation']:.3f}",
                    impact="high", confidence=0.95,
                    tags=["leakage", "correlation"], rule_id="DL-002",
                ))

    # ══════════════════════════════════════════════════════════════
    # 6. MULTICOLLINEARITY RULES (MC-001 → MC-006)
    # ══════════════════════════════════════════════════════════════

    def _rules_multicollinearity(self, ctx: Dict, out: List[Insight]):
        correlations = ctx.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])

        # Exclude near-perfect (those are leakage, handled above)
        moderate_high = [p for p in high_pairs
                         if 0.8 <= p.get("abs_correlation", 0) < 0.98]

        if len(moderate_high) >= 3:
            pair_list = ", ".join(
                f"{p['feature1']}/{p['feature2']} ({p['correlation']:.2f})"
                for p in moderate_high[:5]
            )
            out.append(Insight(
                severity="warning", category="Multicollinearity",
                title=f"{len(moderate_high)} Highly Correlated Feature Pairs Found",
                message=(
                    f"Correlated pairs (|r| ≥ 0.8): {pair_list}. "
                    f"Multicollinearity inflates variance of coefficient estimates in linear "
                    f"models, making them unstable and hard to interpret. Feature importance "
                    f"scores become unreliable as importance is split between correlated features."
                ),
                action=(
                    "Options: (1) Drop one from each correlated pair (keep the one with "
                    "higher correlation to the target). (2) Use PCA to combine correlated "
                    "features into orthogonal components. (3) Use L2 regularization (Ridge) "
                    "which handles multicollinearity naturally. Note: tree-based models are "
                    "less affected but feature importance is still diluted."
                ),
                evidence=f"{len(moderate_high)} pairs with |r| ≥ 0.8",
                impact="medium", confidence=1.0,
                tags=["multicollinearity", "correlation"], rule_id="MC-001",
            ))
        elif len(moderate_high) == 1 or len(moderate_high) == 2:
            for pair in moderate_high:
                out.append(Insight(
                    severity="info", category="Multicollinearity",
                    title=f"Correlated: {pair['feature1']} ↔ {pair['feature2']} ({pair['correlation']:.2f})",
                    message="Consider dropping one to improve model interpretability.",
                    action="Keep the feature with stronger univariate correlation to the target.",
                    evidence=f"r = {pair['correlation']:.3f}",
                    impact="low", tags=["correlation"], rule_id="MC-002",
                ))

    # ══════════════════════════════════════════════════════════════
    # 7. SAMPLE SIZE RULES (SS-001 → SS-006)
    # ══════════════════════════════════════════════════════════════

    def _rules_sample_size(self, ctx: Dict, out: List[Insight]):
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)
        if rows == 0:
            return

        # SS-001: Very small dataset
        if rows < 100:
            out.append(Insight(
                severity="critical", category="Sample Size",
                title=f"Critically Small Dataset ({rows} rows)",
                message=(
                    f"With only {rows} samples, most ML algorithms cannot learn reliable "
                    f"patterns. Cross-validation folds will be tiny, producing highly variable "
                    f"performance estimates. The risk of overfitting is extreme."
                ),
                action=(
                    "Use simple models only: Logistic Regression, Naive Bayes, or Decision "
                    "Stump. Apply leave-one-out cross-validation (LOOCV) for more stable "
                    "estimates. Consider collecting more data before training."
                ),
                evidence=f"n = {rows}",
                metric_key="row_count", metric_value=rows,
                impact="high", tags=["small_data"], rule_id="SS-001",
            ))
        elif rows < 500:
            out.append(Insight(
                severity="warning", category="Sample Size",
                title=f"Small Dataset ({rows} rows)",
                message=(
                    f"{rows} rows limits model complexity. High-capacity models (XGBoost with "
                    f"deep trees, neural networks) will likely overfit."
                ),
                action=(
                    "Prefer simpler models: Logistic Regression, Linear SVM, shallow Decision "
                    "Trees (max_depth=3-5). Use 10-fold CV for more reliable estimates."
                ),
                evidence=f"n = {rows}",
                metric_key="row_count", metric_value=rows,
                impact="medium", tags=["small_data"], rule_id="SS-002",
            ))
        elif rows >= 100000:
            out.append(Insight(
                severity="success", category="Sample Size",
                title=f"Large Dataset ({rows:,} rows)",
                message=(
                    f"With {rows:,} samples, you have enough data for complex models "
                    f"including deep ensembles and neural networks. Overfitting risk is lower."
                ),
                action=None,
                evidence=f"n = {rows:,}",
                impact="low", tags=["large_data"], rule_id="SS-003",
            ))

        # SS-004: Feature-to-sample ratio
        if cols > 0 and rows > 0:
            ratio = rows / cols
            if ratio < 5:
                out.append(Insight(
                    severity="critical", category="Sample Size",
                    title=f"Dangerously Low Sample-to-Feature Ratio ({ratio:.1f}:1)",
                    message=(
                        f"You have only {ratio:.1f} samples per feature. "
                        f"A rule of thumb requires at least 10-20 samples per feature for "
                        f"reliable learning. At this ratio, the model can find spurious "
                        f"patterns in random noise."
                    ),
                    action=(
                        "Reduce features aggressively: use correlation-based filtering, "
                        "mutual information, or recursive feature elimination to get below "
                        f"{cols // 5} features (for a 5:1 ratio). Or collect {cols * 10 - rows} more samples."
                    ),
                    evidence=f"Ratio: {rows}/{cols} = {ratio:.1f} samples per feature",
                    impact="high", tags=["dimensionality"], rule_id="SS-004",
                ))
            elif ratio < 10:
                out.append(Insight(
                    severity="warning", category="Sample Size",
                    title=f"Low Sample-to-Feature Ratio ({ratio:.1f}:1)",
                    message=(
                        f"{ratio:.1f} samples per feature is below the recommended 10:1 minimum. "
                        f"Apply regularization (L1/L2) and consider feature selection."
                    ),
                    action="Use regularized models (Lasso, Ridge, ElasticNet) or reduce feature count.",
                    evidence=f"n/p ratio: {ratio:.1f}",
                    impact="medium", tags=["dimensionality"], rule_id="SS-005",
                ))

    # ══════════════════════════════════════════════════════════════
    # 8. TRAINING CONFIG RULES (TC-001 → TC-012)
    # ══════════════════════════════════════════════════════════════

    def _rules_training_config(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)

        test_size = screen_ctx.get("test_size") or frontend.get("test_size")
        cv_folds = screen_ctx.get("cv_folds") or frontend.get("cv_folds")
        scaling = screen_ctx.get("scaling_method") or frontend.get("scaling_method")
        target = screen_ctx.get("target_column") or frontend.get("target_column")
        selected_features = screen_ctx.get("selected_features") or frontend.get("selected_features", [])

        # TC-001: No target selected
        if not target and ctx.get("screen") in ("mlflow", "training"):
            out.append(Insight(
                severity="critical", category="Training Configuration",
                title="No Target Variable Selected",
                message="You must select a target variable before training. The target is the column your model will learn to predict.",
                action="Select the target column in the training configuration. Choose the column that represents the outcome you want to predict.",
                impact="high", confidence=1.0,
                tags=["target", "configuration"], rule_id="TC-001",
            ))

        # TC-002: Test size too small
        if test_size is not None:
            if test_size < 0.1:
                out.append(Insight(
                    severity="warning", category="Training Configuration",
                    title=f"Test Set Too Small ({test_size*100:.0f}%)",
                    message=(
                        f"A {test_size*100:.0f}% test set means only ~{int(rows * test_size)} rows "
                        f"for final evaluation. Performance estimates will be noisy and unreliable."
                    ),
                    action="Use at least 20% for testing, or rely more on cross-validation for evaluation.",
                    evidence=f"Test size: {test_size*100:.0f}% (~{int(rows * test_size)} rows)",
                    impact="medium", tags=["split"], rule_id="TC-002",
                ))
            elif test_size > 0.4:
                out.append(Insight(
                    severity="warning", category="Training Configuration",
                    title=f"Test Set Too Large ({test_size*100:.0f}%)",
                    message=(
                        f"Using {test_size*100:.0f}% for testing leaves only {(1-test_size)*100:.0f}% "
                        f"(~{int(rows * (1 - test_size))} rows) for training. "
                        f"The model may underfit due to insufficient training data."
                    ),
                    action="Standard practice is 20% test / 80% train, or 20% test / 10% validation / 70% train.",
                    evidence=f"Test size: {test_size*100:.0f}%",
                    impact="medium", tags=["split"], rule_id="TC-003",
                ))

        # TC-004: CV folds + small data
        if cv_folds and rows > 0:
            samples_per_fold = rows / cv_folds
            if samples_per_fold < 30:
                out.append(Insight(
                    severity="warning", category="Training Configuration",
                    title=f"Too Many CV Folds for Dataset Size",
                    message=(
                        f"{cv_folds}-fold CV on {rows} rows means only ~{int(samples_per_fold)} "
                        f"samples per validation fold. Fold-to-fold variance will be high."
                    ),
                    action=f"Use {max(3, min(5, rows // 50))}-fold CV, or switch to Leave-One-Out for very small data.",
                    evidence=f"{rows} rows / {cv_folds} folds = {int(samples_per_fold)} per fold",
                    impact="medium", tags=["cross_validation"], rule_id="TC-004",
                ))

        # TC-005: Scaling method advice
        if scaling:
            if scaling.lower() in ("standard", "standardscaler"):
                features_data = ctx.get("feature_stats", {})
                skewed = features_data.get("skewed_features", [])
                if len(skewed) > 2:
                    out.append(Insight(
                        severity="tip", category="Training Configuration",
                        title="StandardScaler May Not Be Optimal for Skewed Data",
                        message=(
                            f"You selected StandardScaler but {len(skewed)} features have "
                            f"skewed distributions. StandardScaler assumes normally distributed "
                            f"features — heavy skew produces outlier-dominated z-scores."
                        ),
                        action=(
                            "Consider RobustScaler (uses IQR, resistant to outliers) or "
                            "PowerTransformer (applies Yeo-Johnson to normalize before scaling)."
                        ),
                        evidence=f"{len(skewed)} skewed features with StandardScaler",
                        impact="low", tags=["scaling"], rule_id="TC-005",
                    ))

        # TC-006: Too many features selected
        if selected_features and rows > 0:
            n_features = len(selected_features)
            ratio = rows / n_features if n_features > 0 else float('inf')
            if ratio < 10 and n_features > 20:
                out.append(Insight(
                    severity="warning", category="Training Configuration",
                    title=f"Too Many Features Selected ({n_features})",
                    message=(
                        f"You selected {n_features} features for {rows} samples "
                        f"(ratio: {ratio:.1f}:1). With this many features, the curse of "
                        f"dimensionality may degrade model performance."
                    ),
                    action=(
                        f"Reduce to ~{rows // 15} features. Use feature importance from a "
                        f"quick Random Forest run to identify the top features, or apply "
                        f"mutual information scoring."
                    ),
                    evidence=f"{n_features} features / {rows} rows",
                    impact="medium", tags=["feature_selection"], rule_id="TC-006",
                ))

    # ══════════════════════════════════════════════════════════════
    # 9. ALGORITHM SELECTION RULES (AS-001 → AS-010)
    # ══════════════════════════════════════════════════════════════

    def _rules_algorithm_selection(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        algorithm = screen_ctx.get("algorithm") or frontend.get("algorithm")
        selected_algos = screen_ctx.get("selected_algorithms") or frontend.get("selected_algorithms", [])
        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type", "classification")

        all_algos = selected_algos if selected_algos else ([algorithm] if algorithm else [])

        for algo in all_algos:
            algo_lower = (algo or "").lower().replace(" ", "").replace("_", "")

            # AS-001: XGBoost/GBM on tiny data
            if any(x in algo_lower for x in ["xgboost", "xgb", "gradient", "gbm", "lightgbm", "catboost"]):
                if rows > 0 and rows < 500:
                    out.append(Insight(
                        severity="warning", category="Algorithm Selection",
                        title=f"Boosting Algorithm on Small Data ({rows} rows)",
                        message=(
                            f"'{algo}' is a high-capacity ensemble method that typically needs "
                            f"1000+ samples to generalize well. On {rows} rows, it's likely "
                            f"to overfit — showing great training scores but poor test performance."
                        ),
                        action=(
                            "Try simpler models first: Logistic Regression (strong baseline), "
                            "Random Forest (max_depth=5), or SVM with RBF kernel. If boosting "
                            "is necessary, use strong regularization: max_depth=3, n_estimators=50, "
                            "learning_rate=0.01, min_child_weight=5."
                        ),
                        evidence=f"{algo} on {rows} rows",
                        impact="medium", tags=["overfitting", "algorithm"], rule_id="AS-001",
                    ))

            # AS-002: Neural network on tiny data
            if any(x in algo_lower for x in ["neural", "mlp", "deep", "nn"]):
                if rows > 0 and rows < 5000:
                    out.append(Insight(
                        severity="critical" if rows < 1000 else "warning",
                        category="Algorithm Selection",
                        title=f"Neural Network on {'Very ' if rows < 1000 else ''}Small Data",
                        message=(
                            f"Neural networks are data-hungry: they typically need 5,000+ "
                            f"samples per class to learn meaningful representations. With "
                            f"{rows} rows, a neural net will memorize the training set."
                        ),
                        action=(
                            "Use classical ML models instead. If deep learning is required, "
                            "use heavy regularization: dropout=0.5, early stopping, weight decay, "
                            "and a very small architecture (1-2 hidden layers, <64 neurons)."
                        ),
                        evidence=f"Neural network on {rows} samples",
                        impact="high", tags=["deep_learning", "overfitting"], rule_id="AS-002",
                    ))

            # AS-003: SVM on large data
            if any(x in algo_lower for x in ["svm", "svc", "svr", "supportvector"]):
                if rows > 10000:
                    out.append(Insight(
                        severity="warning", category="Algorithm Selection",
                        title=f"SVM on Large Dataset ({rows:,} rows)",
                        message=(
                            f"Standard SVM has O(n²) to O(n³) time complexity. With {rows:,} "
                            f"rows, training may take hours or run out of memory. "
                            f"Kernel SVM scales particularly poorly."
                        ),
                        action=(
                            "Options: (1) Use LinearSVC (O(n) complexity) for linear problems. "
                            "(2) Subsample to ~5000 rows. (3) Use SGDClassifier with hinge loss "
                            "(online SVM that scales linearly). (4) Consider Random Forest or "
                            "XGBoost which scale much better."
                        ),
                        evidence=f"SVM on {rows:,} samples",
                        impact="medium", tags=["scalability", "algorithm"], rule_id="AS-003",
                    ))

            # AS-004: Logistic Regression (always a good choice)
            if any(x in algo_lower for x in ["logistic", "logreg"]):
                out.append(Insight(
                    severity="success", category="Algorithm Selection",
                    title="Logistic Regression — Strong Baseline Choice",
                    message=(
                        "Logistic Regression is an excellent baseline: fast, interpretable, "
                        "and surprisingly competitive. If it performs within 2-3% of more "
                        "complex models, prefer it for production due to lower latency and "
                        "easier monitoring."
                    ),
                    action=None,
                    evidence="Algorithm: Logistic Regression",
                    impact="low", tags=["algorithm", "baseline"], rule_id="AS-004",
                ))

        # AS-005: Algorithm diversity check
        if len(all_algos) == 1:
            out.append(Insight(
                severity="tip", category="Algorithm Selection",
                title="Consider Training Multiple Algorithms",
                message=(
                    "Training a single algorithm risks missing a better-performing model. "
                    "The 'no free lunch' theorem guarantees no single algorithm is best for "
                    "all datasets. A quick comparison across 3-5 algorithms takes minutes "
                    "but can reveal 5-15% performance gains."
                ),
                action=(
                    "Recommended comparison set: Logistic Regression (baseline), "
                    "Random Forest (ensemble), XGBoost (boosting), SVM (kernel). "
                    "The ML Flow wizard can train all of these automatically."
                ),
                impact="medium", tags=["algorithm_comparison"], rule_id="AS-005",
            ))

    # ══════════════════════════════════════════════════════════════
    # 10. EVALUATION METRICS RULES (EM-001 → EM-015)
    # ══════════════════════════════════════════════════════════════

    def _rules_evaluation_metrics(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})
        if not metrics:
            return

        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1_score") or metrics.get("f1")
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        auc_roc = metrics.get("roc_auc") or metrics.get("auc_roc") or metrics.get("auc")
        train_score = metrics.get("train_score") or metrics.get("training_accuracy")
        test_score = metrics.get("test_score") or metrics.get("test_accuracy")

        # EM-001: Accuracy-F1 gap (imbalance trap)
        if accuracy and f1 and accuracy > 0 and f1 > 0:
            gap = accuracy - f1
            if gap > 0.15:
                out.append(Insight(
                    severity="critical", category="Evaluation Metrics",
                    title=f"Misleading Accuracy — {gap*100:.0f}pt Gap with F1",
                    message=(
                        f"Accuracy ({accuracy*100:.1f}%) looks healthy, but F1 ({f1*100:.1f}%) "
                        f"tells a different story. This {gap*100:.0f}-point gap is a classic "
                        f"sign of class imbalance: the model is getting easy negatives right "
                        f"while missing positive cases."
                    ),
                    action=(
                        "Trust F1 over accuracy for this dataset. Focus on improving recall "
                        "of the minority class through class weights, SMOTE, or threshold tuning. "
                        "Also report Precision-Recall AUC and Matthews Correlation Coefficient."
                    ),
                    evidence=f"Accuracy: {accuracy*100:.1f}% vs F1: {f1*100:.1f}% (gap: {gap*100:.0f}pt)",
                    metric_key="accuracy_f1_gap", metric_value=gap,
                    impact="high", confidence=0.95,
                    tags=["imbalance", "metric_trap"], rule_id="EM-001",
                ))

        # EM-002: Low recall
        if recall and recall < 0.5:
            out.append(Insight(
                severity="warning", category="Evaluation Metrics",
                title=f"Low Recall ({recall*100:.1f}%) — Missing Positive Cases",
                message=(
                    f"The model correctly identifies only {recall*100:.1f}% of actual positives. "
                    f"In many business contexts (fraud detection, disease diagnosis, churn prediction), "
                    f"missing a true positive is far more costly than a false alarm."
                ),
                action=(
                    "Lower the classification threshold (currently likely 0.5). A threshold "
                    "of 0.3-0.4 will catch more positives at the cost of some false positives. "
                    "Use the Precision-Recall curve to find the optimal trade-off."
                ),
                evidence=f"Recall: {recall*100:.1f}%",
                metric_key="recall", metric_value=recall,
                impact="high", tags=["recall", "threshold"], rule_id="EM-002",
            ))

        # EM-003: Precision-Recall gap
        if precision and recall and precision > 0 and recall > 0:
            pr_gap = abs(precision - recall)
            if pr_gap > 0.25:
                higher = "Precision" if precision > recall else "Recall"
                lower = "Recall" if precision > recall else "Precision"
                out.append(Insight(
                    severity="warning", category="Evaluation Metrics",
                    title=f"Large Precision-Recall Gap ({pr_gap*100:.0f}pt)",
                    message=(
                        f"{higher} ({max(precision, recall)*100:.1f}%) is much higher than "
                        f"{lower} ({min(precision, recall)*100:.1f}%). "
                        f"This means the model is {'very conservative (few predictions, most correct)' if precision > recall else 'very aggressive (catches most positives but many false alarms)'}."
                    ),
                    action=(
                        f"Adjust the classification threshold to balance Precision and Recall "
                        f"according to your business costs. The optimal F1 threshold often "
                        f"differs significantly from the default 0.5."
                    ),
                    evidence=f"Precision: {precision*100:.1f}% vs Recall: {recall*100:.1f}%",
                    impact="medium", tags=["threshold", "trade_off"], rule_id="EM-003",
                ))

        # EM-004: Overfitting detection (train vs test gap)
        if train_score and test_score and train_score > 0 and test_score > 0:
            overfit_gap = train_score - test_score
            if overfit_gap > 0.1:
                out.append(Insight(
                    severity="critical" if overfit_gap > 0.2 else "warning",
                    category="Evaluation Metrics",
                    title=f"{'Severe ' if overfit_gap > 0.2 else ''}Overfitting Detected ({overfit_gap*100:.0f}pt gap)",
                    message=(
                        f"Training score ({train_score*100:.1f}%) is {overfit_gap*100:.0f} points "
                        f"higher than test score ({test_score*100:.1f}%). The model has memorized "
                        f"training data patterns that don't generalize. {'This level of overfitting typically means the model is not production-ready.' if overfit_gap > 0.2 else ''}"
                    ),
                    action=(
                        "To reduce overfitting: (1) Increase regularization (higher C, lower alpha). "
                        "(2) Reduce model complexity (fewer trees, shallower depth, fewer features). "
                        "(3) Add more training data. (4) Use dropout for neural networks. "
                        "(5) Try cross-validation to get a better estimate."
                    ),
                    evidence=f"Train: {train_score*100:.1f}% → Test: {test_score*100:.1f}% (gap: {overfit_gap*100:.0f}pt)",
                    metric_key="overfit_gap", metric_value=overfit_gap,
                    impact="high", tags=["overfitting"], rule_id="EM-004",
                ))

        # EM-005: Suspiciously perfect scores (likely leakage)
        if auc_roc and auc_roc > 0.99:
            out.append(Insight(
                severity="critical", category="Evaluation Metrics",
                title=f"Suspiciously High AUC ({auc_roc:.4f}) — Possible Leakage",
                message=(
                    f"AUC-ROC of {auc_roc:.4f} is near-perfect. In real-world ML, AUC > 0.99 "
                    f"almost always indicates data leakage (a feature that reveals the target) "
                    f"or evaluation error (test data contaminating training). Legitimate models "
                    f"rarely achieve this on non-trivial problems."
                ),
                action=(
                    "Investigate immediately: (1) Check for features derived from the target "
                    "(post-hoc features). (2) Verify the train/test split has no data leakage. "
                    "(3) Look for ID columns or timestamps that correlate with the target. "
                    "(4) Ensure the target isn't in the feature set."
                ),
                evidence=f"AUC-ROC: {auc_roc:.4f}",
                metric_key="auc_roc", metric_value=auc_roc,
                impact="high", confidence=0.9,
                tags=["leakage", "too_good"], rule_id="EM-005",
            ))

        # EM-006: Strong discrimination
        if auc_roc and 0.85 <= auc_roc <= 0.99:
            out.append(Insight(
                severity="success", category="Evaluation Metrics",
                title=f"Strong Model Discrimination (AUC: {auc_roc:.3f})",
                message=(
                    f"AUC-ROC of {auc_roc:.3f} indicates the model has strong ability to "
                    f"distinguish between positive and negative classes."
                ),
                action=None,
                evidence=f"AUC-ROC: {auc_roc:.3f}",
                metric_key="auc_roc", metric_value=auc_roc,
                impact="low", tags=["auc", "good_model"], rule_id="EM-006",
            ))
        elif auc_roc and auc_roc < 0.6:
            out.append(Insight(
                severity="warning", category="Evaluation Metrics",
                title=f"Weak Model Discrimination (AUC: {auc_roc:.3f})",
                message=(
                    f"AUC of {auc_roc:.3f} is barely above random (0.50). "
                    f"The model is struggling to separate classes."
                ),
                action=(
                    "Try: (1) Better feature engineering. (2) Different algorithms. "
                    "(3) More training data. (4) Check if the problem is genuinely predictable."
                ),
                evidence=f"AUC-ROC: {auc_roc:.3f}",
                metric_key="auc_roc", metric_value=auc_roc,
                impact="high", tags=["weak_model"], rule_id="EM-007",
            ))

        # EM-008: Overall quality assessment
        overall_score = screen_ctx.get("overall_score") or frontend.get("overall_score")
        if overall_score:
            if overall_score >= 80:
                out.append(Insight(
                    severity="success", category="Evaluation Metrics",
                    title=f"Strong Overall Score: {overall_score:.0f}/100",
                    message="The model shows strong performance across multiple evaluation criteria.",
                    action=None,
                    evidence=f"Overall: {overall_score:.0f}/100",
                    impact="low", tags=["overall"], rule_id="EM-008",
                ))
            elif overall_score < 50:
                out.append(Insight(
                    severity="warning", category="Evaluation Metrics",
                    title=f"Low Overall Score: {overall_score:.0f}/100",
                    message="The model underperforms across multiple criteria. Consider retraining with different parameters.",
                    action="Review feature engineering, try different algorithms, and check for data quality issues.",
                    evidence=f"Overall: {overall_score:.0f}/100",
                    impact="high", tags=["overall"], rule_id="EM-009",
                ))

    # ══════════════════════════════════════════════════════════════
    # 11. MODEL COMPARISON RULES (MC-001 → MC-005)
    # ══════════════════════════════════════════════════════════════

    def _rules_model_comparison(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        production_readiness = screen_ctx.get("production_readiness") or frontend.get("production_readiness")

        if production_readiness and isinstance(production_readiness, dict):
            criteria = production_readiness.get("criteria", [])
            passed = sum(1 for c in criteria if c.get("passed", False))
            total = len(criteria)
            if total > 0:
                pct = passed / total * 100
                if pct >= 80:
                    out.append(Insight(
                        severity="success", category="Model Comparison",
                        title=f"Production Readiness: {passed}/{total} Criteria Passed ({pct:.0f}%)",
                        message="Model meets most production readiness criteria.",
                        action=None,
                        evidence=f"{passed}/{total} criteria passed",
                        impact="low", tags=["production"], rule_id="MR-001",
                    ))
                elif pct < 50:
                    failed = [c.get("name", "Unknown") for c in criteria if not c.get("passed", False)]
                    out.append(Insight(
                        severity="warning", category="Model Comparison",
                        title=f"Production Readiness: Only {passed}/{total} Criteria Passed",
                        message=f"Failed criteria: {', '.join(failed[:5])}. Address these before deployment.",
                        action="Review each failed criterion and address the underlying issues.",
                        evidence=f"{passed}/{total} criteria passed",
                        impact="high", tags=["production"], rule_id="MR-002",
                    ))

    # ══════════════════════════════════════════════════════════════
    # 12. PRODUCTION READINESS RULES (PR-001 → PR-008)
    # ══════════════════════════════════════════════════════════════

    def _rules_production_readiness(self, ctx: Dict, out: List[Insight]):
        registry = ctx.get("registry_info", {})

        if not registry:
            return

        deployed = registry.get("deployed_models", 0)
        total = registry.get("total_registered", 0)
        models = registry.get("models", [])

        # PR-001: No deployed models
        if total > 0 and deployed == 0:
            out.append(Insight(
                severity="info", category="Production Readiness",
                title=f"{total} Registered Model(s), None Deployed",
                message="You have registered models but none are deployed to production yet.",
                action=(
                    "To deploy: (1) Promote the best model to 'production' status. "
                    "(2) Run shadow deployment first to validate on real traffic. "
                    "(3) Set up monitoring before going live."
                ),
                evidence=f"{total} registered, {deployed} deployed",
                impact="medium", tags=["deployment"], rule_id="PR-001",
            ))

        # PR-002: Multiple versions - champion/challenger
        for model in models[:3]:
            versions = model.get("total_versions", 0)
            if versions > 1:
                out.append(Insight(
                    severity="success", category="Production Readiness",
                    title=f"Model '{model['name']}' Has {versions} Versions",
                    message=(
                        f"Multiple versions enable champion/challenger testing — "
                        f"serve the current champion while testing the latest challenger "
                        f"on a percentage of traffic."
                    ),
                    action=None,
                    evidence=f"{versions} versions of '{model['name']}'",
                    impact="low", tags=["versioning"], rule_id="PR-002",
                ))

    # ══════════════════════════════════════════════════════════════
    # 13. DEPLOYMENT SAFETY RULES (DS-001 → DS-008)
    # ══════════════════════════════════════════════════════════════

    def _rules_deployment_safety(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        registry = ctx.get("registry_info", {})

        deployed = registry.get("deployed_models", 0)

        # DS-001: Shadow deployment recommendation
        if ctx.get("screen") == "deployment":
            out.append(Insight(
                severity="tip", category="Deployment Safety",
                title="Use Shadow Deployment Before Going Live",
                message=(
                    "Shadow deployment runs the new model alongside the current one, "
                    "comparing predictions without affecting users. This catches "
                    "data distribution shifts, latency issues, and edge cases "
                    "that offline evaluation misses."
                ),
                action=(
                    "Deploy in shadow mode for 1-2 weeks. Monitor: (1) Prediction distribution "
                    "differences. (2) Latency percentiles (p50, p95, p99). (3) Error rates. "
                    "Only promote to production when shadow metrics are stable."
                ),
                impact="medium", tags=["shadow", "safety"], rule_id="DS-001",
            ))

        # DS-002: Rollback plan
        if deployed > 0:
            out.append(Insight(
                severity="tip", category="Deployment Safety",
                title="Ensure Rollback Plan is Ready",
                message=(
                    "Every production deployment needs a rollback plan. If the new model "
                    "degrades performance, you need to revert to the previous version instantly."
                ),
                action=(
                    "Verify: (1) Previous model version is preserved and deployable. "
                    "(2) Rollback can be triggered with one click/command. "
                    "(3) Monitoring alerts are set up to trigger automatic rollback if "
                    "error rate exceeds 2x baseline."
                ),
                impact="medium", tags=["rollback", "safety"], rule_id="DS-002",
            ))

    # ══════════════════════════════════════════════════════════════
    # 14. MONITORING & DRIFT RULES (MD-001 → MD-008)
    # ══════════════════════════════════════════════════════════════

    def _rules_monitoring_drift(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}

        total_predictions = (screen_ctx.get("total_predictions") or
                             frontend.get("total_predictions", 0))
        error_rate = (screen_ctx.get("error_rate") or
                      frontend.get("error_rate", 0))
        avg_latency = (screen_ctx.get("avg_latency") or
                       frontend.get("avg_latency", 0))

        # MD-001: High error rate
        if error_rate and error_rate > 5:
            out.append(Insight(
                severity="critical" if error_rate > 10 else "warning",
                category="Monitoring & Drift",
                title=f"{'High' if error_rate > 10 else 'Elevated'} Error Rate ({error_rate:.1f}%)",
                message=(
                    f"Prediction error rate is {error_rate:.1f}%. "
                    f"{'This exceeds acceptable thresholds and may indicate data drift or model degradation.' if error_rate > 10 else 'Monitor closely for increasing trends.'}"
                ),
                action=(
                    "Investigate: (1) Has the input data distribution changed? "
                    "(2) Are there new categories/values the model hasn't seen? "
                    "(3) Check for data pipeline issues (null values, schema changes). "
                    "Consider retraining on recent data."
                ),
                evidence=f"Error rate: {error_rate:.1f}%",
                metric_key="error_rate", metric_value=error_rate,
                impact="high", tags=["monitoring", "drift"], rule_id="MD-001",
            ))

        # MD-002: High latency
        if avg_latency and avg_latency > 500:
            out.append(Insight(
                severity="warning", category="Monitoring & Drift",
                title=f"High Prediction Latency ({avg_latency:.0f}ms)",
                message=(
                    f"Average prediction latency of {avg_latency:.0f}ms exceeds the "
                    f"recommended 200ms for real-time applications. Users may experience "
                    f"noticeable delays."
                ),
                action=(
                    "Options: (1) Use a simpler model (Logistic Regression < 5ms). "
                    "(2) Reduce feature count. (3) Pre-compute expensive features. "
                    "(4) Use model distillation to create a faster approximation."
                ),
                evidence=f"Avg latency: {avg_latency:.0f}ms",
                metric_key="avg_latency", metric_value=avg_latency,
                impact="medium", tags=["latency", "performance"], rule_id="MD-002",
            ))

        # MD-003: Monitoring best practices
        if ctx.get("screen") == "monitoring":
            out.append(Insight(
                severity="tip", category="Monitoring & Drift",
                title="Set Up Drift Detection",
                message=(
                    "Production models degrade over time as real-world data distributions "
                    "shift. Without drift detection, you won't know your model is failing "
                    "until business metrics drop — which can take weeks."
                ),
                action=(
                    "Monitor these signals: (1) Input feature distributions (PSI > 0.2 = significant drift). "
                    "(2) Prediction distribution (sudden shift = concept drift). "
                    "(3) Feature null rates (schema drift). "
                    "(4) Retrain monthly or when PSI exceeds threshold."
                ),
                impact="medium", tags=["drift", "monitoring_setup"], rule_id="MD-003",
            ))

    # ══════════════════════════════════════════════════════════════
    # 15. PIPELINE LIFECYCLE RULES (PL-001 → PL-008)
    # ══════════════════════════════════════════════════════════════

    def _rules_pipeline_lifecycle(self, ctx: Dict, out: List[Insight]):
        pipeline = ctx.get("pipeline_state", {})
        screen = ctx.get("screen", "")
        training = ctx.get("training_history", {})

        if not pipeline:
            return

        phases = pipeline.get("phases_completed", [])
        next_phase = pipeline.get("next_recommended_phase")
        last_phase = pipeline.get("last_phase")

        # PL-001: No pipeline runs yet
        if not phases and screen in ("dashboard", "mlflow", "training"):
            out.append(Insight(
                severity="info", category="Pipeline Lifecycle",
                title="No Pipeline Runs Detected",
                message=(
                    "No Kedro pipeline phases have been executed yet. "
                    "Start with data loading to bring your data into the platform, "
                    "then proceed through feature engineering and model training."
                ),
                action="Go to ML Flow to start the end-to-end pipeline, or run individual phases.",
                impact="medium", tags=["getting_started"], rule_id="PL-001",
            ))

        # PL-002: Suggest next phase
        if next_phase and screen in ("dashboard", "mlflow"):
            out.append(Insight(
                severity="tip", category="Pipeline Lifecycle",
                title=f"Next Step: {next_phase}",
                message=f"You've completed through {last_phase or 'initial setup'}. The recommended next phase is {next_phase}.",
                action=f"Navigate to ML Flow and run {next_phase}.",
                evidence=f"Completed: {', '.join(phases[:3]) if phases else 'None'}",
                impact="medium", tags=["workflow"], rule_id="PL-002",
            ))

        # PL-003: Failed jobs
        failed = training.get("failed_jobs", 0)
        if failed > 0:
            out.append(Insight(
                severity="warning", category="Pipeline Lifecycle",
                title=f"{failed} Failed Job(s) Detected",
                message=(
                    f"There {'are' if failed > 1 else 'is'} {failed} failed pipeline "
                    f"{'runs' if failed > 1 else 'run'}. Check the error messages and fix "
                    f"underlying issues before continuing."
                ),
                action="Check job logs for error details. Common causes: missing files, memory errors, invalid configurations.",
                evidence=f"{failed} failed jobs",
                impact="medium", tags=["pipeline_errors"], rule_id="PL-003",
            ))

        # PL-004: Success - pipeline completion
        all_phases = [
            "Phase 1: Data Loading",
            "Phase 2: Feature Engineering",
            "Phase 3: Model Training",
            "Phase 4: Algorithm Comparison",
        ]
        completed_main = [p for p in all_phases if p in phases]
        if len(completed_main) >= 3:
            out.append(Insight(
                severity="success", category="Pipeline Lifecycle",
                title=f"Strong Pipeline Progress — {len(completed_main)}/4 Phases Complete",
                message=f"Completed: {', '.join(completed_main)}. Your ML pipeline is well-established.",
                action=None,
                evidence=f"{len(completed_main)} main phases completed",
                impact="low", tags=["progress"], rule_id="PL-004",
            ))
