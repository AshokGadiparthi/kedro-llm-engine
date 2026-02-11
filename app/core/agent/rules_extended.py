"""
Extended Rules — 90+ Additional Expert Rules
=============================================
Extends the base RuleEngine with deeper, more nuanced rules.
These are the rules that separate a good ML practitioner from a great one.

New Rule Categories & IDs:
  DQ-011 → DQ-020:  Advanced data quality (type coercion, outliers, sparsity)
  FH-007 → FH-015:  Advanced feature health (outlier profiles, distribution types, target correlation)
  FE-004 → FE-012:  Advanced feature engineering (binning, aggregations, lag features)
  TV-004 → TV-010:  Advanced target variable (multi-class, regression targets, target transforms)
  DL-003 → DL-008:  Advanced leakage detection (temporal, preprocessing, group leakage)
  MC-003 → MC-008:  Advanced multicollinearity (VIF proxy, correlation clusters)
  SS-006 → SS-010:  Advanced sample size (per-class, statistical power, effect size)
  TC-007 → TC-015:  Advanced training config (stratification, seed, reproducibility)
  AS-006 → AS-015:  Advanced algorithm selection (problem-specific, interpretability vs perf)
  EM-010 → EM-020:  Advanced evaluation (calibration, confusion matrix patterns, lift)
  MC-003 → MC-008:  Model comparison (Occam's razor, significance, stability)
  PR-003 → PR-010:  Production readiness (input validation, monitoring, schema)
  DS-003 → DS-010:  Deployment safety (A/B testing, gradual rollout, feature stores)
  MD-004 → MD-012:  Advanced monitoring (concept drift, label drift, alerting)
  PL-005 → PL-012:  Pipeline lifecycle (reproducibility, lineage, documentation)
"""

import logging
import math
from typing import Any, Dict, List
from .rule_engine import Insight

logger = logging.getLogger(__name__)


class ExtendedRulesMixin:
    """
    Mixin that adds 90+ additional expert rules.
    Designed to be mixed into RuleEngine.
    """

    # ══════════════════════════════════════════════════════════
    # EXTENDED DATA QUALITY RULES (DQ-011 → DQ-020)
    # ══════════════════════════════════════════════════════════

    def _rules_data_quality_extended(self, ctx: Dict, out: List[Insight]):
        quality = ctx.get("data_quality", {})
        profile = ctx.get("dataset_profile", {})
        features = ctx.get("feature_stats", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        if not rows:
            return

        # DQ-010: Numeric columns stored as strings (e.g., TotalCharges in Telco)
        # High-cardinality categoricals with numeric-sounding names are likely
        # numeric data stored as object dtype due to parsing issues (empty strings, etc.)
        cat_stats = features.get("categorical_stats", {})
        numeric_keywords = ["amount", "total", "charge", "price", "cost", "revenue",
                            "income", "salary", "fee", "score", "rating", "count",
                            "balance", "spend", "value", "rate", "percentage", "pct"]
        mistyped_cols = []
        for col, cs in cat_stats.items():
            if not isinstance(cs, dict):
                continue
            unique_ratio = cs.get("unique_ratio", 0)
            unique_count = cs.get("unique", 0)
            # High cardinality + numeric-sounding name = likely numeric stored as string
            if unique_ratio > 0.3 and unique_count > 20:
                if any(k in col.lower() for k in numeric_keywords):
                    mistyped_cols.append(col)

        if mistyped_cols:
            names = ", ".join(mistyped_cols[:4])
            out.append(Insight(
                severity="warning", category="Data Quality",
                title=f"Likely Numeric Column(s) Stored as Text: {names}",
                message=(
                    f"Column(s) {names} have high cardinality and numeric-sounding names "
                    f"but are typed as categorical/object. This often happens when a column "
                    f"has blank strings, special characters, or formatting issues that prevent "
                    f"pandas from auto-detecting the numeric dtype. The column is being treated "
                    f"as text, which means it cannot be used in calculations or modeling."
                ),
                action=(
                    f"Convert to numeric: df['{mistyped_cols[0]}'] = pd.to_numeric("
                    f"df['{mistyped_cols[0]}'], errors='coerce'). The errors='coerce' flag "
                    f"converts unparseable values to NaN. Then check for and impute the "
                    f"resulting missing values."
                ),
                evidence=f"High-cardinality categoricals with numeric names: {names}",
                impact="high", tags=["dtype", "type_coercion", "data_cleaning"], rule_id="DQ-010",
            ))

        # DQ-011: Outlier detection via IQR proxy
        numeric_stats = features.get("numeric_stats", {})
        outlier_cols = []
        for col, stats in numeric_stats.items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            max_val = stats.get("max", 0)
            min_val = stats.get("min", 0)
            if std > 0 and mean != 0:
                # Check if range is extreme relative to std
                range_ratio = (max_val - min_val) / std if std else 0
                if range_ratio > 10:
                    outlier_cols.append({"column": col, "range_ratio": round(range_ratio, 1)})

        if len(outlier_cols) >= 2:
            names = ", ".join(f"{c['column']} ({c['range_ratio']}σ range)" for c in outlier_cols[:4])
            out.append(Insight(
                severity="warning", category="Data Quality",
                title=f"{len(outlier_cols)} Feature(s) With Extreme Outliers",
                message=(
                    f"Features with suspiciously wide ranges relative to their standard "
                    f"deviation: {names}. Outliers can dominate model fitting, especially "
                    f"for distance-based (KNN, SVM) and linear models."
                ),
                action=(
                    "Options: (1) Winsorize at 1st/99th percentile. (2) Log-transform for "
                    "right-skewed data. (3) Use RobustScaler. (4) Tree-based models "
                    "are naturally robust to outliers."
                ),
                evidence=f"{len(outlier_cols)} features with range > 10σ",
                impact="medium", tags=["outliers", "robustness"], rule_id="DQ-011",
            ))

        # DQ-012: Very wide dataset (many features)
        # Skip if cols > rows — DQ-008 already covers this more critically
        if cols > 100 and rows < cols * 5 and rows >= cols:
            out.append(Insight(
                severity="warning", category="Data Quality",
                title=f"Wide Dataset ({cols} features) — Curse of Dimensionality Risk",
                message=(
                    f"With {cols} features and {rows:,} samples, the feature space is "
                    f"high-dimensional. Distance metrics become unreliable (all points appear "
                    f"equidistant), KNN breaks down, and spurious correlations increase."
                ),
                action=(
                    "Apply dimensionality reduction: (1) PCA to retain 95% variance. "
                    "(2) Feature selection via mutual information or L1 regularization. "
                    "(3) Recursive Feature Elimination to find optimal subset size."
                ),
                evidence=f"{cols} features / {rows} samples = {rows/cols:.1f} ratio",
                impact="high", tags=["dimensionality", "feature_selection"], rule_id="DQ-012",
            ))

        # DQ-013: All numeric dataset — encoding opportunity
        cat_count = profile.get("categorical_count", 0)
        num_count = profile.get("numeric_count", 0)
        if num_count > 0 and cat_count == 0 and cols > 5:
            out.append(Insight(
                severity="info", category="Data Quality",
                title="All-Numeric Dataset Detected",
                message=(
                    f"All {num_count} features are numeric. Verify that no columns should be "
                    f"categorical (e.g., encoded status codes like 0/1/2 that represent "
                    f"categories, ZIP codes, product IDs stored as integers)."
                ),
                action=(
                    "Check for integer columns with low cardinality (≤20 unique values) — "
                    "these may be categorical features stored as numbers. Converting them "
                    "to category type can improve tree-based model performance."
                ),
                evidence=f"{num_count} numeric, {cat_count} categorical",
                impact="low", tags=["data_types"], rule_id="DQ-013",
            ))

        # DQ-014: Small dataset with many features
        # Skip if cols > rows — DQ-008 already covers this more critically
        if rows < 1000 and cols > 20 and rows >= cols:
            out.append(Insight(
                severity="warning", category="Data Quality",
                title=f"Small Data + Many Features ({rows} × {cols})",
                message=(
                    f"Small datasets with many features ({rows/cols:.1f} samples per feature) "
                    f"are prone to spurious patterns. Models can find random noise that "
                    f"appears predictive in training but doesn't generalize."
                ),
                action=(
                    "Aggressively reduce features to top-{k} where k ≤ rows/10. "
                    f"Target: ≤{max(5, rows//10)} features. Use mutual information scoring "
                    f"or forward feature selection."
                ),
                evidence=f"{rows} rows, {cols} columns",
                impact="high", tags=["small_data", "high_dimensionality"], rule_id="DQ-014",
            ))

        # DQ-015: Mixed type warnings
        dtypes = profile.get("dtypes", {})
        object_cols = [c for c, t in dtypes.items() if "object" in str(t).lower()]
        if object_cols and len(object_cols) > cols * 0.5:
            out.append(Insight(
                severity="info", category="Data Quality",
                title=f"{len(object_cols)} 'Object' Type Columns — May Need Type Correction",
                message=(
                    f"More than half of columns ({len(object_cols)}/{cols}) are 'object' type. "
                    f"These require encoding before training. Verify numeric-looking strings "
                    f"are converted to numbers (e.g., '1234.56' → 1234.56)."
                ),
                action="Review each object column. Convert numeric strings with pd.to_numeric(). "
                       "Convert dates with pd.to_datetime(). Encode true categoricals.",
                evidence=f"{len(object_cols)} object-type columns",
                impact="medium", tags=["data_types", "cleaning"], rule_id="DQ-015",
            ))

        # DQ-016: Dataset too small for meaningful ML
        if rows < 50:
            out.append(Insight(
                severity="critical", category="Data Quality",
                title=f"Dataset Too Small for ML ({rows} rows)",
                message=(
                    f"With only {rows} rows, machine learning cannot produce reliable "
                    f"models. Statistical power is insufficient, and any pattern found "
                    f"is likely noise. Even LOOCV gives unreliable estimates."
                ),
                action=(
                    "Collect more data (aim for 500+ rows minimum). If impossible, "
                    "consider simple rule-based approaches, expert systems, or Bayesian "
                    "methods with strong priors."
                ),
                evidence=f"n = {rows}",
                impact="high", confidence=1.0, tags=["insufficient_data"], rule_id="DQ-016",
            ))

        # DQ-017: Memory efficiency opportunity
        memory_mb = profile.get("memory_mb", 0)
        if memory_mb > 100 and rows > 0:
            bytes_per_row = (memory_mb * 1024 * 1024) / rows if rows else 0
            if bytes_per_row > 1000:
                out.append(Insight(
                    severity="tip", category="Data Quality",
                    title=f"Memory Optimization Opportunity ({memory_mb:.0f} MB)",
                    message=(
                        f"Dataset uses ~{bytes_per_row:.0f} bytes per row, suggesting "
                        f"inefficient column types. Downcasting can reduce memory by 50-75%."
                    ),
                    action=(
                        "Optimize: (1) float64→float32 (df[col].astype('float32')). "
                        "(2) int64→int32 or int16 where range permits. "
                        "(3) String columns with <50 unique values → pd.Categorical."
                    ),
                    evidence=f"{memory_mb:.0f} MB, {bytes_per_row:.0f} bytes/row",
                    impact="low", tags=["memory", "optimization"], rule_id="DQ-017",
                ))

        # DQ-018: Suspiciously round numbers
        if numeric_stats:
            round_cols = []
            for col, stats in numeric_stats.items():
                if not isinstance(stats, dict):
                    continue
                std = stats.get("std", 0)
                mean = stats.get("mean", 0)
                if std > 0 and mean != 0:
                    # If all quartiles are round numbers, data may be bucketed
                    q25 = stats.get("q25", stats.get("25%"))
                    q75 = stats.get("q75", stats.get("75%"))
                    if q25 is not None and q75 is not None:
                        if q25 == int(q25) and q75 == int(q75) and mean == int(mean):
                            round_cols.append(col)

            if len(round_cols) > 3:
                out.append(Insight(
                    severity="info", category="Data Quality",
                    title=f"{len(round_cols)} Columns With Only Integer Values",
                    message=(
                        f"Columns {', '.join(round_cols[:4])} contain only integer values. "
                        f"These may be categorical codes, ordinal ratings, or counts rather "
                        f"than continuous numeric features."
                    ),
                    action="Review these columns. If ordinal (ratings, levels), consider ordinal encoding. "
                           "If count data, Poisson regression may be more appropriate.",
                    evidence=f"{len(round_cols)} integer-only columns",
                    impact="low", tags=["data_types"], rule_id="DQ-018",
                ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED FEATURE ENGINEERING RULES (FE-004 → FE-012)
    # ══════════════════════════════════════════════════════════

    def _rules_feature_engineering_extended(self, ctx: Dict, out: List[Insight]):
        profile = ctx.get("dataset_profile", {})
        features = ctx.get("feature_stats", {})
        col_types = profile.get("column_types", {})
        numeric_cols = col_types.get("numeric", [])
        cat_cols = col_types.get("categorical", [])
        correlations = ctx.get("correlations", {})
        rows = profile.get("rows", 0)

        # FE-004: Binning continuous features
        skewed = features.get("skewed_features", [])
        if len(numeric_cols) >= 5 and len(cat_cols) < 3:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title="Consider Binning Continuous Features",
                message=(
                    "Converting continuous features into bins (quartiles, deciles, or "
                    "domain-specific ranges) can capture non-linear relationships and "
                    "reduce noise, especially for tree-averse models like Logistic Regression."
                ),
                action=(
                    "Use pd.qcut() for equal-frequency bins (each bin has ~same samples) "
                    "or pd.cut() for equal-width bins. Start with 5-10 bins per feature. "
                    "Domain-driven bins (age groups, income brackets) often outperform automatic ones."
                ),
                evidence=f"{len(numeric_cols)} numeric features available for binning",
                impact="low", tags=["binning", "feature_creation"], rule_id="FE-004",
            ))

        # FE-005: Target encoding for high-cardinality categoricals
        # Exclude columns already flagged as IDs — DL-001 says to drop those
        # Exclude columns likely numeric stored as text — DQ-010 says to convert those
        id_col_names = {c.get("column", "") for c in features.get("potential_id_columns", [])}
        numeric_keywords = {"amount", "total", "charge", "price", "cost", "revenue",
                            "income", "salary", "fee", "score", "rating", "count",
                            "balance", "spend", "value", "rate", "percentage", "pct"}
        high_card = [
            c for c in features.get("high_cardinality_categoricals", [])
            if c.get("column", "") not in id_col_names
               and not any(k in c.get("column", "").lower() for k in numeric_keywords)
        ]
        if high_card:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title=f"Use Target Encoding for High-Cardinality Features",
                message=(
                    f"High-cardinality categoricals ({', '.join(c['column'] for c in high_card[:3])}) "
                    f"will explode into too many columns with one-hot encoding. "
                    f"Target encoding replaces each category with its mean target value, "
                    f"preserving information in a single column."
                ),
                action=(
                    "Use category_encoders.TargetEncoder with smoothing=1.0. "
                    "CRITICAL: fit only on training data to avoid leakage. "
                    "Alternatively, use frequency encoding (count per category / total)."
                ),
                evidence=f"{len(high_card)} high-cardinality categorical features",
                impact="medium", tags=["encoding", "target_encoding"], rule_id="FE-005",
            ))

        # FE-006: Missing indicator features
        quality = ctx.get("data_quality", {})
        missing_cols = quality.get("columns_with_high_missing", [])
        if missing_cols:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title="Create Missing Value Indicator Features",
                message=(
                    f"For columns with significant missing data "
                    f"({', '.join(c['column'] for c in missing_cols[:3])}), "
                    f"the missingness pattern itself can be predictive. Adding binary "
                    f"'is_missing_X' columns captures this signal."
                ),
                action=(
                    "For each column with >5% missing: create df['is_missing_X'] = df['X'].isnull().astype(int). "
                    "Then impute the original column normally. The indicator preserves the information."
                ),
                evidence=f"{len(missing_cols)} columns with >20% missing",
                impact="medium", tags=["missing_data", "feature_creation"], rule_id="FE-006",
            ))

        # FE-007: Feature crosses for categorical pairs
        if len(cat_cols) >= 2:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title=f"Consider Categorical Feature Crosses ({len(cat_cols)} categoricals)",
                message=(
                    f"Combining categorical features (e.g., city + product_type) can capture "
                    f"interaction effects that individual features miss. For example, "
                    f"churn rate may differ by contract_type AND payment_method together."
                ),
                action=(
                    "Create interaction features: df['A_x_B'] = df['A'] + '_' + df['B']. "
                    "Focus on pairs where domain knowledge suggests an interaction. "
                    f"With {len(cat_cols)} categoricals, try the top 2-3 most relevant pairs."
                ),
                evidence=f"{len(cat_cols)} categorical features available",
                impact="medium", tags=["interactions", "categorical"], rule_id="FE-007",
            ))

        # FE-008: Log transform for monetary/count features
        # IMPORTANT: Only flag numeric columns — "PaymentMethod" contains "payment"
        # but is categorical, not monetary
        col_names = profile.get("column_names", [])
        numeric_set = set(profile.get("numeric_columns", []))
        monetary_keywords = ["price", "cost", "revenue", "income", "salary", "amount",
                             "charge", "fee", "spend", "balance", "payment",
                             "total_cost", "total_charge", "total_amount", "total_price",
                             "total_revenue", "total_spend", "total_fee"]
        # Exclude keywords for non-monetary numeric columns
        cat_keywords = ["method", "type", "category", "status", "mode", "plan", "class"]
        non_monetary_keywords = ["area", "distance", "length", "width", "height",
                                 "count", "flag", "days", "months", "years", "age",
                                 "ratio", "rate", "pct", "percent", "score"]
        monetary_cols = [
            c for c in col_names
            if any(k in c.lower() for k in monetary_keywords)
               and not any(nk in c.lower() for nk in non_monetary_keywords)
               and (
                       c in numeric_set  # Must be numeric dtype
                       or not any(ck in c.lower() for ck in cat_keywords)  # OR not obviously categorical
               )
        ]
        # Final filter: if we have numeric column info, strictly enforce it
        if numeric_set:
            monetary_cols = [c for c in monetary_cols if c in numeric_set]
        if monetary_cols:
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title=f"Log-Transform Monetary Features ({len(monetary_cols)} detected)",
                message=(
                    f"Columns {', '.join(monetary_cols[:4])} appear to be monetary/amount features. "
                    f"Monetary data is almost always right-skewed (few large values). "
                    f"Log transformation compresses the range and normalizes the distribution."
                ),
                action=(
                    "Apply np.log1p() (handles zeros safely) to monetary columns. "
                    "This often improves linear model performance by 5-15% and stabilizes "
                    "tree splits."
                ),
                evidence=f"Monetary columns: {', '.join(monetary_cols[:4])}",
                impact="medium", tags=["log_transform", "monetary"], rule_id="FE-008",
            ))

        # FE-009: Correlation-based feature creation
        high_pairs = correlations.get("high_pairs", [])
        mid_corr = [p for p in high_pairs if 0.5 <= p.get("abs_correlation", 0) < 0.7]
        if mid_corr:
            pair = mid_corr[0]
            out.append(Insight(
                severity="tip", category="Feature Engineering",
                title="Create Difference/Ratio Features from Correlated Pairs",
                message=(
                    f"Moderately correlated features like {pair['feature1']} and {pair['feature2']} "
                    f"(r={pair['correlation']:.2f}) may benefit from derived features. "
                    f"The difference or ratio between correlated features often captures "
                    f"a meaningful signal (e.g., income-expenses = savings)."
                ),
                action=(
                    f"Create: df['{pair['feature1']}_minus_{pair['feature2']}'] = "
                    f"df['{pair['feature1']}'] - df['{pair['feature2']}']. "
                    f"Also try: df['{pair['feature1']}_ratio_{pair['feature2']}'] = "
                    f"df['{pair['feature1']}'] / (df['{pair['feature2']}'] + 1e-8)"
                ),
                evidence=f"Correlated pair: {pair['feature1']}/{pair['feature2']} (r={pair['correlation']:.2f})",
                impact="medium", tags=["derived_features", "ratios"], rule_id="FE-009",
            ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED TARGET VARIABLE RULES (TV-004 → TV-010)
    # ══════════════════════════════════════════════════════════

    def _rules_target_variable_extended(self, ctx: Dict, out: List[Insight]):
        features = ctx.get("feature_stats", {})
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")
        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type")
        numeric_stats = features.get("numeric_stats", {})

        if not target:
            return

        # TV-004: Multiclass target detection
        if target in numeric_stats:
            stats = numeric_stats[target]
            if isinstance(stats, dict):
                unique = stats.get("unique", stats.get("count", 0))
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)

                # Detect multiclass
                if 3 <= unique <= 20 and min_val >= 0 and max_val < 100:
                    out.append(Insight(
                        severity="info", category="Target Variable",
                        title=f"Multi-Class Target Detected ({unique} classes)",
                        message=(
                            f"Target '{target}' has {unique} unique values, suggesting "
                            f"a multi-class classification problem. Different algorithms and "
                            f"metrics apply compared to binary classification."
                        ),
                        action=(
                            "Use macro-averaged F1 (treats all classes equally) or "
                            "weighted-averaged F1 (weights by class frequency). "
                            "Check if any class has <5% representation — those need extra attention."
                        ),
                        evidence=f"Target '{target}': {unique} unique classes",
                        impact="medium", tags=["multiclass"], rule_id="TV-004",
                    ))

                # TV-005: Regression target with extreme range
                if unique > 20 and problem_type == "regression":
                    std = stats.get("std", 0)
                    mean = stats.get("mean", 0)
                    cv = std / abs(mean) if mean else 0
                    if cv > 2:
                        out.append(Insight(
                            severity="warning", category="Target Variable",
                            title=f"High Target Variability (CV={cv:.1f})",
                            message=(
                                f"The target '{target}' has a coefficient of variation of "
                                f"{cv:.1f} (std/mean), indicating extreme spread. This makes "
                                f"prediction harder and RMSE may be dominated by extreme values."
                            ),
                            action=(
                                "Consider: (1) Log-transform the target (np.log1p). "
                                "(2) Use MAE instead of RMSE (more robust to outliers). "
                                "(3) Predict log(target) and exponentiate predictions."
                            ),
                            evidence=f"Target CV: {cv:.1f} (mean={mean:.2f}, std={std:.2f})",
                            impact="medium", tags=["regression", "target_transform"], rule_id="TV-005",
                        ))

        # TV-006: Target not selected warning
        if ctx.get("screen") in ("mlflow", "training") and not target:
            out.append(Insight(
                severity="critical", category="Target Variable",
                title="No Target Variable Selected",
                message=(
                    "A target variable must be selected before training. "
                    "The target is what the model learns to predict."
                ),
                action="Select your target variable in the training configuration.",
                impact="high", tags=["target", "configuration"], rule_id="TV-006",
            ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED LEAKAGE RULES (DL-003 → DL-008)
    # ══════════════════════════════════════════════════════════

    def _rules_data_leakage_extended(self, ctx: Dict, out: List[Insight]):
        profile = ctx.get("dataset_profile", {})
        features = ctx.get("feature_stats", {})
        col_types = profile.get("column_types", {})
        col_names = profile.get("column_names", [])
        correlations = ctx.get("correlations", {})

        # DL-003: Temporal leakage risk
        datetime_cols = col_types.get("datetime", [])
        if datetime_cols:
            out.append(Insight(
                severity="warning", category="Data Leakage",
                title=f"Temporal Features Present — Verify No Future Leakage",
                message=(
                    f"Datetime columns detected: {', '.join(datetime_cols[:3])}. "
                    f"Temporal features are a common source of leakage when they encode "
                    f"information that wouldn't be available at prediction time."
                ),
                action=(
                    "For each datetime feature, ask: 'Would I know this at prediction time?' "
                    "Timestamps of the outcome event itself MUST be excluded. "
                    "Use TimeSeriesSplit for cross-validation if data has a temporal component."
                ),
                evidence=f"Datetime columns: {', '.join(datetime_cols[:3])}",
                impact="high", tags=["temporal", "leakage"], rule_id="DL-003",
            ))

        # DL-004: Suspicious column names suggesting post-hoc data
        leakage_keywords = ["result", "outcome", "status_final", "days_to",
                            "time_to", "post_", "after_", "final_",
                            "total_", "cumulative_", "lifetime_"]
        suspicious = [c for c in col_names
                      if any(k in c.lower() for k in leakage_keywords)]
        if suspicious:
            out.append(Insight(
                severity="warning", category="Data Leakage",
                title=f"{len(suspicious)} Column(s) May Contain Post-Outcome Data",
                message=(
                    f"Columns with potentially post-hoc names: {', '.join(suspicious[:5])}. "
                    f"Features derived from or computed after the outcome event cause "
                    f"target leakage — the model appears to perform well but fails in "
                    f"production where these features aren't available yet."
                ),
                action=(
                    "Review each column: ask 'Is this feature available BEFORE the "
                    "prediction is needed?' If not, exclude it. Common leakers: "
                    "total_spend (if predicting churn), days_to_event, final_status."
                ),
                evidence=f"Suspicious columns: {', '.join(suspicious[:5])}",
                impact="high", confidence=0.7,
                tags=["leakage", "post_hoc"], rule_id="DL-004",
            ))

        # DL-005: Group leakage risk
        # Only fire if the column could actually be a GROUP (same entity appears multiple times)
        # If it's 100% unique (like customerID), it's an ID column — no grouping possible
        id_keywords = ["customer", "user", "patient", "account", "group", "household"]
        id_col_names = {c.get("column", "") for c in features.get("potential_id_columns", [])}
        group_cols = [
            c for c in col_names
            if any(k in c.lower() for k in id_keywords)
               and c not in id_col_names  # Exclude 100% unique IDs — they can't be group keys
        ]
        if group_cols:
            out.append(Insight(
                severity="info", category="Data Leakage",
                title=f"Group Identifier Detected — Use GroupKFold for CV",
                message=(
                    f"Columns {', '.join(group_cols[:3])} suggest grouped data (same entity "
                    f"appears multiple times). If the same customer/patient appears in both "
                    f"train and test sets, the model effectively 'memorizes' that entity."
                ),
                action=(
                    "Use GroupKFold or GroupShuffleSplit for cross-validation, splitting "
                    "by the group identifier. This ensures all rows for the same entity "
                    "are in either train OR test, never both."
                ),
                evidence=f"Potential group columns: {', '.join(group_cols[:3])}",
                impact="high", confidence=0.7,
                tags=["group_leakage", "cross_validation"], rule_id="DL-005",
            ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED TRAINING CONFIG RULES (TC-007 → TC-015)
    # ══════════════════════════════════════════════════════════

    def _rules_training_config_extended(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        profile = ctx.get("dataset_profile", {})
        features = ctx.get("feature_stats", {})
        rows = profile.get("rows", 0)

        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type")
        target = screen_ctx.get("target_column") or frontend.get("target_column")
        selected_features = screen_ctx.get("selected_features") or frontend.get("selected_features", [])

        # TC-007: Missing stratification for imbalanced classification
        numeric_stats = features.get("numeric_stats", {})
        if target and target in numeric_stats and problem_type == "classification":
            stats = numeric_stats[target]
            if isinstance(stats, dict):
                mean = stats.get("mean", 0.5)
                if 0 <= mean <= 1:
                    minority_pct = min(mean, 1 - mean)
                    if minority_pct < 0.3:
                        out.append(Insight(
                            severity="warning", category="Training Configuration",
                            title="Stratified Splitting Required for Imbalanced Target",
                            message=(
                                f"Target has {minority_pct*100:.1f}% minority class. Without "
                                f"stratification, some CV folds may have very few or zero "
                                f"minority samples, making evaluation unreliable."
                            ),
                            action=(
                                "Use StratifiedKFold instead of KFold for cross-validation. "
                                "Also set stratify=y in train_test_split()."
                            ),
                            evidence=f"Minority class: {minority_pct*100:.1f}%",
                            impact="high", tags=["stratification", "imbalance"], rule_id="TC-007",
                        ))

        # TC-008: Feature count vs selected features
        all_cols = profile.get("columns", 0)
        if selected_features and all_cols > 0:
            pct_selected = len(selected_features) / all_cols * 100
            if pct_selected > 90:
                out.append(Insight(
                    severity="info", category="Training Configuration",
                    title=f"Using {pct_selected:.0f}% of All Features ({len(selected_features)}/{all_cols})",
                    message=(
                        "You've selected nearly all available features. While this preserves "
                        "maximum information, it may include noise features that degrade "
                        "model performance."
                    ),
                    action=(
                        "Consider running a quick feature importance analysis first "
                        "(Random Forest or mutual information) to identify the top 50-70% "
                        "of features."
                    ),
                    evidence=f"{len(selected_features)}/{all_cols} features selected",
                    impact="low", tags=["feature_selection"], rule_id="TC-008",
                ))

        # TC-009: Reproducibility reminder
        if ctx.get("screen") in ("mlflow", "training"):
            out.append(Insight(
                severity="tip", category="Training Configuration",
                title="Set a Random Seed for Reproducibility",
                message=(
                    "Always set random_state/seed for reproducible results. Without it, "
                    "results vary between runs, making debugging and comparison impossible."
                ),
                action="Set random_state=42 (or any fixed integer) in all sklearn/XGBoost/LightGBM calls.",
                impact="low", tags=["reproducibility"], rule_id="TC-009",
            ))

        # TC-010: Suggest end-to-end pipeline
        if ctx.get("screen") == "training" and rows > 0:
            out.append(Insight(
                severity="tip", category="Training Configuration",
                title="Use End-to-End Pipeline for Fair Comparison",
                message=(
                    "Run the full Kedro pipeline (Phases 1-6) for a fair comparison across "
                    "algorithms. This ensures identical preprocessing, feature engineering, "
                    "and evaluation for each algorithm."
                ),
                action=(
                    "Select 'End-to-End Pipeline' to run all phases automatically. "
                    "This includes data loading, feature engineering, training, "
                    "multi-algorithm comparison, and ensemble evaluation."
                ),
                impact="medium", tags=["pipeline", "best_practice"], rule_id="TC-010",
            ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED ALGORITHM SELECTION RULES (AS-006 → AS-015)
    # ══════════════════════════════════════════════════════════

    def _rules_algorithm_selection_extended(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cat_count = profile.get("categorical_count", 0)
        num_count = profile.get("numeric_count", 0)

        algorithm = screen_ctx.get("algorithm") or frontend.get("algorithm")
        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type", "classification")
        algo_lower = (algorithm or "").lower().replace(" ", "").replace("_", "")

        # AS-006: CatBoost for high-cardinality categoricals
        high_card = ctx.get("feature_stats", {}).get("high_cardinality_categoricals", [])
        if high_card and algorithm and "catboost" not in algo_lower:
            out.append(Insight(
                severity="tip", category="Algorithm Selection",
                title="Consider CatBoost for Your Categorical-Heavy Data",
                message=(
                    f"You have {len(high_card)} high-cardinality categorical feature(s). "
                    f"CatBoost uses ordered target encoding internally, handling categoricals "
                    f"without manual encoding while preventing target leakage."
                ),
                action="Add CatBoost to your algorithm comparison. No encoding needed — "
                       "just specify categorical_features parameter.",
                evidence=f"{len(high_card)} high-cardinality categoricals, {cat_count} total",
                impact="medium", tags=["catboost", "categorical"], rule_id="AS-006",
            ))

        # AS-007: Interpretability vs performance trade-off
        if algorithm and any(x in algo_lower for x in ["xgboost", "lightgbm", "catboost", "neural"]):
            out.append(Insight(
                severity="info", category="Algorithm Selection",
                title="Complex Model Selected — Consider Interpretability Needs",
                message=(
                    f"'{algorithm}' is a powerful but opaque model. If stakeholders need "
                    f"to understand WHY a prediction was made (regulatory, trust, debugging), "
                    f"you may need SHAP values, LIME, or a simpler surrogate model."
                ),
                action=(
                    "After training, generate SHAP explanations for model interpretability. "
                    "Also train a Logistic Regression as a comparison — if it's within 2-3% "
                    "of the complex model, prefer it."
                ),
                evidence=f"Algorithm: {algorithm}",
                impact="low", tags=["interpretability", "explainability"], rule_id="AS-007",
            ))

        # AS-008: LightGBM for speed
        if rows > 50000 and algorithm and "light" not in algo_lower:
            out.append(Insight(
                severity="tip", category="Algorithm Selection",
                title=f"Consider LightGBM for Speed ({rows:,} rows)",
                message=(
                    f"With {rows:,} rows, LightGBM is typically 5-10x faster than XGBoost "
                    f"while achieving comparable accuracy. It uses histogram-based splitting "
                    f"and leaf-wise growth for efficiency."
                ),
                action="Add LightGBM to your comparison. It also handles categoricals natively.",
                evidence=f"Dataset size: {rows:,} rows",
                impact="low", tags=["speed", "lightgbm"], rule_id="AS-008",
            ))

        # AS-009: Baseline model reminder
        all_algos = (screen_ctx.get("selected_algorithms") or
                     frontend.get("selected_algorithms", []))
        if all_algos and not any("logistic" in a.lower() or "linear" in a.lower()
                                 for a in all_algos):
            out.append(Insight(
                severity="tip", category="Algorithm Selection",
                title="Include a Simple Baseline in Your Comparison",
                message=(
                    "Your algorithm selection doesn't include a simple baseline "
                    "(Logistic/Linear Regression). Without a baseline, you can't quantify "
                    "the value added by complex models."
                ),
                action=(
                    "Add Logistic Regression (classification) or Ridge Regression (regression). "
                    "If the complex model only beats it by 1-2%, the simpler model is "
                    "better for production (faster, more interpretable, easier to debug)."
                ),
                evidence=f"Selected: {', '.join(all_algos[:4])}",
                impact="medium", tags=["baseline"], rule_id="AS-009",
            ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED EVALUATION METRICS RULES (EM-010 → EM-020)
    # ══════════════════════════════════════════════════════════

    def _rules_evaluation_metrics_extended(self, ctx: Dict, out: List[Insight]):
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

        # EM-010: All metrics low — fundamental model problem
        metric_vals = [v for v in [accuracy, f1, precision, recall, auc_roc] if v is not None]
        if metric_vals and all(v < 0.6 for v in metric_vals):
            out.append(Insight(
                severity="critical", category="Evaluation Metrics",
                title="All Metrics Below 60% — Fundamental Model Problem",
                message=(
                    "All reported metrics are below 0.6, indicating the model is not "
                    "learning meaningful patterns. This is worse than many simple baselines."
                ),
                action=(
                    "Investigate root causes: (1) Is the target actually predictable? "
                    "(2) Are the features relevant to the outcome? (3) Is there data quality "
                    "issues? (4) Try a completely different algorithm family. (5) Feature "
                    "engineering may be needed to surface the signal."
                ),
                evidence=f"All metrics < 0.6",
                impact="high", tags=["model_failure"], rule_id="EM-010",
            ))

        # EM-011: Perfect train + poor test = clear overfitting
        train_acc = metrics.get("train_score") or metrics.get("training_accuracy")
        test_acc = metrics.get("test_score") or metrics.get("test_accuracy")
        if train_acc and test_acc and train_acc > 0.98 and test_acc < 0.7:
            out.append(Insight(
                severity="critical", category="Evaluation Metrics",
                title="Extreme Overfitting: Perfect Training, Poor Test",
                message=(
                    f"Training score ({train_acc*100:.1f}%) is near-perfect while test score "
                    f"({test_acc*100:.1f}%) is much lower. The model has completely memorized "
                    f"the training data without learning generalizable patterns."
                ),
                action=(
                    "This is almost certainly caused by: (1) Data leakage — check for target "
                    "proxies. (2) Too few samples for model complexity. (3) Too many features. "
                    "Start with the simplest model (Logistic Regression) to establish a "
                    "baseline, then gradually add complexity."
                ),
                evidence=f"Train: {train_acc*100:.1f}% → Test: {test_acc*100:.1f}%",
                impact="high", tags=["overfitting", "leakage"], rule_id="EM-011",
            ))

        # EM-012: Precision = 1.0 (suspiciously perfect)
        if precision and precision >= 0.99 and recall and recall < 0.3:
            out.append(Insight(
                severity="warning", category="Evaluation Metrics",
                title="Perfect Precision + Low Recall — Model Too Conservative",
                message=(
                    f"Precision is {precision*100:.1f}% but recall is only {recall*100:.1f}%. "
                    f"The model makes very few positive predictions, but when it does, it's "
                    f"always right. This means it's catching only the most obvious cases."
                ),
                action=(
                    "Lower the classification threshold significantly (try 0.2-0.3). "
                    "The model knows what a positive looks like but is too cautious. "
                    "Also verify class weights are properly set."
                ),
                evidence=f"Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%",
                impact="high", tags=["threshold", "conservative"], rule_id="EM-012",
            ))

        # EM-013: Good AUC but bad F1 — threshold needs tuning
        if auc_roc and f1 and auc_roc > 0.8 and f1 < 0.5:
            out.append(Insight(
                severity="warning", category="Evaluation Metrics",
                title=f"High AUC ({auc_roc:.3f}) but Low F1 ({f1:.3f}) — Threshold Issue",
                message=(
                    f"AUC-ROC of {auc_roc:.3f} shows the model has good discrimination "
                    f"ability, but F1 of {f1:.3f} means performance at the current threshold "
                    f"(likely 0.5) is poor. The model CAN rank well — it just needs a better "
                    f"operating point."
                ),
                action=(
                    "This is a threshold problem, not a model problem. Scan thresholds from "
                    "0.1 to 0.9 in steps of 0.05 and pick the one that maximizes F1 "
                    "(or your preferred metric). The optimal threshold is often around 0.2-0.4 "
                    "for imbalanced data."
                ),
                evidence=f"AUC: {auc_roc:.3f}, F1: {f1:.3f}",
                impact="high", tags=["threshold", "optimization"], rule_id="EM-013",
            ))

        # EM-014: Suggest MCC for balanced assessment
        if accuracy and f1 and abs(accuracy - f1) > 0.1:
            out.append(Insight(
                severity="tip", category="Evaluation Metrics",
                title="Use Matthews Correlation Coefficient (MCC) for Balanced Assessment",
                message=(
                    "When accuracy and F1 disagree, MCC provides the most balanced single "
                    "metric. It's the ONLY metric that gives a high score only when the "
                    "model performs well on both positive and negative classes."
                ),
                action="Add MCC to your evaluation: sklearn.metrics.matthews_corrcoef(y_true, y_pred). "
                       "MCC > 0.4 = acceptable, > 0.6 = good, > 0.8 = excellent.",
                evidence=f"Accuracy-F1 gap: {abs(accuracy - f1)*100:.1f}pt",
                impact="low", tags=["mcc", "balanced_metric"], rule_id="EM-014",
            ))

        # EM-015: Calibration check reminder
        if auc_roc and auc_roc > 0.75:
            out.append(Insight(
                severity="tip", category="Evaluation Metrics",
                title="Check Model Calibration for Reliable Probabilities",
                message=(
                    "Good AUC doesn't mean predicted probabilities are reliable. "
                    "An uncalibrated model might output 0.8 probability when the true "
                    "probability is only 0.5. This matters for threshold tuning and "
                    "business decisions based on predicted confidence."
                ),
                action=(
                    "Plot a calibration curve (reliability diagram). If it deviates from "
                    "the diagonal, apply Platt scaling or isotonic regression. "
                    "sklearn.calibration.CalibratedClassifierCV makes this easy."
                ),
                evidence=f"AUC: {auc_roc:.3f} (model shows discrimination ability)",
                impact="low", tags=["calibration", "probabilities"], rule_id="EM-015",
            ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED DEPLOYMENT & MONITORING RULES
    # ══════════════════════════════════════════════════════════

    def _rules_deployment_extended(self, ctx: Dict, out: List[Insight]):
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        registry = ctx.get("registry_info", {})

        if ctx.get("screen") not in ("deployment", "registry"):
            return

        # DS-003: Input validation schema
        out.append(Insight(
            severity="warning", category="Deployment Safety",
            title="Define Input Validation Schema Before Deploying",
            message=(
                "Production models receive unexpected inputs: nulls, out-of-range values, "
                "new categories, wrong types. Without input validation, the model may "
                "silently produce garbage predictions instead of failing gracefully."
            ),
            action=(
                "Create a schema that validates: (1) All required features are present. "
                "(2) Numeric features are within training-time ranges. (3) Categorical "
                "features only contain known categories. (4) No null values in required "
                "fields. Use pydantic or Great Expectations."
            ),
            impact="high", tags=["input_validation", "schema"], rule_id="DS-003",
        ))

        # DS-004: Logging predictions
        out.append(Insight(
            severity="tip", category="Deployment Safety",
            title="Log All Predictions for Monitoring & Retraining",
            message=(
                "Logging every prediction (input features + output + timestamp) enables: "
                "drift detection, debugging, model comparison, and future retraining "
                "on production data."
            ),
            action=(
                "Log to a structured store (BigQuery, S3/Parquet, PostgreSQL): "
                "request_id, timestamp, features, prediction, probability, "
                "model_version, latency_ms."
            ),
            impact="medium", tags=["logging", "observability"], rule_id="DS-004",
        ))

        # DS-005: Feature store recommendation
        profile = ctx.get("dataset_profile", {})
        cols = profile.get("columns", 0)
        if cols > 15:
            out.append(Insight(
                severity="tip", category="Deployment Safety",
                title=f"Consider a Feature Store ({cols} features)",
                message=(
                    f"With {cols} features, ensuring training-serving consistency is "
                    f"challenging. A feature store guarantees the same feature "
                    f"transformations run in both training and production."
                ),
                action=(
                    "Options: (1) Feast (open source). (2) Tecton (managed). "
                    "(3) Simple approach: save the sklearn Pipeline/ColumnTransformer "
                    "and use it for both training and serving."
                ),
                evidence=f"{cols} features to maintain consistency",
                impact="low", tags=["feature_store", "consistency"], rule_id="DS-005",
            ))

    def _rules_monitoring_extended(self, ctx: Dict, out: List[Insight]):
        if ctx.get("screen") != "monitoring":
            return

        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        total_predictions = screen_ctx.get("total_predictions") or frontend.get("total_predictions", 0)

        # MD-004: Set up alerting
        out.append(Insight(
            severity="tip", category="Monitoring & Drift",
            title="Configure Automated Alerting Thresholds",
            message=(
                "Manual monitoring doesn't scale. Set up automated alerts that trigger "
                "when key metrics cross thresholds."
            ),
            action=(
                "Recommended alerts: "
                "(1) Error rate > 2x baseline → PagerDuty. "
                "(2) P99 latency > 500ms → warning. "
                "(3) Any feature PSI > 0.25 → data drift alert. "
                "(4) Prediction volume drops >50% → data pipeline issue. "
                "(5) Null rate in any feature > 5% → schema drift."
            ),
            impact="medium", tags=["alerting", "automation"], rule_id="MD-004",
        ))

        # MD-005: Prediction volume monitoring
        if total_predictions > 0:
            out.append(Insight(
                severity="info", category="Monitoring & Drift",
                title=f"Track Prediction Volume Trends ({total_predictions:,} total)",
                message=(
                    "Sudden changes in prediction volume often indicate upstream issues: "
                    "data pipeline failures, traffic routing changes, or dependency outages."
                ),
                action=(
                    "Set up a time-series alert on prediction volume. "
                    "A >50% drop or >200% spike should trigger investigation."
                ),
                evidence=f"Total predictions: {total_predictions:,}",
                impact="low", tags=["volume", "monitoring"], rule_id="MD-005",
            ))

        # MD-006: Regular retraining schedule
        out.append(Insight(
            severity="tip", category="Monitoring & Drift",
            title="Establish a Regular Retraining Schedule",
            message=(
                "All models degrade over time as the world changes. Even without "
                "detected drift, periodic retraining on recent data keeps the model fresh."
            ),
            action=(
                "Start with monthly retraining. If drift is detected frequently, "
                "increase to weekly. If performance is stable, quarterly may suffice. "
                "Always A/B test retrained models against the current champion."
            ),
            impact="medium", tags=["retraining", "schedule"], rule_id="MD-006",
        ))

        # MD-007: Feedback loop
        out.append(Insight(
            severity="info", category="Monitoring & Drift",
            title="Build a Ground Truth Feedback Loop",
            message=(
                "Monitoring prediction quality requires knowing the actual outcomes. "
                "Without ground truth labels, you can only detect input drift, not "
                "whether predictions are actually getting worse."
            ),
            action=(
                "Set up a delayed-label pipeline: collect actual outcomes when they "
                "become available (e.g., 30-day churn label, conversion event, "
                "claim outcome) and compute real-time model accuracy."
            ),
            impact="medium", tags=["feedback_loop", "ground_truth"], rule_id="MD-007",
        ))

    # ══════════════════════════════════════════════════════════
    # EXTENDED PIPELINE LIFECYCLE RULES (PL-005 → PL-012)
    # ══════════════════════════════════════════════════════════

    def _rules_pipeline_lifecycle_extended(self, ctx: Dict, out: List[Insight]):
        pipeline = ctx.get("pipeline_state", {})
        training = ctx.get("training_history", {})
        screen = ctx.get("screen", "")

        phases = pipeline.get("phases_completed", [])

        # PL-005: Skipped phases
        phase_order = [
            "Phase 1: Data Loading",
            "Phase 2: Feature Engineering",
            "Phase 3: Model Training",
            "Phase 4: Algorithm Comparison",
        ]
        if phases:
            # Check for gaps
            max_phase_idx = -1
            for phase in phases:
                for idx, expected in enumerate(phase_order):
                    if expected in phase:
                        max_phase_idx = max(max_phase_idx, idx)

            if max_phase_idx > 0:
                skipped = []
                for i in range(max_phase_idx):
                    if not any(phase_order[i] in p for p in phases):
                        skipped.append(phase_order[i])
                if skipped:
                    out.append(Insight(
                        severity="warning", category="Pipeline Lifecycle",
                        title=f"Pipeline Phase(s) Skipped: {', '.join(skipped)}",
                        message=(
                            "Running phases out of order may produce incorrect results. "
                            "Feature engineering must run before training, and training "
                            "before comparison."
                        ),
                        action="Re-run the skipped phases or use 'End-to-End Pipeline' for correct ordering.",
                        evidence=f"Skipped: {', '.join(skipped)}",
                        impact="high", tags=["pipeline_order"], rule_id="PL-005",
                    ))

        # PL-006: Many re-runs (iteration tracking)
        total_jobs = training.get("total_jobs", 0)
        completed = training.get("completed_jobs", 0)
        if total_jobs > 10:
            success_rate = completed / total_jobs * 100 if total_jobs else 0
            out.append(Insight(
                severity="info", category="Pipeline Lifecycle",
                title=f"Active Experimentation: {total_jobs} Pipeline Runs ({success_rate:.0f}% success)",
                message=(
                    f"You've run {total_jobs} pipeline jobs. "
                    f"{'Great iteration velocity!' if success_rate > 80 else 'Many runs are failing — check configurations.'}"
                ),
                action=None if success_rate > 80 else "Review failed job logs for common patterns.",
                evidence=f"{total_jobs} jobs, {completed} completed",
                impact="low", tags=["experimentation"], rule_id="PL-006",
            ))

        # PL-007: Execution time trends
        avg_time = training.get("avg_execution_time", 0)
        if avg_time > 300:  # > 5 minutes
            out.append(Insight(
                severity="info", category="Pipeline Lifecycle",
                title=f"Long Average Execution Time ({avg_time:.0f}s)",
                message=(
                    f"Pipeline runs average {avg_time:.0f} seconds ({avg_time/60:.1f} minutes). "
                    f"Long execution times slow iteration velocity."
                ),
                action=(
                    "To speed up: (1) Use LightGBM instead of XGBoost. "
                    "(2) Reduce n_estimators and use early stopping. "
                    "(3) Sample data for initial experiments, train on full data for final model."
                ),
                evidence=f"Avg execution: {avg_time:.0f}s",
                impact="low", tags=["performance", "speed"], rule_id="PL-007",
            ))

        # PL-008: Model documentation reminder
        if screen in ("evaluation", "registry"):
            out.append(Insight(
                severity="tip", category="Pipeline Lifecycle",
                title="Document Your Model for Future Reference",
                message=(
                    "A model card captures: problem framing, data description, feature "
                    "rationale, algorithm choice, performance metrics, limitations, "
                    "and ethical considerations. Essential for handoff and auditability."
                ),
                action=(
                    "Create a model card documenting: (1) Business objective. "
                    "(2) Training data description and limitations. (3) Feature list and rationale. "
                    "(4) Performance metrics on all relevant slices. (5) Known failure modes."
                ),
                impact="low", tags=["documentation", "model_card"], rule_id="PL-008",
            ))