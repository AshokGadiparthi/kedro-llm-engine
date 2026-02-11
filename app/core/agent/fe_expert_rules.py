"""
Expert Feature Engineering Rules (FE-010 → FE-025)
====================================================
The rules that separate a staff-level ML scientist from a mid-level one.

These rules fire on the feature_engineering screen and analyze the actual
pipeline execution results — not just the config, but what HAPPENED.

Rule Catalog:
  FE-010: Numeric column misclassified as categorical (TotalCharges bug)
  FE-011: ID column false negative (customerID not detected)
  FE-012: Variance filter removing binary predictors
  FE-013: Feature selection ratio too aggressive
  FE-014: Scaling method vs data distribution mismatch
  FE-015: One-hot encoding explosion (>3x feature expansion)
  FE-016: Rare category complete collapse (>95% grouped to 'Other')
  FE-017: Feature dimensionality vs sample size warning
  FE-018: Binary features scaled then killed by variance
  FE-019: Missing domain-standard interaction features
  FE-020: Post-encoding feature redundancy
  FE-021: Ordinal variable treated as nominal
  FE-022: Pipeline information retention below threshold
  FE-023: Feature type detection inconsistency with EDA
  FE-024: Scaling unnecessary for tree-only workflows
  FE-025: Pipeline completeness — missing recommended steps

Data Requirements:
  These rules need screen_context populated with pipeline execution results.
  The context_compiler's _enrich_feature_engineering() provides this data.
"""

import logging
import math
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Import Insight from rule_engine ──
try:
    from .rule_engine import Insight
except ImportError:
    from app.core.agent.rule_engine import Insight

# ── Domain knowledge constants ──
NUMERIC_NAME_PATTERNS = [
    "charge", "amount", "price", "cost", "fee", "revenue", "salary",
    "income", "balance", "payment", "total", "sum", "count", "quantity",
    "rate", "score", "rating", "weight", "height", "age", "distance",
    "duration", "time", "days", "months", "years", "percent", "ratio",
    "value", "profit", "loss", "margin", "volume", "area", "speed",
]

ID_NAME_PATTERNS = [
    "id", "uuid", "guid", "key", "pk", "idx", "index",
    "customerid", "userid", "accountid", "sessionid",
    "transactionid", "orderid", "productid", "employeeid",
]

KNOWN_CHURN_PREDICTORS = {
    "binary": ["gender", "partner", "dependents", "phoneservice",
               "paperlessbilling", "seniorcitizen"],
    "categorical": ["contract", "internetservice", "paymentmethod"],
    "numeric": ["tenure", "monthlycharges", "totalcharges"],
    "interactions": [
        ("tenure", "monthlycharges", "Customer lifetime value"),
        ("contract", "tenure", "Lock-in effect strength"),
    ],
}


class ExpertFERulesMixin:
    """
    Mixin that adds 16 expert-level feature engineering rules.

    Attach to the RuleEngine class via multiple inheritance.
    The rule engine calls _rules_feature_engineering_expert() via hasattr.
    """

    def _rules_feature_engineering_expert(self, ctx: Dict, out: List):
        """
        Expert FE rules (FE-010 to FE-025).

        Fires on: feature_engineering, eda, ai_insights screens.
        Requires: screen_context with pipeline execution results.
        """
        screen_ctx = ctx.get("screen_context", {}) or {}
        features = ctx.get("feature_stats", {}) or {}
        profile = ctx.get("dataset_profile", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        correlations = ctx.get("correlations", {}) or features.get("correlations", {})

        # Pipeline execution data (from context_compiler enrichment)
        original_columns = screen_ctx.get("original_columns", [])
        selected_features = screen_ctx.get("selected_features", [])
        categorical_features = screen_ctx.get("categorical_features", [])
        numeric_features = screen_ctx.get("numeric_features", [])
        variance_removed = screen_ctx.get("variance_removed", [])
        id_detected = screen_ctx.get("id_columns_detected", [])
        encoding_details = screen_ctx.get("encoding_details", {})
        original_shape = screen_ctx.get("original_shape")
        final_shape = screen_ctx.get("final_shape") or screen_ctx.get("train_shape")
        scaling_method = screen_ctx.get("scaling_method", frontend.get("scaling_method", "standard"))
        create_interactions = screen_ctx.get("create_interactions", frontend.get("create_interactions", False))
        create_polynomial = screen_ctx.get("create_polynomial_features", frontend.get("create_polynomial_features", False))

        n_rows = 0
        n_final_features = 0
        if isinstance(final_shape, (list, tuple)) and len(final_shape) >= 2:
            n_rows = final_shape[0]
            n_final_features = final_shape[1]

        # ── FE-010: Numeric column misclassified as categorical ──
        self._fe010_type_misclassification(
            categorical_features, encoding_details, features, out
        )

        # ── FE-011: ID column false negative ──
        self._fe011_id_false_negative(
            original_columns, id_detected, encoding_details, n_rows, out
        )

        # ── FE-012: Variance filter removing binary predictors ──
        self._fe012_variance_kills_binary(
            variance_removed, original_columns, out
        )

        # ── FE-013: Feature selection too aggressive ──
        self._fe013_aggressive_selection(
            original_columns, selected_features, final_shape, out
        )

        # ── FE-014: Scaling vs distribution mismatch ──
        self._fe014_scaling_mismatch(
            scaling_method, features, profile, out
        )

        # ── FE-015: One-hot encoding explosion ──
        self._fe015_encoding_explosion(
            encoding_details, original_columns, out
        )

        # ── FE-016: Rare category complete collapse ──
        self._fe016_rare_collapse(encoding_details, out)

        # ── FE-017: Dimensionality warning ──
        self._fe017_dimensionality(n_rows, n_final_features, out)

        # ── FE-018: Binary scaled then variance-killed ──
        self._fe018_binary_scale_kill(variance_removed, scaling_method, out)

        # ── FE-019: Missing domain interactions ──
        self._fe019_missing_interactions(
            original_columns, create_interactions, create_polynomial, out
        )

        # ── FE-020: Post-encoding redundancy ──
        self._fe020_encoding_redundancy(encoding_details, correlations, out)

        # ── FE-021: Ordinal treated as nominal ──
        self._fe021_ordinal_as_nominal(categorical_features, encoding_details, out)

        # ── FE-022: Information retention ──
        self._fe022_information_retention(
            original_columns, selected_features, variance_removed,
            encoding_details, out
        )

        # ── FE-023: Type detection vs EDA inconsistency ──
        self._fe023_type_eda_inconsistency(
            categorical_features, numeric_features, features, out
        )

        # ── FE-024: Scaling for tree-only unnecessary ──
        self._fe024_scaling_for_trees(
            scaling_method, frontend, ctx, out
        )

        # ── FE-025: Pipeline completeness check ──
        self._fe025_pipeline_completeness(screen_ctx, frontend, out)

    # ──────────────────────────────────────────────────────────
    # Individual Rule Implementations
    # ──────────────────────────────────────────────────────────

    def _fe010_type_misclassification(self, cat_features, encoding_details, features, out):
        """FE-010: Numeric columns misclassified as categorical."""
        numeric_stats = {}
        stats = features.get("statistics", {}) if isinstance(features, dict) else {}
        if isinstance(stats, dict):
            numeric_stats = stats.get("numeric", {})

        for col in cat_features:
            col_lower = col.lower().strip()
            is_numeric_name = any(p in col_lower for p in NUMERIC_NAME_PATTERNS)

            enc = encoding_details.get(col, {})
            unique = 0
            rare_grouped = 0
            if isinstance(enc, dict):
                unique = _safe_int(enc.get("unique_values") or enc.get("unique", 0))
                rare_grouped = _safe_int(enc.get("rare_grouped", 0))

            if is_numeric_name and unique > 50:
                out.append(Insight(
                    severity="critical",
                    category="Feature Engineering",
                    title=f"⚠️ '{col}' is NUMERIC — Misclassified as Categorical ({unique} unique values)",
                    message=(
                        f"Column '{col}' has {unique} unique values and a name suggesting "
                        f"numeric data (charges/amounts/totals). It was treated as categorical, "
                        f"with {rare_grouped} categories grouped into 'Other' — destroying all "
                        f"ordinal/magnitude information. This column is almost certainly stored "
                        f"as string (likely contains spaces, currency symbols, or commas)."
                    ),
                    action=(
                        f"Fix in data cleaning pipeline (Phase 1c):\n"
                        f"  df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')\n"
                        f"  df['{col}'].fillna(df['{col}'].median(), inplace=True)\n"
                        f"This converts to float, handles non-numeric characters gracefully, "
                        f"and imputes NaN with median. Re-run FE pipeline after fixing."
                    ),
                    evidence=(
                        f"Column '{col}': {unique} unique values, "
                        f"name pattern matches numeric field, "
                        f"{rare_grouped} categories collapsed to 'Other'"
                    ),
                    impact="high",
                    tags=["type_misclassification", "critical_fix", "data_loss"],
                    rule_id="FE-010",
                    confidence=0.95,
                ))

    def _fe011_id_false_negative(self, original_cols, id_detected, encoding_details, n_rows, out):
        """FE-011: ID column not detected."""
        for col in original_cols:
            col_clean = col.lower().strip().replace("_", "").replace("-", "")

            is_id = (
                    col_clean in [p.replace("_", "") for p in ID_NAME_PATTERNS]
                    or (col_clean.endswith("id") and len(col_clean) <= 15)
            )

            if is_id and col not in id_detected:
                enc = encoding_details.get(col, {})
                unique = _safe_int(enc.get("unique_values") or enc.get("unique", 0)) if isinstance(enc, dict) else 0
                cardinality_note = ""
                if unique > 0 and n_rows > 0:
                    card_pct = unique / n_rows * 100
                    cardinality_note = f" Cardinality: {unique}/{n_rows} ({card_pct:.0f}%)."

                out.append(Insight(
                    severity="critical",
                    category="Feature Engineering",
                    title=f"'{col}' is an ID Column — Not Detected",
                    message=(
                        f"Column '{col}' has an ID-pattern name but was not identified as an "
                        f"ID column by the pipeline's ID detection module.{cardinality_note} "
                        f"ID columns have zero predictive power and should be excluded from "
                        f"feature engineering. Instead, '{col}' went through categorical "
                        f"encoding, wasting processing and potentially adding noise."
                    ),
                    action=(
                        f"Fix the ID detection logic to catch '{col}'. Quick fix: add to "
                        f"explicit exclude list in the FE pipeline configuration. The ID "
                        f"detector should match columns ending in 'id'/'ID' with >80% cardinality."
                    ),
                    evidence=f"Column name '{col}' matches ID pattern, not in detected IDs: {id_detected}",
                    impact="medium",
                    tags=["id_detection", "noise_feature"],
                    rule_id="FE-011",
                    confidence=0.9,
                ))

    def _fe012_variance_kills_binary(self, variance_removed, original_cols, out):
        """FE-012: Variance filter removing known binary predictors."""
        if not variance_removed:
            return

        # Check for known binary predictors in removed list
        removed_known = []
        for r in variance_removed:
            base = r.lower().replace("_scaled", "").replace("_encoded", "").strip()
            if base in KNOWN_CHURN_PREDICTORS.get("binary", []):
                removed_known.append((r, base))

        if removed_known:
            names = [f"'{r}' (was '{b}')" for r, b in removed_known]
            out.append(Insight(
                severity="critical",
                category="Feature Engineering",
                title=f"Variance Filter Removed {len(removed_known)} Known Predictors",
                message=(
                    f"Binary features {', '.join(names)} were removed by the variance filter. "
                    f"These features are well-established predictors in churn/classification "
                    f"models. The issue: binary features (0/1) have inherently low variance "
                    f"(max 0.25 for balanced, much lower for imbalanced). After StandardScaling, "
                    f"variance may appear even lower, triggering removal. "
                    f"Variance ≠ predictive power — a feature with 0.01 variance can still "
                    f"have high mutual information with the target."
                ),
                action=(
                    "Fix options (pick one):\n"
                    "  1. Lower variance threshold: variance_threshold: 0.001\n"
                    "  2. Skip binary features in scaling: protect_binary: true\n"
                    "  3. Switch to mutual information filtering (target-aware):\n"
                    "     from sklearn.feature_selection import mutual_info_classif\n"
                    "     mi = mutual_info_classif(X, y)\n"
                    "     # Only remove features with MI ≈ 0\n"
                    "  4. Whitelist known predictors from variance filtering"
                ),
                evidence=f"Removed by variance filter: {', '.join(variance_removed)}",
                impact="high",
                tags=["variance_filter", "binary_predictors", "feature_recovery"],
                rule_id="FE-012",
                confidence=0.92,
            ))
        elif len(variance_removed) > 3:
            removal_pct = len(variance_removed) / max(len(original_cols), 1) * 100
            if removal_pct > 15:
                out.append(Insight(
                    severity="warning",
                    category="Feature Engineering",
                    title=f"Variance Filter Removed {len(variance_removed)} Features ({removal_pct:.0f}%)",
                    message=(
                        f"The variance filter removed {len(variance_removed)} features: "
                        f"{', '.join(variance_removed[:4])}{'...' if len(variance_removed) > 4 else ''}. "
                        f"This is {removal_pct:.0f}% of the feature space. Some of these "
                        f"may be predictive despite low variance."
                    ),
                    action=(
                        "Review removed features — if any are domain-relevant, lower the "
                        "variance threshold or switch to target-aware feature selection."
                    ),
                    evidence=f"Removed: {', '.join(variance_removed[:5])}",
                    impact="medium",
                    tags=["variance_filter", "feature_recovery"],
                    rule_id="FE-012",
                    confidence=0.75,
                ))

    def _fe013_aggressive_selection(self, original_cols, selected_features, final_shape, out):
        """FE-013: Feature selection too aggressive."""
        n_original = len(original_cols)
        n_selected = len(selected_features)

        # Try final_shape if selected_features is empty
        if n_selected == 0 and isinstance(final_shape, (list, tuple)) and len(final_shape) >= 2:
            n_selected = final_shape[1]

        if n_original == 0 or n_selected == 0:
            return

        retention = n_selected / n_original * 100

        if retention < 40:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"Aggressive Selection: {n_original} → {n_selected} features ({retention:.0f}% kept)",
                message=(
                    f"Feature selection retained only {retention:.0f}% of features. While "
                    f"dimensionality reduction prevents overfitting, dropping >{60}% risks "
                    f"discarding secondary predictors and interaction effects that contribute "
                    f"to ensemble diversity and edge-case coverage."
                ),
                action=(
                    f"Consider increasing selected features to {min(n_original, max(n_selected + 5, int(n_original * 0.6)))}. "
                    f"Let the model handle feature importance via built-in mechanisms "
                    f"(L1 regularization, tree splitting, dropout). Or use recursive feature "
                    f"elimination (RFE) with cross-validation for data-driven selection count."
                ),
                evidence=f"Original: {n_original} columns, Selected: {n_selected} features",
                impact="medium",
                tags=["feature_selection", "aggressiveness"],
                rule_id="FE-013",
                confidence=0.8,
            ))

    def _fe014_scaling_mismatch(self, scaling_method, features, profile, out):
        """FE-014: Scaling method vs data distribution mismatch."""
        quality = features.get("data_quality", {}) if isinstance(features, dict) else {}
        outlier_pct = _safe_float(quality.get("outlier_percentage", 0))
        skewed = features.get("skewed_features", []) if isinstance(features, dict) else []

        if scaling_method == "standard" and outlier_pct > 5:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"StandardScaler on Data With {outlier_pct:.1f}% Outliers",
                message=(
                    f"StandardScaler uses mean & standard deviation, both highly sensitive to "
                    f"outliers. With {outlier_pct:.1f}% outliers, the mean is pulled toward "
                    f"extreme values and std is inflated, compressing the majority of data "
                    f"into a narrow range and reducing discriminative power."
                ),
                action=(
                    "Switch to RobustScaler (uses median & IQR):\n"
                    "  scaling_method: 'robust'\n"
                    "RobustScaler has a breakdown point of 25% vs 0% for StandardScaler."
                ),
                evidence=f"Outlier percentage: {outlier_pct:.1f}%, Scaling: {scaling_method}",
                impact="medium",
                tags=["scaling", "outliers", "distribution"],
                rule_id="FE-014",
                confidence=0.85,
            ))

    def _fe015_encoding_explosion(self, encoding_details, original_cols, out):
        """FE-015: One-hot encoding creating too many features."""
        total_encoded = 0
        worst_col = None
        worst_count = 0

        for col, details in encoding_details.items():
            if not isinstance(details, dict):
                continue
            unique = _safe_int(details.get("unique_values") or details.get("unique", 0))
            strategy = details.get("strategy", "")
            if "one_hot" in strategy.lower():
                total_encoded += unique
                if unique > worst_count:
                    worst_count = unique
                    worst_col = col

        n_original = len(original_cols) or 1
        if total_encoded > n_original * 3:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"Encoding Explosion: {total_encoded} Features From Categorical Encoding",
                message=(
                    f"One-hot encoding created {total_encoded} features from categorical "
                    f"columns — {total_encoded / n_original:.1f}x expansion from {n_original} "
                    f"original columns. "
                    f"{'Worst offender: ' + worst_col + ' (' + str(worst_count) + ' categories).' if worst_col else ''}"
                ),
                action=(
                    "For high-cardinality categoricals (>10 unique values), use:\n"
                    "  • Target encoding (1 column per feature, target-aware)\n"
                    "  • Hash encoding (fixed-size, memory-efficient)\n"
                    "  • Leave-one-out encoding (prevents leakage)\n"
                    "Only use one-hot for ≤10 categories."
                ),
                evidence=f"Total encoded features: {total_encoded}, Original columns: {n_original}",
                impact="medium",
                tags=["encoding", "dimensionality"],
                rule_id="FE-015",
                confidence=0.8,
            ))

    def _fe016_rare_collapse(self, encoding_details, out):
        """FE-016: >95% of categories collapsed to 'Other'."""
        for col, details in encoding_details.items():
            if not isinstance(details, dict):
                continue
            unique = _safe_int(details.get("unique_values") or details.get("unique", 0))
            rare = _safe_int(details.get("rare_grouped", 0))

            if unique > 10 and rare > 0 and rare / max(unique, 1) > 0.95:
                collapse_pct = rare / unique * 100
                out.append(Insight(
                    severity="critical",
                    category="Feature Engineering",
                    title=f"'{col}' — {collapse_pct:.0f}% Categories Collapsed to 'Other'",
                    message=(
                        f"Column '{col}' had {unique} categories; {rare} ({collapse_pct:.0f}%) "
                        f"were below the rare threshold and merged into 'Other'. The resulting "
                        f"feature is near-constant (almost every row = 'Other'), carrying "
                        f"approximately zero predictive information."
                    ),
                    action=(
                        f"This column is likely: (a) a misclassified numeric column — convert "
                        f"with pd.to_numeric(), (b) an ID column — exclude entirely, or "
                        f"(c) genuinely high-cardinality — use target encoding instead."
                    ),
                    evidence=f"'{col}': {rare}/{unique} categories rare ({collapse_pct:.0f}%)",
                    impact="high",
                    tags=["encoding", "information_loss", "rare_categories"],
                    rule_id="FE-016",
                    confidence=0.9,
                ))

    def _fe017_dimensionality(self, n_rows, n_features, out):
        """FE-017: Feature dimensionality vs sample size."""
        if n_rows == 0 or n_features == 0:
            return

        ratio = n_rows / n_features

        if ratio < 5:
            out.append(Insight(
                severity="critical",
                category="Feature Engineering",
                title=f"Dangerous Dimensionality: {ratio:.0f} Samples per Feature",
                message=(
                    f"With {n_rows} samples and {n_features} features, the ratio is only "
                    f"{ratio:.0f}:1. Statistical learning theory recommends ≥20:1 for reliable "
                    f"generalization. Below 5:1, models almost certainly overfit — memorizing "
                    f"training noise rather than learning real patterns."
                ),
                action=(
                    f"Reduce features to {max(5, n_rows // 20)} (for 20:1 ratio) or collect "
                    f"more data. Use strong regularization: L1 penalty >0.1, tree max_depth ≤5, "
                    f"dropout >0.3."
                ),
                evidence=f"Shape: ({n_rows}, {n_features}), Ratio: {ratio:.1f}:1",
                impact="high",
                tags=["dimensionality", "overfitting"],
                rule_id="FE-017",
                confidence=0.95,
            ))
        elif ratio < 10:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"Low Sample-to-Feature Ratio: {ratio:.0f}:1",
                message=(
                    f"Ratio of {ratio:.0f}:1 is below the recommended 20:1 minimum. "
                    f"Cross-validation scores may be unstable."
                ),
                action=f"Consider reducing to {max(5, n_rows // 20)} features.",
                evidence=f"Shape: ({n_rows}, {n_features})",
                impact="medium",
                tags=["dimensionality"],
                rule_id="FE-017",
                confidence=0.85,
            ))

    def _fe018_binary_scale_kill(self, variance_removed, scaling_method, out):
        """FE-018: Binary features scaled then killed by variance."""
        if not variance_removed or scaling_method not in ("standard", "minmax"):
            return

        scaled_then_killed = [
            v for v in variance_removed
            if "_scaled" in v.lower()
        ]

        if scaled_then_killed:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"{len(scaled_then_killed)} Features: Scaled → Low Variance → Removed (Cascade Error)",
                message=(
                    f"Features {', '.join(scaled_then_killed[:4])} were first scaled "
                    f"('{scaling_method}') then removed by the variance filter. This is a "
                    f"cascading error: scaling binary features can artificially reduce their "
                    f"apparent variance below the threshold, causing premature removal."
                ),
                action=(
                    "Either: (1) don't apply scaling to binary features, (2) lower the "
                    "variance threshold to 0.001, or (3) apply variance filter BEFORE scaling."
                ),
                evidence=f"Scaled then removed: {', '.join(scaled_then_killed[:5])}",
                impact="medium",
                tags=["cascade_error", "scaling", "variance"],
                rule_id="FE-018",
                confidence=0.85,
            ))

    def _fe019_missing_interactions(self, original_cols, create_interactions, create_polynomial, out):
        """FE-019: Missing domain-standard interaction features."""
        if create_interactions or create_polynomial:
            return

        cols_lower = [c.lower().replace("_", "") for c in original_cols]
        applicable = []

        for feat_a, feat_b, reason in KNOWN_CHURN_PREDICTORS.get("interactions", []):
            a_found = any(feat_a.lower().replace("_", "") in c for c in cols_lower)
            b_found = any(feat_b.lower().replace("_", "") in c for c in cols_lower)
            if a_found and b_found:
                applicable.append(f"{feat_a} × {feat_b} ({reason})")

        if applicable:
            out.append(Insight(
                severity="tip",
                category="Feature Engineering",
                title=f"{len(applicable)} Standard Interaction Features Not Created",
                message=(
                    f"Interaction features disabled but {len(applicable)} domain-standard "
                    f"interactions are available: {'; '.join(applicable)}. These capture "
                    f"non-linear relationships that individual features miss."
                ),
                action=(
                    "Enable interaction features:\n"
                    "  create_interactions: true\n"
                    "Expected improvement: 1-4% accuracy for classification models."
                ),
                evidence=f"Available interactions: {', '.join(applicable)}",
                impact="medium",
                tags=["interactions", "domain_knowledge"],
                rule_id="FE-019",
                confidence=0.75,
            ))

    def _fe020_encoding_redundancy(self, encoding_details, correlations, out):
        """FE-020: Post-encoding feature redundancy."""
        # Check for one-hot columns from the same category that are perfectly
        # anti-correlated (e.g., InternetService_Yes and InternetService_No)
        onehot_groups = {}
        for col, details in encoding_details.items():
            if isinstance(details, dict) and "one_hot" in str(details.get("strategy", "")).lower():
                unique = _safe_int(details.get("unique_values") or details.get("unique", 0))
                if unique == 2:
                    onehot_groups[col] = unique

        if len(onehot_groups) >= 3:
            out.append(Insight(
                severity="tip",
                category="Feature Engineering",
                title=f"{len(onehot_groups)} Binary Categoricals One-Hot Encoded (Redundant)",
                message=(
                    f"Columns {', '.join(list(onehot_groups.keys())[:4])} have only 2 "
                    f"categories each but were one-hot encoded (creating 2 columns each). "
                    f"For binary categoricals, label encoding (0/1) is sufficient and avoids "
                    f"the redundant anti-correlated column."
                ),
                action=(
                    "Use label encoding for binary categoricals:\n"
                    "  for col in binary_categoricals:\n"
                    "      df[col] = (df[col] == 'Yes').astype(int)\n"
                    f"This saves {len(onehot_groups)} features."
                ),
                evidence=f"Binary categoricals: {', '.join(list(onehot_groups.keys())[:5])}",
                impact="low",
                tags=["encoding", "redundancy"],
                rule_id="FE-020",
                confidence=0.8,
            ))

    def _fe021_ordinal_as_nominal(self, cat_features, encoding_details, out):
        """FE-021: Ordinal variable treated as nominal."""
        ordinal_keywords = [
            "level", "grade", "tier", "stage", "rank", "priority",
            "severity", "rating", "quality", "experience", "education",
            "contract",  # Common ordinal: month-to-month < one-year < two-year
        ]

        for col in cat_features:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ordinal_keywords):
                enc = encoding_details.get(col, {})
                strategy = enc.get("strategy", "") if isinstance(enc, dict) else ""
                if "one_hot" in strategy.lower():
                    out.append(Insight(
                        severity="tip",
                        category="Feature Engineering",
                        title=f"'{col}' May Be Ordinal — One-Hot Loses Order",
                        message=(
                            f"Column '{col}' has a name suggesting ordinal data (levels, "
                            f"grades, contracts, tiers). One-hot encoding treats categories "
                            f"as unrelated, losing the natural ordering (e.g., month-to-month "
                            f"< one-year < two-year)."
                        ),
                        action=(
                            f"If '{col}' is ordinal, use ordinal encoding:\n"
                            f"  mapping = {{'Month-to-month': 0, 'One year': 1, 'Two year': 2}}\n"
                            f"  df['{col}'] = df['{col}'].map(mapping)"
                        ),
                        evidence=f"Column '{col}' name matches ordinal pattern",
                        impact="low",
                        tags=["encoding", "ordinal"],
                        rule_id="FE-021",
                        confidence=0.65,
                    ))

    def _fe022_information_retention(self, original_cols, selected_features,
                                     variance_removed, encoding_details, out):
        """FE-022: Pipeline information retention check."""
        n_original = len(original_cols)
        n_selected = len(selected_features)
        n_var_removed = len(variance_removed)

        if n_original == 0:
            return

        # Count columns that were completely destroyed
        destroyed = 0
        for col, details in encoding_details.items():
            if isinstance(details, dict):
                unique = _safe_int(details.get("unique_values") or details.get("unique", 0))
                rare = _safe_int(details.get("rare_grouped", 0))
                if unique > 0 and rare / max(unique, 1) > 0.95:
                    destroyed += 1

        effective = max(0, n_selected - destroyed) if n_selected else n_original - n_var_removed - destroyed
        retention = effective / n_original * 100

        if retention < 40:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"Low Information Retention: {retention:.0f}% of Original Columns Used",
                message=(
                    f"Starting with {n_original} columns: {n_var_removed} removed by variance "
                    f"filter, {destroyed} destroyed by encoding collapse, "
                    f"{max(0, n_original - (n_selected or n_original))} dropped by selection. "
                    f"Effective retention: {retention:.0f}%."
                ),
                action=(
                    "Review each loss source. Fix type misclassification first (highest impact), "
                    "then adjust variance threshold, then reconsider selection count."
                ),
                evidence=f"Original: {n_original}, Effective: {effective}, Retention: {retention:.0f}%",
                impact="high",
                tags=["information_loss", "pipeline_quality"],
                rule_id="FE-022",
                confidence=0.85,
            ))

    def _fe023_type_eda_inconsistency(self, cat_features, numeric_features, features, out):
        """FE-023: FE type detection disagrees with EDA."""
        stats = features.get("statistics", {}) if isinstance(features, dict) else {}
        eda_numeric = set(stats.get("numeric", {}).keys()) if isinstance(stats, dict) else set()
        eda_categorical = set(stats.get("categorical", {}).keys()) if isinstance(stats, dict) else set()

        # Columns EDA says numeric but FE says categorical
        mismatches = []
        for col in cat_features:
            if col in eda_numeric:
                mismatches.append(col)

        if mismatches:
            out.append(Insight(
                severity="warning",
                category="Feature Engineering",
                title=f"Type Conflict: {len(mismatches)} Columns — EDA=Numeric, FE=Categorical",
                message=(
                    f"Columns {', '.join(mismatches[:3])} were identified as numeric by EDA "
                    f"but classified as categorical by the FE pipeline. This inconsistency "
                    f"usually means the data was modified between EDA and FE (e.g., string "
                    f"conversion, missing value markers)."
                ),
                action=(
                    "Check the data pipeline between EDA and FE. Ensure numeric columns "
                    "retain their numeric dtype through all preprocessing steps."
                ),
                evidence=f"Mismatched columns: {', '.join(mismatches[:5])}",
                impact="high",
                tags=["type_inconsistency", "data_pipeline"],
                rule_id="FE-023",
                confidence=0.8,
            ))

    def _fe024_scaling_for_trees(self, scaling_method, frontend, ctx, out):
        """FE-024: Scaling is unnecessary for tree-only workflows."""
        selected_algos = frontend.get("selected_algorithms", [])
        if not selected_algos:
            return

        tree_algos = {"random_forest", "xgboost", "lightgbm", "catboost",
                      "gradient_boosting", "decision_tree", "extra_trees"}

        algos_lower = {a.lower().replace(" ", "_") for a in selected_algos}
        all_trees = algos_lower.issubset(tree_algos)

        if all_trees and scaling_method in ("standard", "robust", "minmax"):
            out.append(Insight(
                severity="tip",
                category="Feature Engineering",
                title="Scaling Unnecessary — All Selected Algorithms Are Tree-Based",
                message=(
                    f"All selected algorithms ({', '.join(selected_algos)}) are tree-based. "
                    f"Trees split on feature values regardless of scale, so "
                    f"StandardScaler/RobustScaler/MinMaxScaler add computation without benefit."
                ),
                action=(
                    "Either: (a) skip scaling for tree-only workflows, or "
                    "(b) keep scaling if you plan to add distance-based models later "
                    "(SVM, KNN, Logistic Regression)."
                ),
                evidence=f"Algorithms: {', '.join(selected_algos)}, Scaling: {scaling_method}",
                impact="low",
                tags=["scaling", "optimization"],
                rule_id="FE-024",
                confidence=0.9,
            ))

    def _fe025_pipeline_completeness(self, screen_ctx, frontend, out):
        """FE-025: Pipeline completeness check."""
        missing_steps = []

        if not screen_ctx.get("handle_missing_values") and not frontend.get("handle_missing_values"):
            missing_steps.append("Missing value handling")
        if not screen_ctx.get("handle_outliers") and not frontend.get("handle_outliers"):
            missing_steps.append("Outlier handling")
        if not screen_ctx.get("encode_categories") and not frontend.get("encode_categories"):
            missing_steps.append("Categorical encoding")

        if missing_steps:
            out.append(Insight(
                severity="info",
                category="Feature Engineering",
                title=f"Pipeline Missing {len(missing_steps)} Recommended Step{'s' if len(missing_steps) > 1 else ''}",
                message=(
                    f"The following FE steps are disabled: {', '.join(missing_steps)}. "
                    f"These are typically recommended for production-grade pipelines."
                ),
                action=f"Consider enabling: {', '.join(missing_steps)}",
                evidence=f"Disabled steps: {', '.join(missing_steps)}",
                impact="low",
                tags=["completeness", "best_practice"],
                rule_id="FE-025",
                confidence=0.7,
            ))


def _safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default