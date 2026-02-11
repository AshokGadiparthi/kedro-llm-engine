"""
Recommendation Engine — Actionable ML Strategy Generator
==========================================================
Generates specific, quantified recommendations based on data characteristics.
Every recommendation includes WHY, WHAT, and exact parameters to use.

Capabilities:
  1. Feature Selection Strategy  — Which features to keep/drop and why
  2. Encoding Strategy           — How to encode each categorical feature
  3. Scaling Strategy            — Which scaler for which features
  4. Imputation Strategy         — How to handle missing data per column
  5. Algorithm Recommendation    — Ranked algorithms with configurations
  6. Hyperparameter Search Space — Exact search ranges for selected algorithm
  7. Threshold Optimization      — Business-optimal classification threshold
  8. Ensemble Strategy           — When and how to combine models
  9. Cross-Validation Strategy   — CV configuration for the data shape
  10. Retraining Strategy        — When and how to retrain in production
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generates precise, data-driven recommendations for every ML decision.
    All methods work from compiled context — no database access.
    """

    # ──────────────────────────────────────────────────────────
    # 1. FEATURE SELECTION STRATEGY
    # ──────────────────────────────────────────────────────────

    def recommend_feature_selection(self, context: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Recommend which features to keep, drop, or transform.
        Uses correlation clusters, cardinality, target scores, and data shape.
        """
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)
        feature_stats = context.get("feature_stats", {})

        # From statistical analysis
        cardinality = analysis.get("cardinality", [])
        feature_target = analysis.get("feature_target_scores", [])
        type_mismatches = analysis.get("type_mismatches", [])
        data_shape = analysis.get("data_shape", {})
        clusters = analysis.get("correlation_clusters", [])

        max_features = data_shape.get("max_safe_features", cols)

        # Build recommendations
        to_drop = []
        to_keep = []
        to_transform = []
        to_investigate = []

        # Drop: ID columns
        id_cols = feature_stats.get("potential_id_columns", [])
        for col_info in id_cols:
            to_drop.append({
                "column": col_info["column"],
                "reason": "Identified as ID/index column — provides no predictive value",
                "confidence": 0.85,
            })

        # Drop: constant features
        for card in cardinality:
            if card.get("is_constant"):
                to_drop.append({
                    "column": card["column"],
                    "reason": "Zero variance — only one unique value",
                    "confidence": 1.0,
                })

        # Drop: very high cardinality categoricals (likely IDs)
        for card in cardinality:
            if card.get("cardinality_class") == "likely_id":
                to_drop.append({
                    "column": card["column"],
                    "reason": f"Likely ID column ({card.get('unique_count', 0)} unique values, "
                              f"{card.get('unique_ratio', 0)*100:.0f}% unique ratio)",
                    "confidence": 0.75,
                })

        # Drop: redundant features from correlation clusters
        dropped_names = set(d["column"] for d in to_drop)
        for cluster in clusters:
            features = cluster.get("features", [])
            # Keep the feature with highest target correlation, drop rest
            best_feat = None
            best_corr = -1
            for feat in features:
                for ft in feature_target:
                    if ft.get("column") == feat:
                        if ft.get("correlation_abs", 0) > best_corr:
                            best_corr = ft.get("correlation_abs", 0)
                            best_feat = feat
                        break

            for feat in features:
                if feat in dropped_names:
                    continue
                if feat == best_feat:
                    to_keep.append({
                        "column": feat,
                        "reason": f"Best in correlation cluster (r={best_corr:.3f} with target)",
                        "from_cluster": cluster.get("cluster_id"),
                    })
                else:
                    to_drop.append({
                        "column": feat,
                        "reason": (
                            f"Redundant with '{best_feat}' (cluster avg r={cluster.get('avg_internal_correlation', 0):.2f}). "
                            f"Keeping '{best_feat}' which has stronger target correlation."
                        ),
                        "confidence": 0.7,
                    })

        # Transform: type mismatches
        for mismatch in type_mismatches:
            to_transform.append({
                "column": mismatch["column"],
                "action": f"Convert from {mismatch['current_type']} to {mismatch['suggested_type']}",
                "reason": mismatch["reason"],
            })

        # Investigate: noise features (very low target correlation)
        for ft in feature_target:
            if ft.get("predictive_class") == "noise" and ft["column"] not in dropped_names:
                to_investigate.append({
                    "column": ft["column"],
                    "reason": f"Very low target correlation (r={ft.get('correlation_abs', 0):.4f})",
                    "suggestion": "May be noise — consider dropping if feature engineering doesn't help",
                })

        # Investigate: leakage suspects
        for ft in feature_target:
            if ft.get("predictive_class") == "leakage_suspect":
                to_investigate.append({
                    "column": ft["column"],
                    "reason": f"Suspiciously high target correlation (r={ft.get('correlation_abs', 0):.3f})",
                    "suggestion": "Verify this isn't derived from the target or collected post-outcome",
                })

        # Compute final count
        remaining = cols - len(to_drop)
        needs_reduction = remaining > max_features

        return {
            "total_features": cols,
            "recommended_max": max_features,
            "to_drop": to_drop,
            "to_keep": to_keep,
            "to_transform": to_transform,
            "to_investigate": to_investigate,
            "needs_further_reduction": needs_reduction,
            "features_after_dropping": remaining,
            "reduction_method": (
                "Use mutual information scoring or L1 regularization to further reduce"
                if needs_reduction else "Current feature count is acceptable after dropping recommended columns"
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 2. ENCODING STRATEGY
    # ──────────────────────────────────────────────────────────

    def recommend_encoding(self, context: Dict, analysis: Dict) -> List[Dict[str, Any]]:
        """
        Recommend encoding strategy for each categorical feature.
        Considers cardinality, algorithm compatibility, and data characteristics.
        """
        recommendations = []
        cardinality = analysis.get("cardinality", [])

        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        algorithm = screen_ctx.get("algorithm") or frontend.get("algorithm", "")
        algo_lower = algorithm.lower() if algorithm else ""

        is_tree_based = any(x in algo_lower for x in [
            "tree", "forest", "xgboost", "xgb", "lightgbm", "lgbm", "catboost", "gbm"
        ])

        for card in cardinality:
            col = card.get("column", "")
            unique = card.get("unique_count", 0)
            card_class = card.get("cardinality_class", "low")

            rec = {"column": col, "unique_values": unique}

            if card_class == "constant":
                rec["encoding"] = "drop"
                rec["reason"] = "Only 1 unique value — no information"
                rec["code_snippet"] = f"df = df.drop(columns=['{col}'])"

            elif card_class == "binary":
                rec["encoding"] = "label_encode"
                rec["reason"] = "Binary feature — simple 0/1 encoding"
                rec["code_snippet"] = f"df['{col}'] = df['{col}'].map({{val1: 0, val2: 1}})"

            elif card_class == "likely_id":
                rec["encoding"] = "drop"
                rec["reason"] = "Likely ID column — drop before training"
                rec["code_snippet"] = f"df = df.drop(columns=['{col}'])"

            elif card_class == "low":
                if is_tree_based:
                    rec["encoding"] = "ordinal_encode"
                    rec["reason"] = f"Low cardinality ({unique} values) + tree-based model → ordinal is sufficient"
                    rec["code_snippet"] = f"from sklearn.preprocessing import OrdinalEncoder\noe = OrdinalEncoder()\ndf['{col}'] = oe.fit_transform(df[['{col}']])"
                else:
                    rec["encoding"] = "one_hot_encode"
                    rec["reason"] = f"Low cardinality ({unique} values) + linear model → one-hot for proper representation"
                    rec["code_snippet"] = f"df = pd.get_dummies(df, columns=['{col}'], drop_first=True)"

            elif card_class == "medium":
                if is_tree_based:
                    rec["encoding"] = "ordinal_encode"
                    rec["reason"] = f"Medium cardinality ({unique}) + tree model → ordinal"
                else:
                    rec["encoding"] = "target_encode"
                    rec["reason"] = (
                        f"Medium cardinality ({unique}) + linear model → target encoding avoids "
                        f"dimensionality explosion from one-hot ({unique} new columns)"
                    )
                    rec["code_snippet"] = (
                        f"from sklearn.preprocessing import TargetEncoder\n"
                        f"te = TargetEncoder(smooth='auto')\n"
                        f"df['{col}'] = te.fit_transform(df[['{col}']], y)"
                    )

            elif card_class in ("high", "very_high"):
                rec["encoding"] = "target_encode"
                rec["reason"] = (
                    f"High cardinality ({unique}) — one-hot would create {unique} columns. "
                    f"Target encoding compresses to 1 column while preserving predictive signal."
                )
                rec["code_snippet"] = (
                    f"from sklearn.preprocessing import TargetEncoder\n"
                    f"te = TargetEncoder(smooth='auto')\n"
                    f"df['{col}'] = te.fit_transform(df[['{col}']], y)"
                )
                rec["warning"] = "Use K-fold target encoding during CV to prevent leakage"

            recommendations.append(rec)

        return recommendations

    # ──────────────────────────────────────────────────────────
    # 3. SCALING STRATEGY
    # ──────────────────────────────────────────────────────────

    def recommend_scaling(self, context: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Recommend scaling strategy based on distributions and algorithm.
        """
        distributions = analysis.get("distributions", [])
        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        algorithm = screen_ctx.get("algorithm") or frontend.get("algorithm", "")
        algo_lower = algorithm.lower() if algorithm else ""

        is_tree_based = any(x in algo_lower for x in [
            "tree", "forest", "xgboost", "xgb", "lightgbm", "lgbm", "catboost", "gbm"
        ])

        if is_tree_based:
            return {
                "primary_scaler": "none",
                "reason": (
                    "Tree-based models (including gradient boosting) are invariant to "
                    "monotonic transformations and do not require feature scaling. "
                    "Scaling adds computational overhead without benefit."
                ),
                "exception": "Only needed if using distance-based pre-processing (e.g., KNN imputation)",
                "per_feature": [],
            }

        # Analyze distribution characteristics
        has_outliers = any(d.get("outlier_fraction", 0) > 0.05 for d in distributions)
        extreme_skew = [d for d in distributions if d.get("skew_class") in ("high_skew", "extreme_skew")]
        normal_ish = [d for d in distributions if d.get("normality_score", 0) > 0.7]

        per_feature = []
        primary_scaler = "standard"
        reason = ""

        if has_outliers and len(extreme_skew) > len(distributions) * 0.3:
            primary_scaler = "robust"
            reason = (
                "RobustScaler recommended: many features have outliers and skewed distributions. "
                "RobustScaler uses median and IQR instead of mean and std, making it robust to outliers."
            )
        elif len(extreme_skew) > len(distributions) * 0.5:
            primary_scaler = "power_transform"
            reason = (
                "PowerTransformer (Yeo-Johnson) recommended: majority of features are heavily skewed. "
                "This transform normalizes distributions before scaling."
            )
        elif len(normal_ish) > len(distributions) * 0.7:
            primary_scaler = "standard"
            reason = (
                "StandardScaler recommended: most features are approximately normally distributed. "
                "Standard scaling (zero mean, unit variance) is optimal for these distributions."
            )
        else:
            primary_scaler = "robust"
            reason = "RobustScaler recommended as a safe default for mixed distribution types."

        # Per-feature overrides
        for dist in distributions:
            col = dist.get("column", "")
            transform = dist.get("suggested_transform", "none")
            if transform and transform != "none":
                per_feature.append({
                    "column": col,
                    "pre_transform": transform,
                    "reason": f"Skewness: {dist.get('skewness', 0):.2f}, class: {dist.get('skew_class', '')}",
                })

        return {
            "primary_scaler": primary_scaler,
            "reason": reason,
            "per_feature_overrides": per_feature,
            "code_snippet": self._scaling_code(primary_scaler),
            "important_notes": [
                "Always fit scaler on TRAINING data only, then transform test data",
                "Store the fitted scaler for production inference",
                "Apply the same transforms at prediction time",
            ],
        }

    @staticmethod
    def _scaling_code(scaler: str) -> str:
        snippets = {
            "standard": (
                "from sklearn.preprocessing import StandardScaler\n"
                "scaler = StandardScaler()\n"
                "X_train_scaled = scaler.fit_transform(X_train)\n"
                "X_test_scaled = scaler.transform(X_test)  # Note: .transform() only"
            ),
            "robust": (
                "from sklearn.preprocessing import RobustScaler\n"
                "scaler = RobustScaler()\n"
                "X_train_scaled = scaler.fit_transform(X_train)\n"
                "X_test_scaled = scaler.transform(X_test)"
            ),
            "power_transform": (
                "from sklearn.preprocessing import PowerTransformer\n"
                "scaler = PowerTransformer(method='yeo-johnson', standardize=True)\n"
                "X_train_scaled = scaler.fit_transform(X_train)\n"
                "X_test_scaled = scaler.transform(X_test)"
            ),
            "none": "# No scaling needed for tree-based models",
        }
        return snippets.get(scaler, snippets["standard"])

    # ──────────────────────────────────────────────────────────
    # 4. IMPUTATION STRATEGY
    # ──────────────────────────────────────────────────────────

    def recommend_imputation(self, context: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Recommend imputation strategy per column based on missing mechanism,
        distribution shape, and feature type.
        """
        quality = context.get("data_quality", {})
        missing_by_col = quality.get("missing_by_column", {})
        missing_mechanism = analysis.get("missing_mechanism", {})
        distributions = {d["column"]: d for d in analysis.get("distributions", [])}
        profile = context.get("dataset_profile", {})
        col_types = profile.get("column_types", {})
        rows = profile.get("rows", 1)

        mechanism = missing_mechanism.get("mechanism", "unknown")
        per_column = []

        for col, info in missing_by_col.items():
            pct = info.get("percent", 0) if isinstance(info, dict) else 0
            count = info.get("count", 0) if isinstance(info, dict) else 0
            if pct == 0:
                continue

            is_numeric = col in col_types.get("numeric", [])
            is_categorical = col in col_types.get("categorical", [])
            dist = distributions.get(col, {})
            is_skewed = dist.get("skew_class", "") in ("high_skew", "extreme_skew")

            rec = {
                "column": col,
                "missing_pct": round(pct, 1),
                "missing_count": count,
            }

            # Drop threshold: >60% missing
            if pct > 60:
                rec["strategy"] = "drop_column"
                rec["reason"] = (
                    f"{pct:.0f}% missing — imputing would fabricate more than half the values. "
                    f"Unless this feature is critically important, dropping is safer."
                )
                rec["code"] = f"df = df.drop(columns=['{col}'])"

            # High missing: multivariate imputation
            elif pct > 20:
                rec["strategy"] = "iterative_imputer"
                rec["reason"] = (
                    f"{pct:.0f}% missing — too much for simple imputation. "
                    f"IterativeImputer models this column as a function of other columns."
                )
                rec["add_indicator"] = True
                rec["code"] = (
                    f"from sklearn.impute import IterativeImputer\n"
                    f"imputer = IterativeImputer(max_iter=10, random_state=42)\n"
                    f"df['{col}'] = imputer.fit_transform(df[['{col}', ...other_cols...]])"
                )

            # Moderate missing: depends on type and distribution
            elif pct > 5:
                if is_numeric:
                    if is_skewed:
                        rec["strategy"] = "median"
                        rec["reason"] = f"Skewed distribution — median preserves central tendency better than mean"
                    else:
                        rec["strategy"] = "median"
                        rec["reason"] = "Moderate missing — median imputation (robust to outliers)"
                    rec["add_indicator"] = True
                    rec["code"] = (
                        f"from sklearn.impute import SimpleImputer\n"
                        f"imp = SimpleImputer(strategy='median')\n"
                        f"df['{col}'] = imp.fit_transform(df[['{col}']])"
                    )
                else:
                    rec["strategy"] = "mode"
                    rec["reason"] = "Categorical column — fill with most frequent value"
                    rec["add_indicator"] = True
                    rec["code"] = f"df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)"

            # Low missing: simple imputation
            else:
                if is_numeric:
                    rec["strategy"] = "median"
                    rec["reason"] = f"Low missing ({pct:.1f}%) — simple median fill is sufficient"
                else:
                    rec["strategy"] = "mode"
                    rec["reason"] = f"Low missing ({pct:.1f}%) — fill with most frequent value"
                rec["add_indicator"] = False

            per_column.append(rec)

        # Sort by missing % descending
        per_column.sort(key=lambda x: -x["missing_pct"])

        add_indicators_for = [c["column"] for c in per_column if c.get("add_indicator")]

        return {
            "overall_mechanism": mechanism,
            "total_columns_affected": len(per_column),
            "columns_to_drop": [c["column"] for c in per_column if c.get("strategy") == "drop_column"],
            "per_column": per_column,
            "add_missing_indicators": add_indicators_for,
            "indicator_note": (
                "Binary missing indicators (1=was missing, 0=was present) can capture "
                "informative missingness patterns. Add these as new features."
            ) if add_indicators_for else None,
        }

    # ──────────────────────────────────────────────────────────
    # 5. ALGORITHM RECOMMENDATION (ADVANCED)
    # ──────────────────────────────────────────────────────────

    def recommend_algorithms(self, context: Dict, analysis: Dict) -> List[Dict[str, Any]]:
        """
        Data-driven algorithm recommendation with specific configs.
        Goes beyond simple rules — considers data shape, feature composition,
        target characteristics, and computational constraints.
        """
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)
        numeric = profile.get("numeric_count", 0)
        categorical = profile.get("categorical_count", 0)

        data_shape = analysis.get("data_shape", {})
        readiness = analysis.get("data_readiness", {})
        difficulty = readiness.get("estimated_difficulty", "medium")

        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type", "classification")

        recommendations = []

        # ── TIER 1: Always include baseline ──
        if problem_type == "classification":
            # Data-aware description: warn if categorical-heavy
            if categorical > 0 and numeric > 0 and categorical > numeric * 2:
                logreg_reason = (
                    f"Linear baseline. ⚠️ NOTE: Your dataset has {categorical} categorical vs "
                    f"{numeric} numeric features — LogReg requires ALL categoricals to be encoded "
                    f"first (target encoding recommended for high-cardinality). Consider CatBoost "
                    f"as your primary model since it handles categoricals natively."
                )
            elif categorical > 10:
                logreg_reason = (
                    f"Linear baseline. With {categorical} categorical features, use ordinal or "
                    f"target encoding — one-hot would create too many sparse columns."
                )
            else:
                logreg_reason = "Every comparison needs a strong linear baseline. Fast, interpretable, production-friendly."

            recommendations.append({
                "algorithm": "LogisticRegression",
                "tier": "baseline",
                "priority": 1,
                "reason": logreg_reason,
                "config": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs" if cols < 1000 else "saga",
                    "max_iter": 1000,
                    "class_weight": "balanced" if difficulty in ("hard", "very_hard") else None,
                },
                "expected_training_time": "seconds",
                "inference_latency": "<1ms",
            })
        else:
            if categorical > numeric * 2:
                ridge_reason = (
                    f"Regularized linear baseline. ⚠️ With {categorical} categorical features, "
                    f"encode them first (target encoding recommended). Consider gradient boosting "
                    f"as primary model for categorical-heavy data."
                )
            else:
                ridge_reason = "Regularized linear regression baseline. Stable, interpretable."

            recommendations.append({
                "algorithm": "Ridge",
                "tier": "baseline",
                "priority": 1,
                "reason": ridge_reason,
                "config": {"alpha": 1.0},
                "expected_training_time": "seconds",
                "inference_latency": "<1ms",
            })

        # ── TIER 2: Best for data size ──
        if rows < 500:
            recommendations.extend([
                {
                    "algorithm": "SVM",
                    "tier": "small_data",
                    "priority": 2,
                    "reason": f"Effective on small datasets ({rows} rows). Kernel trick captures non-linearity.",
                    "config": {
                        "kernel": "rbf",
                        "C": 1.0,
                        "gamma": "scale",
                        "probability": True,
                    },
                    "expected_training_time": "seconds",
                },
                {
                    "algorithm": "RandomForest",
                    "tier": "small_data",
                    "priority": 2,
                    "reason": "Robust ensemble for small data. Limit depth to prevent overfitting.",
                    "config": {
                        "n_estimators": 100,
                        "max_depth": 5,
                        "min_samples_leaf": max(5, rows // 50),
                        "class_weight": "balanced" if problem_type == "classification" else None,
                    },
                    "expected_training_time": "seconds",
                },
            ])

        elif rows < 5000:
            rf_reason = (
                    f"Solid performance with minimal tuning for {rows:,} rows."
                    + (f" Handles mixed types but needs ordinal encoding for {categorical} categoricals." if categorical > 5 else "")
                    + f" max_features='sqrt' → ~{max(1, int(cols**0.5))} features per split."
            )
            xgb_reason = (
                    f"State-of-the-art on tabular data."
                    + (f" Needs encoding for {categorical} categoricals — use ordinal." if categorical > 5 else "")
                    + f" Regularization important at {rows:,} rows to prevent overfitting."
            )
            recommendations.extend([
                {
                    "algorithm": "RandomForest",
                    "tier": "medium_data",
                    "priority": 2,
                    "reason": rf_reason,
                    "config": {
                        "n_estimators": 200,
                        "max_depth": None,
                        "min_samples_leaf": 5,
                        "max_features": "sqrt",
                    },
                    "expected_training_time": "10-30 seconds",
                },
                {
                    "algorithm": "XGBoost",
                    "tier": "medium_data",
                    "priority": 2,
                    "reason": xgb_reason,
                    "config": {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "min_child_weight": 5,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                    },
                    "expected_training_time": "10-60 seconds",
                },
            ])

        else:  # rows >= 5000
            lgbm_reason = (
                    f"Fastest boosting for {rows:,} rows."
                    + (f" Native categorical support — no encoding needed for {categorical} categorical features." if categorical > 3 else " Native categorical support.")
                    + " Leaf-wise growth is faster than level-wise."
            )
            xgb_large_reason = (
                    f"Strong accuracy on {rows:,} rows, excellent SHAP support for interpretability."
                    + (f" Needs ordinal encoding for {categorical} categoricals." if categorical > 5 else "")
            )
            recommendations.extend([
                {
                    "algorithm": "LightGBM",
                    "tier": "large_data",
                    "priority": 2,
                    "reason": lgbm_reason,
                    "config": {
                        "n_estimators": 500,
                        "num_leaves": 31,
                        "learning_rate": 0.05,
                        "min_child_samples": 20,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                    },
                    "expected_training_time": "30-120 seconds",
                },
                {
                    "algorithm": "XGBoost",
                    "tier": "large_data",
                    "priority": 2,
                    "reason": xgb_large_reason,
                    "config": {
                        "n_estimators": 500,
                        "max_depth": 8,
                        "learning_rate": 0.05,
                        "min_child_weight": 10,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                    "expected_training_time": "1-5 minutes",
                },
            ])

        # ── TIER 3: Special considerations ──
        if categorical > 0 and categorical >= numeric and rows > 1000:
            recommendations.append({
                "algorithm": "CatBoost",
                "tier": "categorical_heavy",
                "priority": 3,
                "reason": (
                    f"Best native categorical handling ({categorical} cat features). "
                    f"No encoding needed — CatBoost handles it internally with ordered target statistics."
                ),
                "config": {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "cat_features": "auto",
                },
                "expected_training_time": "1-5 minutes",
            })

        # Deduplicate
        seen = set()
        unique_recs = []
        for r in recommendations:
            if r["algorithm"] not in seen:
                seen.add(r["algorithm"])
                unique_recs.append(r)

        unique_recs.sort(key=lambda x: x["priority"])
        return unique_recs

    # ──────────────────────────────────────────────────────────
    # 6. HYPERPARAMETER SEARCH SPACE
    # ──────────────────────────────────────────────────────────

    def recommend_hyperparameter_space(self, algorithm: str, context: Dict) -> Dict[str, Any]:
        """
        Generate the optimal hyperparameter search space for a specific algorithm.
        Adapts ranges based on data size and characteristics.
        """
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 1000)
        cols = profile.get("columns", 10)
        algo_lower = algorithm.lower().replace(" ", "").replace("_", "")

        spaces = {}

        if "xgboost" in algo_lower or "xgb" in algo_lower:
            max_depth_upper = min(12, max(4, int(math.log2(rows / 10))))
            spaces = {
                "algorithm": "XGBoost",
                "method": "Optuna or RandomizedSearchCV (50-100 iterations)",
                "parameters": {
                    "n_estimators": {"type": "int", "range": [100, 1000], "step": 100},
                    "max_depth": {"type": "int", "range": [3, max_depth_upper]},
                    "learning_rate": {"type": "float", "range": [0.01, 0.3], "log_scale": True},
                    "min_child_weight": {"type": "int", "range": [1, max(10, rows // 100)]},
                    "subsample": {"type": "float", "range": [0.6, 1.0]},
                    "colsample_bytree": {"type": "float", "range": [0.5, 1.0]},
                    "reg_alpha": {"type": "float", "range": [1e-8, 10], "log_scale": True},
                    "reg_lambda": {"type": "float", "range": [1e-8, 10], "log_scale": True},
                    "gamma": {"type": "float", "range": [0, 5]},
                },
                "notes": [
                    "Use early_stopping_rounds=50 to prevent overfitting",
                    f"max_depth capped at {max_depth_upper} based on {rows} samples",
                    "Lower learning_rate + higher n_estimators = better generalization",
                ],
            }

        elif "lightgbm" in algo_lower or "lgbm" in algo_lower:
            max_leaves = min(128, max(15, int(rows ** 0.5 / 2)))
            spaces = {
                "algorithm": "LightGBM",
                "method": "Optuna or RandomizedSearchCV",
                "parameters": {
                    "n_estimators": {"type": "int", "range": [100, 1000]},
                    "num_leaves": {"type": "int", "range": [15, max_leaves]},
                    "learning_rate": {"type": "float", "range": [0.01, 0.3], "log_scale": True},
                    "min_child_samples": {"type": "int", "range": [5, max(50, rows // 50)]},
                    "subsample": {"type": "float", "range": [0.6, 1.0]},
                    "colsample_bytree": {"type": "float", "range": [0.5, 1.0]},
                    "reg_alpha": {"type": "float", "range": [1e-8, 10], "log_scale": True},
                    "reg_lambda": {"type": "float", "range": [1e-8, 10], "log_scale": True},
                },
                "notes": [
                    f"num_leaves capped at {max_leaves} to prevent overfitting on {rows} rows",
                    "Keep num_leaves < 2^max_depth equivalent",
                ],
            }

        elif "randomforest" in algo_lower or "random_forest" in algo_lower:
            spaces = {
                "algorithm": "RandomForest",
                "method": "RandomizedSearchCV (30-50 iterations)",
                "parameters": {
                    "n_estimators": {"type": "int", "range": [100, 500], "step": 50},
                    "max_depth": {"type": "int_or_none", "range": [5, 30, None]},
                    "min_samples_split": {"type": "int", "range": [2, 20]},
                    "min_samples_leaf": {"type": "int", "range": [1, max(10, rows // 100)]},
                    "max_features": {"type": "categorical", "values": ["sqrt", "log2", 0.5, 0.8]},
                },
                "notes": [
                    "More trees is almost always better (no overfitting risk from n_estimators)",
                    "max_depth=None lets trees grow fully — control via min_samples_leaf instead",
                ],
            }

        elif "logistic" in algo_lower:
            spaces = {
                "algorithm": "LogisticRegression",
                "method": "GridSearchCV",
                "parameters": {
                    "C": {"type": "float", "range": [0.001, 100], "log_scale": True,
                          "grid": [0.001, 0.01, 0.1, 1.0, 10, 100]},
                    "penalty": {"type": "categorical", "values": ["l1", "l2", "elasticnet"]},
                    "solver": {"type": "categorical", "values": ["lbfgs", "saga"]},
                    "l1_ratio": {"type": "float", "range": [0.0, 1.0],
                                 "note": "Only when penalty=elasticnet"},
                },
                "notes": [
                    "Small C = more regularization (simpler model)",
                    "L1 penalty enables automatic feature selection",
                    "ElasticNet combines L1 + L2 benefits",
                ],
            }

        elif "svm" in algo_lower or "svc" in algo_lower:
            spaces = {
                "algorithm": "SVM",
                "method": "GridSearchCV (SVM is sensitive to hyperparameters)",
                "parameters": {
                    "C": {"type": "float", "grid": [0.1, 1, 10, 100]},
                    "kernel": {"type": "categorical", "values": ["rbf", "linear", "poly"]},
                    "gamma": {"type": "categorical", "values": ["scale", "auto", 0.001, 0.01, 0.1]},
                },
                "notes": [
                    f"With {rows} rows, training time ≈ O(n²). Consider LinearSVC for speed.",
                    "Always scale features before SVM",
                ],
            }

        else:
            spaces = {
                "algorithm": algorithm,
                "method": "RandomizedSearchCV",
                "parameters": {},
                "notes": [f"No pre-built search space for '{algorithm}'. Use algorithm documentation."],
            }

        return spaces

    # ──────────────────────────────────────────────────────────
    # 7. THRESHOLD OPTIMIZATION
    # ──────────────────────────────────────────────────────────

    def recommend_threshold(self, context: Dict) -> Dict[str, Any]:
        """
        Recommend optimal classification threshold based on business context.
        """
        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})

        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1_score") or metrics.get("f1", 0)
        auc = metrics.get("roc_auc") or metrics.get("auc_roc", 0)

        recommendation = {
            "current_threshold": 0.5,
            "analysis": {},
            "strategies": [],
        }

        if not precision or not recall:
            recommendation["analysis"]["note"] = "Insufficient metrics for threshold analysis"
            return recommendation

        # Analyze current balance
        if precision > recall * 1.5:
            recommendation["analysis"] = {
                "current_bias": "conservative",
                "explanation": (
                    f"Model is conservative: high precision ({precision:.1%}) but low recall "
                    f"({recall:.1%}). It rarely makes false positives but misses many true positives."
                ),
            }
            recommendation["strategies"] = [
                {
                    "name": "Maximize F1",
                    "threshold_direction": "lower (try 0.3-0.4)",
                    "effect": "Catches more positives at cost of some false alarms",
                    "best_for": "General purpose when FP and FN costs are similar",
                },
                {
                    "name": "Maximize Recall (catch all positives)",
                    "threshold_direction": "lower (try 0.2-0.3)",
                    "effect": "Catches >80% of positives, more false alarms",
                    "best_for": "Fraud detection, disease screening, churn prevention",
                },
            ]

        elif recall > precision * 1.5:
            recommendation["analysis"] = {
                "current_bias": "aggressive",
                "explanation": (
                    f"Model is aggressive: high recall ({recall:.1%}) but low precision "
                    f"({precision:.1%}). It catches most positives but generates many false alarms."
                ),
            }
            recommendation["strategies"] = [
                {
                    "name": "Maximize F1",
                    "threshold_direction": "higher (try 0.6-0.7)",
                    "effect": "Reduces false alarms while keeping reasonable recall",
                    "best_for": "General purpose balanced approach",
                },
                {
                    "name": "Maximize Precision (reduce false alarms)",
                    "threshold_direction": "higher (try 0.7-0.8)",
                    "effect": "Very few false positives, will miss some true positives",
                    "best_for": "Spam filtering, recommendation systems",
                },
            ]

        else:
            recommendation["analysis"] = {
                "current_bias": "balanced",
                "explanation": f"Precision ({precision:.1%}) and recall ({recall:.1%}) are reasonably balanced.",
            }
            recommendation["strategies"] = [
                {
                    "name": "Current threshold is reasonable",
                    "threshold_direction": "keep at 0.5",
                    "effect": "Balanced trade-off",
                    "best_for": "Most use cases",
                },
            ]

        recommendation["how_to_optimize"] = (
            "Use sklearn.metrics.precision_recall_curve to plot P-R at every threshold. "
            "Find the threshold that maximizes your chosen metric (F1, F-beta, or custom cost)."
        )

        return recommendation

    # ──────────────────────────────────────────────────────────
    # 8. CROSS-VALIDATION STRATEGY
    # ──────────────────────────────────────────────────────────

    def recommend_cv_strategy(self, context: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Recommend cross-validation configuration based on data characteristics.
        """
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        temporal = analysis.get("temporal_patterns", {})
        readiness = analysis.get("data_readiness", {})

        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")

        is_timeseries = temporal.get("is_likely_timeseries", False)

        if is_timeseries:
            return {
                "strategy": "TimeSeriesSplit",
                "n_splits": min(5, max(3, rows // 500)),
                "reason": (
                    "Temporal data detected — random K-fold would leak future information. "
                    "TimeSeriesSplit ensures training always uses past data to predict future."
                ),
                "code": (
                    "from sklearn.model_selection import TimeSeriesSplit\n"
                    f"tscv = TimeSeriesSplit(n_splits={min(5, max(3, rows // 500))})\n"
                    "for train_idx, val_idx in tscv.split(X):\n"
                    "    X_train, X_val = X[train_idx], X[val_idx]"
                ),
                "warnings": [
                    "Sort data by time BEFORE splitting",
                    "Each fold uses more training data than the last",
                ],
            }

        if rows < 100:
            return {
                "strategy": "LeaveOneOut",
                "n_splits": rows,
                "reason": f"Very small dataset ({rows} rows) — LOO uses maximum training data per fold.",
                "code": (
                    "from sklearn.model_selection import LeaveOneOut\n"
                    "loo = LeaveOneOut()\n"
                    "scores = cross_val_score(model, X, y, cv=loo)"
                ),
                "warnings": ["Computationally expensive for complex models", "High variance in estimates"],
            }

        if rows < 500:
            n_folds = 10
        elif rows < 5000:
            n_folds = 5
        else:
            n_folds = 5

        samples_per_fold = rows // n_folds

        return {
            "strategy": "StratifiedKFold" if target else "KFold",
            "n_splits": n_folds,
            "shuffle": True,
            "reason": (
                f"{'Stratified ' if target else ''}K-Fold with {n_folds} folds "
                f"(~{samples_per_fold} samples per fold). "
                f"{'Stratification maintains class ratio in each fold.' if target else ''}"
            ),
            "code": (
                f"from sklearn.model_selection import {'StratifiedKFold' if target else 'KFold'}\n"
                f"cv = {'StratifiedKFold' if target else 'KFold'}(n_splits={n_folds}, shuffle=True, random_state=42)\n"
                "scores = cross_val_score(model, X, y, cv=cv, scoring='f1')"
            ),
            "repeat_recommendation": (
                f"For more robust estimates, use RepeatedStratifiedKFold with n_repeats=3"
                if rows < 2000 else None
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 9. ENSEMBLE STRATEGY
    # ──────────────────────────────────────────────────────────

    def recommend_ensemble(self, context: Dict) -> Dict[str, Any]:
        """
        Recommend whether and how to build an ensemble.
        """
        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 0)

        f1 = metrics.get("f1_score") or metrics.get("f1", 0)
        auc = metrics.get("roc_auc") or metrics.get("auc_roc", 0)

        if rows < 500:
            return {
                "recommend_ensemble": False,
                "reason": (
                    f"With only {rows} samples, ensembles risk overfitting. "
                    f"Focus on a single well-tuned model instead."
                ),
            }

        if f1 and f1 > 0.9:
            return {
                "recommend_ensemble": False,
                "reason": f"F1 of {f1:.3f} is already excellent. Ensemble gains would be marginal.",
            }

        return {
            "recommend_ensemble": True,
            "reason": "Ensembles typically improve by 1-5% over single best model.",
            "strategies": [
                {
                    "name": "Voting Ensemble",
                    "description": "Combine predictions from diverse models via majority vote (hard) or probability averaging (soft).",
                    "best_when": "You have 3-5 diverse models trained independently.",
                    "code": (
                        "from sklearn.ensemble import VotingClassifier\n"
                        "ensemble = VotingClassifier(\n"
                        "    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],\n"
                        "    voting='soft'\n"
                        ")"
                    ),
                },
                {
                    "name": "Stacking",
                    "description": "Train a meta-learner on base model predictions. More powerful but more complex.",
                    "best_when": "Maximum accuracy is the goal and you have enough data for a meta-learner.",
                    "code": (
                        "from sklearn.ensemble import StackingClassifier\n"
                        "stacking = StackingClassifier(\n"
                        "    estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],\n"
                        "    final_estimator=LogisticRegression(),\n"
                        "    cv=5\n"
                        ")"
                    ),
                },
            ],
            "diversity_principle": (
                "Ensemble power comes from DIVERSITY. Combine models that make different errors: "
                "linear (LogReg) + tree (RF) + boosting (XGB) works better than 3 tree-based models."
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 10. AGGREGATE RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────

    def full_recommendations(self, context: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Generate all recommendations in one call.
        Main entry point for the orchestrator.
        """
        return {
            "feature_selection": self.recommend_feature_selection(context, analysis),
            "encoding": self.recommend_encoding(context, analysis),
            "scaling": self.recommend_scaling(context, analysis),
            "imputation": self.recommend_imputation(context, analysis),
            "algorithms": self.recommend_algorithms(context, analysis),
            "threshold": self.recommend_threshold(context),
            "cv_strategy": self.recommend_cv_strategy(context, analysis),
            "ensemble": self.recommend_ensemble(context),
        }