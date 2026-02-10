"""
Q&A Engine — Deterministic Question Answering
===============================================
Answers ML questions WITHOUT an LLM using:
  1. Intent classification (regex + keyword matching)
  2. Knowledge base lookup
  3. Context-aware response templates
  4. Insight synthesis (summarizing rule engine findings)
  5. Recommendation extraction

Intent Categories:
  - algorithm_selection:  "which algorithm", "what model should I use"
  - feature_importance:   "which features matter", "feature importance"
  - metric_interpretation: "what does F1 mean", "is 0.87 accuracy good"
  - data_quality:         "how is my data", "data quality issues"
  - class_imbalance:      "imbalanced", "class distribution"
  - correlation:          "correlated features", "multicollinearity"
  - missing_data:         "missing values", "null", "NaN"
  - overfitting:          "overfitting", "generalization"
  - threshold:            "threshold", "cutoff", "precision vs recall"
  - deployment:           "deploy", "production", "serve"
  - drift:               "drift", "monitoring", "performance decay"
  - hyperparameters:      "hyperparameter", "tune", "grid search"
  - cross_validation:     "cross-validation", "CV", "k-fold"
  - encoding:             "encoding", "categorical", "one-hot"
  - scaling:              "scaling", "normalize", "standardize"
  - ensemble:             "ensemble", "stacking", "blending"
  - leakage:              "leakage", "data leakage", "target leakage"
  - general_help:         "help", "what should I do", "next step"

Confidence Levels:
  1.0  — Exact match on known topic with context data
  0.8  — Intent matched, context partially available
  0.5  — Intent matched, generic response (no context)
  0.3  — No intent match, best-effort response
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# INTENT PATTERNS
# ═══════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "algorithm_selection": [
        r"\b(which|what|best|recommend)\b.*\b(algorithm|model|classifier|regressor)\b",
        r"\bshould\s+i\s+use\b.*\b(xgboost|random\s*forest|logistic|svm|lightgbm|neural|catboost)\b",
        r"\b(compare|versus|vs)\b.*\b(algorithm|model)\b",
        r"\bwhat\s+(model|algorithm)\b",
        r"\b(pick|choose|select)\b.*\b(algorithm|model)\b",
    ],
    "feature_importance": [
        r"\b(which|what|most)\b.*\b(feature|variable|column)s?\b.*\b(important|matter|impact|predict)\b",
        r"\bfeature\s*(importance|ranking|relevance)\b",
        r"\bwhich\s+(feature|column)s?\s+(should|to)\s+(keep|drop|remove)\b",
        r"\b(top|best|key|critical)\b.*\bfeature\b",
    ],
    "metric_interpretation": [
        r"\b(what\s+does|explain|interpret|meaning)\b.*\b(f1|auc|roc|precision|recall|accuracy|mcc|rmse|mae|r2)\b",
        r"\bis\s+[\d.]+\s*(accuracy|f1|auc|precision|recall)\b.*\bgood\b",
        r"\b(good|bad|acceptable)\b.*\b(accuracy|f1|auc|precision|recall|score)\b",
        r"\bmetric\b.*\b(mean|significance|interpret)\b",
        r"\bwhy\b.*\b(accuracy|f1)\b.*\b(different|misleading|low|high)\b",
    ],
    "data_quality": [
        r"\b(how|what)\b.*\b(data\s+quality|quality\s+of)\b",
        r"\bdata\b.*\b(clean|dirty|issues|problems)\b",
        r"\b(quality|health)\s*(score|check|assessment)\b",
        r"\bready\s+(for|to)\s+train\b",
    ],
    "class_imbalance": [
        r"\b(class|target)\s*(imbalance|distribution|ratio|balanced)\b",
        r"\b(imbalanced|skewed)\s*(data|dataset|classes|target)\b",
        r"\b(oversamp|undersamp|smote|balanced)\b",
        r"\bminority\s*(class)?\b",
    ],
    "correlation": [
        r"\b(correlat|multicollinear)\b",
        r"\bhighly?\s+correlat\b",
        r"\bremove\b.*\bcorrelat\b",
        r"\b(redundant|duplicate)\s+feature\b",
    ],
    "missing_data": [
        r"\bmissing\s*(data|values?|cells?)?\b",
        r"\b(null|nan|none|na)\s*(values?)?\b",
        r"\bimput(e|ation|ing)\b",
        r"\bfill\b.*\bmissing\b",
    ],
    "overfitting": [
        r"\bover\s*fit(ting)?\b",
        r"\bunder\s*fit(ting)?\b",
        r"\bgeneraliz(e|ation)\b",
        r"\btrain\b.*\b(vs|versus)\b.*\b(test|validation)\b.*\b(gap|difference)\b",
        r"\bvariance\b.*\bbias\b",
    ],
    "threshold": [
        r"\bthreshold\b",
        r"\bcutoff\b.*\b(point|value)\b",
        r"\bprecision\b.*\b(vs|versus|or)\b.*\brecall\b",
        r"\b(optimize|optimal)\b.*\b(threshold|cutoff)\b",
        r"\bfalse\s*(positive|negative)\b.*\b(trade|balance)\b",
    ],
    "deployment": [
        r"\bdeploy(ment|ing)?\b",
        r"\bproduction\b",
        r"\bserv(e|ing)\b.*\bmodel\b",
        r"\b(shadow|canary|blue.green|rolling)\b",
        r"\bmodel\s+serving\b",
    ],
    "drift": [
        r"\bdrift\b",
        r"\bmonitor(ing)?\b.*\b(model|performance|data)\b",
        r"\bperformance\s*(decay|degrad)\b",
        r"\bretrain\b.*\bwhen\b",
        r"\bstale\s+model\b",
    ],
    "hyperparameters": [
        r"\bhyperparameter\b",
        r"\b(tune|tuning)\b",
        r"\b(grid|random|bayesian)\s*search\b",
        r"\b(learning\s+rate|max\s*depth|n\s*estimators|regulariz)\b",
    ],
    "cross_validation": [
        r"\bcross[\s-]*validat\b",
        r"\b(k[\s-]*fold|cv|stratif)\b",
        r"\b(train|test|validation)\s*split\b",
    ],
    "encoding": [
        r"\b(encod(e|ing)|one[\s-]*hot|label\s+encod|target\s+encod|ordinal\s+encod)\b",
        r"\bcategorical\b.*\b(handle|convert|transform|encode)\b",
    ],
    "scaling": [
        r"\b(scal(e|ing)|normaliz|standardiz)\b",
        r"\b(standard\s*scaler|min\s*max|robust\s*scaler)\b",
    ],
    "ensemble": [
        r"\b(ensemble|stack(ing)?|blend(ing)?|bag(ging)?|boost(ing)?)\b",
        r"\bcombine\b.*\bmodel\b",
        r"\bvoting\b.*\b(classifier|model)\b",
    ],
    "leakage": [
        r"\b(data\s+)?leakage\b",
        r"\btarget\s+leakage\b",
        r"\bfuture\s+data\b.*\btrain\b",
    ],
    "general_help": [
        r"\b(help|guide|what\s+should|next\s+step|where\s+to\s+start|what\s+now)\b",
        r"\bwhat\s+do\s+you\s+(recommend|suggest|think)\b",
        r"\bhow\s+(do|should)\s+i\s+(start|proceed|begin)\b",
    ],
}


# ═══════════════════════════════════════════════════════════════
# Q&A ENGINE
# ═══════════════════════════════════════════════════════════════

class QAEngine:
    """
    Deterministic question answering engine.
    Classifies intent, extracts context, generates grounded answers.
    """

    def answer(
        self,
        question: str,
        screen: str,
        context: Dict[str, Any],
        analysis: Optional[Dict] = None,
        insights: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using deterministic logic.
        Returns: { answer: str, confidence: float, follow_ups: [str] }
        """
        analysis = analysis or {}
        insights = insights or []

        # Step 1: Classify intent
        intent, intent_confidence = self._classify_intent(question)

        # Step 2: Route to handler
        handler = self._get_handler(intent)
        result = handler(
            question=question,
            screen=screen,
            context=context,
            analysis=analysis,
            insights=insights,
        )

        # Adjust confidence based on data availability
        has_data = bool(context.get("dataset_profile", {}).get("rows"))
        if has_data:
            result["confidence"] = min(1.0, result["confidence"] + 0.2)
        else:
            result["confidence"] = max(0.2, result["confidence"] - 0.2)

        result["intent"] = intent
        return result

    # ──────────────────────────────────────────────────────────
    # INTENT CLASSIFICATION
    # ──────────────────────────────────────────────────────────

    def _classify_intent(self, question: str) -> Tuple[str, float]:
        """Classify question intent using pattern matching."""
        q_lower = question.lower().strip()

        best_intent = "general_help"
        best_score = 0.0

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, q_lower)
                if match:
                    # Longer matches are more specific
                    score = len(match.group()) / max(len(q_lower), 1)
                    score = max(score, 0.6)  # minimum score for any match
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        return best_intent, best_score

    def _get_handler(self, intent: str):
        """Route intent to handler."""
        handlers = {
            "algorithm_selection": self._answer_algorithm_selection,
            "feature_importance": self._answer_feature_importance,
            "metric_interpretation": self._answer_metric_interpretation,
            "data_quality": self._answer_data_quality,
            "class_imbalance": self._answer_class_imbalance,
            "correlation": self._answer_correlation,
            "missing_data": self._answer_missing_data,
            "overfitting": self._answer_overfitting,
            "threshold": self._answer_threshold,
            "deployment": self._answer_deployment,
            "drift": self._answer_drift,
            "hyperparameters": self._answer_hyperparameters,
            "cross_validation": self._answer_cross_validation,
            "encoding": self._answer_encoding,
            "scaling": self._answer_scaling,
            "ensemble": self._answer_ensemble,
            "leakage": self._answer_leakage,
            "general_help": self._answer_general_help,
        }
        return handlers.get(intent, self._answer_general_help)

    # ══════════════════════════════════════════════════════════
    # INTENT HANDLERS
    # ══════════════════════════════════════════════════════════

    def _answer_algorithm_selection(self, **kw) -> Dict:
        ctx = kw["context"]
        analysis = kw["analysis"]
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)
        num_count = profile.get("numeric_count", 0)
        cat_count = profile.get("categorical_count", 0)
        quality = ctx.get("data_quality", {})

        # Determine problem characteristics
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type", "classification")

        parts = []
        follow_ups = []

        if rows > 0:
            parts.append(f"Based on your dataset ({rows:,} rows × {cols} columns, "
                        f"{num_count} numeric / {cat_count} categorical features):")

            recs = []
            if problem_type == "classification":
                if rows < 500:
                    recs = [
                        ("**Logistic Regression**", "Best for small data. Interpretable, fast, "
                         "strong regularization prevents overfitting."),
                        ("**Random Forest** (max_depth=5)", "Handles mixed types, resistant to "
                         "outliers, but limit depth to avoid memorization."),
                    ]
                    parts.append("With small data, simpler models generalize better.")
                elif rows < 10000:
                    recs = [
                        ("**XGBoost**", f"Sweet spot for {rows:,} rows. Strong regularization "
                         "options, handles missing values natively."),
                        ("**Random Forest**", "Excellent baseline — robust, parallelizable, "
                         "minimal tuning needed."),
                        ("**Logistic Regression**", "Always include as a baseline. If it "
                         "performs within 2% of complex models, prefer it for interpretability."),
                    ]
                    if cat_count > num_count:
                        recs.insert(0, ("**LightGBM** or **CatBoost**",
                                       "Native categorical support — no encoding needed. "
                                       "Ideal for your predominantly categorical data."))
                else:
                    recs = [
                        ("**LightGBM**", f"Fastest gradient boosting for large data ({rows:,} rows). "
                         "Histogram-based splitting, lower memory usage."),
                        ("**XGBoost**", "Gold standard for structured/tabular data."),
                        ("**CatBoost**", "Best if you have many categorical features — "
                         "ordered target encoding prevents leakage."),
                    ]

                for name, reason in recs:
                    parts.append(f"  • {name} — {reason}")

            else:  # regression
                if rows < 1000:
                    recs = [
                        ("**Ridge Regression**", "L2 regularization handles multicollinearity."),
                        ("**Lasso Regression**", "L1 regularization performs automatic feature selection."),
                        ("**Random Forest Regressor**", "Non-linear relationships without overfitting risk."),
                    ]
                else:
                    recs = [
                        ("**XGBoost Regressor**", "Top performer on tabular regression benchmarks."),
                        ("**LightGBM Regressor**", "Faster training, excellent for large datasets."),
                        ("**Ridge Regression**", "Strong interpretable baseline."),
                    ]
                for name, reason in recs:
                    parts.append(f"  • {name} — {reason}")

            # Check for issues that affect algorithm choice
            completeness = quality.get("completeness", 100)
            if completeness < 85:
                parts.append(f"\n⚠ Note: With {completeness:.0f}% completeness, "
                            "prefer tree-based models (XGBoost, LightGBM) which handle "
                            "missing values natively, or impute first for linear models.")

            follow_ups = [
                "What hyperparameters should I use?",
                "Should I use an ensemble approach?",
                "How should I compare model performance?",
            ]

        else:
            parts.append("I need a dataset loaded to make specific algorithm recommendations. "
                        "In general, for structured/tabular data, XGBoost and LightGBM are "
                        "strong defaults for classification, with Logistic Regression as "
                        "an interpretable baseline.")
            follow_ups = ["What data should I upload?"]

        return {
            "answer": "\n".join(parts),
            "confidence": 0.85 if rows > 0 else 0.5,
            "follow_ups": follow_ups,
        }

    def _answer_feature_importance(self, **kw) -> Dict:
        ctx = kw["context"]
        analysis = kw["analysis"]
        insights = kw["insights"]
        profile = ctx.get("dataset_profile", {})
        correlations = ctx.get("correlations", {})
        features = ctx.get("feature_stats", {})

        parts = []
        follow_ups = []

        # Check for feature-target scores in analysis
        ft_scores = analysis.get("feature_target_scores", [])
        if ft_scores:
            parts.append("**Feature Relevance Analysis** (from statistical analysis):")
            for score_info in ft_scores[:10]:
                if isinstance(score_info, dict):
                    name = score_info.get("feature", score_info.get("column", "?"))
                    score = score_info.get("score", score_info.get("relevance", 0))
                    method = score_info.get("method", "mutual_information")
                    parts.append(f"  • **{name}** — score: {score:.4f} ({method})")

        # Check for correlation-based insights
        high_pairs = correlations.get("high_pairs", [])
        if high_pairs:
            parts.append("\n**Correlated Features** (consider dropping one from each pair):")
            for pair in high_pairs[:5]:
                parts.append(
                    f"  • {pair['feature1']} ↔ {pair['feature2']} "
                    f"(r = {pair['correlation']:.3f})"
                )

        # Check for features to drop
        id_cols = features.get("potential_id_columns", [])
        constant = features.get("constant_columns", [])
        drops = []
        if id_cols:
            drops.extend([f"**{c['column']}** (ID column)" for c in id_cols[:3]])
        if constant:
            drops.extend([f"**{c}** (zero variance)" for c in constant[:3]])
        if drops:
            parts.append("\n**Recommended Drops:** " + ", ".join(drops))

        if not parts:
            parts.append("Feature importance analysis requires a trained model. "
                        "To get feature rankings:\n"
                        "  1. Train a Random Forest first (it computes feature importance natively)\n"
                        "  2. Use mutual information scoring (works before training)\n"
                        "  3. Check the Evaluation screen for feature importance charts after training")

        follow_ups = [
            "Should I remove correlated features?",
            "How many features should I keep?",
            "What encoding should I use for categorical features?",
        ]

        return {
            "answer": "\n".join(parts),
            "confidence": 0.8 if ft_scores else 0.5,
            "follow_ups": follow_ups,
        }

    def _answer_metric_interpretation(self, **kw) -> Dict:
        ctx = kw["context"]
        insights = kw["insights"]
        question = kw["question"]
        q_lower = question.lower()

        parts = []
        follow_ups = []
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})

        # Detect which metric they're asking about
        metric_defs = {
            "accuracy": {
                "name": "Accuracy",
                "formula": "correct predictions / total predictions",
                "good_for": "balanced datasets where both classes matter equally",
                "bad_for": "imbalanced datasets — a model predicting all majority class gets high accuracy",
                "range": "0 to 1 (higher is better)",
            },
            "f1": {
                "name": "F1 Score",
                "formula": "2 × (precision × recall) / (precision + recall)",
                "good_for": "imbalanced datasets. Balances precision and recall into one number.",
                "bad_for": "when precision and recall have very different business costs",
                "range": "0 to 1 (higher is better)",
            },
            "precision": {
                "name": "Precision",
                "formula": "true positives / (true positives + false positives)",
                "good_for": "when false positives are expensive (e.g., spam detection, fraud alerts)",
                "bad_for": "when missing positive cases is costly (use recall instead)",
                "range": "0 to 1 (higher is better)",
            },
            "recall": {
                "name": "Recall (Sensitivity)",
                "formula": "true positives / (true positives + false negatives)",
                "good_for": "when missing positive cases is dangerous (medical diagnosis, fraud detection)",
                "bad_for": "when false alarms are costly (it ignores false positives)",
                "range": "0 to 1 (higher is better)",
            },
            "auc": {
                "name": "AUC-ROC",
                "formula": "area under the ROC curve (TPR vs FPR at all thresholds)",
                "good_for": "overall model discrimination ability. Threshold-independent.",
                "bad_for": "highly imbalanced data — use PR-AUC instead",
                "range": "0.5 (random) to 1.0 (perfect)",
            },
            "roc": {
                "name": "AUC-ROC",
                "formula": "area under the ROC curve",
                "good_for": "overall ranking ability",
                "bad_for": "class-imbalanced problems",
                "range": "0.5 (random) to 1.0 (perfect)",
            },
            "mcc": {
                "name": "Matthews Correlation Coefficient",
                "formula": "(TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
                "good_for": "imbalanced datasets. Only metric that gives high score when both positive and negative predictions are accurate.",
                "bad_for": "nothing — it's the most balanced single metric for binary classification",
                "range": "-1 to +1 (0 = random, 1 = perfect)",
            },
            "rmse": {
                "name": "Root Mean Squared Error",
                "formula": "√(mean of squared residuals)",
                "good_for": "regression. Penalizes large errors more than MAE.",
                "bad_for": "datasets with many outliers (use MAE instead)",
                "range": "0 to ∞ (lower is better, in same units as target)",
            },
            "mae": {
                "name": "Mean Absolute Error",
                "formula": "mean of |actual - predicted|",
                "good_for": "regression, especially with outliers. More robust than RMSE.",
                "bad_for": "nothing — universally useful, but doesn't penalize large errors extra",
                "range": "0 to ∞ (lower is better, in same units as target)",
            },
            "r2": {
                "name": "R² (Coefficient of Determination)",
                "formula": "1 - (sum of squared residuals / total sum of squares)",
                "good_for": "regression. Shows proportion of variance explained by the model.",
                "bad_for": "comparing models with different numbers of features (use adjusted R² instead)",
                "range": "-∞ to 1 (higher is better, 1 = perfect, <0 = worse than mean)",
            },
        }

        found_metric = None
        for metric_key in metric_defs:
            if metric_key in q_lower:
                found_metric = metric_key
                break

        if found_metric:
            md = metric_defs[found_metric]
            parts.append(f"**{md['name']}**")
            parts.append(f"  Formula: {md['formula']}")
            parts.append(f"  Range: {md['range']}")
            parts.append(f"  Best for: {md['good_for']}")
            parts.append(f"  Watch out: {md['bad_for']}")

            # Add context-specific interpretation
            if metrics:
                val = metrics.get(found_metric) or metrics.get(md["name"].lower().replace(" ", "_"))
                if val is not None:
                    parts.append(f"\n**Your value: {val:.4f}**")
                    interpretation = self._interpret_metric_value(found_metric, val, metrics)
                    if interpretation:
                        parts.append(interpretation)
        elif metrics:
            parts.append("**Your Current Metrics:**")
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    parts.append(f"  • **{key}**: {val:.4f}")
                    interp = self._interpret_metric_value(key.lower(), val, metrics)
                    if interp:
                        parts.append(f"    {interp}")
        else:
            parts.append("Which metric would you like explained? I can cover: "
                        "accuracy, F1, precision, recall, AUC-ROC, MCC, RMSE, MAE, R².")

        follow_ups = [
            "Should I optimize for precision or recall?",
            "What threshold should I use?",
            "Is my model good enough for production?",
        ]

        return {
            "answer": "\n".join(parts),
            "confidence": 0.9 if found_metric else 0.6,
            "follow_ups": follow_ups,
        }

    def _answer_data_quality(self, **kw) -> Dict:
        ctx = kw["context"]
        insights = kw["insights"]
        quality = ctx.get("data_quality", {})
        profile = ctx.get("dataset_profile", {})

        parts = []
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        if rows > 0:
            score = quality.get("overall_quality_score", 0)
            completeness = quality.get("completeness", 100)
            dup_pct = quality.get("duplicate_pct", 0)

            parts.append(f"**Data Quality Report** for {profile.get('file_name', 'your dataset')}:")
            parts.append(f"  • Shape: {rows:,} rows × {cols} columns")
            parts.append(f"  • Quality Score: **{score:.0f}/100**")
            parts.append(f"  • Completeness: {completeness:.1f}%")
            parts.append(f"  • Duplicates: {dup_pct:.1f}%")

            # Highlight issues from insights
            criticals = [i for i in insights if i.get("severity") == "critical"
                        and i.get("category", "").startswith("Data")]
            warnings = [i for i in insights if i.get("severity") == "warning"
                       and i.get("category", "").startswith("Data")]

            if criticals:
                parts.append(f"\n🚫 **{len(criticals)} Critical Issues:**")
                for c in criticals[:3]:
                    parts.append(f"  • {c['title']}")
            if warnings:
                parts.append(f"\n⚠ **{len(warnings)} Warnings:**")
                for w in warnings[:3]:
                    parts.append(f"  • {w['title']}")

            if score >= 90 and not criticals:
                parts.append("\n✅ Your data quality is strong. Ready for model training.")
            elif score >= 70:
                parts.append("\n🔶 Address the issues above to improve model quality.")
            else:
                parts.append("\n🔴 Significant data issues found. Fix critical issues before training.")

        else:
            parts.append("No dataset loaded. Upload a dataset to see quality analysis.")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.9 if rows > 0 else 0.3,
            "follow_ups": [
                "How should I handle missing values?",
                "Should I remove duplicates?",
                "Is my data ready for training?",
            ],
        }

    def _answer_class_imbalance(self, **kw) -> Dict:
        ctx = kw["context"]
        features = ctx.get("feature_stats", {})
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")

        parts = []
        numeric_stats = features.get("numeric_stats", {})

        # Try to detect imbalance from target stats
        if target and target in numeric_stats:
            stats = numeric_stats[target]
            mean_val = stats.get("mean", 0.5)
            if 0 <= mean_val <= 1:
                minority_pct = min(mean_val, 1 - mean_val) * 100
                majority_pct = max(mean_val, 1 - mean_val) * 100
                parts.append(f"**Class Distribution for '{target}':**")
                parts.append(f"  • Positive class: {mean_val*100:.1f}%")
                parts.append(f"  • Negative class: {(1-mean_val)*100:.1f}%")
                parts.append(f"  • Imbalance ratio: 1:{majority_pct/minority_pct:.1f}")

                if minority_pct < 5:
                    parts.append(f"\n🚫 **Severe imbalance** ({minority_pct:.1f}% minority). Actions:")
                    parts.append("  1. **SMOTE** — synthetic oversampling of minority class")
                    parts.append("  2. **class_weight='balanced'** — algorithmic cost adjustment")
                    parts.append("  3. **Threshold tuning** — lower from 0.5 to increase recall")
                    parts.append("  4. **Evaluate with F1, PR-AUC, MCC** — never accuracy alone")
                    parts.append("  5. **Stratified CV** — maintain ratio in each fold")
                elif minority_pct < 20:
                    parts.append(f"\n⚠ **Moderate imbalance** ({minority_pct:.1f}%). Actions:")
                    parts.append("  1. Use **class_weight='balanced'** in your algorithm")
                    parts.append("  2. Evaluate with **F1 score** as primary metric")
                    parts.append("  3. Use **stratified** train/test split and CV")
                else:
                    parts.append(f"\n✅ **Balanced** — standard methods work fine.")
            else:
                parts.append("Target appears to be a regression target (not binary). "
                            "Class imbalance applies to classification problems.")
        else:
            parts.append("**Handling Class Imbalance:**\n"
                        "  1. **Data-level:** SMOTE, ADASYN, random oversampling/undersampling\n"
                        "  2. **Algorithm-level:** class_weight='balanced', focal loss\n"
                        "  3. **Threshold-level:** tune classification threshold (often < 0.5)\n"
                        "  4. **Evaluation:** always use F1, PR-AUC, MCC for imbalanced data\n"
                        "  5. **Stratification:** maintain class ratio in CV folds and splits")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.85 if target else 0.6,
            "follow_ups": [
                "Should I use SMOTE or class weights?",
                "What threshold should I use?",
                "Which metric is best for imbalanced data?",
            ],
        }

    def _answer_correlation(self, **kw) -> Dict:
        ctx = kw["context"]
        correlations = ctx.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])

        parts = []
        if high_pairs:
            parts.append(f"**Found {len(high_pairs)} highly correlated feature pairs:**")
            for pair in high_pairs[:8]:
                severity_icon = "🚫" if pair["abs_correlation"] >= 0.95 else "⚠" if pair["abs_correlation"] >= 0.8 else "ℹ"
                parts.append(
                    f"  {severity_icon} **{pair['feature1']}** ↔ **{pair['feature2']}**: "
                    f"r = {pair['correlation']:.3f}"
                )

            parts.append("\n**What to do:**")
            parts.append("  • **For |r| ≥ 0.95:** Almost certainly duplicate/derived — drop one")
            parts.append("  • **For 0.8 ≤ |r| < 0.95:** Keep the one with higher target correlation")
            parts.append("  • **For 0.7 ≤ |r| < 0.8:** Keep both unless using linear models")
            parts.append("\nTree-based models (XGBoost, RF) are less affected by multicollinearity, "
                        "but feature importance becomes diluted across correlated pairs.")
        else:
            parts.append("No highly correlated feature pairs detected (|r| < 0.7). "
                        "Multicollinearity is not a concern for your current features.")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.9,
            "follow_ups": [
                "Which features should I drop?",
                "Should I use PCA to handle correlation?",
                "How does correlation affect model performance?",
            ],
        }

    def _answer_missing_data(self, **kw) -> Dict:
        ctx = kw["context"]
        quality = ctx.get("data_quality", {})
        completeness = quality.get("completeness", 100)
        high_missing = quality.get("columns_with_high_missing", [])
        missing_by_col = quality.get("missing_by_column", {})

        parts = []
        if completeness < 100:
            parts.append(f"**Missing Data Analysis** (completeness: {completeness:.1f}%):")

            if high_missing:
                parts.append(f"\n**Columns with >20% missing:**")
                for col in high_missing[:5]:
                    parts.append(f"  • **{col['column']}**: {col['missing_pct']:.1f}% missing")

            parts.append("\n**Imputation Strategy by Column Type:**")
            parts.append("  • **Numeric (low missing <10%):** Median (robust to outliers)")
            parts.append("  • **Numeric (high missing 10-50%):** IterativeImputer (multivariate)")
            parts.append("  • **Numeric (>50% missing):** DROP the column")
            parts.append("  • **Categorical:** Mode (most frequent value)")
            parts.append("  • **Any type:** Add binary indicator column (is_missing) — "
                        "the missingness pattern itself can be predictive")

            parts.append("\n**Important:** Tree-based models (XGBoost, LightGBM) handle "
                        "missing values natively — you may not need to impute for them.")
        else:
            parts.append("✅ No missing values detected. Your dataset is complete.")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.9,
            "follow_ups": [
                "Should I drop columns with lots of missing data?",
                "Which imputation method is best?",
                "Can the model handle missing values directly?",
            ],
        }

    def _answer_overfitting(self, **kw) -> Dict:
        ctx = kw["context"]
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)
        screen_ctx = ctx.get("screen_context", {}) or {}
        metrics = screen_ctx.get("metrics", {})

        parts = []
        parts.append("**Detecting & Preventing Overfitting:**\n")

        if rows > 0 and cols > 0:
            ratio = rows / cols
            if ratio < 10:
                parts.append(f"⚠ **High overfitting risk:** Your sample-to-feature ratio is "
                            f"only {ratio:.1f}:1 ({rows:,} rows / {cols} columns). "
                            f"Aim for at least 10:1.")

        parts.append("**Signs of overfitting:**")
        parts.append("  • Training accuracy >> test accuracy (gap > 5%)")
        parts.append("  • Perfect or near-perfect training scores")
        parts.append("  • High variance in cross-validation fold scores")

        parts.append("\n**Prevention techniques:**")
        parts.append("  1. **Regularization:** L1/L2 penalties, dropout")
        parts.append("  2. **Simpler model:** Fewer parameters, shallower trees")
        parts.append("  3. **More data:** Collect more samples or augment")
        parts.append("  4. **Feature selection:** Reduce to most important features")
        parts.append("  5. **Cross-validation:** Use k-fold CV to detect it early")
        parts.append("  6. **Early stopping:** Stop training when validation loss increases")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.75,
            "follow_ups": [
                "How should I regularize my model?",
                "What's a good number of features to keep?",
                "How do I interpret cross-validation results?",
            ],
        }

    def _answer_threshold(self, **kw) -> Dict:
        ctx = kw["context"]
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})

        parts = []
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        if precision and recall:
            parts.append(f"**Current Performance at Default Threshold (0.5):**")
            parts.append(f"  • Precision: {precision:.3f} — "
                        f"of predicted positives, {precision*100:.0f}% are correct")
            parts.append(f"  • Recall: {recall:.3f} — "
                        f"of actual positives, {recall*100:.0f}% are caught")

            gap = abs(precision - recall)
            if gap > 0.2:
                parts.append(f"\n⚠ **{gap:.0%} gap** between precision and recall. "
                            "Threshold tuning can help balance these.")

            parts.append("\n**Threshold Guidelines:**")
            if recall < precision:
                parts.append("  → **Lower the threshold** (e.g., 0.3–0.4) to catch more positives")
                parts.append("    Trade-off: more false positives, but fewer missed cases")
            else:
                parts.append("  → **Raise the threshold** (e.g., 0.6–0.7) for higher precision")
                parts.append("    Trade-off: more missed positives, but fewer false alarms")

        parts.append("\n**Business-Driven Threshold Selection:**")
        parts.append("  • **Cost of missing a positive >> cost of false alarm:** Lower threshold (0.2–0.4)")
        parts.append("    Examples: cancer detection, fraud, churn prevention")
        parts.append("  • **Cost of false alarm >> cost of missing:** Raise threshold (0.6–0.8)")
        parts.append("    Examples: spam filtering, arrest warrant, loan approval")
        parts.append("  • **Equal costs:** Use the threshold that maximizes F1 score")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.85 if (precision and recall) else 0.6,
            "follow_ups": [
                "What's the business impact of different thresholds?",
                "How do I find the optimal threshold automatically?",
                "What's the precision-recall trade-off curve?",
            ],
        }

    def _answer_deployment(self, **kw) -> Dict:
        ctx = kw["context"]
        registry = ctx.get("registry_info", {})

        parts = []
        parts.append("**Model Deployment Best Practices:**\n")

        parts.append("**1. Pre-Deployment Checklist:**")
        parts.append("  • Model passes all validation checks (accuracy, fairness, latency)")
        parts.append("  • Test set performance is acceptable and stable")
        parts.append("  • Model is versioned and reproducible")
        parts.append("  • Input schema is validated (feature types, ranges)")
        parts.append("  • Monitoring dashboards are configured")

        parts.append("\n**2. Deployment Strategies (in order of safety):**")
        parts.append("  • **Shadow Mode** — Model runs alongside production, predictions logged but not served")
        parts.append("  • **Canary** — 5-10% of traffic routes to new model, monitor for regressions")
        parts.append("  • **Blue/Green** — Full switchover with instant rollback capability")
        parts.append("  • **Rolling** — Gradual replacement across serving instances")

        parts.append("\n**3. Post-Deployment Monitoring:**")
        parts.append("  • Track prediction distribution (PSI for drift)")
        parts.append("  • Monitor feature distributions vs training data")
        parts.append("  • Set up alerts for latency spikes and error rate increases")
        parts.append("  • Schedule regular retraining (weekly/monthly based on drift)")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.7,
            "follow_ups": [
                "How do I set up shadow deployment?",
                "When should I retrain the model?",
                "How do I monitor for data drift?",
            ],
        }

    def _answer_drift(self, **kw) -> Dict:
        parts = []
        parts.append("**Data & Model Drift Monitoring:**\n")

        parts.append("**Types of Drift:**")
        parts.append("  • **Data Drift** — Input feature distributions change (most common)")
        parts.append("  • **Concept Drift** — Relationship between features and target changes")
        parts.append("  • **Label Drift** — Target distribution shifts")

        parts.append("\n**Detection Methods:**")
        parts.append("  • **PSI (Population Stability Index)** — compare feature distributions")
        parts.append("    PSI < 0.1: No drift | 0.1-0.25: Moderate | > 0.25: Significant")
        parts.append("  • **KS Test** — statistical test for distribution difference")
        parts.append("  • **Performance monitoring** — track F1/accuracy on recent predictions")
        parts.append("  • **Feature importance shift** — compare with training-time importances")

        parts.append("\n**When to Retrain:**")
        parts.append("  • PSI > 0.25 on any top-5 feature")
        parts.append("  • Performance drops >5% from baseline")
        parts.append("  • Prediction distribution shifts significantly")
        parts.append("  • Business rules or external factors change")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.7,
            "follow_ups": [
                "How do I calculate PSI?",
                "How often should I retrain?",
                "What monitoring alerts should I set up?",
            ],
        }

    def _answer_hyperparameters(self, **kw) -> Dict:
        ctx = kw["context"]
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        algo = (screen_ctx.get("algorithm") or frontend.get("algorithm", "")).lower()

        parts = []
        hp_guides = {
            "xgboost": {
                "name": "XGBoost",
                "params": {
                    "n_estimators": "100-1000 (start 300, use early stopping)",
                    "max_depth": "3-8 (start 6, lower for small data)",
                    "learning_rate": "0.01-0.3 (start 0.1, reduce with more estimators)",
                    "min_child_weight": "1-10 (higher = more conservative, reduce overfitting)",
                    "subsample": "0.6-1.0 (start 0.8, lower = more regularization)",
                    "colsample_bytree": "0.6-1.0 (start 0.8)",
                    "reg_alpha (L1)": "0-10 (start 0)",
                    "reg_lambda (L2)": "1-10 (start 1)",
                },
            },
            "random": {
                "name": "Random Forest",
                "params": {
                    "n_estimators": "100-500 (diminishing returns after 300)",
                    "max_depth": "None or 5-20 (None lets trees grow fully)",
                    "min_samples_split": "2-20 (higher = less overfitting)",
                    "min_samples_leaf": "1-10 (higher = smoother predictions)",
                    "max_features": "'sqrt' for classification, 'log2' or 0.33 for regression",
                },
            },
            "logistic": {
                "name": "Logistic Regression",
                "params": {
                    "C": "0.001-100 (inverse regularization, start 1.0)",
                    "penalty": "'l1' (feature selection), 'l2' (default), 'elasticnet'",
                    "solver": "'lbfgs' (default), 'saga' (for l1/elasticnet)",
                    "max_iter": "1000 (increase if convergence warning)",
                },
            },
            "lightgbm": {
                "name": "LightGBM",
                "params": {
                    "n_estimators": "100-1000 (use early stopping)",
                    "num_leaves": "31-127 (start 31, increase for complex data)",
                    "max_depth": "-1 to 8 (-1 = no limit)",
                    "learning_rate": "0.01-0.3 (start 0.1)",
                    "min_child_samples": "5-100 (higher for small data)",
                    "subsample": "0.6-1.0",
                    "colsample_bytree": "0.6-1.0",
                },
            },
        }

        matched = None
        for key, guide in hp_guides.items():
            if key in algo:
                matched = guide
                break

        if matched:
            parts.append(f"**{matched['name']} Hyperparameter Guide:**\n")
            for param, advice in matched["params"].items():
                parts.append(f"  • **{param}**: {advice}")
            parts.append("\n**Tuning Strategy:** Use Optuna (Bayesian) for 50-100 trials, "
                        "or RandomizedSearchCV for faster exploration.")
        else:
            parts.append("**General Hyperparameter Tuning Advice:**\n")
            parts.append("  1. Start with defaults — they're usually good baselines")
            parts.append("  2. Tune the most impactful parameters first (learning_rate, max_depth)")
            parts.append("  3. Use early stopping to determine optimal n_estimators")
            parts.append("  4. Bayesian optimization (Optuna) > Grid Search for efficiency")
            parts.append("  5. Always evaluate on a held-out test set, not CV score")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.8 if matched else 0.6,
            "follow_ups": [
                "How do I set up Optuna for tuning?",
                "What's the best search strategy?",
                "How many tuning iterations should I run?",
            ],
        }

    def _answer_cross_validation(self, **kw) -> Dict:
        ctx = kw["context"]
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)

        parts = []
        parts.append("**Cross-Validation Guide:**\n")

        if rows > 0:
            if rows < 200:
                rec_folds = "Leave-One-Out (LOOCV)"
                reason = f"with only {rows} samples, standard k-fold gives tiny validation sets"
            elif rows < 1000:
                rec_folds = "10-fold stratified"
                reason = f"{rows} samples benefits from more folds for stable estimates"
            elif rows < 50000:
                rec_folds = "5-fold stratified"
                reason = "standard choice, good bias-variance balance"
            else:
                rec_folds = "3-fold stratified (or single hold-out)"
                reason = f"with {rows:,} samples, even 3-fold gives large validation sets"

            parts.append(f"**Recommended for your data ({rows:,} rows):** {rec_folds}")
            parts.append(f"  Reason: {reason}")
        else:
            parts.append("Standard recommendation: **5-fold stratified** cross-validation")

        parts.append("\n**Key Rules:**")
        parts.append("  • **Always stratify** for classification (maintains class ratio per fold)")
        parts.append("  • **Shuffle** the data before splitting (unless time-series)")
        parts.append("  • For time-series: use **TimeSeriesSplit** (never random splits)")
        parts.append("  • **Repeated CV** (3×5-fold) gives more stable estimates at 3× the cost")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.85,
            "follow_ups": [
                "How do I handle time-series in CV?",
                "What if my CV scores vary a lot?",
                "Should I use nested CV for hyperparameter tuning?",
            ],
        }

    def _answer_encoding(self, **kw) -> Dict:
        ctx = kw["context"]
        profile = ctx.get("dataset_profile", {})
        features = ctx.get("feature_stats", {})
        cat_count = profile.get("categorical_count", 0)
        high_card = features.get("high_cardinality_categoricals", [])

        parts = []
        parts.append("**Categorical Encoding Guide:**\n")

        if cat_count > 0:
            parts.append(f"Your dataset has **{cat_count} categorical features**.\n")

        parts.append("**Encoding Methods (when to use each):**")
        parts.append("  • **One-Hot (pd.get_dummies)** — ≤10 unique values. Creates binary columns per category.")
        parts.append("  • **Target Encoding** — >10 unique values. Replaces category with mean target value. Use with regularization.")
        parts.append("  • **Ordinal Encoding** — Categories have natural order (low/medium/high).")
        parts.append("  • **Frequency Encoding** — Replace with category frequency. Simple, no leakage risk.")
        parts.append("  • **Hash Encoding** — Very high cardinality (1000+ values). Fixed-size output.")

        if high_card:
            parts.append(f"\n**High-Cardinality Columns (avoid one-hot):**")
            for col in high_card[:5]:
                parts.append(f"  • **{col['column']}**: {col['unique_count']} unique values → use target or hash encoding")

        parts.append("\n**Pro Tip:** LightGBM and CatBoost handle categoricals natively — "
                    "no encoding needed. Just specify `categorical_feature` parameter.")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.85,
            "follow_ups": [
                "What's the difference between target encoding and one-hot?",
                "How do I avoid leakage with target encoding?",
                "Should I combine rare categories?",
            ],
        }

    def _answer_scaling(self, **kw) -> Dict:
        ctx = kw["context"]
        features = ctx.get("feature_stats", {})
        skewed = features.get("skewed_features", [])

        parts = []
        parts.append("**Feature Scaling Guide:**\n")

        parts.append("**Scalers and When to Use:**")
        parts.append("  • **StandardScaler** — Normal-ish distributions. Subtracts mean, divides by std.")
        parts.append("  • **RobustScaler** — Data with outliers. Uses IQR instead of std.")
        parts.append("  • **MinMaxScaler** — Need 0-1 range (neural networks, image data).")
        parts.append("  • **PowerTransformer** — Skewed features. Applies Yeo-Johnson to normalize first.")

        if skewed:
            parts.append(f"\n⚠ **{len(skewed)} skewed features detected** — consider RobustScaler "
                        "or PowerTransformer instead of StandardScaler.")

        parts.append("\n**Algorithms That Need Scaling:**")
        parts.append("  ✅ Must scale: Logistic Regression, SVM, KNN, Neural Networks, PCA")
        parts.append("  ❌ No scaling needed: Decision Trees, Random Forest, XGBoost, LightGBM")

        parts.append("\n**Critical Rule:** Fit scaler on training data only, then transform "
                    "test data with the same scaler. Never fit on the full dataset (data leakage).")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.85,
            "follow_ups": [
                "Should I scale before or after splitting?",
                "What about scaling categorical features?",
                "How does scaling affect tree-based models?",
            ],
        }

    def _answer_ensemble(self, **kw) -> Dict:
        parts = []
        parts.append("**Ensemble Methods Guide:**\n")

        parts.append("**Types of Ensembles:**")
        parts.append("  • **Bagging** (Random Forest) — Reduces variance. Trains parallel models on data subsets.")
        parts.append("  • **Boosting** (XGBoost, LightGBM) — Reduces bias. Sequential models correct predecessors.")
        parts.append("  • **Stacking** — Different algorithms as base, meta-learner combines. Highest potential.")
        parts.append("  • **Voting** — Simple average or majority vote of diverse models. Low effort, decent gains.")
        parts.append("  • **Blending** — Like stacking but uses held-out set instead of CV for meta-features.")

        parts.append("\n**When to Ensemble:**")
        parts.append("  ✅ Different models make different errors (low correlation between predictions)")
        parts.append("  ✅ You need to squeeze 1-2% more performance")
        parts.append("  ❌ Single model already performs near-perfectly")
        parts.append("  ❌ Interpretability is more important than performance")

        parts.append("\n**Quick Win:** Soft voting of top 3 diverse models "
                    "(e.g., XGBoost + LightGBM + Logistic Regression) often adds 0.5-2% improvement.")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.7,
            "follow_ups": [
                "How do I set up stacking?",
                "Which models should I combine?",
                "Is ensemble worth the complexity?",
            ],
        }

    def _answer_leakage(self, **kw) -> Dict:
        ctx = kw["context"]
        features = ctx.get("feature_stats", {})
        correlations = ctx.get("correlations", {})
        id_cols = features.get("potential_id_columns", [])
        perfect_corr = [p for p in correlations.get("high_pairs", [])
                       if p.get("abs_correlation", 0) >= 0.95]

        parts = []
        parts.append("**Data Leakage Detection & Prevention:**\n")

        if id_cols or perfect_corr:
            parts.append("⚠ **Potential leakage signals found:**")
            for col in id_cols[:3]:
                parts.append(f"  • **{col['column']}** — appears to be an identifier")
            for pair in perfect_corr[:3]:
                parts.append(f"  • **{pair['feature1']}** ↔ **{pair['feature2']}** "
                            f"(r={pair['correlation']:.3f}) — may be derived/proxy")

        parts.append("\n**Common Leakage Sources:**")
        parts.append("  1. **ID columns** — model memorizes row identifiers")
        parts.append("  2. **Future features** — data not available at prediction time")
        parts.append("  3. **Target proxies** — features derived from the target (e.g., 'days_to_churn')")
        parts.append("  4. **Aggregated features** — using post-event aggregations")
        parts.append("  5. **Preprocessing on full data** — fitting scalers/encoders before splitting")

        parts.append("\n**Prevention:**")
        parts.append("  • Split data FIRST, then preprocess each set independently")
        parts.append("  • Ask for each feature: 'Would I have this at prediction time?'")
        parts.append("  • Suspiciously high training accuracy (>99%) is a leakage red flag")

        return {
            "answer": "\n".join(parts),
            "confidence": 0.8 if (id_cols or perfect_corr) else 0.7,
            "follow_ups": [
                "How do I verify if a feature is leaking?",
                "What if removing suspected features hurts accuracy?",
                "How do I prevent preprocessing leakage?",
            ],
        }

    def _answer_general_help(self, **kw) -> Dict:
        screen = kw["screen"]
        ctx = kw["context"]
        insights = kw["insights"]
        pipeline = ctx.get("pipeline_state", {})

        parts = []

        # Give screen-specific guidance
        screen_guidance = {
            "dashboard": "Start by uploading a dataset, then run EDA to understand your data before training.",
            "data": "Upload your dataset. The platform supports CSV files. Larger datasets may take a moment to process.",
            "eda": "Review the data quality score, check for missing values, and examine feature distributions. Address any critical issues before training.",
            "mlflow": "Select your target variable, choose features, pick algorithms, and configure training parameters. The agent will validate your choices.",
            "training": "Select your target variable, choose features, pick algorithms, and configure training. Run Phase 1-3 for basic training, or end-to-end for full pipeline.",
            "evaluation": "Review model metrics, check the confusion matrix, and examine feature importance. Focus on F1 and AUC-ROC for imbalanced data.",
            "registry": "Register your best model with a version tag. Only promote to production after shadow testing.",
            "deployment": "Start with shadow deployment to validate predictions match training performance. Then move to canary (5-10% traffic).",
            "predictions": "Run predictions on new data. Monitor prediction distributions for drift.",
            "monitoring": "Track prediction quality, feature drift (PSI), and latency. Set up alerts for anomalies.",
        }

        parts.append(f"**On the {screen.title()} screen:**")
        parts.append(screen_guidance.get(screen, "Ask me anything about your ML workflow."))

        # Summarize critical issues if any
        criticals = [i for i in insights if i.get("severity") == "critical"]
        if criticals:
            parts.append(f"\n🚫 **{len(criticals)} critical issue(s) found:**")
            for c in criticals[:3]:
                parts.append(f"  • {c['title']}")
            parts.append("Address these before proceeding.")

        # Suggest next phase
        next_phase = pipeline.get("next_recommended_phase")
        if next_phase:
            parts.append(f"\n**Next recommended step:** {next_phase}")

        follow_ups = {
            "eda": ["How is my data quality?", "Which features should I keep?", "Is there class imbalance?"],
            "training": ["Which algorithm should I use?", "What hyperparameters do you recommend?", "How should I split the data?"],
            "evaluation": ["Is my model good enough?", "Should I optimize the threshold?", "How do I improve recall?"],
        }

        return {
            "answer": "\n".join(parts),
            "confidence": 0.5,
            "follow_ups": follow_ups.get(screen, [
                "How is my data quality?",
                "Which algorithm should I use?",
                "What should I do next?",
            ]),
        }

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _interpret_metric_value(self, metric: str, value: float, all_metrics: Dict) -> str:
        """Provide contextual interpretation of a metric value."""
        if metric in ("accuracy", "acc"):
            if value > 0.95:
                return "Excellent, but verify this isn't due to class imbalance or leakage."
            elif value > 0.85:
                return "Good performance. Check F1 to confirm on both classes."
            elif value > 0.70:
                return "Moderate. Room for improvement via feature engineering or algorithm tuning."
            else:
                return "Low. Consider better features, different algorithm, or data quality issues."

        elif metric in ("f1", "f1_score"):
            if value > 0.80:
                return "Strong F1 — model is balanced on both precision and recall."
            elif value > 0.60:
                return "Moderate F1 — check precision vs recall to see which side needs work."
            elif value > 0.40:
                return "Low F1 — model is struggling. Check class imbalance and feature quality."
            else:
                return "Very low F1 — model may not be useful. Significant improvements needed."

        elif metric in ("precision",):
            if value > 0.85:
                return "High precision — few false positives. Check if recall is acceptable."
            elif value > 0.60:
                return "Moderate precision. Balance depends on false positive cost."

        elif metric in ("recall", "sensitivity"):
            if value > 0.85:
                return "High recall — catching most positives. Verify precision isn't too low."
            elif value > 0.60:
                return "Moderate recall. Consider lowering threshold if missing positives is costly."
            else:
                return "Low recall — missing many positive cases. Lower the threshold or use SMOTE."

        elif metric in ("auc", "auc_roc", "roc_auc"):
            if value > 0.90:
                return "Excellent discrimination. Model ranks positives well above negatives."
            elif value > 0.80:
                return "Good. Model has useful discriminative ability."
            elif value > 0.70:
                return "Fair. Model is better than random but has room to improve."
            elif value > 0.50:
                return "Barely above random chance. Model needs significant improvement."

        return ""
