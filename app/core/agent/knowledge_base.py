"""
ML Knowledge Base — Expert Decision Frameworks
================================================
Curated knowledge that powers the LLM reasoner with real ML expertise.
This is NOT a rules engine — it provides structured knowledge that the
LLM uses to generate contextual, nuanced advice.

Contents:
  1. Algorithm Selection Matrix
  2. Metric Interpretation Guide
  3. Common Pitfall Patterns
  4. Feature Engineering Playbook
  5. Production Readiness Checklist
  6. Debugging Decision Trees
"""

from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# 1. ALGORITHM SELECTION MATRIX
# ═══════════════════════════════════════════════════════════════════

ALGORITHM_PROFILES = {
    "LogisticRegression": {
        "family": "linear",
        "best_for": "Binary/multiclass classification with interpretable results",
        "strengths": [
            "Fast training and inference (<5ms)",
            "Highly interpretable coefficients",
            "Works well with small datasets (100+ rows)",
            "Natural probability calibration",
            "Low memory footprint",
        ],
        "weaknesses": [
            "Cannot capture non-linear relationships without feature engineering",
            "Sensitive to multicollinearity",
            "Requires feature scaling",
        ],
        "ideal_data_size": {"min": 50, "sweet_spot": "500-50K", "handles_large": True},
        "handles_missing": False,
        "handles_categorical": False,
        "interpretability": "high",
        "production_latency": "very_low",
        "overfitting_risk": "low",
        "hyperparameters": {
            "critical": ["C (regularization strength)", "penalty (L1/L2/ElasticNet)"],
            "tip": "Start with C=1.0 and L2 penalty. Use L1 for automatic feature selection.",
        },
    },
    "RandomForest": {
        "family": "ensemble_bagging",
        "best_for": "General-purpose classification and regression with mixed feature types",
        "strengths": [
            "Handles mixed types (numeric + categorical) naturally",
            "Robust to outliers and noisy features",
            "Built-in feature importance",
            "Minimal hyperparameter tuning needed",
            "Parallelizable training",
        ],
        "weaknesses": [
            "Slower inference than linear models (~50-200ms)",
            "Can overfit on small datasets with deep trees",
            "Black-box predictions",
            "Memory-heavy for large forests",
        ],
        "ideal_data_size": {"min": 200, "sweet_spot": "1K-500K", "handles_large": True},
        "handles_missing": False,
        "handles_categorical": False,
        "interpretability": "medium",
        "production_latency": "medium",
        "overfitting_risk": "low_to_medium",
        "hyperparameters": {
            "critical": ["n_estimators", "max_depth", "min_samples_leaf"],
            "tip": "Start with n_estimators=100, max_depth=None, min_samples_leaf=5. Increase n_estimators until OOB score plateaus.",
        },
    },
    "XGBoost": {
        "family": "ensemble_boosting",
        "best_for": "Structured/tabular data where maximum accuracy is the goal",
        "strengths": [
            "State-of-the-art accuracy on tabular data",
            "Built-in regularization (L1 + L2)",
            "Handles missing values natively",
            "Feature importance via SHAP values",
            "GPU acceleration available",
        ],
        "weaknesses": [
            "Requires careful hyperparameter tuning",
            "Can overfit on small datasets (<1000 rows)",
            "Slower training than Random Forest",
            "Sequential nature limits parallelization",
        ],
        "ideal_data_size": {"min": 1000, "sweet_spot": "5K-1M", "handles_large": True},
        "handles_missing": True,
        "handles_categorical": False,
        "interpretability": "medium",
        "production_latency": "medium",
        "overfitting_risk": "medium",
        "hyperparameters": {
            "critical": ["learning_rate", "max_depth", "n_estimators", "min_child_weight"],
            "tip": "Start with learning_rate=0.1, max_depth=6, n_estimators=100. Lower learning_rate and increase n_estimators for better generalization.",
        },
    },
    "LightGBM": {
        "family": "ensemble_boosting",
        "best_for": "Large datasets requiring fast training with high accuracy",
        "strengths": [
            "Fastest gradient boosting implementation",
            "Native categorical feature support",
            "Handles missing values",
            "Lower memory usage than XGBoost",
            "Leaf-wise growth produces better fits",
        ],
        "weaknesses": [
            "Can overfit on small datasets (more aggressive than XGBoost)",
            "Leaf-wise growth needs careful num_leaves tuning",
            "Less stable than XGBoost on noisy data",
        ],
        "ideal_data_size": {"min": 2000, "sweet_spot": "10K-10M", "handles_large": True},
        "handles_missing": True,
        "handles_categorical": True,
        "interpretability": "medium",
        "production_latency": "low_to_medium",
        "overfitting_risk": "medium_to_high",
        "hyperparameters": {
            "critical": ["num_leaves", "learning_rate", "n_estimators", "min_child_samples"],
            "tip": "num_leaves < 2^max_depth to prevent overfitting. Start with num_leaves=31, learning_rate=0.1.",
        },
    },
    "SVM": {
        "family": "kernel",
        "best_for": "Small to medium datasets, especially with clear margin of separation",
        "strengths": [
            "Effective in high-dimensional spaces",
            "Memory efficient (uses support vectors only)",
            "Versatile kernel functions (RBF, polynomial, linear)",
        ],
        "weaknesses": [
            "O(n²) to O(n³) training complexity — doesn't scale",
            "Sensitive to feature scaling",
            "No probability estimates without extra computation",
            "Difficult to interpret",
        ],
        "ideal_data_size": {"min": 100, "sweet_spot": "500-10K", "handles_large": False},
        "handles_missing": False,
        "handles_categorical": False,
        "interpretability": "low",
        "production_latency": "low_to_medium",
        "overfitting_risk": "low_to_medium",
        "hyperparameters": {
            "critical": ["C", "kernel", "gamma"],
            "tip": "Start with RBF kernel, C=1.0, gamma='scale'. Use GridSearch over C=[0.1, 1, 10, 100] and gamma=['scale', 'auto'].",
        },
    },
    "NaiveBayes": {
        "family": "probabilistic",
        "best_for": "Text classification, high-dimensional sparse data, quick baselines",
        "strengths": [
            "Extremely fast training and inference",
            "Works well with very small datasets",
            "Handles high dimensionality naturally",
            "Good probability calibration",
        ],
        "weaknesses": [
            "Assumes feature independence (usually violated)",
            "Cannot capture feature interactions",
            "Poor accuracy on complex non-linear problems",
        ],
        "ideal_data_size": {"min": 20, "sweet_spot": "100-100K", "handles_large": True},
        "handles_missing": False,
        "handles_categorical": True,
        "interpretability": "high",
        "production_latency": "very_low",
        "overfitting_risk": "very_low",
        "hyperparameters": {
            "critical": ["var_smoothing (Gaussian)", "alpha (Multinomial)"],
            "tip": "Use GaussianNB for continuous features, MultinomialNB for counts/frequencies, BernoulliNB for binary features.",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# 2. METRIC INTERPRETATION GUIDE
# ═══════════════════════════════════════════════════════════════════

METRIC_GUIDE = {
    "accuracy": {
        "what": "Percentage of correct predictions (both classes)",
        "when_useful": "Only when classes are balanced (45-55% split)",
        "when_misleading": "Imbalanced datasets — a model predicting all majority class gets high accuracy",
        "range": "0-100%. Random baseline = majority class percentage",
        "improve": "Better features, class weights, threshold tuning",
    },
    "f1_score": {
        "what": "Harmonic mean of Precision and Recall. Balances both.",
        "when_useful": "Imbalanced datasets, when both FP and FN matter",
        "when_misleading": "Rarely — F1 is robust for most classification tasks",
        "range": "0-100%. Higher = better balance of Precision and Recall",
        "improve": "Threshold tuning, SMOTE, class weights, better features",
    },
    "precision": {
        "what": "Of all predicted positives, how many are actually positive",
        "when_useful": "When false positives are costly (spam filter, fraud alert)",
        "when_misleading": "Can be 100% if model predicts only when very certain (but misses many)",
        "range": "0-100%. Trade-off with Recall",
        "improve": "Raise threshold to be more conservative",
    },
    "recall": {
        "what": "Of all actual positives, how many did the model catch",
        "when_useful": "When missing positives is costly (disease, fraud, churn)",
        "when_misleading": "Can be 100% if model predicts everything as positive",
        "range": "0-100%. Trade-off with Precision",
        "improve": "Lower threshold, SMOTE, class weights, ensemble methods",
    },
    "auc_roc": {
        "what": "Model's ability to distinguish classes across all thresholds",
        "when_useful": "Overall model quality assessment, comparing models",
        "when_misleading": "Severely imbalanced data (use PR-AUC instead)",
        "range": "0.5 (random) to 1.0 (perfect). >0.8 = good, >0.9 = excellent",
        "improve": "Better features, more complex models, ensemble methods",
    },
}


# ═══════════════════════════════════════════════════════════════════
# 3. COMMON PITFALL PATTERNS
# ═══════════════════════════════════════════════════════════════════

PITFALL_PATTERNS = {
    "accuracy_illusion": {
        "pattern": "High accuracy + Low F1",
        "diagnosis": "Class imbalance — model predicts majority class",
        "fix": "Use F1, class weights, SMOTE, and stratified CV",
    },
    "leakage_signal": {
        "pattern": "AUC > 0.99 or accuracy > 99%",
        "diagnosis": "Data leakage — feature derived from target",
        "fix": "Audit features for post-hoc variables, check temporal ordering",
    },
    "overfitting_classic": {
        "pattern": "Train score >> Test score (gap > 10%)",
        "diagnosis": "Model memorized training data",
        "fix": "More regularization, simpler model, more data, feature reduction",
    },
    "underfitting": {
        "pattern": "Both train and test scores are low (<60%)",
        "diagnosis": "Model too simple or features lack signal",
        "fix": "More complex model, better features, polynomial features",
    },
    "data_drift": {
        "pattern": "Production accuracy drops over weeks/months",
        "diagnosis": "Real-world data distribution shifted from training data",
        "fix": "Retrain on recent data, set up PSI monitoring, use online learning",
    },
}


# ═══════════════════════════════════════════════════════════════════
# 4. DECISION FRAMEWORK — ALGORITHM RECOMMENDER
# ═══════════════════════════════════════════════════════════════════

def recommend_algorithms(
    rows: int,
    cols: int,
    problem_type: str = "classification",
    has_categorical: bool = False,
    has_missing: bool = False,
    interpretability_needed: bool = False,
    latency_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """
    Recommend algorithms based on dataset characteristics.
    Returns prioritized list with reasoning.
    """
    recommendations = []

    # Tier 1: Always start with a strong baseline
    if problem_type == "classification":
        recommendations.append({
            "algorithm": "LogisticRegression",
            "tier": "baseline",
            "reason": "Fast, interpretable baseline. If it's within 2-3% of complex models, prefer it.",
            "priority": 1,
        })
    else:
        recommendations.append({
            "algorithm": "Ridge / Lasso Regression",
            "tier": "baseline",
            "reason": "Regularized linear regression baseline. Interpretable and fast.",
            "priority": 1,
        })

    # Tier 2: Based on data size
    if rows < 500:
        recommendations.extend([
            {"algorithm": "NaiveBayes", "tier": "small_data", "reason": "Works well with limited samples", "priority": 2},
            {"algorithm": "SVM (RBF)", "tier": "small_data", "reason": "Effective with few samples, high dimensions", "priority": 2},
        ])
    elif rows < 5000:
        recommendations.extend([
            {"algorithm": "RandomForest", "tier": "medium_data", "reason": "Robust ensemble, minimal tuning needed", "priority": 2},
            {"algorithm": "XGBoost", "tier": "medium_data", "reason": "Top accuracy with proper regularization", "priority": 2},
        ])
    else:
        recommendations.extend([
            {"algorithm": "LightGBM", "tier": "large_data", "reason": "Fastest boosting for large data, native categorical support", "priority": 2},
            {"algorithm": "XGBoost", "tier": "large_data", "reason": "State-of-the-art tabular accuracy", "priority": 2},
            {"algorithm": "RandomForest", "tier": "large_data", "reason": "Parallelizable, robust baseline", "priority": 3},
        ])

    # Tier 3: Special considerations
    if has_categorical and rows > 2000:
        recommendations.append({
            "algorithm": "CatBoost",
            "tier": "special",
            "reason": "Best native categorical feature handling, no encoding needed",
            "priority": 2,
        })

    if interpretability_needed:
        recommendations.append({
            "algorithm": "DecisionTree (max_depth=5)",
            "tier": "interpretable",
            "reason": "Fully transparent decision rules, perfect for stakeholder explanations",
            "priority": 3,
        })

    if latency_sensitive:
        recommendations = [r for r in recommendations if r["algorithm"] not in ("SVM (RBF)",)]
        recommendations.insert(0, {
            "algorithm": "LogisticRegression",
            "tier": "low_latency",
            "reason": "Sub-millisecond inference, ideal for real-time serving",
            "priority": 1,
        })

    # Deduplicate and sort
    seen = set()
    unique = []
    for r in recommendations:
        if r["algorithm"] not in seen:
            seen.add(r["algorithm"])
            unique.append(r)

    unique.sort(key=lambda x: x["priority"])
    return unique


# ═══════════════════════════════════════════════════════════════════
# 5. SCREEN-SPECIFIC KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════

SCREEN_KNOWLEDGE = {
    "eda": {
        "focus": "Understanding your data before modeling",
        "key_questions": [
            "What is the target distribution? (balanced vs imbalanced)",
            "Which features have the most missing data?",
            "Are there highly correlated features that should be removed?",
            "Do any features look like ID columns (potential leakage)?",
            "Are numeric features normally distributed or heavily skewed?",
        ],
        "common_mistakes": [
            "Skipping EDA and jumping to training",
            "Ignoring missing data patterns",
            "Not checking for duplicate rows",
            "Missing data leakage signals (near-perfect correlations)",
        ],
    },
    "training": {
        "focus": "Configuring and running model training",
        "key_questions": [
            "Is the test size appropriate (typically 20%)?",
            "Are you using stratified splitting for imbalanced targets?",
            "Have you selected the right evaluation metric (not just accuracy)?",
            "Is the scaling method appropriate for your data distribution?",
        ],
        "common_mistakes": [
            "Using accuracy as primary metric for imbalanced data",
            "Not scaling features before SVM or Logistic Regression",
            "Using too many features without regularization",
            "Not running cross-validation",
        ],
    },
    "evaluation": {
        "focus": "Understanding if your model is ready for production",
        "key_questions": [
            "Is the accuracy-F1 gap acceptable?",
            "Is the model overfitting (train >> test performance)?",
            "Are AUC scores suspiciously high (possible leakage)?",
            "What is the optimal threshold for your business context?",
            "Has the model passed production readiness criteria?",
        ],
        "common_mistakes": [
            "Relying solely on accuracy",
            "Ignoring overfitting signals",
            "Not tuning the classification threshold",
            "Not considering business costs of FP vs FN",
        ],
    },
    "deployment": {
        "focus": "Safely deploying models to production",
        "key_questions": [
            "Have you run shadow deployment first?",
            "Is monitoring in place before deployment?",
            "Do you have a rollback plan?",
            "What is the expected prediction latency?",
        ],
        "common_mistakes": [
            "Deploying without monitoring",
            "No rollback plan",
            "Not testing with real-world data distribution",
            "Ignoring latency requirements",
        ],
    },
    "monitoring": {
        "focus": "Keeping deployed models healthy",
        "key_questions": [
            "Is the prediction distribution stable?",
            "Has the input feature distribution shifted?",
            "Are error rates within acceptable bounds?",
            "When was the model last retrained?",
        ],
        "common_mistakes": [
            "Not monitoring for data drift",
            "Waiting for business metrics to drop before acting",
            "Not tracking prediction distribution over time",
            "Ignoring schema changes in input data",
        ],
    },
}


def get_algorithm_profile(name: str) -> Optional[Dict]:
    """Look up algorithm profile by partial name match."""
    name_lower = name.lower().replace(" ", "").replace("_", "")
    for key, profile in ALGORITHM_PROFILES.items():
        if key.lower().replace(" ", "") in name_lower or name_lower in key.lower().replace(" ", ""):
            return {**profile, "name": key}
    return None


def get_screen_knowledge(screen: str) -> Optional[Dict]:
    """Get knowledge for a specific screen."""
    return SCREEN_KNOWLEDGE.get(screen)


def get_metric_guide(metric: str) -> Optional[Dict]:
    """Get interpretation guide for a metric."""
    metric_lower = metric.lower().replace(" ", "_")
    for key, guide in METRIC_GUIDE.items():
        if key in metric_lower or metric_lower in key:
            return {**guide, "name": key}
    return None


def detect_pitfall(metrics: Dict[str, float]) -> List[Dict]:
    """Check metrics against known pitfall patterns."""
    detected = []

    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_score") or metrics.get("f1", 0)
    auc = metrics.get("auc_roc") or metrics.get("roc_auc", 0)
    train_score = metrics.get("train_score", 0)
    test_score = metrics.get("test_score", 0)

    if accuracy > 0 and f1 > 0 and (accuracy - f1) > 0.15:
        detected.append(PITFALL_PATTERNS["accuracy_illusion"])

    if auc > 0.99 or accuracy > 0.99:
        detected.append(PITFALL_PATTERNS["leakage_signal"])

    if train_score > 0 and test_score > 0 and (train_score - test_score) > 0.1:
        detected.append(PITFALL_PATTERNS["overfitting_classic"])

    if train_score > 0 and test_score > 0 and train_score < 0.6 and test_score < 0.6:
        detected.append(PITFALL_PATTERNS["underfitting"])

    return detected
