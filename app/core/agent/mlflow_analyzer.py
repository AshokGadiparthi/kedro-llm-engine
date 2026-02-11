"""
ML Flow Analyzer — Multi-Model Comparison & Training Intelligence
===================================================================
Dedicated service for analyzing ML Flow training results:

  • analyze_leaderboard()    — Full leaderboard analysis with rankings, overfitting, recommendations
  • compare_models()         — Head-to-head model comparison (model A vs model B)
  • explain_winner()         — Why the best model won (data-driven explanation)
  • smart_config()           — Pre-training: optimal configuration for dataset
  • detect_training_issues() — Find problems: failed algos, overfitting, underfitting
  • generate_suggested_questions() — Context-aware questions for the ML Flow AI chat

Data Source:
  The frontend passes model_results[] in the extra dict when calling /insights or /mlflow/compare.
  Each model_result has: {algorithm, train_score, test_score, accuracy, precision, recall, f1_score, roc_auc, training_time, status}

  Additionally, the registered model's data comes from the model_versions table:
  accuracy, precision, recall, f1_score, train_score, test_score, roc_auc, algorithm,
  hyperparameters, feature_names, feature_importances, confusion_matrix, training_config
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Overfitting thresholds
OVERFIT_MILD = 0.02       # 2% train-test gap = mild
OVERFIT_MODERATE = 0.05   # 5% = moderate
OVERFIT_SEVERE = 0.10     # 10% = severe

# Performance thresholds
SCORE_POOR = 0.60
SCORE_FAIR = 0.70
SCORE_GOOD = 0.80
SCORE_EXCELLENT = 0.90

# Spread thresholds
SPREAD_TIGHT = 0.01       # Top models within 1%
SPREAD_MODERATE = 0.05    # Within 5%

# Algorithm categories
LINEAR_MODELS = {
    "logisticregression", "ridgeclassifier", "sgdclassifier",
    "passiveaggressiveclassifier", "perceptron", "linearsvc",
    "linearregression", "ridge", "lasso", "elasticnet", "sgdregressor",
}
TREE_MODELS = {
    "decisiontreeclassifier", "randomforestclassifier", "extratreesclassifier",
    "gradientboostingclassifier", "adaboostclassifier", "baggingclassifier",
    "decisiontreeregressor", "randomforestregressor", "extratreesregressor",
    "gradientboostingregressor", "adaboostregressor", "baggingregressor",
}
BOOSTING_MODELS = {
    "gradientboostingclassifier", "adaboostclassifier",
    "xgbclassifier", "lgbmclassifier", "catboostclassifier",
    "gradientboostingregressor", "adaboostregressor",
    "xgbregressor", "lgbmregressor", "catboostregressor",
}
NAIVE_BAYES_MODELS = {
    "gaussiannb", "multinomialnb", "bernoullinb", "complementnb", "categoricalnb",
}
ENSEMBLE_MODELS = {
    "votingensemble", "votingclassifier", "votingregressor",
    "stackingclassifier", "stackingregressor",
}
SVM_MODELS = {
    "svc", "linearsvc", "svr", "linearsvr",
}
NEIGHBOR_MODELS = {
    "kneighborsclassifier", "kneighborsregressor",
}

INTERPRETABLE_MODELS = LINEAR_MODELS | {"decisiontreeclassifier", "decisiontreeregressor"}
COMPLEX_MODELS = BOOSTING_MODELS | ENSEMBLE_MODELS


def _algo_key(name: str) -> str:
    """Normalize algorithm name for comparison."""
    return name.lower().replace(" ", "").replace("_", "")


def _algo_category(name: str) -> str:
    """Get human-readable category for an algorithm."""
    key = _algo_key(name)
    if key in LINEAR_MODELS:
        return "linear"
    elif key in BOOSTING_MODELS:
        return "boosting"
    elif key in TREE_MODELS:
        return "tree"
    elif key in NAIVE_BAYES_MODELS:
        return "naive_bayes"
    elif key in ENSEMBLE_MODELS:
        return "ensemble"
    elif key in SVM_MODELS:
        return "svm"
    elif key in NEIGHBOR_MODELS:
        return "neighbor"
    return "other"


def _generalization_label(gap: float) -> str:
    """Human label for train-test gap."""
    if gap < OVERFIT_MILD:
        return "Excellent"
    elif gap < OVERFIT_MODERATE:
        return "Good"
    elif gap < OVERFIT_SEVERE:
        return "Moderate"
    else:
        return "Poor"


def _score_label(score: float) -> str:
    """Human label for model performance."""
    if score >= SCORE_EXCELLENT:
        return "excellent"
    elif score >= SCORE_GOOD:
        return "good"
    elif score >= SCORE_FAIR:
        return "fair"
    else:
        return "poor"


# ═══════════════════════════════════════════════════════════════
# CORE ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════

class MLFlowAnalyzer:
    """
    Analyzes ML Flow training results: multi-model comparison,
    overfitting detection, winner explanation, smart config.
    """

    # ─── LEADERBOARD ANALYSIS ─────────────────────────────────

    def analyze_leaderboard(self, model_results: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Full analysis of training leaderboard.

        Args:
            model_results: List of dicts with algorithm, train_score, test_score, etc.
            context: Optional dataset context (rows, cols, target, problem_type)

        Returns:
            Comprehensive analysis dict with rankings, comparisons, issues, recommendations.
        """
        if not model_results:
            return {"error": "No model results provided", "insights": []}

        # Normalize and filter valid results
        models = self._normalize_results(model_results)
        if not models:
            return {"error": "No valid model results after normalization", "insights": []}

        # Sort by test_score descending
        models.sort(key=lambda m: m.get("test_score", 0), reverse=True)

        winner = models[0]
        total = len(models)
        failed = [m for m in model_results if m.get("status") == "failed"]

        # Build analysis
        analysis = {
            "total_models_trained": total,
            "total_models_failed": len(failed),
            "failed_algorithms": [f.get("algorithm", "unknown") for f in failed],
            "winner": self._build_winner_summary(winner, models),
            "rankings": self._build_rankings(models),
            "overfitting_analysis": self._analyze_overfitting(models),
            "performance_spread": self._analyze_spread(models),
            "category_analysis": self._analyze_by_category(models),
            "recommendations": self._build_recommendations(models, context),
            "insights": [],  # Will be populated by rules
            "suggested_questions": self._generate_post_training_questions(winner, models, context),
        }

        # Generate narrative insights
        analysis["insights"] = self._generate_narrative_insights(analysis, context)

        return analysis

    # ─── NORMALIZE RESULTS ────────────────────────────────────

    def _normalize_results(self, raw: List[Dict]) -> List[Dict]:
        """Normalize model results to consistent format."""
        normalized = []
        for r in raw:
            if r.get("status") == "failed":
                continue

            algo = r.get("algorithm", r.get("name", "Unknown"))
            train = r.get("train_score", r.get("training_score"))
            test = r.get("test_score", r.get("accuracy", r.get("score")))

            if test is None:
                continue

            # Convert percentage to ratio if needed
            if isinstance(train, (int, float)) and train > 1:
                train = train / 100.0
            if isinstance(test, (int, float)) and test > 1:
                test = test / 100.0

            model = {
                "algorithm": algo,
                "train_score": float(train) if train is not None else None,
                "test_score": float(test),
                "gap": abs(float(train) - float(test)) if train is not None else None,
                "precision": self._to_float(r.get("precision")),
                "recall": self._to_float(r.get("recall")),
                "f1_score": self._to_float(r.get("f1_score")),
                "roc_auc": self._to_float(r.get("roc_auc")),
                "training_time": self._to_float(r.get("training_time", r.get("training_time_seconds"))),
                "category": _algo_category(algo),
                "generalization": _generalization_label(abs(float(train) - float(test))) if train else "Unknown",
            }
            normalized.append(model)

        return normalized

    @staticmethod
    def _to_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            v = float(val)
            return v if not math.isnan(v) else None
        except (ValueError, TypeError):
            return None

    # ─── WINNER SUMMARY ───────────────────────────────────────

    def _build_winner_summary(self, winner: Dict, all_models: List[Dict]) -> Dict:
        """Build detailed summary of the winning model."""
        algo = winner["algorithm"]
        test = winner["test_score"]
        train = winner.get("train_score")
        gap = winner.get("gap")
        category = winner["category"]

        # Runner-up comparison
        runner_up = all_models[1] if len(all_models) > 1 else None
        margin = (test - runner_up["test_score"]) if runner_up else 0

        summary = {
            "algorithm": algo,
            "test_score": test,
            "test_score_pct": f"{test * 100:.2f}%",
            "train_score": train,
            "generalization_gap": gap,
            "generalization_label": winner["generalization"],
            "category": category,
            "margin_over_runner_up": margin,
            "margin_pct": f"{margin * 100:.2f}%",
            "runner_up": runner_up["algorithm"] if runner_up else None,
            "score_label": _score_label(test),
            "is_interpretable": _algo_key(algo) in INTERPRETABLE_MODELS,
            "is_complex": _algo_key(algo) in COMPLEX_MODELS,
        }

        # Precision/Recall trade-off
        if winner.get("precision") and winner.get("recall"):
            p, r = winner["precision"], winner["recall"]
            summary["precision"] = p
            summary["recall"] = r
            summary["precision_recall_balance"] = "balanced" if abs(p - r) < 0.05 else (
                "precision-heavy" if p > r else "recall-heavy"
            )

        return summary

    # ─── RANKINGS ─────────────────────────────────────────────

    def _build_rankings(self, models: List[Dict]) -> List[Dict]:
        """Build ranked list with relative analysis."""
        rankings = []
        best_score = models[0]["test_score"] if models else 0

        for rank, m in enumerate(models, 1):
            entry = {
                "rank": rank,
                "algorithm": m["algorithm"],
                "test_score": m["test_score"],
                "test_score_pct": f"{m['test_score'] * 100:.2f}%",
                "train_score": m.get("train_score"),
                "gap": m.get("gap"),
                "generalization": m["generalization"],
                "category": m["category"],
                "distance_from_best": best_score - m["test_score"],
                "distance_pct": f"{(best_score - m['test_score']) * 100:.2f}%",
            }
            if m.get("roc_auc"):
                entry["roc_auc"] = m["roc_auc"]
            rankings.append(entry)

        return rankings

    # ─── OVERFITTING ANALYSIS ─────────────────────────────────

    def _analyze_overfitting(self, models: List[Dict]) -> Dict:
        """Analyze overfitting patterns across all models."""
        gaps = [(m["algorithm"], m["gap"], m["category"]) for m in models if m.get("gap") is not None]
        if not gaps:
            return {"status": "unknown", "message": "No train/test gap data available"}

        avg_gap = sum(g[1] for g in gaps) / len(gaps)
        max_gap = max(gaps, key=lambda x: x[1])
        min_gap = min(gaps, key=lambda x: x[1])

        severe = [(algo, gap) for algo, gap, _ in gaps if gap >= OVERFIT_SEVERE]
        moderate = [(algo, gap) for algo, gap, _ in gaps if OVERFIT_MODERATE <= gap < OVERFIT_SEVERE]
        mild = [(algo, gap) for algo, gap, _ in gaps if OVERFIT_MILD <= gap < OVERFIT_MODERATE]

        # Category-level analysis
        tree_gaps = [gap for _, gap, cat in gaps if cat in ("tree", "boosting")]
        linear_gaps = [gap for _, gap, cat in gaps if cat == "linear"]

        result = {
            "average_gap": avg_gap,
            "average_gap_pct": f"{avg_gap * 100:.2f}%",
            "worst_offender": {"algorithm": max_gap[0], "gap": max_gap[1], "gap_pct": f"{max_gap[1] * 100:.2f}%"},
            "best_generalizer": {"algorithm": min_gap[0], "gap": min_gap[1], "gap_pct": f"{min_gap[1] * 100:.2f}%"},
            "severe_overfitting": [{"algorithm": a, "gap_pct": f"{g * 100:.2f}%"} for a, g in severe],
            "moderate_overfitting": [{"algorithm": a, "gap_pct": f"{g * 100:.2f}%"} for a, g in moderate],
            "count_severe": len(severe),
            "count_moderate": len(moderate),
            "count_mild": len(mild),
        }

        # Tree vs Linear comparison
        if tree_gaps and linear_gaps:
            tree_avg = sum(tree_gaps) / len(tree_gaps)
            linear_avg = sum(linear_gaps) / len(linear_gaps)
            result["tree_vs_linear"] = {
                "tree_avg_gap": f"{tree_avg * 100:.2f}%",
                "linear_avg_gap": f"{linear_avg * 100:.2f}%",
                "trees_overfit_more": tree_avg > linear_avg,
                "difference_pct": f"{abs(tree_avg - linear_avg) * 100:.2f}%",
            }

        # Overall status
        if len(severe) > 0:
            result["status"] = "concerning"
            result["message"] = f"{len(severe)} model(s) severely overfitting (>{OVERFIT_SEVERE*100}% gap)"
        elif len(moderate) > 0:
            result["status"] = "moderate"
            result["message"] = f"{len(moderate)} model(s) with moderate overfitting"
        elif avg_gap < OVERFIT_MILD:
            result["status"] = "excellent"
            result["message"] = "All models generalize well"
        else:
            result["status"] = "good"
            result["message"] = "Minor overfitting in some models, generally healthy"

        return result

    # ─── PERFORMANCE SPREAD ───────────────────────────────────

    def _analyze_spread(self, models: List[Dict]) -> Dict:
        """Analyze how close/far models are from each other."""
        scores = [m["test_score"] for m in models]
        if len(scores) < 2:
            return {"status": "insufficient_data"}

        best = max(scores)
        worst = min(scores)
        spread = best - worst

        # Top-3 spread (more meaningful than full spread)
        top3 = sorted(scores, reverse=True)[:3]
        top3_spread = top3[0] - top3[-1] if len(top3) >= 2 else 0

        result = {
            "total_spread": spread,
            "total_spread_pct": f"{spread * 100:.2f}%",
            "top3_spread": top3_spread,
            "top3_spread_pct": f"{top3_spread * 100:.2f}%",
            "best_score": best,
            "worst_score": worst,
        }

        if top3_spread < SPREAD_TIGHT:
            result["status"] = "tight"
            result["message"] = (
                f"Top models are within {top3_spread * 100:.1f}% of each other. "
                f"Performance differences are NOT statistically significant — "
                f"choose the simplest/fastest model."
            )
            result["recommendation"] = "prefer_simpler"
        elif top3_spread < SPREAD_MODERATE:
            result["status"] = "moderate"
            result["message"] = (
                f"Top models spread across {top3_spread * 100:.1f}%. "
                f"The winner has a meaningful but small advantage."
            )
            result["recommendation"] = "winner_justified"
        else:
            result["status"] = "wide"
            result["message"] = (
                f"Large performance gap ({top3_spread * 100:.1f}%) between top models. "
                f"The winner clearly outperforms alternatives."
            )
            result["recommendation"] = "clear_winner"

        return result

    # ─── CATEGORY ANALYSIS ────────────────────────────────────

    def _analyze_by_category(self, models: List[Dict]) -> Dict:
        """Group models by category and compare performance."""
        categories = {}
        for m in models:
            cat = m["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(m)

        result = {}
        for cat, group in categories.items():
            scores = [m["test_score"] for m in group]
            gaps = [m["gap"] for m in group if m.get("gap") is not None]
            best = max(group, key=lambda x: x["test_score"])

            result[cat] = {
                "count": len(group),
                "best_algorithm": best["algorithm"],
                "best_score": best["test_score"],
                "best_score_pct": f"{best['test_score'] * 100:.2f}%",
                "avg_score": sum(scores) / len(scores),
                "avg_gap": sum(gaps) / len(gaps) if gaps else None,
                "algorithms": [m["algorithm"] for m in group],
            }

        return result

    # ─── MODEL COMPARISON ─────────────────────────────────────

    def compare_models(self, model_a: Dict, model_b: Dict) -> Dict:
        """
        Detailed head-to-head comparison of two models.

        Args:
            model_a, model_b: Model result dicts with algorithm, metrics, etc.

        Returns:
            Comparison analysis with winner, advantages, trade-offs.
        """
        a_name = model_a.get("algorithm", "Model A")
        b_name = model_b.get("algorithm", "Model B")

        a_test = float(model_a.get("test_score", model_a.get("accuracy", 0)))
        b_test = float(model_b.get("test_score", model_b.get("accuracy", 0)))
        a_train = self._to_float(model_a.get("train_score"))
        b_train = self._to_float(model_b.get("train_score"))

        score_diff = a_test - b_test
        winner = a_name if score_diff >= 0 else b_name

        comparison = {
            "model_a": a_name,
            "model_b": b_name,
            "winner": winner,
            "score_difference": abs(score_diff),
            "score_difference_pct": f"{abs(score_diff) * 100:.2f}%",
            "statistically_significant": abs(score_diff) > 0.01,
            "metrics_comparison": {},
            "advantages": {a_name: [], b_name: []},
            "recommendation": "",
        }

        # Compare each metric
        for metric in ["test_score", "precision", "recall", "f1_score", "roc_auc"]:
            a_val = self._to_float(model_a.get(metric))
            b_val = self._to_float(model_b.get(metric))
            if a_val is not None and b_val is not None:
                comparison["metrics_comparison"][metric] = {
                    a_name: a_val, b_name: b_val,
                    "winner": a_name if a_val >= b_val else b_name,
                    "difference": abs(a_val - b_val),
                }
                better = a_name if a_val > b_val else b_name
                if abs(a_val - b_val) > 0.005:
                    comparison["advantages"][better].append(f"Better {metric} ({abs(a_val - b_val) * 100:.1f}% higher)")

        # Overfitting comparison
        if a_train is not None and b_train is not None:
            a_gap = abs(a_train - a_test)
            b_gap = abs(b_train - b_test)
            better_gen = a_name if a_gap < b_gap else b_name
            comparison["overfitting"] = {
                a_name: {"gap": a_gap, "label": _generalization_label(a_gap)},
                b_name: {"gap": b_gap, "label": _generalization_label(b_gap)},
                "better_generalizer": better_gen,
            }
            if abs(a_gap - b_gap) > 0.01:
                comparison["advantages"][better_gen].append(
                    f"Better generalization ({_generalization_label(min(a_gap, b_gap))} vs {_generalization_label(max(a_gap, b_gap))})"
                )

        # Complexity comparison
        a_cat = _algo_category(a_name)
        b_cat = _algo_category(b_name)
        a_complex = _algo_key(a_name) in COMPLEX_MODELS
        b_complex = _algo_key(b_name) in COMPLEX_MODELS

        if a_complex != b_complex:
            simpler = a_name if not a_complex else b_name
            comparison["advantages"][simpler].append("Simpler model (easier to interpret and deploy)")

        # Build recommendation
        if abs(score_diff) < 0.01:
            simpler = a_name if not a_complex else b_name
            comparison["recommendation"] = (
                f"Performance is nearly identical ({abs(score_diff) * 100:.2f}% difference). "
                f"Consider {simpler} for its simplicity and interpretability."
            )
        else:
            comparison["recommendation"] = (
                f"{winner} outperforms by {abs(score_diff) * 100:.2f}%. "
                f"{'This is a meaningful difference.' if abs(score_diff) > 0.02 else 'The difference is small but consistent.'}"
            )

        return comparison

    # ─── WINNER EXPLANATION ───────────────────────────────────

    def explain_winner(self, models: List[Dict], context: Optional[Dict] = None) -> Dict:
        """
        Generate a human-readable explanation of why the winner won.

        Returns explanation text and supporting data.
        """
        if not models:
            return {"explanation": "No models to analyze."}

        normalized = self._normalize_results(models)
        if not normalized:
            return {"explanation": "No valid model results."}

        normalized.sort(key=lambda m: m["test_score"], reverse=True)
        winner = normalized[0]
        algo = winner["algorithm"]
        test = winner["test_score"]
        cat = winner["category"]

        reasons = []

        # 1. Absolute performance
        reasons.append(f"Achieved the highest test accuracy of {test * 100:.2f}%")

        # 2. Margin over runner-up
        if len(normalized) > 1:
            runner = normalized[1]
            margin = test - runner["test_score"]
            if margin > 0.02:
                reasons.append(
                    f"Outperformed runner-up ({runner['algorithm']}) by {margin * 100:.2f}% — a clear advantage"
                )
            elif margin > 0.005:
                reasons.append(
                    f"Edged out {runner['algorithm']} by {margin * 100:.2f}%"
                )
            else:
                reasons.append(
                    f"Essentially tied with {runner['algorithm']} (only {margin * 100:.2f}% difference)"
                )

        # 3. Generalization
        gap = winner.get("gap")
        if gap is not None:
            if gap < OVERFIT_MILD:
                reasons.append(f"Excellent generalization (train-test gap of only {gap * 100:.2f}%)")
            elif gap < OVERFIT_MODERATE:
                reasons.append(f"Good generalization with {gap * 100:.2f}% train-test gap")
            else:
                reasons.append(f"⚠️ Shows overfitting with {gap * 100:.2f}% train-test gap")

        # 4. Algorithm family strength
        if cat == "boosting":
            reasons.append("Gradient boosting models excel at capturing non-linear patterns and feature interactions")
        elif cat == "linear":
            reasons.append("Linear models suggest the data has strong linear relationships — a good sign for interpretability")
        elif cat == "ensemble":
            reasons.append("Ensemble combines multiple models' strengths, reducing individual model weaknesses")

        # 5. Context-aware reasoning
        if context:
            rows = context.get("rows", 0)
            cols = context.get("columns", 0)
            if cat == "boosting" and rows < 1000:
                reasons.append(f"Note: Boosting models can overfit on small datasets ({rows} rows) — validate with cross-validation")
            if cat == "linear" and cols > rows:
                reasons.append("Linear models are a smart choice for high-dimensional data (more features than samples)")

        return {
            "winner": algo,
            "score": test,
            "score_pct": f"{test * 100:.2f}%",
            "category": cat,
            "reasons": reasons,
            "explanation": f"{algo} won with {test * 100:.2f}% accuracy. " + " ".join(reasons[1:]),
        }

    # ─── SMART CONFIG ─────────────────────────────────────────

    def smart_config(self, dataset_profile: Dict, recommendations: Dict) -> Dict:
        """
        Wrap all recommend_* outputs into a single smart configuration.

        Args:
            dataset_profile: rows, columns, dtypes, missing_pct, etc.
            recommendations: Output from orchestrator's recommend_* methods

        Returns:
            Ready-to-use config for ML Flow training.
        """
        rows = dataset_profile.get("rows", 0)
        cols = dataset_profile.get("columns", 0)
        target = dataset_profile.get("target_column")
        problem_type = dataset_profile.get("problem_type", "classification")

        config = {
            "problem_type": problem_type,
            "target_column": target,
            "recommended_algorithms": [],
            "scaling_method": "standard",
            "encoding_strategy": "auto",
            "cv_folds": 5,
            "test_size": 0.2,
            "stratified": problem_type == "classification",
            "preprocessing": {
                "handle_missing": True,
                "handle_outliers": True,
                "encode_categoricals": True,
            },
            "estimated_training_time": "~2-5 minutes",
            "rationale": [],
        }

        # Merge recommendations
        if recommendations.get("algorithms"):
            algos = recommendations["algorithms"]
            config["recommended_algorithms"] = [
                {"algorithm": a.get("algorithm", a.get("name")), "tier": a.get("tier", "recommended"),
                 "reason": a.get("reason", "")}
                for a in algos[:6]
            ]

        if recommendations.get("scaling"):
            sc = recommendations["scaling"]
            if isinstance(sc, dict):
                config["scaling_method"] = sc.get("method", "standard")
            elif isinstance(sc, str):
                config["scaling_method"] = sc

        if recommendations.get("cv_strategy"):
            cv = recommendations["cv_strategy"]
            if isinstance(cv, dict):
                config["cv_folds"] = cv.get("folds", 5)
                config["test_size"] = cv.get("test_size", 0.2)

        if recommendations.get("encoding"):
            config["encoding_strategy"] = recommendations["encoding"]

        # Data-aware adjustments
        if rows < 500:
            config["cv_folds"] = 3
            config["rationale"].append(f"Reduced CV folds to 3 (small dataset: {rows} rows)")
        elif rows > 50000:
            config["cv_folds"] = 3
            config["rationale"].append(f"Reduced CV folds to 3 (large dataset: {rows} rows — speed optimization)")

        if rows < 1000:
            config["test_size"] = 0.15
            config["rationale"].append("Reduced test size to 15% to maximize training data")

        if cols > 50:
            config["rationale"].append(f"High-dimensional data ({cols} features) — feature selection recommended before training")

        if not config["rationale"]:
            config["rationale"].append("Standard configuration — dataset shape is typical for ML training")

        return config

    # ─── TRAINING ISSUES DETECTION ────────────────────────────

    def detect_training_issues(self, model_results: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """
        Detect issues in training results.

        Returns list of issues with severity, description, action.
        """
        issues = []
        models = self._normalize_results(model_results)
        failed = [m for m in model_results if m.get("status") == "failed"]

        if not models:
            issues.append({
                "type": "no_models",
                "severity": "critical",
                "title": "No models trained successfully",
                "description": f"All {len(model_results)} algorithms failed.",
                "action": "Check data preprocessing — ensure no NaN/Inf values, proper encoding, and scaling."
            })
            return issues

        best = models[0]["test_score"]
        worst = models[-1]["test_score"]

        # Issue: Failed algorithms
        if failed:
            fail_algos = [f.get("algorithm", "unknown") for f in failed]
            nb_failures = [a for a in fail_algos if _algo_key(a) in NAIVE_BAYES_MODELS]
            if nb_failures:
                issues.append({
                    "type": "nb_failure",
                    "severity": "info",
                    "title": f"{len(nb_failures)} Naive Bayes algorithm(s) failed",
                    "description": (
                        f"{', '.join(nb_failures)} failed due to negative values in scaled data. "
                        "This is expected when using StandardScaler (which produces negative values) "
                        "with Naive Bayes models that require non-negative input."
                    ),
                    "action": "Not a problem — these models are incompatible with standard scaling. The other algorithms compensate."
                })
            non_nb = [a for a in fail_algos if _algo_key(a) not in NAIVE_BAYES_MODELS]
            if non_nb:
                issues.append({
                    "type": "unexpected_failure",
                    "severity": "warning",
                    "title": f"{len(non_nb)} algorithm(s) failed unexpectedly",
                    "description": f"Failed: {', '.join(non_nb)}",
                    "action": "Check backend logs for error details. May indicate data issues."
                })

        # Issue: All models underperforming
        if best < SCORE_POOR:
            issues.append({
                "type": "underfitting",
                "severity": "critical",
                "title": f"All models underperforming (best: {best * 100:.1f}%)",
                "description": (
                    "No algorithm achieved above 60% accuracy. This suggests fundamental data issues: "
                    "wrong target column, insufficient signal in features, or data quality problems."
                ),
                "action": "Return to EDA — check feature correlations with target, examine data quality, consider feature engineering."
            })
        elif best < SCORE_FAIR:
            issues.append({
                "type": "low_performance",
                "severity": "warning",
                "title": f"Best model achieves only {best * 100:.1f}% accuracy",
                "description": "Performance is below 70%, suggesting the features may not have enough predictive power.",
                "action": "Consider feature engineering: create interaction features, try different encodings, or add external data."
            })

        # Issue: Severe overfitting in winner
        winner_gap = models[0].get("gap")
        if winner_gap and winner_gap >= OVERFIT_SEVERE:
            issues.append({
                "type": "winner_overfitting",
                "severity": "warning",
                "title": f"Best model is overfitting ({models[0]['algorithm']})",
                "description": (
                    f"Train-test gap of {winner_gap * 100:.1f}% indicates the model memorizes training data "
                    f"rather than learning patterns."
                ),
                "action": (
                    "Consider: (1) More regularization, (2) Reduce model complexity, "
                    "(3) Get more training data, (4) Use the 2nd-best model with better generalization."
                )
            })

        # Issue: Ensemble didn't improve
        ensemble_models = [m for m in models if m["category"] == "ensemble"]
        non_ensemble = [m for m in models if m["category"] != "ensemble"]
        if ensemble_models and non_ensemble:
            best_single = non_ensemble[0]["test_score"]
            best_ensemble = ensemble_models[0]["test_score"]
            if best_ensemble < best_single:
                issues.append({
                    "type": "ensemble_underperform",
                    "severity": "info",
                    "title": "Ensemble didn't outperform best single model",
                    "description": (
                        f"VotingEnsemble ({best_ensemble * 100:.2f}%) scored below "
                        f"{non_ensemble[0]['algorithm']} ({best_single * 100:.2f}%). "
                        "This can happen when the top models are too similar or when one model dominates."
                    ),
                    "action": "Try stacking or blending with diverse model types for better ensemble performance."
                })

        # Issue: Linear model competitive with complex models
        linear_models = [m for m in models if m["category"] == "linear"]
        complex_models = [m for m in models if m["category"] in ("boosting", "tree")]
        if linear_models and complex_models:
            best_linear = linear_models[0]["test_score"]
            best_complex = complex_models[0]["test_score"]
            if best_linear >= best_complex - 0.01:
                issues.append({
                    "type": "linear_competitive",
                    "severity": "info",
                    "title": "Linear model matches complex models",
                    "description": (
                        f"{linear_models[0]['algorithm']} ({best_linear * 100:.2f}%) is competitive with "
                        f"{complex_models[0]['algorithm']} ({best_complex * 100:.2f}%). "
                        "This suggests the data has strong linear patterns."
                    ),
                    "action": (
                        "Consider deploying the linear model — it's faster, more interpretable, "
                        "and less prone to overfitting."
                    )
                })

        # Issue: Top models very close
        if len(models) >= 3:
            top3_spread = models[0]["test_score"] - models[2]["test_score"]
            if top3_spread < SPREAD_TIGHT:
                issues.append({
                    "type": "tight_spread",
                    "severity": "info",
                    "title": f"Top 3 models within {top3_spread * 100:.2f}% of each other",
                    "description": "The performance difference is likely NOT statistically significant.",
                    "action": "Choose based on secondary criteria: speed, interpretability, deployment simplicity."
                })

        return issues

    # ─── RECOMMENDATIONS ──────────────────────────────────────

    def _build_recommendations(self, models: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """Generate actionable recommendations based on training results."""
        recs = []
        if not models:
            return recs

        winner = models[0]
        best_score = winner["test_score"]

        # Recommendation: Deploy or Iterate?
        if best_score >= SCORE_GOOD:
            recs.append({
                "type": "deploy",
                "priority": "high",
                "title": "Ready for Deployment",
                "description": (
                    f"{winner['algorithm']} at {best_score * 100:.1f}% is production-ready. "
                    "Deploy and monitor performance on real data."
                ),
            })
        elif best_score >= SCORE_FAIR:
            recs.append({
                "type": "iterate",
                "priority": "high",
                "title": "Good Start — Room for Improvement",
                "description": (
                    f"{best_score * 100:.1f}% accuracy is decent but can likely be improved. "
                    "Try: (1) Feature engineering, (2) Hyperparameter tuning, (3) Ensemble methods."
                ),
            })
        else:
            recs.append({
                "type": "rethink",
                "priority": "critical",
                "title": "Revisit Data & Features",
                "description": (
                    f"Best accuracy of {best_score * 100:.1f}% suggests fundamental issues. "
                    "Go back to EDA and check: feature correlations with target, data quality, "
                    "whether the problem is solvable with available features."
                ),
            })

        # Recommendation: Simplify if linear is competitive
        linear = [m for m in models if m["category"] == "linear"]
        if linear and linear[0]["test_score"] >= best_score - 0.015:
            recs.append({
                "type": "simplify",
                "priority": "medium",
                "title": "Consider Simpler Model",
                "description": (
                    f"{linear[0]['algorithm']} achieves {linear[0]['test_score'] * 100:.1f}%, "
                    f"nearly matching the winner. It's faster to deploy, easier to explain, "
                    "and less likely to break in production."
                ),
            })

        # Recommendation: Hyperparameter tuning if gap is large
        if winner.get("gap") and winner["gap"] > OVERFIT_MODERATE:
            recs.append({
                "type": "tune",
                "priority": "high",
                "title": "Hyperparameter Tuning Needed",
                "description": (
                    f"{winner['algorithm']} has {winner['gap'] * 100:.1f}% overfitting gap. "
                    "Tune regularization parameters to reduce overfitting before deploying."
                ),
            })

        return recs

    # ─── NARRATIVE INSIGHTS ───────────────────────────────────

    def _generate_narrative_insights(self, analysis: Dict, context: Optional[Dict] = None) -> List[str]:
        """Generate human-readable narrative insights."""
        insights = []
        winner = analysis.get("winner", {})
        spread = analysis.get("performance_spread", {})
        overfit = analysis.get("overfitting_analysis", {})

        # 1. Winner summary
        algo = winner.get("algorithm", "Unknown")
        score = winner.get("test_score_pct", "N/A")
        margin = winner.get("margin_pct", "0%")
        insights.append(
            f"🏆 {algo} wins with {score} accuracy, "
            f"{margin} ahead of {winner.get('runner_up', 'the field')}."
        )

        # 2. Spread insight
        spread_status = spread.get("status", "")
        if spread_status == "tight":
            insights.append(
                f"📊 Top models are extremely close ({spread.get('top3_spread_pct', 'N/A')} spread). "
                "Choose based on speed and interpretability, not just accuracy."
            )
        elif spread_status == "wide":
            insights.append(
                f"📊 Clear performance hierarchy — {spread.get('top3_spread_pct', 'N/A')} gap between top models."
            )

        # 3. Overfitting insight
        overfit_status = overfit.get("status", "")
        if overfit_status == "excellent":
            insights.append("✅ All models generalize well — no overfitting concerns.")
        elif overfit_status == "concerning":
            worst = overfit.get("worst_offender", {})
            insights.append(
                f"⚠️ {worst.get('algorithm', 'Unknown')} shows {worst.get('gap_pct', 'N/A')} overfitting gap. "
                "Consider regularization or more training data."
            )

        # 4. Category insight
        cats = analysis.get("category_analysis", {})
        if "linear" in cats and "boosting" in cats:
            lin_score = cats["linear"]["best_score"]
            boost_score = cats["boosting"]["best_score"]
            if lin_score >= boost_score - 0.01:
                insights.append(
                    f"💡 Linear models match boosting ({lin_score * 100:.1f}% vs {boost_score * 100:.1f}%) — "
                    "your data has strong linear patterns. Simpler models may be the best choice."
                )

        # 5. Failed algorithms insight
        total_failed = analysis.get("total_models_failed", 0)
        if total_failed > 0:
            insights.append(
                f"ℹ️ {total_failed} algorithm(s) failed: {', '.join(analysis.get('failed_algorithms', []))}. "
                "See training issues for details."
            )

        return insights

    # ─── SUGGESTED QUESTIONS ──────────────────────────────────

    def _generate_post_training_questions(self, winner: Dict, models: List[Dict],
                                          context: Optional[Dict] = None) -> List[str]:
        """Generate context-aware suggested questions for the AI chat."""
        questions = []
        algo = winner.get("algorithm", "the best model")
        score = winner.get("test_score", 0)

        questions.append(f"Why did {algo} outperform the other algorithms?")
        questions.append(f"Is {score * 100:.1f}% accuracy good enough to deploy?")

        if winner.get("gap") and winner["gap"] > OVERFIT_MILD:
            questions.append(f"How can I reduce overfitting in {algo}?")

        if len(models) > 1:
            runner = models[1]["algorithm"] if len(models) > 1 else None
            if runner:
                questions.append(f"What's the trade-off between {algo} and {runner}?")

        questions.append("What should I try next to improve model performance?")
        questions.append("Which model is best for production deployment?")

        return questions[:6]

    # ─── PRE-TRAINING QUESTIONS ───────────────────────────────

    def generate_pre_training_questions(self, context: Optional[Dict] = None) -> List[str]:
        """Generate suggested questions before training starts."""
        questions = [
            "Which algorithms should I select for my dataset?",
            "What scaling method is best for my data?",
            "Should I enable automated feature engineering?",
            "How many algorithms should I compare?",
            "What problem type should I choose for my target?",
        ]

        if context:
            rows = context.get("rows", 0)
            if rows < 1000:
                questions.insert(0, f"Is {rows} rows enough data for reliable training?")
            cols = context.get("columns", 0)
            if cols > 50:
                questions.insert(0, f"Should I reduce {cols} features before training?")

        return questions[:6]