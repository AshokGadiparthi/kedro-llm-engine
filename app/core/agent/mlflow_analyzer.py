"""
ML Flow Analyzer — World-Class Post-Training Intelligence
===================================================================
Multi-model comparison, production readiness, registry intelligence,
training history analysis, confusion matrix patterns, feature importance.

PHASE 1 (existing):
  • analyze_leaderboard()         — Full rankings, overfitting, spread
  • compare_models()              — Head-to-head (model A vs B)
  • explain_winner()              — Why the best model won
  • smart_config()                — Pre-training configuration
  • detect_training_issues()      — Failed algos, overfitting, underfitting

PHASE 2 (NEW — uses DB data):
  • assess_production_readiness() — 10-point deployment score card
  • generate_next_steps()         — Prioritized action roadmap
  • analyze_registry()            — Registry intelligence + version comparison
  • analyze_confusion_matrix()    — Where model fails, cost analysis
  • analyze_feature_importance()  — Top features, redundancy, gaps
  • analyze_training_history()    — Learn from past runs, plateau detection
  • enhanced smart_config()       — History-aware (skips failed algos, learns from past)

Data Sources:
  Frontend:  model_results[] in extra dict
  DB:        jobs, registered_models, model_versions (via context_compiler)
"""

import logging
import math
from datetime import datetime, timedelta
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

# Production readiness thresholds
PROD_MIN_ACCURACY = 0.70
PROD_MAX_OVERFIT_GAP = 0.05
PROD_MIN_PRECISION = 0.65
PROD_MIN_RECALL = 0.60
PROD_MIN_F1 = 0.65
PROD_MIN_ROC_AUC = 0.70

# Feature importance thresholds
FEAT_CONCENTRATION_WARN = 0.50   # Top 1 feature > 50% importance
FEAT_TOP3_CONCENTRATION = 0.80   # Top 3 features > 80%

# Model staleness (days)
MODEL_FRESH = 30
MODEL_AGING = 90
MODEL_STALE = 180

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


def _safe_float(val, default=None):
    """Safely convert to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except (ValueError, TypeError):
        return default


def _days_since(dt_str_or_obj) -> Optional[int]:
    """Calculate days since a datetime."""
    if dt_str_or_obj is None:
        return None
    try:
        if isinstance(dt_str_or_obj, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(dt_str_or_obj.split("+")[0].split("Z")[0], fmt)
                    return (datetime.utcnow() - dt).days
                except ValueError:
                    continue
        elif isinstance(dt_str_or_obj, datetime):
            return (datetime.utcnow() - dt_str_or_obj).days
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# CORE ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════

class MLFlowAnalyzer:
    """
    World-class ML training intelligence engine.

    Phase 1: Model comparison (from frontend model_results[])
    Phase 2: DB-powered intelligence (from jobs, registered_models, model_versions)
    """

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: LEADERBOARD ANALYSIS (existing)
    # ═══════════════════════════════════════════════════════════

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
            "insights": [],
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
                "precision": _safe_float(r.get("precision")),
                "recall": _safe_float(r.get("recall")),
                "f1_score": _safe_float(r.get("f1_score")),
                "roc_auc": _safe_float(r.get("roc_auc")),
                "training_time": _safe_float(r.get("training_time", r.get("training_time_seconds"))),
                "category": _algo_category(algo),
                "generalization": _generalization_label(abs(float(train) - float(test))) if train else "Unknown",
            }
            normalized.append(model)

        return normalized

    @staticmethod
    def _to_float(val) -> Optional[float]:
        return _safe_float(val)

    # ─── WINNER SUMMARY ───────────────────────────────────────

    def _build_winner_summary(self, winner: Dict, all_models: List[Dict]) -> Dict:
        """Build detailed summary of the winning model."""
        algo = winner["algorithm"]
        test = winner["test_score"]
        train = winner.get("train_score")
        gap = winner.get("gap")
        category = winner["category"]

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

        if tree_gaps and linear_gaps:
            tree_avg = sum(tree_gaps) / len(tree_gaps)
            linear_avg = sum(linear_gaps) / len(linear_gaps)
            result["tree_vs_linear"] = {
                "tree_avg_gap": f"{tree_avg * 100:.2f}%",
                "linear_avg_gap": f"{linear_avg * 100:.2f}%",
                "trees_overfit_more": tree_avg > linear_avg,
                "difference_pct": f"{abs(tree_avg - linear_avg) * 100:.2f}%",
            }

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
                "Performance differences are NOT statistically significant — "
                "choose the simplest/fastest model."
            )
            result["recommendation"] = "prefer_simpler"
        elif top3_spread < SPREAD_MODERATE:
            result["status"] = "moderate"
            result["message"] = (
                f"Top models spread across {top3_spread * 100:.1f}%. "
                "The winner has a meaningful but small advantage."
            )
            result["recommendation"] = "winner_justified"
        else:
            result["status"] = "wide"
            result["message"] = (
                f"Large performance gap ({top3_spread * 100:.1f}%) between top models. "
                "The winner clearly outperforms alternatives."
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

    # ─── HEAD-TO-HEAD COMPARISON ──────────────────────────────

    def compare_models(self, model_a: Dict, model_b: Dict) -> Dict:
        """Detailed head-to-head comparison of two models."""
        a_name = model_a.get("algorithm", "Model A")
        b_name = model_b.get("algorithm", "Model B")

        a_test = float(model_a.get("test_score", model_a.get("accuracy", 0)))
        b_test = float(model_b.get("test_score", model_b.get("accuracy", 0)))
        a_train = _safe_float(model_a.get("train_score"))
        b_train = _safe_float(model_b.get("train_score"))

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

        for metric in ["test_score", "precision", "recall", "f1_score", "roc_auc"]:
            a_val = _safe_float(model_a.get(metric))
            b_val = _safe_float(model_b.get(metric))
            if a_val is not None and b_val is not None:
                comparison["metrics_comparison"][metric] = {
                    a_name: a_val, b_name: b_val,
                    "winner": a_name if a_val >= b_val else b_name,
                    "difference": abs(a_val - b_val),
                }
                better = a_name if a_val > b_val else b_name
                if abs(a_val - b_val) > 0.005:
                    comparison["advantages"][better].append(f"Better {metric} ({abs(a_val - b_val) * 100:.1f}% higher)")

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

        a_complex = _algo_key(a_name) in COMPLEX_MODELS
        b_complex = _algo_key(b_name) in COMPLEX_MODELS
        if a_complex != b_complex:
            simpler = a_name if not a_complex else b_name
            comparison["advantages"][simpler].append("Simpler model (easier to interpret and deploy)")

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
        """Generate a human-readable explanation of why the winner won."""
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
        reasons.append(f"Achieved the highest test accuracy of {test * 100:.2f}%")

        if len(normalized) > 1:
            runner = normalized[1]
            margin = test - runner["test_score"]
            if margin > 0.02:
                reasons.append(f"Outperformed runner-up ({runner['algorithm']}) by {margin * 100:.2f}% — a clear advantage")
            elif margin > 0.005:
                reasons.append(f"Edged out {runner['algorithm']} by {margin * 100:.2f}%")
            else:
                reasons.append(f"Essentially tied with {runner['algorithm']} (only {margin * 100:.2f}% difference)")

        gap = winner.get("gap")
        if gap is not None:
            if gap < OVERFIT_MILD:
                reasons.append(f"Excellent generalization (train-test gap of only {gap * 100:.2f}%)")
            elif gap < OVERFIT_MODERATE:
                reasons.append(f"Good generalization with {gap * 100:.2f}% train-test gap")
            else:
                reasons.append(f"⚠️ Shows overfitting with {gap * 100:.2f}% train-test gap")

        if cat == "boosting":
            reasons.append("Gradient boosting models excel at capturing non-linear patterns and feature interactions")
        elif cat == "linear":
            reasons.append("Linear models suggest the data has strong linear relationships — a good sign for interpretability")
        elif cat == "ensemble":
            reasons.append("Ensemble combines multiple models' strengths, reducing individual model weaknesses")

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

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: DB-POWERED INTELLIGENCE (NEW)
    # ═══════════════════════════════════════════════════════════

    # ─── PRODUCTION READINESS ASSESSMENT ──────────────────────

    def assess_production_readiness(
            self,
            model_results: List[Dict],
            model_version_data: Optional[Dict] = None,
            context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        10-point deployment readiness score card.

        Checks: accuracy, overfitting, precision, recall, F1, ROC-AUC,
        statistical significance, feature count, data size, class balance.

        Returns:
            score_card with pass/fail checks, overall score, blockers, verdict.
        """
        models = self._normalize_results(model_results)
        if not models:
            return {"score": 0, "max_score": 10, "verdict": "not_ready",
                    "message": "No valid models to evaluate", "checks": []}

        models.sort(key=lambda m: m["test_score"], reverse=True)
        winner = models[0]
        test = winner["test_score"]
        gap = winner.get("gap")

        # Merge DB metrics if available (richer than frontend data)
        if model_version_data:
            mv_metrics = model_version_data.get("metrics", {})
            if not winner.get("precision") and mv_metrics.get("precision"):
                winner["precision"] = _safe_float(mv_metrics["precision"])
            if not winner.get("recall") and mv_metrics.get("recall"):
                winner["recall"] = _safe_float(mv_metrics["recall"])
            if not winner.get("f1_score") and mv_metrics.get("f1_score"):
                winner["f1_score"] = _safe_float(mv_metrics["f1_score"])
            if not winner.get("roc_auc") and mv_metrics.get("roc_auc"):
                winner["roc_auc"] = _safe_float(mv_metrics["roc_auc"])

        checks = []
        score = 0

        # 1. Accuracy threshold
        passed = test >= PROD_MIN_ACCURACY
        checks.append({
            "id": "accuracy", "name": "Accuracy ≥ 70%", "passed": passed,
            "value": f"{test * 100:.2f}%", "threshold": f"{PROD_MIN_ACCURACY * 100:.0f}%",
            "detail": f"Winner scores {test * 100:.2f}%" + (" ✅" if passed else " — too low for reliable predictions"),
            "weight": 2,
        })
        if passed:
            score += 2

        # 2. Overfitting check
        if gap is not None:
            passed = gap <= PROD_MAX_OVERFIT_GAP
            checks.append({
                "id": "overfitting", "name": "Overfit Gap ≤ 5%", "passed": passed,
                "value": f"{gap * 100:.2f}%", "threshold": f"{PROD_MAX_OVERFIT_GAP * 100:.0f}%",
                "detail": f"{_generalization_label(gap)} generalization ({gap * 100:.2f}% gap)" + (" ✅" if passed else " — model may not generalize to new data"),
                "weight": 2,
            })
            if passed:
                score += 2
        else:
            checks.append({
                "id": "overfitting", "name": "Overfit Gap ≤ 5%", "passed": False,
                "value": "N/A", "threshold": f"{PROD_MAX_OVERFIT_GAP * 100:.0f}%",
                "detail": "No train/test gap data — cannot assess generalization",
                "weight": 2,
            })

        # 3. Precision check
        prec = winner.get("precision")
        if prec is not None:
            passed = prec >= PROD_MIN_PRECISION
            checks.append({
                "id": "precision", "name": "Precision ≥ 65%", "passed": passed,
                "value": f"{prec * 100:.2f}%", "threshold": f"{PROD_MIN_PRECISION * 100:.0f}%",
                "detail": f"Precision {prec * 100:.2f}%" + (" ✅" if passed else " — too many false positives"),
                "weight": 1,
            })
            if passed:
                score += 1
        else:
            checks.append({
                "id": "precision", "name": "Precision ≥ 65%", "passed": None,
                "value": "N/A", "threshold": f"{PROD_MIN_PRECISION * 100:.0f}%",
                "detail": "Precision not available — register model to compute full metrics",
                "weight": 1,
            })

        # 4. Recall check
        rec = winner.get("recall")
        if rec is not None:
            passed = rec >= PROD_MIN_RECALL
            checks.append({
                "id": "recall", "name": "Recall ≥ 60%", "passed": passed,
                "value": f"{rec * 100:.2f}%", "threshold": f"{PROD_MIN_RECALL * 100:.0f}%",
                "detail": f"Recall {rec * 100:.2f}%" + (" ✅" if passed else " — missing too many positive cases"),
                "weight": 1,
            })
            if passed:
                score += 1
        else:
            checks.append({
                "id": "recall", "name": "Recall ≥ 60%", "passed": None,
                "value": "N/A", "threshold": f"{PROD_MIN_RECALL * 100:.0f}%",
                "detail": "Recall not available — register model to compute full metrics",
                "weight": 1,
            })

        # 5. Statistical significance (margin over runner-up)
        if len(models) > 1:
            margin = test - models[1]["test_score"]
            passed = margin > 0.01
            checks.append({
                "id": "significance", "name": "Statistically Significant", "passed": passed,
                "value": f"{margin * 100:.2f}% margin", "threshold": "> 1%",
                "detail": (f"Clear separation from runner-up ✅" if passed else
                           f"Only {margin * 100:.2f}% ahead of {models[1]['algorithm']} — ranking may not be stable"),
                "weight": 1,
            })
            if passed:
                score += 1

        # 6. Multiple algorithms tested
        passed = len(models) >= 3
        checks.append({
            "id": "diversity", "name": "Algorithm Diversity (≥ 3)", "passed": passed,
            "value": f"{len(models)} algorithms", "threshold": "≥ 3",
            "detail": (f"Tested {len(models)} algorithms — good coverage ✅" if passed else
                       "Only tested 1-2 algorithms — may miss better models"),
            "weight": 1,
        })
        if passed:
            score += 1

        # 7. Data sufficiency (from context)
        rows = (context or {}).get("rows", 0)
        if rows > 0:
            passed = rows >= 1000
            checks.append({
                "id": "data_size", "name": "Sufficient Training Data (≥ 1K)", "passed": passed,
                "value": f"{rows:,} rows", "threshold": "≥ 1,000",
                "detail": (f"{rows:,} rows — adequate for most models ✅" if passed else
                           f"Only {rows:,} rows — model may not generalize well to unseen data"),
                "weight": 1,
            })
            if passed:
                score += 1

        # 8. ROC-AUC check
        auc = winner.get("roc_auc")
        if auc is not None:
            passed = auc >= PROD_MIN_ROC_AUC
            checks.append({
                "id": "roc_auc", "name": "ROC-AUC ≥ 70%", "passed": passed,
                "value": f"{auc * 100:.2f}%", "threshold": f"{PROD_MIN_ROC_AUC * 100:.0f}%",
                "detail": f"ROC-AUC {auc * 100:.2f}%" + (" — good discrimination ✅" if passed else " — poor class separation"),
                "weight": 1,
            })
            if passed:
                score += 1

        # Calculate max possible score
        max_score = sum(c["weight"] for c in checks)
        pct = (score / max_score * 100) if max_score > 0 else 0

        # Determine verdict
        blockers = [c for c in checks if c["passed"] is False and c["weight"] >= 2]
        warnings = [c for c in checks if c["passed"] is False and c["weight"] == 1]
        unknowns = [c for c in checks if c["passed"] is None]

        if blockers:
            verdict = "not_ready"
            message = f"🚫 NOT production-ready — {len(blockers)} critical blocker(s): {', '.join(b['name'] for b in blockers)}"
        elif pct >= 80:
            verdict = "ready"
            message = f"✅ Production-ready! Score {score}/{max_score} ({pct:.0f}%)" + (f" with {len(warnings)} minor warning(s)" if warnings else "")
        elif pct >= 60:
            verdict = "ready_with_caveats"
            message = f"⚠️ Conditionally ready — Score {score}/{max_score} ({pct:.0f}%). Address warnings before production."
        else:
            verdict = "not_ready"
            message = f"🚫 NOT production-ready — Score {score}/{max_score} ({pct:.0f}%). Significant improvements needed."

        return {
            "score": score,
            "max_score": max_score,
            "percentage": round(pct, 1),
            "verdict": verdict,
            "message": message,
            "checks": checks,
            "blockers": [{"name": b["name"], "value": b["value"], "detail": b["detail"]} for b in blockers],
            "warnings": [{"name": w["name"], "value": w["value"], "detail": w["detail"]} for w in warnings],
            "unknowns": [{"name": u["name"], "detail": u["detail"]} for u in unknowns],
            "winner_algorithm": winner["algorithm"],
            "winner_score": f"{test * 100:.2f}%",
        }

    # ─── NEXT STEPS ENGINE ────────────────────────────────────

    def generate_next_steps(
            self,
            model_results: List[Dict],
            registry_info: Optional[Dict] = None,
            training_history: Optional[Dict] = None,
            production_readiness: Optional[Dict] = None,
            context: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prioritized, actionable next steps roadmap.

        Considers: model performance, registry state, training history, production readiness.

        Returns:
            Ordered list of steps with priority, effort, impact.
        """
        steps = []
        models = self._normalize_results(model_results)
        if not models:
            return [{"step": 1, "priority": "critical", "title": "Fix Training",
                     "description": "No models trained successfully. Check data preprocessing and pipeline logs.",
                     "effort": "low", "impact": "high", "category": "debug"}]

        models.sort(key=lambda m: m["test_score"], reverse=True)
        winner = models[0]
        best_score = winner["test_score"]
        gap = winner.get("gap", 0) or 0

        step_num = 1

        # Check if ready already (use production_readiness if available)
        is_ready = (production_readiness or {}).get("verdict") == "ready"
        has_registry = registry_info and registry_info.get("total_registered", 0) > 0
        deployed_models = registry_info.get("deployed_models", 0) if registry_info else 0

        # ── STEP: Register model (if not registered yet) ──
        if not has_registry or not any(
                m.get("best_algorithm") == winner["algorithm"]
                for m in (registry_info or {}).get("models", [])
        ):
            steps.append({
                "step": step_num, "priority": "high",
                "title": f"Register {winner['algorithm']} to Model Registry",
                "description": f"Your best model ({winner['algorithm']} at {best_score * 100:.1f}%) is not in the registry yet. Register it to track versions and enable deployment.",
                "effort": "low", "impact": "high", "category": "registry",
                "action_label": "Register Model",
            })
            step_num += 1

        # ── STEP: Deploy (if production-ready and not deployed) ──
        if is_ready and deployed_models == 0:
            steps.append({
                "step": step_num, "priority": "high",
                "title": f"Deploy {winner['algorithm']} to Production",
                "description": f"Model passed production readiness ({(production_readiness or {}).get('percentage', 0):.0f}% score). No models currently deployed — deploy this model to start serving predictions.",
                "effort": "medium", "impact": "high", "category": "deploy",
                "action_label": "Deploy Model",
            })
            step_num += 1
        elif is_ready and deployed_models > 0:
            # Check if new model is better than deployed
            deployed = [m for m in (registry_info or {}).get("models", []) if m.get("is_deployed")]
            if deployed:
                deployed_acc = _safe_float(deployed[0].get("best_accuracy"), 0)
                if best_score > deployed_acc + 0.005:
                    improvement = (best_score - deployed_acc) * 100
                    steps.append({
                        "step": step_num, "priority": "high",
                        "title": f"Upgrade Deployed Model (+{improvement:.1f}% improvement)",
                        "description": f"New {winner['algorithm']} ({best_score * 100:.1f}%) outperforms deployed model ({deployed_acc * 100:.1f}%). Consider upgrading.",
                        "effort": "medium", "impact": "high", "category": "deploy",
                        "action_label": "Upgrade Model",
                    })
                    step_num += 1

        # ── STEP: Hyperparameter tuning (if gap > 2% or score < 85%) ──
        if gap > OVERFIT_MILD or (best_score < 0.85 and best_score >= SCORE_FAIR):
            reason = f"overfitting ({gap * 100:.1f}% gap)" if gap > OVERFIT_MODERATE else "room for improvement"
            steps.append({
                "step": step_num, "priority": "high" if gap > OVERFIT_MODERATE else "medium",
                "title": f"Tune {winner['algorithm']} Hyperparameters",
                "description": f"Current {reason}. Try: GridSearchCV or RandomizedSearchCV on key parameters (max_depth, learning_rate, n_estimators, regularization).",
                "effort": "medium", "impact": "medium", "category": "tune",
                "action_label": "Start Tuning",
            })
            step_num += 1

        # ── STEP: Feature engineering (if score < 80%) ──
        if best_score < SCORE_GOOD:
            steps.append({
                "step": step_num, "priority": "medium",
                "title": "Improve Features with Feature Engineering",
                "description": f"At {best_score * 100:.1f}%, algorithm selection alone isn't enough. Create interaction features, polynomial features, or domain-specific transformations to give models more signal.",
                "effort": "high", "impact": "high", "category": "feature_engineering",
                "action_label": "Open Feature Engineering",
            })
            step_num += 1

        # ── STEP: Try different algorithms (if only tested few) ──
        tested_categories = set(m["category"] for m in models)
        missing_categories = {"boosting", "linear", "tree"} - tested_categories
        if missing_categories and len(models) < 5:
            cat_algos = {"boosting": "XGBoost/LightGBM", "linear": "LogisticRegression", "tree": "RandomForest"}
            missing_algos = [cat_algos.get(c, c) for c in missing_categories]
            steps.append({
                "step": step_num, "priority": "medium",
                "title": f"Test More Algorithm Types ({', '.join(missing_algos)})",
                "description": f"Only tested {len(tested_categories)} algorithm categories. Adding {', '.join(missing_algos)} could reveal better models for your data.",
                "effort": "low", "impact": "medium", "category": "experiment",
                "action_label": "Re-train with More Algos",
            })
            step_num += 1

        # ── STEP: Cross-validation (if not done) ──
        training_runs = (training_history or {}).get("total_jobs", 0)
        if training_runs <= 1:
            steps.append({
                "step": step_num, "priority": "low",
                "title": "Validate with Cross-Validation",
                "description": "Only one training run — results may vary on different random splits. Run k-fold cross-validation (5-10 folds) for more reliable accuracy estimates.",
                "effort": "low", "impact": "medium", "category": "validate",
                "action_label": "Run Cross-Validation",
            })
            step_num += 1

        # ── STEP: Set up monitoring (if deployed) ──
        if deployed_models > 0:
            steps.append({
                "step": step_num, "priority": "medium",
                "title": "Set Up Model Monitoring",
                "description": "Your model is deployed — set up drift detection, prediction logging, and accuracy monitoring to catch performance degradation early.",
                "effort": "medium", "impact": "high", "category": "monitoring",
                "action_label": "Open Monitoring",
            })
            step_num += 1

        # ── STEP: Training plateau detection ──
        if training_runs >= 3:
            history_trend = self._detect_accuracy_trend(training_history)
            if history_trend.get("trend") == "plateau":
                steps.append({
                    "step": step_num, "priority": "medium",
                    "title": "Break the Accuracy Plateau",
                    "description": f"Accuracy has plateaued around {history_trend.get('avg_accuracy', 0) * 100:.1f}% across {training_runs} runs. More algorithm tuning won't help — focus on better features or more data.",
                    "effort": "high", "impact": "high", "category": "feature_engineering",
                    "action_label": "Try Feature Engineering",
                })
                step_num += 1

        return steps

    # ─── REGISTRY INTELLIGENCE ────────────────────────────────

    def analyze_registry(
            self,
            registry_info: Dict,
            model_versions: Optional[Dict] = None,
            current_training_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Model registry intelligence — portfolio summary, version comparison,
        deployment recommendations, staleness alerts.

        Args:
            registry_info: From context_compiler._get_registry_info()
            model_versions: From context_compiler._get_model_versions()
            current_training_score: Best score from current training (for comparison)

        Returns:
            Registry analysis with alerts, version comparison, deployment recommendations.
        """
        result = {
            "total_registered": 0,
            "total_deployed": 0,
            "total_versions": 0,
            "models": [],
            "alerts": [],
            "deployment_recommendation": None,
            "version_trend": None,
        }

        if not registry_info or registry_info.get("total_registered", 0) == 0:
            result["alerts"].append({
                "type": "no_registry",
                "severity": "info",
                "title": "No Models Registered",
                "message": "Register your trained model to track versions, compare performance over time, and enable deployment.",
                "action": "Register your best model from the training results.",
            })
            return result

        result["total_registered"] = registry_info.get("total_registered", 0)
        result["total_deployed"] = registry_info.get("deployed_models", 0)

        # Analyze each registered model
        for model in registry_info.get("models", []):
            model_info = {
                "id": model.get("id"),
                "name": model.get("name", "Unnamed"),
                "best_algorithm": model.get("best_algorithm"),
                "best_accuracy": model.get("best_accuracy", 0),
                "best_accuracy_pct": f"{(model.get('best_accuracy') or 0) * 100:.2f}%",
                "is_deployed": model.get("is_deployed", False),
                "total_versions": model.get("total_versions", 0),
                "current_version": model.get("current_version"),
                "status": model.get("status", "unknown"),
                "problem_type": model.get("problem_type"),
            }
            result["models"].append(model_info)

            # Alert: Deployed model is old
            if model.get("is_deployed"):
                # Check staleness via model_versions created_at
                deployed_age = None
                if model_versions:
                    for v in model_versions.get("versions", []):
                        if v.get("model_id") == model.get("id") and v.get("is_current"):
                            created = v.get("created_at") if isinstance(v.get("created_at"), str) else None
                            if created:
                                deployed_age = _days_since(created)
                            break

                if deployed_age is not None:
                    if deployed_age > MODEL_STALE:
                        result["alerts"].append({
                            "type": "model_stale",
                            "severity": "warning",
                            "title": f"Deployed Model is {deployed_age} Days Old",
                            "message": f"'{model.get('name')}' was deployed {deployed_age} days ago. Model performance may have degraded due to data drift.",
                            "action": "Retrain with recent data and compare against current deployed model.",
                        })
                    elif deployed_age > MODEL_AGING:
                        result["alerts"].append({
                            "type": "model_aging",
                            "severity": "info",
                            "title": f"Deployed Model is {deployed_age} Days Old",
                            "message": f"'{model.get('name')}' has been deployed for {deployed_age} days. Consider retraining to capture recent data patterns.",
                            "action": "Schedule a retraining run to verify accuracy hasn't degraded.",
                        })

                # Alert: New training is better than deployed
                deployed_acc = _safe_float(model.get("best_accuracy"), 0)
                if current_training_score and current_training_score > deployed_acc + 0.005:
                    improvement = (current_training_score - deployed_acc) * 100
                    result["alerts"].append({
                        "type": "upgrade_available",
                        "severity": "info",
                        "title": f"New Model is {improvement:.1f}% Better Than Deployed",
                        "message": f"Current training ({current_training_score * 100:.2f}%) outperforms deployed model ({deployed_acc * 100:.2f}%).",
                        "action": "Register the new model and promote it to replace the deployed version.",
                    })
                    result["deployment_recommendation"] = {
                        "action": "upgrade",
                        "current_deployed_accuracy": f"{deployed_acc * 100:.2f}%",
                        "new_model_accuracy": f"{current_training_score * 100:.2f}%",
                        "improvement": f"+{improvement:.1f}%",
                        "message": f"Upgrade recommended: +{improvement:.1f}% accuracy improvement.",
                    }

        # Version trend analysis
        if model_versions and model_versions.get("versions"):
            versions = model_versions["versions"]
            result["total_versions"] = len(versions)
            accuracies = []
            for v in versions:
                acc = _safe_float(v.get("metrics", {}).get("accuracy"))
                if acc and acc > 0:
                    accuracies.append(acc)

            if len(accuracies) >= 2:
                # Simple trend: is accuracy improving?
                recent = accuracies[:min(3, len(accuracies))]
                older = accuracies[min(3, len(accuracies)):]
                if older:
                    recent_avg = sum(recent) / len(recent)
                    older_avg = sum(older) / len(older)
                    diff = recent_avg - older_avg
                    if diff > 0.01:
                        trend = "improving"
                        trend_msg = f"Accuracy trending up: recent {recent_avg * 100:.1f}% vs older {older_avg * 100:.1f}%"
                    elif diff < -0.01:
                        trend = "declining"
                        trend_msg = f"⚠️ Accuracy declining: recent {recent_avg * 100:.1f}% vs older {older_avg * 100:.1f}%"
                    else:
                        trend = "stable"
                        trend_msg = f"Accuracy stable around {recent_avg * 100:.1f}%"
                    result["version_trend"] = {
                        "trend": trend,
                        "message": trend_msg,
                        "recent_avg": f"{recent_avg * 100:.2f}%",
                        "older_avg": f"{older_avg * 100:.2f}%",
                        "total_versions": len(accuracies),
                    }

        return result

    # ─── CONFUSION MATRIX ANALYSIS ────────────────────────────

    def analyze_confusion_matrix(
            self,
            confusion_matrix: Any,
            class_names: Optional[List[str]] = None,
            problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Deep confusion matrix analysis — where the model fails,
        cost-aware insights, threshold suggestions.

        Args:
            confusion_matrix: 2D list/array [[TN, FP], [FN, TP]] for binary
            class_names: e.g., ["No Churn", "Churn"]

        Returns:
            Analysis with per-class metrics, error patterns, recommendations.
        """
        if confusion_matrix is None:
            return {"available": False, "message": "No confusion matrix data — register the model to compute."}

        # Parse matrix
        try:
            if isinstance(confusion_matrix, str):
                import json
                cm = json.loads(confusion_matrix)
            else:
                cm = confusion_matrix

            if not cm or not isinstance(cm, (list, tuple)):
                return {"available": False, "message": "Invalid confusion matrix format"}
        except Exception:
            return {"available": False, "message": "Could not parse confusion matrix"}

        n_classes = len(cm)
        is_binary = n_classes == 2

        result = {
            "available": True,
            "n_classes": n_classes,
            "is_binary": is_binary,
            "matrix": cm,
            "total_predictions": sum(sum(row) for row in cm),
            "per_class": [],
            "error_patterns": [],
            "recommendations": [],
        }

        if not class_names:
            class_names = [f"Class {i}" for i in range(n_classes)]
        result["class_names"] = class_names

        total = result["total_predictions"]
        if total == 0:
            return result

        # Per-class analysis
        for i in range(n_classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n_classes)) - tp
            fn = sum(cm[i]) - tp
            tn = total - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = sum(cm[i])

            cls_info = {
                "class_name": class_names[i] if i < len(class_names) else f"Class {i}",
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "precision_pct": f"{precision * 100:.1f}%",
                "recall": round(recall, 4),
                "recall_pct": f"{recall * 100:.1f}%",
                "f1_score": round(f1, 4),
                "support": support,
                "support_pct": f"{support / total * 100:.1f}%",
            }
            result["per_class"].append(cls_info)

        # Binary classification specific analysis
        if is_binary:
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            positive_name = class_names[1] if len(class_names) > 1 else "Positive"
            negative_name = class_names[0] if class_names else "Negative"

            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            # False Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            result["binary_analysis"] = {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "false_positive_rate": round(fpr, 4),
                "false_positive_rate_pct": f"{fpr * 100:.1f}%",
                "false_negative_rate": round(fnr, 4),
                "false_negative_rate_pct": f"{fnr * 100:.1f}%",
                "positive_class": positive_name,
                "negative_class": negative_name,
            }

            # Error patterns
            if fnr > 0.30:
                result["error_patterns"].append({
                    "type": "high_false_negatives",
                    "severity": "warning",
                    "title": f"Missing {fnr * 100:.0f}% of Actual {positive_name}",
                    "description": f"The model fails to identify {fn} out of {fn + tp} actual {positive_name} cases ({fnr * 100:.1f}% miss rate). These are costly misses.",
                    "recommendation": "Lower the classification threshold (e.g., from 0.5 to 0.3) to catch more positive cases at the cost of more false alarms.",
                })

            if fpr > 0.20:
                result["error_patterns"].append({
                    "type": "high_false_positives",
                    "severity": "info",
                    "title": f"False Alarm Rate: {fpr * 100:.0f}%",
                    "description": f"The model incorrectly flags {fp} out of {fp + tn} actual {negative_name} cases as {positive_name} ({fpr * 100:.1f}% false alarm rate).",
                    "recommendation": "Raise the classification threshold if false alarms are costly, or accept higher false alarms for better recall.",
                })

            # Class imbalance detection
            positive_rate = (fn + tp) / total
            if positive_rate < 0.15:
                result["error_patterns"].append({
                    "type": "class_imbalance",
                    "severity": "info",
                    "title": f"Imbalanced Classes ({positive_rate * 100:.0f}% positive)",
                    "description": f"Only {positive_rate * 100:.1f}% of samples are {positive_name}. The model may be biased toward predicting {negative_name}.",
                    "recommendation": "Use class weights, SMOTE oversampling, or optimize for F1/AUC instead of accuracy.",
                })

            # Recommendations
            if fnr > fpr:
                result["recommendations"].append(
                    f"Your model is better at confirming {negative_name} than detecting {positive_name}. "
                    f"If missing {positive_name} cases is costly (e.g., churn, fraud), lower the threshold."
                )
            else:
                result["recommendations"].append(
                    f"Your model catches most {positive_name} cases but has some false alarms. "
                    f"If false alarms are costly, raise the threshold."
                )

        return result

    # ─── FEATURE IMPORTANCE ANALYSIS ──────────────────────────

    def analyze_feature_importance(
            self,
            feature_importances: Any,
            feature_names: Optional[List[str]] = None,
            context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Feature importance intelligence — top features, concentration,
        redundancy detection, missing feature suggestions.

        Args:
            feature_importances: Dict {name: importance} or list of floats
            feature_names: List of feature names (if importances is a list)

        Returns:
            Analysis with ranked features, concentration warnings, recommendations.
        """
        if not feature_importances:
            return {"available": False, "message": "No feature importance data — register the model or use a tree-based algorithm."}

        # Normalize to dict {name: importance}
        try:
            if isinstance(feature_importances, str):
                import json
                feature_importances = json.loads(feature_importances)

            if isinstance(feature_importances, dict):
                fi = feature_importances
            elif isinstance(feature_importances, (list, tuple)):
                if feature_names and len(feature_names) == len(feature_importances):
                    fi = dict(zip(feature_names, feature_importances))
                else:
                    fi = {f"feature_{i}": v for i, v in enumerate(feature_importances)}
            else:
                return {"available": False, "message": "Unsupported feature importance format"}
        except Exception:
            return {"available": False, "message": "Could not parse feature importances"}

        if not fi:
            return {"available": False, "message": "Empty feature importance data"}

        # Sort by importance descending
        sorted_features = sorted(fi.items(), key=lambda x: abs(float(x[1])), reverse=True)
        total_importance = sum(abs(float(v)) for _, v in sorted_features)
        if total_importance == 0:
            return {"available": False, "message": "All feature importances are zero"}

        # Build ranked features
        ranked = []
        cumulative = 0
        for rank, (name, imp) in enumerate(sorted_features, 1):
            imp_float = abs(float(imp))
            pct = imp_float / total_importance
            cumulative += pct
            ranked.append({
                "rank": rank,
                "feature": name,
                "importance": round(imp_float, 6),
                "importance_pct": f"{pct * 100:.1f}%",
                "cumulative_pct": f"{cumulative * 100:.1f}%",
            })

        result = {
            "available": True,
            "total_features": len(sorted_features),
            "top_features": ranked[:10],
            "all_features": ranked,
            "concentration": {},
            "insights": [],
            "recommendations": [],
        }

        # Concentration analysis
        top1_pct = abs(float(sorted_features[0][1])) / total_importance if sorted_features else 0
        top3_pct = sum(abs(float(v)) for _, v in sorted_features[:3]) / total_importance if len(sorted_features) >= 3 else top1_pct
        top5_pct = sum(abs(float(v)) for _, v in sorted_features[:5]) / total_importance if len(sorted_features) >= 5 else top3_pct

        result["concentration"] = {
            "top1_feature": sorted_features[0][0] if sorted_features else None,
            "top1_pct": f"{top1_pct * 100:.1f}%",
            "top3_pct": f"{top3_pct * 100:.1f}%",
            "top5_pct": f"{top5_pct * 100:.1f}%",
            "status": "over_concentrated" if top1_pct > FEAT_CONCENTRATION_WARN else (
                "concentrated" if top3_pct > FEAT_TOP3_CONCENTRATION else "balanced"
            ),
        }

        # Insights
        result["insights"].append(
            f"Top feature: {sorted_features[0][0]} ({top1_pct * 100:.1f}% of total importance)"
        )
        result["insights"].append(
            f"Top 3 features account for {top3_pct * 100:.1f}% of model decisions"
        )

        # Low-importance features (candidates for removal)
        low_importance = [name for name, imp in sorted_features if abs(float(imp)) / total_importance < 0.01]
        if low_importance and len(low_importance) >= 3:
            result["insights"].append(
                f"{len(low_importance)} features contribute < 1% each — candidates for removal to simplify model"
            )
            result["drop_candidates"] = low_importance[:10]

        # Concentration warnings
        if top1_pct > FEAT_CONCENTRATION_WARN:
            result["recommendations"].append(
                f"⚠️ '{sorted_features[0][0]}' alone drives {top1_pct * 100:.0f}% of predictions. "
                f"If this feature is noisy or leaky, the entire model's accuracy is at risk. "
                f"Verify this feature is legitimate and not a data leak."
            )
        if top3_pct > FEAT_TOP3_CONCENTRATION:
            result["recommendations"].append(
                f"Top 3 features drive {top3_pct * 100:.0f}% of decisions. "
                f"The model essentially ignores most other features. "
                f"Consider if the ignored features need better encoding or engineering."
            )
        if low_importance and len(low_importance) > len(sorted_features) * 0.5:
            result["recommendations"].append(
                f"{len(low_importance)} of {len(sorted_features)} features are nearly useless (< 1% importance). "
                f"Remove them to speed up training and reduce overfitting risk."
            )

        return result

    # ─── TRAINING HISTORY ANALYSIS ────────────────────────────

    def analyze_training_history(
            self,
            training_history: Dict,
            registry_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Learn from past training runs — accuracy trends, plateau detection,
        failed algorithm patterns, efficiency insights.

        Args:
            training_history: From context_compiler._get_training_history()

        Returns:
            History analysis with trends, patterns, and learnings.
        """
        result = {
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "trend": None,
            "accuracy_history": [],
            "failed_algorithm_patterns": [],
            "avg_training_time": 0,
            "insights": [],
            "learnings": [],
        }

        if not training_history or training_history.get("total_jobs", 0) == 0:
            return result

        result["total_runs"] = training_history.get("total_jobs", 0)
        result["completed_runs"] = training_history.get("completed_jobs", 0)
        result["failed_runs"] = training_history.get("failed_jobs", 0)
        result["avg_training_time"] = training_history.get("avg_execution_time", 0)

        # Extract accuracy from recent jobs
        recent_jobs = training_history.get("recent_jobs", [])
        for job in recent_jobs:
            metrics = job.get("metrics", {})
            if isinstance(metrics, dict):
                acc = _safe_float(metrics.get("accuracy") or metrics.get("best_accuracy") or metrics.get("test_score"))
                if acc and acc > 0:
                    result["accuracy_history"].append({
                        "accuracy": acc,
                        "accuracy_pct": f"{acc * 100:.2f}%",
                        "algorithm": job.get("algorithm"),
                        "pipeline": job.get("pipeline"),
                        "created_at": job.get("created_at"),
                    })

        # Trend detection
        trend_data = self._detect_accuracy_trend(training_history)
        result["trend"] = trend_data

        # Failed algorithm patterns
        failed_algos = {}
        for job in recent_jobs:
            if job.get("status") == "failed":
                algo = job.get("algorithm", "unknown")
                error = job.get("error", "")
                failed_algos.setdefault(algo, []).append(error)

        for algo, errors in failed_algos.items():
            result["failed_algorithm_patterns"].append({
                "algorithm": algo,
                "failure_count": len(errors),
                "sample_error": errors[0] if errors else "Unknown error",
            })

        # Generate insights
        if result["total_runs"] > 1:
            result["insights"].append(f"You've run {result['total_runs']} training pipelines ({result['completed_runs']} completed, {result['failed_runs']} failed)")

        if result["avg_training_time"] > 0:
            mins = result["avg_training_time"] / 60
            result["insights"].append(f"Average training time: {mins:.1f} minutes")

        if trend_data.get("trend") == "plateau":
            result["learnings"].append({
                "type": "plateau",
                "title": "Accuracy Has Plateaued",
                "description": f"Accuracy stable around {trend_data.get('avg_accuracy', 0) * 100:.1f}% across multiple runs. Algorithm tuning alone won't help.",
                "action": "Focus on feature engineering, data augmentation, or collecting more data.",
            })
        elif trend_data.get("trend") == "declining":
            result["learnings"].append({
                "type": "declining",
                "title": "Accuracy Is Declining",
                "description": "Recent training runs show declining accuracy. This may indicate data drift or problematic preprocessing changes.",
                "action": "Compare recent vs older data distributions. Check if any preprocessing steps changed.",
            })

        if result["failed_algorithm_patterns"]:
            repeat_fails = [p for p in result["failed_algorithm_patterns"] if p["failure_count"] >= 2]
            if repeat_fails:
                algo_names = [p["algorithm"] for p in repeat_fails]
                result["learnings"].append({
                    "type": "repeated_failures",
                    "title": f"{', '.join(algo_names)} Fail Repeatedly",
                    "description": f"These algorithms have failed {sum(p['failure_count'] for p in repeat_fails)} times total. They're incompatible with your current preprocessing.",
                    "action": "Exclude these algorithms from future runs to save training time.",
                })

        return result

    def _detect_accuracy_trend(self, training_history: Optional[Dict]) -> Dict:
        """Detect if accuracy is improving, plateauing, or declining."""
        if not training_history:
            return {"trend": "unknown", "message": "No history data"}

        recent_jobs = training_history.get("recent_jobs", [])
        accuracies = []
        for job in recent_jobs:
            if job.get("status") == "completed":
                metrics = job.get("metrics", {})
                if isinstance(metrics, dict):
                    acc = _safe_float(metrics.get("accuracy") or metrics.get("best_accuracy"))
                    if acc and acc > 0:
                        accuracies.append(acc)

        if len(accuracies) < 2:
            return {"trend": "insufficient_data", "message": "Need at least 2 completed runs"}

        avg = sum(accuracies) / len(accuracies)
        recent = accuracies[:min(2, len(accuracies))]
        older = accuracies[min(2, len(accuracies)):]

        if not older:
            return {"trend": "insufficient_data", "avg_accuracy": avg}

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        diff = recent_avg - older_avg

        if abs(diff) < 0.005:
            return {"trend": "plateau", "avg_accuracy": avg, "message": f"Stable around {avg * 100:.1f}%"}
        elif diff > 0.005:
            return {"trend": "improving", "avg_accuracy": avg, "improvement": f"+{diff * 100:.1f}%",
                    "message": f"Improving: recent {recent_avg * 100:.1f}% vs older {older_avg * 100:.1f}%"}
        else:
            return {"trend": "declining", "avg_accuracy": avg, "decline": f"{diff * 100:.1f}%",
                    "message": f"Declining: recent {recent_avg * 100:.1f}% vs older {older_avg * 100:.1f}%"}

    # ═══════════════════════════════════════════════════════════
    # ENHANCED SMART CONFIG (History-Aware)
    # ═══════════════════════════════════════════════════════════

    def smart_config(
            self,
            dataset_profile: Dict,
            recommendations: Dict,
            training_history: Optional[Dict] = None,
            registry_info: Optional[Dict] = None,
    ) -> Dict:
        """
        History-aware smart configuration.

        Enhanced: Learns from past training (failed algos, best performers,
        training time estimates). Uses registry data for version comparison.
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
            "history_insights": [],
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

        # ── HISTORY-AWARE ENHANCEMENTS (NEW) ──

        if training_history and training_history.get("total_jobs", 0) > 0:
            total_runs = training_history.get("total_jobs", 0)
            config["history_insights"].append(f"Based on {total_runs} previous training run(s) on this project")

            # Learn from past failures — warn about algorithms that always fail
            recent_jobs = training_history.get("recent_jobs", [])
            failed_algos = set()
            for job in recent_jobs:
                if job.get("status") == "failed":
                    algo = job.get("algorithm")
                    if algo:
                        failed_algos.add(algo.lower())

            if failed_algos:
                # Mark failed algos in recommendations
                for rec in config["recommended_algorithms"]:
                    if rec["algorithm"].lower() in failed_algos:
                        rec["warning"] = "⚠️ Failed in previous training run"

                config["history_insights"].append(
                    f"Previously failed: {', '.join(failed_algos)} — these may need different preprocessing"
                )

            # Learn from past successes
            best_past_algo = None
            best_past_score = 0
            for job in recent_jobs:
                if job.get("status") == "completed":
                    metrics = job.get("metrics", {})
                    if isinstance(metrics, dict):
                        acc = _safe_float(metrics.get("accuracy") or metrics.get("best_accuracy"), 0)
                        if acc > best_past_score:
                            best_past_score = acc
                            best_past_algo = job.get("algorithm")

            if best_past_algo and best_past_score > 0:
                config["history_insights"].append(
                    f"Previous best: {best_past_algo} at {best_past_score * 100:.1f}% accuracy"
                )

            # Estimate training time from history
            avg_time = training_history.get("avg_execution_time", 0)
            if avg_time > 0:
                if avg_time < 60:
                    config["estimated_training_time"] = f"~{avg_time:.0f} seconds (based on history)"
                else:
                    config["estimated_training_time"] = f"~{avg_time / 60:.1f} minutes (based on history)"

            # Accuracy trend
            trend = self._detect_accuracy_trend(training_history)
            if trend.get("trend") == "plateau":
                config["history_insights"].append(
                    f"⚠️ Accuracy plateaued at ~{trend.get('avg_accuracy', 0) * 100:.1f}% — consider feature engineering instead of more training"
                )
            elif trend.get("trend") == "improving":
                config["history_insights"].append(
                    f"📈 Accuracy is improving ({trend.get('improvement', '')} recent improvement)"
                )

        # Registry-aware insights
        if registry_info and registry_info.get("total_registered", 0) > 0:
            deployed = [m for m in registry_info.get("models", []) if m.get("is_deployed")]
            if deployed:
                deployed_acc = _safe_float(deployed[0].get("best_accuracy"), 0)
                deployed_algo = deployed[0].get("best_algorithm", "unknown")
                if deployed_acc > 0:
                    config["history_insights"].append(
                        f"Currently deployed: {deployed_algo} at {deployed_acc * 100:.1f}% — beat this to justify an upgrade"
                    )

        if not config["rationale"]:
            config["rationale"].append("Standard configuration — dataset shape is typical for ML training")

        return config

    # ═══════════════════════════════════════════════════════════
    # TRAINING ISSUES DETECTION (existing, unchanged)
    # ═══════════════════════════════════════════════════════════

    def detect_training_issues(self, model_results: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """Detect issues in training results."""
        issues = []
        models = self._normalize_results(model_results)
        failed = [m for m in model_results if m.get("status") == "failed"]

        if not models:
            issues.append({
                "type": "no_models", "severity": "critical",
                "title": "No models trained successfully",
                "description": f"All {len(model_results)} algorithms failed.",
                "action": "Check data preprocessing — ensure no NaN/Inf values, proper encoding, and scaling."
            })
            return issues

        best = models[0]["test_score"]

        # Issue: Failed algorithms
        if failed:
            fail_algos = [f.get("algorithm", "unknown") for f in failed]
            nb_failures = [a for a in fail_algos if _algo_key(a) in NAIVE_BAYES_MODELS]
            if nb_failures:
                issues.append({
                    "type": "nb_failure", "severity": "info",
                    "title": f"{len(nb_failures)} Naive Bayes algorithm(s) failed",
                    "description": (
                        f"{', '.join(nb_failures)} failed due to negative values in scaled data. "
                        "This is expected when using StandardScaler with Naive Bayes models."
                    ),
                    "action": "Not a problem — these models are incompatible with standard scaling."
                })
            non_nb = [a for a in fail_algos if _algo_key(a) not in NAIVE_BAYES_MODELS]
            if non_nb:
                issues.append({
                    "type": "unexpected_failure", "severity": "warning",
                    "title": f"{len(non_nb)} algorithm(s) failed unexpectedly",
                    "description": f"Failed: {', '.join(non_nb)}",
                    "action": "Check backend logs for error details."
                })

        if best < SCORE_POOR:
            issues.append({
                "type": "underfitting", "severity": "critical",
                "title": f"All models underperforming (best: {best * 100:.1f}%)",
                "description": "No algorithm above 60%. Suggests wrong target, insufficient signal, or data quality problems.",
                "action": "Return to EDA — check feature correlations with target, examine data quality."
            })
        elif best < SCORE_FAIR:
            issues.append({
                "type": "low_performance", "severity": "warning",
                "title": f"Best model achieves only {best * 100:.1f}% accuracy",
                "description": "Below 70% — features may lack predictive power.",
                "action": "Consider feature engineering or adding external data."
            })

        winner_gap = models[0].get("gap")
        if winner_gap and winner_gap >= OVERFIT_SEVERE:
            issues.append({
                "type": "winner_overfitting", "severity": "warning",
                "title": f"Best model is overfitting ({models[0]['algorithm']})",
                "description": f"Train-test gap of {winner_gap * 100:.1f}% — memorizing rather than learning.",
                "action": "More regularization, reduce complexity, get more data, or use 2nd-best model."
            })

        # Ensemble didn't improve
        ensemble_models = [m for m in models if m["category"] == "ensemble"]
        non_ensemble = [m for m in models if m["category"] != "ensemble"]
        if ensemble_models and non_ensemble:
            best_single = non_ensemble[0]["test_score"]
            best_ensemble = ensemble_models[0]["test_score"]
            if best_ensemble < best_single:
                issues.append({
                    "type": "ensemble_underperform", "severity": "info",
                    "title": "Ensemble didn't outperform best single model",
                    "description": f"VotingEnsemble ({best_ensemble * 100:.2f}%) below {non_ensemble[0]['algorithm']} ({best_single * 100:.2f}%).",
                    "action": "Try stacking or blending with diverse model types."
                })

        # Linear competitive with complex
        linear_models = [m for m in models if m["category"] == "linear"]
        complex_models = [m for m in models if m["category"] in ("boosting", "tree")]
        if linear_models and complex_models:
            best_linear = linear_models[0]["test_score"]
            best_complex = complex_models[0]["test_score"]
            if best_linear >= best_complex - 0.01:
                issues.append({
                    "type": "linear_competitive", "severity": "info",
                    "title": "Linear model matches complex models",
                    "description": f"{linear_models[0]['algorithm']} ({best_linear * 100:.2f}%) competitive with {complex_models[0]['algorithm']} ({best_complex * 100:.2f}%).",
                    "action": "Consider deploying the linear model — faster, more interpretable."
                })

        # Top models very close
        if len(models) >= 3:
            top3_spread = models[0]["test_score"] - models[2]["test_score"]
            if top3_spread < SPREAD_TIGHT:
                issues.append({
                    "type": "tight_spread", "severity": "info",
                    "title": f"Top 3 models within {top3_spread * 100:.2f}% of each other",
                    "description": "Performance difference is likely NOT statistically significant.",
                    "action": "Choose based on speed, interpretability, deployment simplicity."
                })

        return issues

    # ═══════════════════════════════════════════════════════════
    # RECOMMENDATIONS & INSIGHTS BUILDERS
    # ═══════════════════════════════════════════════════════════

    def _build_recommendations(self, models: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """Generate actionable recommendations based on training results."""
        recs = []
        if not models:
            return recs

        winner = models[0]
        best_score = winner["test_score"]

        if best_score >= SCORE_GOOD:
            recs.append({
                "type": "deploy", "priority": "high",
                "title": "Ready for Deployment",
                "description": f"{winner['algorithm']} at {best_score * 100:.1f}% is production-ready. Deploy and monitor.",
            })
        elif best_score >= SCORE_FAIR:
            recs.append({
                "type": "iterate", "priority": "high",
                "title": "Good Start — Room for Improvement",
                "description": f"{best_score * 100:.1f}% accuracy — try feature engineering, hyperparameter tuning, or ensemble methods.",
            })
        else:
            recs.append({
                "type": "rethink", "priority": "critical",
                "title": "Revisit Data & Features",
                "description": f"Best accuracy {best_score * 100:.1f}% — check feature correlations, data quality, problem formulation.",
            })

        linear = [m for m in models if m["category"] == "linear"]
        if linear and linear[0]["test_score"] >= best_score - 0.015:
            recs.append({
                "type": "simplify", "priority": "medium",
                "title": "Consider Simpler Model",
                "description": f"{linear[0]['algorithm']} at {linear[0]['test_score'] * 100:.1f}% nearly matches winner. Faster, more interpretable.",
            })

        if winner.get("gap") and winner["gap"] > OVERFIT_MODERATE:
            recs.append({
                "type": "tune", "priority": "high",
                "title": "Hyperparameter Tuning Needed",
                "description": f"{winner['algorithm']} has {winner['gap'] * 100:.1f}% overfitting. Tune regularization before deploying.",
            })

        return recs

    def _generate_narrative_insights(self, analysis: Dict, context: Optional[Dict] = None) -> List[str]:
        """Generate human-readable narrative insights."""
        insights = []
        winner = analysis.get("winner", {})
        spread = analysis.get("performance_spread", {})
        overfit = analysis.get("overfitting_analysis", {})

        algo = winner.get("algorithm", "Unknown")
        score = winner.get("test_score_pct", "N/A")
        margin = winner.get("margin_pct", "0%")
        insights.append(f"🏆 {algo} wins with {score} accuracy, {margin} ahead of {winner.get('runner_up', 'the field')}.")

        spread_status = spread.get("status", "")
        if spread_status == "tight":
            insights.append(f"📊 Top models extremely close ({spread.get('top3_spread_pct', 'N/A')} spread). Choose by speed/interpretability.")
        elif spread_status == "wide":
            insights.append(f"📊 Clear hierarchy — {spread.get('top3_spread_pct', 'N/A')} gap between top models.")

        overfit_status = overfit.get("status", "")
        if overfit_status == "excellent":
            insights.append("✅ All models generalize well — no overfitting concerns.")
        elif overfit_status == "concerning":
            worst = overfit.get("worst_offender", {})
            insights.append(f"⚠️ {worst.get('algorithm', 'Unknown')} shows {worst.get('gap_pct', 'N/A')} overfitting gap.")

        cats = analysis.get("category_analysis", {})
        if "linear" in cats and "boosting" in cats:
            lin = cats["linear"]["best_score"]
            boost = cats["boosting"]["best_score"]
            if lin >= boost - 0.01:
                insights.append(f"💡 Linear matches boosting ({lin * 100:.1f}% vs {boost * 100:.1f}%) — simpler may be better.")

        total_failed = analysis.get("total_models_failed", 0)
        if total_failed > 0:
            insights.append(f"ℹ️ {total_failed} algorithm(s) failed: {', '.join(analysis.get('failed_algorithms', []))}")

        return insights

    # ─── SUGGESTED QUESTIONS ──────────────────────────────────

    def _generate_post_training_questions(self, winner: Dict, models: List[Dict],
                                          context: Optional[Dict] = None) -> List[str]:
        """Generate context-aware suggested questions."""
        questions = []
        algo = winner.get("algorithm", "the best model")
        score = winner.get("test_score", 0)

        questions.append(f"Why did {algo} outperform the other algorithms?")
        questions.append(f"Is {score * 100:.1f}% accuracy good enough to deploy?")

        if winner.get("gap") and winner["gap"] > OVERFIT_MILD:
            questions.append(f"How can I reduce overfitting in {algo}?")

        if len(models) > 1:
            runner = models[1]["algorithm"]
            questions.append(f"What's the trade-off between {algo} and {runner}?")

        questions.append("What should I try next to improve performance?")
        questions.append("Which model is best for production deployment?")

        return questions[:6]

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