"""
Threshold Optimizer — Business-Optimal Classification Threshold
=================================================================
Finds the classification threshold that maximizes business value,
not just statistical metrics.

Capabilities:
  1. Metric-Optimal Thresholds    — Maximize F1, F-beta, balanced accuracy
  2. Cost-Optimal Threshold       — Minimize total cost given cost matrix
  3. Revenue-Optimal Threshold    — Maximize revenue given revenue model
  4. Constraint-Optimal Threshold — Maximize X subject to Y >= min_Y
  5. Multi-Objective Pareto       — Find Pareto frontier of precision/recall
  6. Calibration Analysis         — Is the model well-calibrated?
  7. Operating Point Analysis     — Full confusion matrix at any threshold
  8. Threshold Sensitivity        — How sensitive are metrics to threshold changes
  9. Confidence Bands             — Uncertainty estimates via bootstrap-like analysis
  10. Business Scenario Simulator — What-if analysis for different thresholds

Works from pre-computed ROC/PR curves (arrays of thresholds, tprs, fprs, precisions, recalls).
NO access to raw predictions.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ThresholdPoint:
    """Metrics at a specific threshold."""
    threshold: float
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fbeta: float = 0.0
    fpr: float = 0.0
    tpr: float = 0.0
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    total_cost: float = 0.0
    net_revenue: float = 0.0

    def to_dict(self):
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""
    optimal_threshold: float
    optimization_criterion: str
    optimal_value: float
    current_threshold: float = 0.5
    current_value: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0
    optimal_point: Optional[ThresholdPoint] = None
    current_point: Optional[ThresholdPoint] = None
    pareto_frontier: List[ThresholdPoint] = field(default_factory=list)
    all_points: List[ThresholdPoint] = field(default_factory=list)
    sensitivity: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self):
        return {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "optimization_criterion": self.optimization_criterion,
            "optimal_value": round(self.optimal_value, 4),
            "current_threshold": round(self.current_threshold, 4),
            "current_value": round(self.current_value, 4),
            "improvement": round(self.improvement, 4),
            "improvement_pct": round(self.improvement_pct, 2),
            "optimal_point": self.optimal_point.to_dict() if self.optimal_point else None,
            "current_point": self.current_point.to_dict() if self.current_point else None,
            "pareto_frontier": [p.to_dict() for p in self.pareto_frontier],
            "sensitivity": self.sensitivity,
            "recommendation": self.recommendation,
        }


class ThresholdOptimizer:
    """
    Production threshold optimizer that finds business-optimal thresholds.
    
    Usage:
        optimizer = ThresholdOptimizer()
        result = optimizer.optimize(
            thresholds=[0.1, 0.2, ...],
            precisions=[0.9, 0.85, ...],
            recalls=[0.3, 0.5, ...],
            total_samples=7043,
            positive_rate=0.265,
        )
    """

    def optimize(
        self,
        thresholds: List[float],
        precisions: List[float],
        recalls: List[float],
        tprs: List[float] = None,
        fprs: List[float] = None,
        total_samples: int = 1000,
        positive_rate: float = 0.5,
        current_threshold: float = 0.5,
        criterion: str = "f1",
        beta: float = 1.0,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
        revenue_per_tp: float = 100.0,
        min_precision: float = None,
        min_recall: float = None,
    ) -> OptimizationResult:
        """
        Find optimal threshold for the given criterion.
        
        Criteria:
            f1: Maximize F1 score
            fbeta: Maximize F-beta score  
            cost: Minimize total cost
            revenue: Maximize net revenue
            balanced: Maximize balanced accuracy
            mcc: Maximize Matthews Correlation Coefficient
            precision_at_recall: Max precision with recall >= min_recall
            recall_at_precision: Max recall with precision >= min_precision
        """
        if not thresholds or not precisions or not recalls:
            return OptimizationResult(
                optimal_threshold=0.5, optimization_criterion=criterion,
                optimal_value=0, recommendation="Insufficient data for optimization."
            )

        n_pos = int(total_samples * positive_rate)
        n_neg = total_samples - n_pos

        # Build ThresholdPoint for each threshold
        points = []
        for i, thresh in enumerate(thresholds):
            prec = precisions[min(i, len(precisions) - 1)]
            rec = recalls[min(i, len(recalls) - 1)]

            tp = int(rec * n_pos)
            fp = int(tp / max(prec, 0.001) - tp) if prec > 0 else 0
            fn = n_pos - tp
            tn = n_neg - fp

            # Clamp
            tp = max(0, tp)
            fp = max(0, min(fp, n_neg))
            fn = max(0, fn)
            tn = max(0, n_neg - fp)

            specificity = tn / max(1, tn + fp)
            fpr_val = fp / max(1, fp + tn)
            bal_acc = (rec + specificity) / 2

            f1 = 2 * prec * rec / max(prec + rec, 0.001)
            fbeta_val = (1 + beta**2) * prec * rec / max(beta**2 * prec + rec, 0.001)

            # MCC
            denom = math.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            mcc_val = (tp * tn - fp * fn) / denom if denom > 0 else 0

            total_cost = cost_fp * fp + cost_fn * fn + cost_tp * tp + cost_tn * tn
            net_rev = revenue_per_tp * tp - cost_fp * fp - cost_fn * fn

            point = ThresholdPoint(
                threshold=thresh, precision=prec, recall=rec,
                f1=f1, fbeta=fbeta_val, fpr=fpr_val, tpr=rec,
                specificity=specificity, balanced_accuracy=bal_acc,
                mcc=mcc_val, tp=tp, fp=fp, fn=fn, tn=tn,
                total_cost=total_cost, net_revenue=net_rev,
            )
            points.append(point)

        # Find optimal by criterion
        criterion_map = {
            "f1": lambda p: p.f1,
            "fbeta": lambda p: p.fbeta,
            "cost": lambda p: -p.total_cost,  # minimize cost = maximize negative cost
            "revenue": lambda p: p.net_revenue,
            "balanced": lambda p: p.balanced_accuracy,
            "mcc": lambda p: p.mcc,
            "precision_at_recall": lambda p: p.precision if min_recall and p.recall >= min_recall else -1,
            "recall_at_precision": lambda p: p.recall if min_precision and p.precision >= min_precision else -1,
        }

        score_fn = criterion_map.get(criterion, criterion_map["f1"])
        best_point = max(points, key=score_fn)
        best_score = score_fn(best_point)

        # Current threshold point
        current_point = min(points, key=lambda p: abs(p.threshold - current_threshold))
        current_score = score_fn(current_point)

        improvement = best_score - current_score
        improvement_pct = (improvement / max(abs(current_score), 0.001)) * 100

        # Pareto frontier (precision vs recall)
        pareto = self._compute_pareto_frontier(points)

        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(points, best_point)

        # Generate recommendation
        rec = self._generate_recommendation(
            criterion, best_point, current_point, improvement_pct,
            cost_fn=cost_fn, cost_fp=cost_fp, total_samples=total_samples,
        )

        result = OptimizationResult(
            optimal_threshold=best_point.threshold,
            optimization_criterion=criterion,
            optimal_value=best_score if criterion != "cost" else best_point.total_cost,
            current_threshold=current_threshold,
            current_value=current_score if criterion != "cost" else current_point.total_cost,
            improvement=improvement,
            improvement_pct=improvement_pct,
            optimal_point=best_point,
            current_point=current_point,
            pareto_frontier=pareto,
            all_points=points,
            sensitivity=sensitivity,
            recommendation=rec,
        )

        return result

    def multi_objective_optimize(
        self,
        thresholds: List[float],
        precisions: List[float],
        recalls: List[float],
        total_samples: int = 1000,
        positive_rate: float = 0.5,
        objectives: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Optimize for multiple business objectives simultaneously.
        
        objectives example:
            [
                {"criterion": "f1", "weight": 0.3},
                {"criterion": "cost", "weight": 0.4, "cost_fp": 50, "cost_fn": 200},
                {"criterion": "recall", "weight": 0.3, "min_value": 0.6},
            ]
        """
        if not objectives:
            objectives = [
                {"criterion": "f1", "weight": 0.5},
                {"criterion": "balanced", "weight": 0.3},
                {"criterion": "cost", "weight": 0.2, "cost_fp": 10, "cost_fn": 100},
            ]

        # Run optimization for each objective
        results_by_criterion = {}
        for obj in objectives:
            result = self.optimize(
                thresholds=thresholds, precisions=precisions, recalls=recalls,
                total_samples=total_samples, positive_rate=positive_rate,
                criterion=obj["criterion"],
                cost_fp=obj.get("cost_fp", 10),
                cost_fn=obj.get("cost_fn", 100),
                min_recall=obj.get("min_value") if obj["criterion"] == "precision_at_recall" else None,
                min_precision=obj.get("min_value") if obj["criterion"] == "recall_at_precision" else None,
            )
            results_by_criterion[obj["criterion"]] = {
                "optimal_threshold": result.optimal_threshold,
                "optimal_value": result.optimal_value,
                "weight": obj["weight"],
            }

        # Weighted consensus threshold
        weighted_threshold = sum(
            r["optimal_threshold"] * r["weight"]
            for r in results_by_criterion.values()
        ) / sum(obj["weight"] for obj in objectives)

        # Find closest actual threshold
        closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - weighted_threshold))
        consensus_threshold = thresholds[closest_idx]

        return {
            "consensus_threshold": round(consensus_threshold, 4),
            "weighted_threshold": round(weighted_threshold, 4),
            "by_criterion": results_by_criterion,
            "recommendation": (
                f"Multi-objective consensus: threshold={consensus_threshold:.3f}. "
                f"This balances {', '.join(results_by_criterion.keys())} "
                f"with weights {[obj['weight'] for obj in objectives]}."
            ),
        }

    def business_scenario_analysis(
        self,
        thresholds: List[float],
        precisions: List[float],
        recalls: List[float],
        total_samples: int = 1000,
        positive_rate: float = 0.5,
        scenarios: List[Dict] = None,
    ) -> List[Dict]:
        """
        Run what-if analysis across multiple business scenarios.
        
        scenarios example:
            [
                {"name": "Conservative", "cost_fp": 200, "cost_fn": 50, "revenue_per_tp": 100},
                {"name": "Aggressive", "cost_fp": 10, "cost_fn": 500, "revenue_per_tp": 200},
            ]
        """
        if not scenarios:
            scenarios = [
                {"name": "Low FP cost", "cost_fp": 10, "cost_fn": 100, "revenue_per_tp": 50},
                {"name": "Balanced", "cost_fp": 50, "cost_fn": 100, "revenue_per_tp": 100},
                {"name": "High FP cost", "cost_fp": 200, "cost_fn": 100, "revenue_per_tp": 150},
                {"name": "High FN cost", "cost_fp": 50, "cost_fn": 500, "revenue_per_tp": 200},
            ]

        results = []
        for scenario in scenarios:
            result = self.optimize(
                thresholds=thresholds, precisions=precisions, recalls=recalls,
                total_samples=total_samples, positive_rate=positive_rate,
                criterion="revenue",
                cost_fp=scenario.get("cost_fp", 10),
                cost_fn=scenario.get("cost_fn", 100),
                revenue_per_tp=scenario.get("revenue_per_tp", 100),
            )

            results.append({
                "scenario": scenario["name"],
                "optimal_threshold": round(result.optimal_threshold, 4),
                "max_revenue": round(result.optimal_point.net_revenue, 2) if result.optimal_point else 0,
                "precision_at_optimal": round(result.optimal_point.precision, 4) if result.optimal_point else 0,
                "recall_at_optimal": round(result.optimal_point.recall, 4) if result.optimal_point else 0,
                "cost_fp": scenario.get("cost_fp"),
                "cost_fn": scenario.get("cost_fn"),
                "revenue_per_tp": scenario.get("revenue_per_tp"),
            })

        return results

    def calibration_analysis(
        self,
        predicted_probs: List[float] = None,
        actual_positives: List[float] = None,
        n_bins: int = 10,
        bin_means: List[float] = None,
        bin_fractions: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze model calibration (are predicted probabilities reliable?).
        Can work from either raw data or pre-binned data.
        """
        if bin_means is not None and bin_fractions is not None:
            means = bin_means
            fractions = bin_fractions
        elif predicted_probs and actual_positives:
            # Bin the predictions
            means = []
            fractions = []
            for i in range(n_bins):
                lo = i / n_bins
                hi = (i + 1) / n_bins
                indices = [j for j in range(len(predicted_probs)) if lo <= predicted_probs[j] < hi]
                if indices:
                    mean_pred = sum(predicted_probs[j] for j in indices) / len(indices)
                    frac_pos = sum(actual_positives[j] for j in indices) / len(indices)
                    means.append(mean_pred)
                    fractions.append(frac_pos)
        else:
            return {"error": "Need either raw data or pre-binned data"}

        if not means or not fractions:
            return {"error": "Empty calibration data"}

        # Expected Calibration Error (ECE)
        ece = sum(abs(means[i] - fractions[i]) for i in range(len(means))) / len(means)

        # Maximum Calibration Error
        mce = max(abs(means[i] - fractions[i]) for i in range(len(means)))

        # Brier-like score from bins
        brier = sum((means[i] - fractions[i])**2 for i in range(len(means))) / len(means)

        # Calibration quality
        if ece < 0.05:
            quality = "excellent"
            rec = "Model is well-calibrated. Predicted probabilities are trustworthy."
        elif ece < 0.10:
            quality = "good"
            rec = "Model is reasonably calibrated. Consider Platt scaling for fine-tuning."
        elif ece < 0.20:
            quality = "fair"
            rec = "Model needs calibration. Apply Platt scaling or isotonic regression."
        else:
            quality = "poor"
            rec = "Model is poorly calibrated. DO NOT use raw probabilities for decisions. Apply isotonic regression."

        # Direction of miscalibration
        over_confident = sum(1 for i in range(len(means)) if means[i] > fractions[i])
        under_confident = len(means) - over_confident
        bias = "overconfident" if over_confident > under_confident else "underconfident"

        return {
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier, 4),
            "quality": quality,
            "bias": bias,
            "calibration_curve": [
                {"bin_mean_predicted": round(means[i], 4), "actual_positive_rate": round(fractions[i], 4)}
                for i in range(len(means))
            ],
            "recommendation": rec,
        }

    # ──────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    def _compute_pareto_frontier(self, points: List[ThresholdPoint]) -> List[ThresholdPoint]:
        """Find Pareto-optimal points in precision-recall space."""
        sorted_points = sorted(points, key=lambda p: p.recall)
        pareto = []
        max_precision = -1

        for p in sorted_points:
            if p.precision > max_precision:
                pareto.append(p)
                max_precision = p.precision

        return pareto

    def _sensitivity_analysis(self, points: List[ThresholdPoint], optimal: ThresholdPoint) -> Dict:
        """How much do metrics change around the optimal threshold?"""
        optimal_idx = None
        for i, p in enumerate(points):
            if abs(p.threshold - optimal.threshold) < 0.001:
                optimal_idx = i
                break

        if optimal_idx is None or len(points) < 3:
            return {}

        # Look ±5% around optimal
        nearby = [p for p in points if abs(p.threshold - optimal.threshold) <= 0.05]
        if len(nearby) < 2:
            return {}

        f1_range = max(p.f1 for p in nearby) - min(p.f1 for p in nearby)
        prec_range = max(p.precision for p in nearby) - min(p.precision for p in nearby)
        rec_range = max(p.recall for p in nearby) - min(p.recall for p in nearby)

        return {
            "window": "±0.05 around optimal",
            "f1_sensitivity": round(f1_range, 4),
            "precision_sensitivity": round(prec_range, 4),
            "recall_sensitivity": round(rec_range, 4),
            "is_stable": f1_range < 0.05,
            "note": "Stable" if f1_range < 0.05 else "Sensitive — small threshold changes cause significant metric shifts",
        }

    def _generate_recommendation(self, criterion, optimal, current, improvement_pct, **kwargs):
        """Generate human-readable recommendation."""
        parts = []

        if abs(optimal.threshold - current.threshold) < 0.01:
            parts.append(f"Current threshold ({current.threshold:.3f}) is already near-optimal for {criterion}.")
        else:
            direction = "↑" if optimal.threshold > current.threshold else "↓"
            parts.append(
                f"Move threshold from {current.threshold:.3f} to {optimal.threshold:.3f} {direction} "
                f"for {improvement_pct:.1f}% improvement in {criterion}."
            )

        # Precision-recall trade-off summary
        prec_change = optimal.precision - current.precision
        rec_change = optimal.recall - current.recall
        if prec_change > 0 and rec_change < 0:
            parts.append(f"Trade-off: +{prec_change:.1%} precision, {rec_change:.1%} recall.")
        elif prec_change < 0 and rec_change > 0:
            parts.append(f"Trade-off: {prec_change:.1%} precision, +{rec_change:.1%} recall.")

        # Confusion matrix impact
        fp_change = optimal.fp - current.fp
        fn_change = optimal.fn - current.fn
        if fp_change != 0 or fn_change != 0:
            total = kwargs.get("total_samples", 1000)
            parts.append(f"Impact: {fp_change:+d} false positives, {fn_change:+d} false negatives per {total:,} samples.")

        return " ".join(parts)
