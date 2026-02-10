"""
Business Impact Calculator — ML Metrics → Business Value
============================================================
Translates model performance metrics into quantifiable business impact.
Enables stakeholders to understand ML improvements in dollar terms.

Capabilities:
  1. Cost Matrix Analysis         — FP/FN cost-weighted evaluation
  2. Revenue Impact Estimation    — Dollar impact of model improvement
  3. ROI Calculator               — Return on investment of ML effort
  4. Threshold Business Optimizer — Find threshold maximizing business value
  5. Churn Prevention Calculator  — Specific to customer churn use cases
  6. Fraud Detection Calculator   — Specific to fraud detection use cases
  7. Lift Analysis                — Quantify model lift over random baseline
  8. Capacity Planning            — How many predictions can be acted upon
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BusinessImpactCalculator:
    """
    Calculates business impact from model metrics and business parameters.
    All methods are pure computation — no database access.
    """

    # ──────────────────────────────────────────────────────────
    # 1. COST MATRIX ANALYSIS
    # ──────────────────────────────────────────────────────────

    def cost_matrix_analysis(
        self,
        metrics: Dict[str, float],
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
        total_predictions: int = 10000,
        positive_rate: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Evaluate model using a business cost matrix.
        
        Args:
            metrics: Model metrics (precision, recall, f1, etc.)
            cost_fp: Cost of a false positive (e.g., unnecessary intervention)
            cost_fn: Cost of a false negative (e.g., missed churn customer)
            cost_tp: Benefit/cost of true positive (often negative = revenue saved)
            cost_tn: Benefit/cost of true negative (usually 0)
            total_predictions: Expected number of predictions
            positive_rate: Expected positive class rate
        """
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        if not precision or not recall:
            return {"error": "Need precision and recall for cost analysis"}

        total_positive = int(total_predictions * positive_rate)
        total_negative = total_predictions - total_positive

        # Derive confusion matrix counts
        tp = int(total_positive * recall)
        fn = total_positive - tp
        fp = int(tp / precision - tp) if precision > 0 else 0
        tn = total_negative - fp

        # Calculate costs
        total_cost = (tp * cost_tp + tn * cost_tn + fp * cost_fp + fn * cost_fn)
        baseline_cost = total_positive * cost_fn  # No model = miss all positives
        savings = baseline_cost - total_cost

        return {
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": max(0, tn),
                "false_negative": fn,
            },
            "costs": {
                "fp_cost": round(fp * cost_fp, 2),
                "fn_cost": round(fn * cost_fn, 2),
                "tp_benefit": round(tp * abs(cost_tp), 2) if cost_tp < 0 else 0,
                "total_model_cost": round(total_cost, 2),
                "baseline_cost_no_model": round(baseline_cost, 2),
                "savings_vs_no_model": round(savings, 2),
            },
            "cost_matrix_used": {
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "cost_tp": cost_tp,
                "cost_tn": cost_tn,
            },
            "interpretation": self._interpret_costs(fp, fn, cost_fp, cost_fn, savings),
            "recommendations": self._cost_recommendations(precision, recall, cost_fp, cost_fn),
        }

    def _interpret_costs(self, fp, fn, cost_fp, cost_fn, savings):
        """Generate human-readable cost interpretation."""
        if savings > 0:
            return (
                f"The model saves ~${savings:,.0f} compared to no model. "
                f"False positives cost ${fp * cost_fp:,.0f} ({fp} × ${cost_fp:.0f}), "
                f"while missed cases cost ${fn * cost_fn:,.0f} ({fn} × ${cost_fn:.0f})."
            )
        else:
            return (
                f"Warning: The model costs ${abs(savings):,.0f} MORE than having no model. "
                f"This typically means the threshold needs adjustment or the model "
                f"needs improvement."
            )

    def _cost_recommendations(self, precision, recall, cost_fp, cost_fn):
        """Recommend threshold direction based on cost ratio."""
        fn_fp_ratio = cost_fn / cost_fp if cost_fp > 0 else 10

        recs = []
        if fn_fp_ratio > 5 and recall < 0.8:
            recs.append({
                "action": "Lower threshold to increase recall",
                "reason": (
                    f"Missing a positive case costs {fn_fp_ratio:.0f}× more than a false alarm. "
                    f"Current recall ({recall:.1%}) means you're missing too many. "
                    f"Even doubling false positives would be worth it."
                ),
            })
        elif fn_fp_ratio < 2 and precision < 0.7:
            recs.append({
                "action": "Raise threshold to increase precision",
                "reason": (
                    f"FP and FN costs are similar. With precision at {precision:.1%}, "
                    f"too many false alarms are diluting the value of each alert."
                ),
            })

        return recs

    # ──────────────────────────────────────────────────────────
    # 2. ROI CALCULATOR
    # ──────────────────────────────────────────────────────────

    def calculate_roi(
        self,
        model_savings_per_year: float,
        development_cost: float = 50000,
        infrastructure_cost_per_year: float = 12000,
        maintenance_cost_per_year: float = 24000,
        time_horizon_years: int = 3,
    ) -> Dict[str, Any]:
        """
        Calculate ROI of the ML model over a time horizon.
        """
        total_cost = development_cost + (infrastructure_cost_per_year + maintenance_cost_per_year) * time_horizon_years
        total_savings = model_savings_per_year * time_horizon_years
        net_benefit = total_savings - total_cost
        roi_pct = (net_benefit / total_cost * 100) if total_cost > 0 else 0

        # Payback period
        annual_cost = infrastructure_cost_per_year + maintenance_cost_per_year
        net_annual_benefit = model_savings_per_year - annual_cost
        if net_annual_benefit > 0:
            payback_months = development_cost / net_annual_benefit * 12
        else:
            payback_months = float('inf')

        return {
            "total_investment": round(total_cost, 0),
            "total_savings": round(total_savings, 0),
            "net_benefit": round(net_benefit, 0),
            "roi_percent": round(roi_pct, 1),
            "payback_period_months": round(payback_months, 1) if payback_months != float('inf') else "Never",
            "profitable": net_benefit > 0,
            "breakdown": {
                "development": development_cost,
                "infrastructure": infrastructure_cost_per_year * time_horizon_years,
                "maintenance": maintenance_cost_per_year * time_horizon_years,
                "annual_savings": model_savings_per_year,
            },
        }

    # ──────────────────────────────────────────────────────────
    # 3. CHURN PREVENTION CALCULATOR
    # ──────────────────────────────────────────────────────────

    def churn_impact(
        self,
        metrics: Dict[str, float],
        total_customers: int = 10000,
        churn_rate: float = 0.15,
        avg_customer_value: float = 500,
        intervention_cost: float = 50,
        intervention_success_rate: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Calculate business impact of a churn prediction model.
        
        Args:
            metrics: Model performance metrics
            total_customers: Total customer base
            churn_rate: Expected churn rate
            avg_customer_value: Annual value per customer
            intervention_cost: Cost of retention intervention per customer
            intervention_success_rate: Success rate of retention efforts
        """
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        if not precision or not recall:
            return {"error": "Need precision and recall for churn analysis"}

        expected_churners = int(total_customers * churn_rate)
        non_churners = total_customers - expected_churners

        # Model predictions
        caught_churners = int(expected_churners * recall)  # TP
        missed_churners = expected_churners - caught_churners  # FN
        false_alarms = int(caught_churners / precision - caught_churners) if precision > 0 else 0  # FP

        # Revenue impact
        total_contacted = caught_churners + false_alarms
        successfully_retained = int(caught_churners * intervention_success_rate)
        revenue_saved = successfully_retained * avg_customer_value
        intervention_total_cost = total_contacted * intervention_cost
        revenue_lost_missed = missed_churners * avg_customer_value

        net_impact = revenue_saved - intervention_total_cost

        # Baseline: no model
        baseline_loss = expected_churners * avg_customer_value

        return {
            "scenario": {
                "total_customers": total_customers,
                "churn_rate": churn_rate,
                "expected_churners": expected_churners,
                "avg_customer_value": avg_customer_value,
            },
            "model_performance": {
                "churners_identified": caught_churners,
                "churners_missed": missed_churners,
                "false_alarms": false_alarms,
                "total_contacted": total_contacted,
            },
            "financial_impact": {
                "customers_retained": successfully_retained,
                "revenue_saved": round(revenue_saved, 0),
                "intervention_cost": round(intervention_total_cost, 0),
                "net_impact": round(net_impact, 0),
                "revenue_lost_from_missed": round(revenue_lost_missed, 0),
                "baseline_loss_no_model": round(baseline_loss, 0),
                "improvement_vs_baseline": round(net_impact / baseline_loss * 100, 1) if baseline_loss > 0 else 0,
            },
            "per_point_improvement": {
                "recall_1pct_worth": round(
                    expected_churners * 0.01 * intervention_success_rate * avg_customer_value, 0
                ),
                "precision_1pct_worth": round(
                    total_contacted * 0.01 * intervention_cost, 0
                ) if total_contacted > 0 else 0,
            },
            "recommendation": self._churn_recommendation(
                recall, precision, avg_customer_value, intervention_cost
            ),
        }

    def _churn_recommendation(self, recall, precision, customer_value, intervention_cost):
        """Generate churn-specific recommendation."""
        value_ratio = customer_value / intervention_cost if intervention_cost > 0 else 10

        if value_ratio > 10 and recall < 0.7:
            return (
                f"Customer value (${customer_value:,.0f}) is {value_ratio:.0f}× the intervention cost "
                f"(${intervention_cost:,.0f}). Prioritize RECALL — lowering the threshold to catch "
                f"more churners is worth the extra false alarms."
            )
        elif value_ratio < 3 and precision < 0.5:
            return (
                f"Intervention cost (${intervention_cost:,.0f}) is significant relative to customer "
                f"value (${customer_value:,.0f}). Prioritize PRECISION to avoid costly false interventions."
            )
        else:
            return (
                "Balance precision and recall. Use the F1-optimal threshold, "
                "then fine-tune based on your specific budget constraints."
            )

    # ──────────────────────────────────────────────────────────
    # 4. LIFT ANALYSIS
    # ──────────────────────────────────────────────────────────

    def calculate_lift(
        self,
        metrics: Dict[str, float],
        positive_rate: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Calculate model lift over random baseline.
        Lift = precision / positive_rate
        """
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        if not precision:
            return {"error": "Need precision for lift analysis"}

        lift = precision / positive_rate if positive_rate > 0 else 0

        # Cumulative gains at different percentiles
        gains = []
        for pct in [10, 20, 30, 50, 100]:
            # At top pct% of predictions, what fraction of positives do we capture?
            if recall > 0 and precision > 0:
                # Simplified: assume ranked predictions
                captured = min(recall * (pct / 100) * lift, 1.0)
                gains.append({
                    "percentile": pct,
                    "estimated_capture_rate": round(captured, 3),
                    "random_capture_rate": round(pct / 100, 3),
                })

        return {
            "lift": round(lift, 2),
            "interpretation": (
                f"The model is {lift:.1f}× better than random at identifying positives. "
                f"For every 100 predictions flagged as positive, "
                f"~{precision*100:.0f} are actually positive (vs ~{positive_rate*100:.0f} by random chance)."
            ),
            "lift_quality": (
                "excellent" if lift > 4 else
                "good" if lift > 2.5 else
                "moderate" if lift > 1.5 else
                "poor"
            ),
            "cumulative_gains": gains,
        }

    # ──────────────────────────────────────────────────────────
    # 5. MODEL IMPROVEMENT VALUE
    # ──────────────────────────────────────────────────────────

    def improvement_value(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        positive_rate: float = 0.2,
        total_predictions: int = 10000,
        cost_fn: float = 100,
        cost_fp: float = 10,
    ) -> Dict[str, Any]:
        """
        Calculate the dollar value of improving metrics from current to target.
        """
        current_cost = self.cost_matrix_analysis(
            current_metrics, cost_fp=cost_fp, cost_fn=cost_fn,
            total_predictions=total_predictions, positive_rate=positive_rate,
        )
        target_cost = self.cost_matrix_analysis(
            target_metrics, cost_fp=cost_fp, cost_fn=cost_fn,
            total_predictions=total_predictions, positive_rate=positive_rate,
        )

        current_total = current_cost.get("costs", {}).get("total_model_cost", 0)
        target_total = target_cost.get("costs", {}).get("total_model_cost", 0)
        savings = current_total - target_total

        improvements = {}
        for metric in ["precision", "recall", "f1_score", "roc_auc"]:
            current_val = current_metrics.get(metric, 0)
            target_val = target_metrics.get(metric, 0)
            if current_val and target_val:
                improvements[metric] = {
                    "current": round(current_val, 4),
                    "target": round(target_val, 4),
                    "improvement": round(target_val - current_val, 4),
                }

        return {
            "annual_savings_from_improvement": round(savings, 0),
            "metric_improvements": improvements,
            "worth_investing": savings > 0,
            "recommendation": (
                f"Improving to target metrics would save ~${savings:,.0f} annually."
                if savings > 0 else
                "Target metrics don't improve business outcome — focus elsewhere."
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 6. CAPACITY PLANNING
    # ──────────────────────────────────────────────────────────

    def capacity_planning(
        self,
        metrics: Dict[str, float],
        total_predictions: int = 10000,
        positive_rate: float = 0.2,
        team_capacity: int = 100,
    ) -> Dict[str, Any]:
        """
        Calculate how many model predictions a team can realistically act upon.
        
        Args:
            team_capacity: How many cases can the team investigate per period
        """
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        if not precision:
            return {"error": "Need precision for capacity planning"}

        expected_positives = int(total_predictions * positive_rate)

        # Model flags
        tp = int(expected_positives * recall)
        fp = int(tp / precision - tp) if precision > 0 else 0
        total_flagged = tp + fp

        # Can the team handle all flags?
        can_handle_all = total_flagged <= team_capacity
        utilization = total_flagged / team_capacity if team_capacity > 0 else float('inf')

        if can_handle_all:
            efficiency = tp / total_flagged if total_flagged > 0 else 0
            recommendation = f"Team can handle all {total_flagged} flags. {efficiency:.0%} will be true positives."
        else:
            # Need to prioritize — effectively raising the threshold
            effective_capacity_ratio = team_capacity / total_flagged
            effective_precision = min(1.0, precision / effective_capacity_ratio) if effective_capacity_ratio > 0 else precision
            cases_handled = min(team_capacity, total_flagged)
            true_positives_handled = int(cases_handled * precision)
            recommendation = (
                f"Model flags {total_flagged} cases but team can only handle {team_capacity}. "
                f"Prioritize by prediction probability to maximize true positives in the top {team_capacity}."
            )

        return {
            "total_predictions": total_predictions,
            "model_flags": total_flagged,
            "team_capacity": team_capacity,
            "utilization": round(utilization, 2),
            "can_handle_all": can_handle_all,
            "true_positives_in_flags": tp,
            "false_positives_in_flags": fp,
            "recommendation": recommendation,
            "optimization_tip": (
                "If overloaded: raise the prediction threshold to flag fewer, "
                "higher-confidence cases that fit within team capacity."
                if not can_handle_all else
                "If underutilized: lower the threshold to capture more potential positives."
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 7. AGGREGATE BUSINESS REPORT
    # ──────────────────────────────────────────────────────────

    def full_business_report(
        self,
        context: Dict[str, Any],
        business_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive business impact report from context.
        Business params can be passed from frontend or use defaults.
        """
        screen_ctx = context.get("screen_context", {}) or {}
        frontend = context.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})

        if not metrics:
            return {"error": "No model metrics available for business analysis"}

        params = business_params or {}
        positive_rate = params.get("positive_rate", 0.2)

        report = {
            "cost_analysis": self.cost_matrix_analysis(
                metrics,
                cost_fp=params.get("cost_fp", 10),
                cost_fn=params.get("cost_fn", 100),
                total_predictions=params.get("total_predictions", 10000),
                positive_rate=positive_rate,
            ),
            "lift": self.calculate_lift(metrics, positive_rate),
        }

        # Add churn-specific analysis if detected
        problem_type = screen_ctx.get("problem_type") or frontend.get("problem_type", "")
        dataset_name = context.get("dataset_profile", {}).get("file_name", "").lower()

        if "churn" in dataset_name or "churn" in problem_type.lower():
            report["churn_analysis"] = self.churn_impact(
                metrics,
                total_customers=params.get("total_customers", 10000),
                churn_rate=positive_rate,
                avg_customer_value=params.get("avg_customer_value", 500),
                intervention_cost=params.get("intervention_cost", 50),
            )

        return report
