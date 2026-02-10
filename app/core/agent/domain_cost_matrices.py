"""
Domain Cost Matrices — Industry-Specific Business Impact Models
================================================================
Replaces generic cost formulas with domain-calibrated cost matrices.

SOLVES: "Healthcare vs fintech vs retail have completely different FP/FN costs."

Each domain profile includes:
  - Cost per false positive / false negative (calibrated to industry)
  - Revenue impact models specific to the use case
  - Capacity planning constraints
  - Regulatory penalty multipliers
  - Time-to-action windows

Usage:
  matrix = DomainCostMatrix.get("healthcare", "diagnosis")
  impact = matrix.compute_impact(precision=0.85, recall=0.92, n_predictions=10000)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CostMatrix:
    """Industry-calibrated cost matrix for a specific use case."""
    domain: str
    use_case: str
    description: str

    # Core costs (in USD)
    cost_fp: float = 10.0     # Cost of a false positive
    cost_fn: float = 100.0    # Cost of a false negative
    benefit_tp: float = 0.0   # Revenue saved/gained per true positive
    cost_tn: float = 0.0      # Cost of true negative (usually 0)

    # Multipliers
    regulatory_penalty_multiplier: float = 1.0  # Extra penalty for regulated industries
    reputational_cost_multiplier: float = 1.0   # Brand damage multiplier
    time_sensitivity_hours: float = 24.0         # Hours before cost compounds

    # Volume assumptions
    typical_monthly_volume: int = 10000
    positive_rate: float = 0.1  # Expected positive class rate

    # Capacity constraints
    max_actions_per_day: int = 100  # How many positives can be acted on
    action_cost_per_case: float = 0.0  # Cost to investigate/act on a flagged case

    # Notes for user
    cost_justification: str = ""

    def compute_impact(
        self,
        precision: float,
        recall: float,
        n_predictions: Optional[int] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Compute full business impact for this domain/use case."""
        n = n_predictions or self.typical_monthly_volume
        n_positive = int(n * self.positive_rate)
        n_negative = n - n_positive

        tp = int(n_positive * recall)
        fn = n_positive - tp
        fp = int(tp / max(precision, 0.01) - tp)
        tn = max(0, n_negative - fp)

        # Core costs
        fp_cost = fp * self.cost_fp
        fn_cost = fn * self.cost_fn * self.regulatory_penalty_multiplier
        tp_benefit = tp * self.benefit_tp
        action_cost = (tp + fp) * self.action_cost_per_case

        total_model_cost = fp_cost + fn_cost + action_cost - tp_benefit
        no_model_cost = n_positive * self.cost_fn * self.regulatory_penalty_multiplier
        savings = no_model_cost - total_model_cost

        # Reputational impact for false negatives
        reputational_fn_cost = fn * self.cost_fn * self.reputational_cost_multiplier

        # Capacity check
        daily_flags = (tp + fp) / 30  # monthly to daily
        capacity_exceeded = daily_flags > self.max_actions_per_day

        return {
            "domain": self.domain,
            "use_case": self.use_case,
            "confusion_matrix": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            },
            "costs": {
                "false_positive_cost": round(fp_cost, 2),
                "false_negative_cost": round(fn_cost, 2),
                "true_positive_benefit": round(tp_benefit, 2),
                "action_investigation_cost": round(action_cost, 2),
                "total_model_cost": round(total_model_cost, 2),
                "cost_without_model": round(no_model_cost, 2),
                "net_savings": round(savings, 2),
                "reputational_risk": round(reputational_fn_cost, 2),
            },
            "roi": {
                "monthly_savings": round(savings, 2),
                "annual_savings": round(savings * 12, 2),
                "roi_pct": round(savings / max(total_model_cost, 1) * 100, 1),
            },
            "capacity": {
                "daily_flags": round(daily_flags, 1),
                "max_daily_capacity": self.max_actions_per_day,
                "capacity_exceeded": capacity_exceeded,
                "utilization_pct": round(daily_flags / max(self.max_actions_per_day, 1) * 100, 1),
            },
            "recommendations": self._generate_recommendations(
                precision, recall, fp, fn, capacity_exceeded, savings
            ),
        }

    def _generate_recommendations(
        self, precision, recall, fp, fn, capacity_exceeded, savings
    ) -> List[str]:
        recs = []
        if fn > fp * 2 and self.cost_fn > self.cost_fp * 5:
            recs.append(
                f"Lower the threshold to catch more {self.use_case} cases. "
                f"Each missed case costs ${self.cost_fn:,.0f}."
            )
        if fp > fn * 3:
            recs.append(
                f"Raise the threshold to reduce false alarms. "
                f"Currently generating {fp:,} unnecessary investigations per period."
            )
        if capacity_exceeded:
            recs.append(
                f"Model flags more cases than your team can handle. "
                f"Raise threshold or increase team capacity."
            )
        if savings < 0:
            recs.append(
                f"Model costs more than it saves (net loss: ${abs(savings):,.0f}). "
                f"Review model quality or cost assumptions."
            )
        elif savings > 0:
            recs.append(
                f"Model saves ${savings:,.0f}/month (${savings*12:,.0f}/year)."
            )
        return recs

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════
# DOMAIN COST MATRIX REGISTRY
# ═══════════════════════════════════════════════════════════════

DOMAIN_COST_MATRICES: Dict[str, Dict[str, CostMatrix]] = {
    "healthcare": {
        "diagnosis": CostMatrix(
            domain="healthcare", use_case="diagnosis",
            description="Disease diagnosis / screening",
            cost_fp=200, cost_fn=15000, benefit_tp=5000,
            regulatory_penalty_multiplier=2.0,
            reputational_cost_multiplier=3.0,
            time_sensitivity_hours=4,
            typical_monthly_volume=5000,
            positive_rate=0.05,
            max_actions_per_day=50,
            action_cost_per_case=100,
            cost_justification=(
                "FN cost includes delayed treatment, malpractice risk, and regulatory fines. "
                "FP cost is follow-up testing. TP benefit is early intervention savings."
            ),
        ),
        "readmission": CostMatrix(
            domain="healthcare", use_case="readmission",
            description="Hospital readmission prediction",
            cost_fp=500, cost_fn=25000, benefit_tp=8000,
            regulatory_penalty_multiplier=1.5,
            typical_monthly_volume=2000,
            positive_rate=0.15,
            max_actions_per_day=30,
            action_cost_per_case=200,
        ),
    },
    "finance": {
        "fraud": CostMatrix(
            domain="finance", use_case="fraud",
            description="Fraud detection",
            cost_fp=25, cost_fn=5000, benefit_tp=4500,
            regulatory_penalty_multiplier=2.5,
            reputational_cost_multiplier=2.0,
            time_sensitivity_hours=1,
            typical_monthly_volume=100000,
            positive_rate=0.02,
            max_actions_per_day=500,
            action_cost_per_case=15,
            cost_justification=(
                "FN = full fraud amount + chargeback fees + regulatory penalty. "
                "FP = customer friction + investigation cost."
            ),
        ),
        "credit": CostMatrix(
            domain="finance", use_case="credit",
            description="Credit risk / loan default prediction",
            cost_fp=200, cost_fn=10000, benefit_tp=1500,
            regulatory_penalty_multiplier=1.5,
            typical_monthly_volume=5000,
            positive_rate=0.08,
            max_actions_per_day=200,
            action_cost_per_case=50,
        ),
    },
    "retail": {
        "churn": CostMatrix(
            domain="retail", use_case="churn",
            description="Customer churn prediction",
            cost_fp=5, cost_fn=200, benefit_tp=150,
            typical_monthly_volume=50000,
            positive_rate=0.15,
            max_actions_per_day=1000,
            action_cost_per_case=5,
            cost_justification=(
                "FN = lost customer lifetime value. FP = wasted retention offer. "
                "TP = saved customer minus retention cost."
            ),
        ),
        "recommendation": CostMatrix(
            domain="retail", use_case="recommendation",
            description="Product recommendation",
            cost_fp=0.10, cost_fn=2.0, benefit_tp=5.0,
            typical_monthly_volume=500000,
            positive_rate=0.05,
            max_actions_per_day=100000,
            action_cost_per_case=0.01,
        ),
    },
    "manufacturing": {
        "predictive_maintenance": CostMatrix(
            domain="manufacturing", use_case="predictive_maintenance",
            description="Equipment failure prediction",
            cost_fp=500, cost_fn=50000, benefit_tp=40000,
            time_sensitivity_hours=8,
            typical_monthly_volume=1000,
            positive_rate=0.03,
            max_actions_per_day=10,
            action_cost_per_case=200,
            cost_justification=(
                "FN = unplanned downtime + repair + production loss. "
                "FP = unnecessary maintenance window. "
                "TP = planned maintenance vs emergency repair savings."
            ),
        ),
        "quality": CostMatrix(
            domain="manufacturing", use_case="quality",
            description="Product quality / defect detection",
            cost_fp=20, cost_fn=1000, benefit_tp=500,
            typical_monthly_volume=10000,
            positive_rate=0.02,
            max_actions_per_day=100,
            action_cost_per_case=10,
        ),
    },
    "cybersecurity": {
        "intrusion": CostMatrix(
            domain="cybersecurity", use_case="intrusion",
            description="Intrusion / threat detection",
            cost_fp=10, cost_fn=100000, benefit_tp=90000,
            regulatory_penalty_multiplier=3.0,
            reputational_cost_multiplier=5.0,
            time_sensitivity_hours=0.5,
            typical_monthly_volume=1000000,
            positive_rate=0.001,
            max_actions_per_day=200,
            action_cost_per_case=50,
        ),
    },
    "insurance": {
        "claims": CostMatrix(
            domain="insurance", use_case="claims",
            description="Fraudulent claims detection",
            cost_fp=200, cost_fn=8000, benefit_tp=7000,
            regulatory_penalty_multiplier=1.5,
            typical_monthly_volume=10000,
            positive_rate=0.05,
            max_actions_per_day=50,
            action_cost_per_case=100,
        ),
    },
    "hr": {
        "attrition": CostMatrix(
            domain="hr", use_case="attrition",
            description="Employee attrition prediction",
            cost_fp=500, cost_fn=15000, benefit_tp=10000,
            typical_monthly_volume=500,
            positive_rate=0.12,
            max_actions_per_day=20,
            action_cost_per_case=200,
        ),
    },
    "marketing": {
        "conversion": CostMatrix(
            domain="marketing", use_case="conversion",
            description="Conversion prediction / lead scoring",
            cost_fp=0.50, cost_fn=50, benefit_tp=30,
            typical_monthly_volume=100000,
            positive_rate=0.03,
            max_actions_per_day=5000,
            action_cost_per_case=1.0,
        ),
    },
}


class DomainCostMatrixManager:
    """Manager for looking up and using domain cost matrices."""

    @staticmethod
    def get(domain: str, use_case: Optional[str] = None) -> Optional[CostMatrix]:
        """Get a specific cost matrix."""
        domain_matrices = DOMAIN_COST_MATRICES.get(domain, {})
        if use_case and use_case in domain_matrices:
            return domain_matrices[use_case]
        # Return first available for domain
        if domain_matrices:
            return next(iter(domain_matrices.values()))
        return None

    @staticmethod
    def get_default_for_domain(domain: str) -> CostMatrix:
        """Get default cost matrix for domain, falling back to general."""
        matrix = DomainCostMatrixManager.get(domain)
        if matrix:
            return matrix
        return CostMatrix(domain="general", use_case="default",
                         description="General purpose defaults")

    @staticmethod
    def list_available() -> Dict[str, List[Dict[str, str]]]:
        """List all available domain cost matrices."""
        result = {}
        for domain, cases in DOMAIN_COST_MATRICES.items():
            result[domain] = [
                {"use_case": uc, "description": matrix.description}
                for uc, matrix in cases.items()
            ]
        return result

    @staticmethod
    def auto_select(domain: str, context: Dict[str, Any]) -> CostMatrix:
        """Auto-select the best cost matrix based on domain + context clues."""
        domain_matrices = DOMAIN_COST_MATRICES.get(domain, {})
        if not domain_matrices:
            return CostMatrix(domain=domain, use_case="auto",
                             description="Auto-generated defaults")

        # Try to match based on context clues
        target = (context.get("screen_context", {}) or {}).get("target_column", "")
        target_lower = target.lower() if target else ""

        # Heuristic matching
        for use_case, matrix in domain_matrices.items():
            keywords = use_case.lower().split("_")
            if any(kw in target_lower for kw in keywords):
                return matrix

        # Return first available
        return next(iter(domain_matrices.values()))
