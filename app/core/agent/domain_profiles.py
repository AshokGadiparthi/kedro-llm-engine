"""
Domain Profiles — Context-Adaptive Threshold System
=====================================================
Replaces hardcoded magic numbers with domain-aware thresholds.
Each domain profile calibrates what counts as "critical" vs "warning"
based on industry-specific norms and tolerances.

SOLVES: "50% missing is normal in medical data but a crisis in financial data."

Architecture:
  - 12 pre-built domain profiles (healthcare, finance, retail, etc.)
  - Auto-detection from dataset metadata (column names, value ranges)
  - User-configurable overrides
  - Composable: combine base + domain + user layers
  - Every threshold in the RuleEngine becomes a lookup, not a literal

Usage:
  profile = DomainProfileManager.detect(context)
  threshold = profile.get_threshold("missing_pct_critical")
  # Returns 70 for healthcare, 30 for finance, 50 for general
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# THRESHOLD DEFINITIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class DomainThresholds:
    """All configurable thresholds used across the engine."""

    # ── Data Quality ──
    missing_pct_critical: float = 70.0
    missing_pct_warning: float = 85.0
    missing_pct_column_drop: float = 50.0
    missing_pct_column_warn: float = 20.0
    duplicate_pct_critical: float = 10.0
    duplicate_pct_warning: float = 3.0
    quality_score_good: float = 90.0

    # ── Feature Health ──
    skew_threshold_moderate: float = 1.0
    skew_threshold_high: float = 2.0
    high_cardinality_unique_count: int = 50
    near_unique_ratio: float = 0.8
    zero_variance_tolerance: float = 0.001

    # ── Class Imbalance ──
    imbalance_severe_pct: float = 5.0
    imbalance_moderate_pct: float = 20.0

    # ── Sample Size ──
    min_rows_critical: int = 100
    min_rows_warning: int = 500
    min_rows_complex_model: int = 1000
    min_rows_neural: int = 5000
    sample_feature_ratio_critical: float = 5.0
    sample_feature_ratio_warning: float = 10.0

    # ── Correlation / Leakage ──
    correlation_leakage: float = 0.98
    correlation_high: float = 0.80
    correlation_moderate: float = 0.70

    # ── Evaluation ──
    accuracy_f1_gap_critical: float = 0.15
    precision_recall_gap_warning: float = 0.25
    overfit_gap_critical: float = 0.20
    overfit_gap_warning: float = 0.10
    auc_suspicious: float = 0.99
    auc_strong: float = 0.85
    auc_weak: float = 0.60
    recall_low: float = 0.50

    # ── Training ──
    test_size_min: float = 0.10
    test_size_max: float = 0.40
    cv_min_samples_per_fold: int = 30
    svm_max_rows: int = 10000

    # ── Monitoring ──
    error_rate_critical: float = 10.0
    error_rate_warning: float = 5.0
    latency_warning_ms: float = 500.0
    psi_significant: float = 0.25
    psi_moderate: float = 0.10

    # ── Business Impact ──
    default_cost_fp: float = 10.0
    default_cost_fn: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def override(self, overrides: Dict[str, Any]) -> 'DomainThresholds':
        """Return a new DomainThresholds with overrides applied."""
        new = deepcopy(self)
        for k, v in overrides.items():
            if hasattr(new, k):
                setattr(new, k, v)
        return new


# ═══════════════════════════════════════════════════════════════
# DOMAIN PROFILES
# ═══════════════════════════════════════════════════════════════

DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "healthcare": {
        "name": "Healthcare / Clinical",
        "description": "Medical data with expected sparsity, critical recall requirements",
        "thresholds": {
            # Medical data commonly has many optional tests/measurements
            "missing_pct_critical": 85.0,
            "missing_pct_warning": 95.0,
            "missing_pct_column_drop": 70.0,
            "missing_pct_column_warn": 40.0,
            # Diagnosing disease: missing a positive case is catastrophic
            "imbalance_severe_pct": 2.0,
            "imbalance_moderate_pct": 10.0,
            "recall_low": 0.70,
            # More conservative on overfitting for patient safety
            "overfit_gap_critical": 0.15,
            "overfit_gap_warning": 0.08,
            "auc_strong": 0.80,
            # False negatives cost lives
            "default_cost_fn": 1000.0,
            "default_cost_fp": 50.0,
            "accuracy_f1_gap_critical": 0.10,
        },
        "detection_hints": {
            "column_patterns": [
                r"patient", r"diagnos", r"symptom", r"icd[_-]?\d", r"lab[_-]?result",
                r"blood[_-]?pressure", r"bmi", r"glucose", r"heart[_-]?rate",
                r"medical", r"clinical", r"prescription", r"dosage", r"hospital",
                r"treatment", r"prognosis", r"mortality", r"survival",
            ],
            "value_hints": {"has_icd_codes": True},
        },
    },

    "finance": {
        "name": "Finance / Banking",
        "description": "Financial data with strict accuracy, regulatory compliance",
        "thresholds": {
            # Financial data should be complete and clean
            "missing_pct_critical": 30.0,
            "missing_pct_warning": 60.0,
            "missing_pct_column_drop": 25.0,
            "missing_pct_column_warn": 10.0,
            "duplicate_pct_critical": 5.0,
            "duplicate_pct_warning": 1.0,
            # Fraud is rare — severe imbalance is expected
            "imbalance_severe_pct": 1.0,
            "imbalance_moderate_pct": 5.0,
            # Tight overfitting control
            "overfit_gap_critical": 0.12,
            "overfit_gap_warning": 0.06,
            # Regulatory: must explain decisions
            "correlation_high": 0.75,
            # Fraud: missing fraud is expensive
            "default_cost_fn": 5000.0,
            "default_cost_fp": 25.0,
            "recall_low": 0.60,
            "accuracy_f1_gap_critical": 0.08,
        },
        "detection_hints": {
            "column_patterns": [
                r"transaction", r"amount", r"balance", r"credit", r"debit",
                r"account", r"fraud", r"risk[_-]?score", r"loan", r"interest",
                r"revenue", r"profit", r"portfolio", r"asset", r"liability",
                r"currency", r"exchange[_-]?rate", r"stock", r"price",
            ],
        },
    },

    "retail": {
        "name": "Retail / E-commerce",
        "description": "Customer and product data, churn prediction, recommendation",
        "thresholds": {
            "missing_pct_critical": 60.0,
            "missing_pct_warning": 80.0,
            "missing_pct_column_drop": 45.0,
            "duplicate_pct_critical": 15.0,  # Repeat purchases are normal
            "duplicate_pct_warning": 5.0,
            "imbalance_severe_pct": 8.0,
            "imbalance_moderate_pct": 25.0,
            "high_cardinality_unique_count": 100,  # Product IDs, SKUs
            "default_cost_fn": 200.0,  # Lost customer
            "default_cost_fp": 5.0,    # Wasted promo
        },
        "detection_hints": {
            "column_patterns": [
                r"product", r"sku", r"order", r"cart", r"purchase", r"customer",
                r"churn", r"lifetime[_-]?value", r"ltv", r"click", r"conversion",
                r"category", r"price", r"quantity", r"discount", r"coupon",
            ],
        },
    },

    "manufacturing": {
        "name": "Manufacturing / IoT",
        "description": "Sensor data, predictive maintenance, quality control",
        "thresholds": {
            "missing_pct_critical": 40.0,
            "missing_pct_warning": 65.0,
            "duplicate_pct_critical": 20.0,  # Sensor readings can repeat
            "imbalance_severe_pct": 3.0,     # Failures are rare events
            "imbalance_moderate_pct": 10.0,
            "min_rows_critical": 500,         # Need more data for temporal patterns
            "latency_warning_ms": 100.0,      # Real-time monitoring
            "default_cost_fn": 10000.0,       # Equipment failure
            "default_cost_fp": 200.0,         # Unnecessary maintenance
            "recall_low": 0.85,               # Must catch failures
            "psi_significant": 0.15,          # Tighter drift for sensor data
        },
        "detection_hints": {
            "column_patterns": [
                r"sensor", r"temperature", r"pressure", r"vibration", r"rpm",
                r"voltage", r"current", r"humidity", r"machine", r"equipment",
                r"maintenance", r"failure", r"anomaly", r"defect", r"quality",
            ],
        },
    },

    "marketing": {
        "name": "Marketing / Advertising",
        "description": "Campaign data, CTR prediction, attribution",
        "thresholds": {
            "missing_pct_critical": 60.0,
            "missing_pct_warning": 80.0,
            "imbalance_severe_pct": 2.0,     # Click rates are tiny
            "imbalance_moderate_pct": 10.0,
            "high_cardinality_unique_count": 200,
            "default_cost_fn": 0.50,         # Missed click
            "default_cost_fp": 0.10,         # Wasted impression
            "auc_strong": 0.75,              # Lower bar — prediction is hard
        },
        "detection_hints": {
            "column_patterns": [
                r"campaign", r"impression", r"click", r"ctr", r"conversion",
                r"ad[_-]?group", r"keyword", r"channel", r"attribution",
                r"engagement", r"reach", r"cpm", r"cpc", r"roas",
            ],
        },
    },

    "insurance": {
        "name": "Insurance / Actuarial",
        "description": "Claims data, risk assessment, underwriting",
        "thresholds": {
            "missing_pct_critical": 40.0,
            "missing_pct_warning": 70.0,
            "imbalance_severe_pct": 5.0,
            "imbalance_moderate_pct": 15.0,
            "overfit_gap_critical": 0.15,
            "default_cost_fn": 2000.0,       # Missed claim
            "default_cost_fp": 100.0,        # Over-pricing
            "recall_low": 0.65,
        },
        "detection_hints": {
            "column_patterns": [
                r"claim", r"premium", r"policy", r"coverage", r"underwrite",
                r"actuarial", r"exposure", r"loss[_-]?ratio", r"deductible",
                r"insured", r"beneficiary", r"risk[_-]?class",
            ],
        },
    },

    "telecom": {
        "name": "Telecommunications",
        "description": "Network data, churn prediction, usage analysis",
        "thresholds": {
            "missing_pct_critical": 50.0,
            "missing_pct_warning": 75.0,
            "imbalance_severe_pct": 10.0,
            "imbalance_moderate_pct": 25.0,
            "default_cost_fn": 500.0,       # Lost subscriber
            "default_cost_fp": 20.0,        # Unnecessary retention offer
        },
        "detection_hints": {
            "column_patterns": [
                r"call[_-]?duration", r"data[_-]?usage", r"subscriber",
                r"plan", r"churn", r"network", r"signal", r"roaming",
                r"sms", r"minutes", r"bandwidth", r"latency",
            ],
        },
    },

    "hr": {
        "name": "Human Resources",
        "description": "Employee data, attrition prediction, performance",
        "thresholds": {
            "missing_pct_critical": 50.0,
            "missing_pct_warning": 75.0,
            "imbalance_severe_pct": 10.0,
            "imbalance_moderate_pct": 25.0,
            "min_rows_critical": 50,          # Small org datasets
            "min_rows_warning": 200,
            "default_cost_fn": 15000.0,       # Replacement cost
            "default_cost_fp": 500.0,
        },
        "detection_hints": {
            "column_patterns": [
                r"employee", r"salary", r"department", r"tenure", r"attrition",
                r"performance", r"satisfaction", r"overtime", r"promotion",
                r"hiring", r"turnover", r"engagement",
            ],
        },
    },

    "cybersecurity": {
        "name": "Cybersecurity",
        "description": "Intrusion detection, threat analysis",
        "thresholds": {
            "missing_pct_critical": 30.0,
            "missing_pct_warning": 50.0,
            "imbalance_severe_pct": 0.5,     # Attacks are very rare
            "imbalance_moderate_pct": 5.0,
            "latency_warning_ms": 50.0,       # Real-time detection
            "default_cost_fn": 50000.0,       # Missed breach
            "default_cost_fp": 10.0,          # Alert fatigue
            "recall_low": 0.90,               # Must catch attacks
            "psi_significant": 0.10,
        },
        "detection_hints": {
            "column_patterns": [
                r"intrusion", r"attack", r"threat", r"malware", r"packet",
                r"ip[_-]?address", r"port", r"protocol", r"firewall",
                r"anomaly", r"log[_-]?entry", r"session",
            ],
        },
    },

    "nlp": {
        "name": "NLP / Text",
        "description": "Text classification, sentiment analysis, NER",
        "thresholds": {
            "missing_pct_critical": 50.0,
            "missing_pct_warning": 75.0,
            "high_cardinality_unique_count": 1000,
            "min_rows_critical": 500,
            "min_rows_warning": 2000,
            "min_rows_neural": 10000,
            "auc_strong": 0.80,
        },
        "detection_hints": {
            "column_patterns": [
                r"text", r"content", r"review", r"comment", r"tweet",
                r"sentiment", r"label", r"language", r"token", r"embedding",
                r"title", r"description", r"body", r"document",
            ],
        },
    },

    "energy": {
        "name": "Energy / Utilities",
        "description": "Load forecasting, consumption prediction, grid management",
        "thresholds": {
            "missing_pct_critical": 40.0,
            "missing_pct_warning": 65.0,
            "duplicate_pct_critical": 20.0,
            "imbalance_severe_pct": 5.0,
            "latency_warning_ms": 200.0,
            "default_cost_fn": 5000.0,
            "default_cost_fp": 100.0,
            "psi_significant": 0.20,
        },
        "detection_hints": {
            "column_patterns": [
                r"consumption", r"kwh", r"energy", r"power", r"load",
                r"generation", r"grid", r"meter", r"utility", r"solar",
                r"wind", r"peak[_-]?demand", r"outage",
            ],
        },
    },

    "education": {
        "name": "Education",
        "description": "Student performance, dropout prediction, course recommendation",
        "thresholds": {
            "missing_pct_critical": 60.0,
            "missing_pct_warning": 80.0,
            "imbalance_severe_pct": 10.0,
            "imbalance_moderate_pct": 25.0,
            "min_rows_critical": 50,
            "min_rows_warning": 200,
            "default_cost_fn": 1000.0,
            "default_cost_fp": 50.0,
        },
        "detection_hints": {
            "column_patterns": [
                r"student", r"grade", r"gpa", r"course", r"enrollment",
                r"dropout", r"attendance", r"score", r"exam", r"assignment",
                r"graduation", r"semester", r"academic",
            ],
        },
    },
}


# ═══════════════════════════════════════════════════════════════
# DOMAIN PROFILE RESULT
# ═══════════════════════════════════════════════════════════════

@dataclass
class DomainProfile:
    """Detected or configured domain profile with calibrated thresholds."""
    domain: str = "general"
    domain_name: str = "General Purpose"
    detection_confidence: float = 0.0
    detection_evidence: List[str] = field(default_factory=list)
    thresholds: DomainThresholds = field(default_factory=DomainThresholds)
    user_overrides: Dict[str, Any] = field(default_factory=dict)

    def get_threshold(self, key: str, default: Any = None) -> Any:
        """Get a threshold value. User overrides > domain > base."""
        if key in self.user_overrides:
            return self.user_overrides[key]
        return self.thresholds.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "domain_name": self.domain_name,
            "detection_confidence": round(self.detection_confidence, 3),
            "detection_evidence": self.detection_evidence,
            "thresholds": self.thresholds.to_dict(),
            "user_overrides": self.user_overrides,
        }


# ═══════════════════════════════════════════════════════════════
# DOMAIN PROFILE MANAGER
# ═══════════════════════════════════════════════════════════════

class DomainProfileManager:
    """
    Detects domain from dataset metadata and provides calibrated thresholds.

    Detection uses column name pattern matching + value range heuristics.
    Falls back to 'general' profile (original hardcoded thresholds).
    """

    @staticmethod
    def detect(context: Dict[str, Any], user_domain: Optional[str] = None,
               user_overrides: Optional[Dict[str, Any]] = None) -> DomainProfile:
        """
        Detect or load domain profile from context.

        Priority: user_domain > auto-detect > general
        """
        user_overrides = user_overrides or {}

        # 1. User explicitly set domain
        if user_domain and user_domain in DOMAIN_PROFILES:
            profile_def = DOMAIN_PROFILES[user_domain]
            base = DomainThresholds()
            thresholds = base.override(profile_def.get("thresholds", {}))
            if user_overrides:
                thresholds = thresholds.override(user_overrides)
            return DomainProfile(
                domain=user_domain,
                domain_name=profile_def["name"],
                detection_confidence=1.0,
                detection_evidence=["user_specified"],
                thresholds=thresholds,
                user_overrides=user_overrides,
            )

        # 2. Auto-detect from dataset metadata
        detected_domain, confidence, evidence = DomainProfileManager._auto_detect(context)

        if detected_domain and confidence >= 0.3:
            profile_def = DOMAIN_PROFILES[detected_domain]
            base = DomainThresholds()
            thresholds = base.override(profile_def.get("thresholds", {}))
            if user_overrides:
                thresholds = thresholds.override(user_overrides)
            return DomainProfile(
                domain=detected_domain,
                domain_name=profile_def["name"],
                detection_confidence=confidence,
                detection_evidence=evidence,
                thresholds=thresholds,
                user_overrides=user_overrides,
            )

        # 3. General purpose defaults (original behavior)
        base = DomainThresholds()
        if user_overrides:
            base = base.override(user_overrides)
        return DomainProfile(
            domain="general",
            domain_name="General Purpose",
            detection_confidence=0.0,
            detection_evidence=[],
            thresholds=base,
            user_overrides=user_overrides,
        )

    @staticmethod
    def _auto_detect(context: Dict[str, Any]) -> Tuple[Optional[str], float, List[str]]:
        """Auto-detect domain from column names and data characteristics."""
        profile = context.get("dataset_profile", {})
        feature_stats = context.get("feature_stats", {})
        dtypes = profile.get("dtypes", {})
        col_types = profile.get("column_types", {})

        # Gather all column names
        all_columns = set()
        for k, v in dtypes.items():
            all_columns.add(k.lower())
        for type_group in col_types.values():
            if isinstance(type_group, list):
                for c in type_group:
                    all_columns.add(str(c).lower())

        # Also check numeric_stats keys
        numeric_stats = feature_stats.get("numeric_stats", {})
        for k in numeric_stats:
            all_columns.add(str(k).lower())

        if not all_columns:
            return None, 0.0, []

        # Score each domain
        scores: Dict[str, Tuple[float, List[str]]] = {}
        columns_str = " ".join(all_columns)

        for domain_key, domain_def in DOMAIN_PROFILES.items():
            hints = domain_def.get("detection_hints", {})
            patterns = hints.get("column_patterns", [])
            matches = []

            for pattern in patterns:
                for col in all_columns:
                    if re.search(pattern, col, re.IGNORECASE):
                        matches.append(f"column '{col}' matches '{pattern}'")
                        break  # One match per pattern is enough

            if matches:
                score = len(matches) / max(len(patterns), 1)
                # Bonus for multiple matches
                if len(matches) >= 3:
                    score = min(1.0, score + 0.15)
                if len(matches) >= 5:
                    score = min(1.0, score + 0.15)
                scores[domain_key] = (score, matches)

        if not scores:
            return None, 0.0, []

        # Pick best domain
        best_domain = max(scores, key=lambda k: scores[k][0])
        best_score, best_evidence = scores[best_domain]

        return best_domain, best_score, best_evidence[:10]

    @staticmethod
    def list_domains() -> List[Dict[str, str]]:
        """List all available domain profiles."""
        result = [{"key": "general", "name": "General Purpose",
                    "description": "Default balanced thresholds for any dataset"}]
        for key, profile in DOMAIN_PROFILES.items():
            result.append({
                "key": key,
                "name": profile["name"],
                "description": profile["description"],
            })
        return result

    @staticmethod
    def get_profile_details(domain: str) -> Optional[Dict[str, Any]]:
        """Get full details of a domain profile."""
        if domain == "general":
            return {
                "key": "general",
                "name": "General Purpose",
                "description": "Default balanced thresholds",
                "thresholds": DomainThresholds().to_dict(),
            }
        if domain in DOMAIN_PROFILES:
            profile = DOMAIN_PROFILES[domain]
            base = DomainThresholds()
            calibrated = base.override(profile.get("thresholds", {}))
            return {
                "key": domain,
                "name": profile["name"],
                "description": profile["description"],
                "thresholds": calibrated.to_dict(),
                "detection_hints": profile.get("detection_hints", {}),
            }
        return None


# ═══════════════════════════════════════════════════════════════
# HELPER: Apply thresholds to rule engine
# ═══════════════════════════════════════════════════════════════

def inject_thresholds(context: Dict[str, Any], profile: DomainProfile) -> Dict[str, Any]:
    """
    Inject domain thresholds into context so the rule engine can use them.
    The rule engine reads context["domain_thresholds"] instead of hardcoded values.
    """
    context["domain_profile"] = profile.to_dict()
    context["domain_thresholds"] = profile.thresholds.to_dict()
    context["_threshold_getter"] = profile.get_threshold
    return context
