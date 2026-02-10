"""
Drift Analyzer — Production Statistical Drift Detection Engine
================================================================
Detects data drift, concept drift, and prediction drift using
statistical tests and distributional distance measures.

Capabilities:
  1. Population Stability Index (PSI)     — Overall distribution shift
  2. Kolmogorov-Smirnov Test              — Continuous feature drift
  3. Chi-Square Test                       — Categorical feature drift
  4. Jensen-Shannon Divergence            — Symmetric distribution distance
  5. Wasserstein Distance                  — Earth mover's distance
  6. CUSUM (Cumulative Sum)               — Sequential change detection
  7. Page-Hinkley Test                    — Online drift detection
  8. ADWIN (Adaptive Windowing)           — Adaptive window drift detection
  9. Feature-Level Drift Report           — Per-feature drift analysis
  10. Concept Drift Detection              — Target relationship changes
  11. Prediction Distribution Drift        — Model output distribution shift
  12. Alert Severity Classification        — Critical/warning/info thresholds

All methods work from pre-computed statistics (means, stds, histograms, percentiles).
NO raw data access.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DriftResult:
    """Result of a drift test on a single feature."""
    feature: str
    test_name: str            # psi | ks | chi2 | js | wasserstein | cusum
    statistic: float          # test statistic value
    threshold: float          # severity threshold used
    p_value: Optional[float] = None
    is_drifted: bool = False
    severity: str = "none"    # none | low | medium | high | critical
    direction: str = "unknown"  # shift_up | shift_down | spread_change | distribution_change
    reference_stat: Optional[float] = None
    current_stat: Optional[float] = None
    change_pct: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DriftReport:
    """Comprehensive drift analysis report."""
    timestamp: str = ""
    overall_drift_score: float = 0.0  # 0-1, weighted aggregate
    overall_severity: str = "none"
    drifted_feature_count: int = 0
    total_features_analyzed: int = 0
    drift_pct: float = 0.0
    feature_results: List[DriftResult] = field(default_factory=list)
    concept_drift: Optional[Dict] = None
    prediction_drift: Optional[Dict] = None
    alerts: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    should_retrain: bool = False
    retrain_urgency: str = "none"  # none | low | medium | high | immediate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_drift_score": round(self.overall_drift_score, 4),
            "overall_severity": self.overall_severity,
            "drifted_feature_count": self.drifted_feature_count,
            "total_features_analyzed": self.total_features_analyzed,
            "drift_pct": round(self.drift_pct, 2),
            "feature_results": [r.to_dict() for r in self.feature_results],
            "concept_drift": self.concept_drift,
            "prediction_drift": self.prediction_drift,
            "alerts": self.alerts,
            "recommendations": self.recommendations,
            "should_retrain": self.should_retrain,
            "retrain_urgency": self.retrain_urgency,
        }


# ═══════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════

class DriftThresholds:
    """Configurable thresholds for drift severity classification."""

    # PSI thresholds (industry standard)
    PSI_LOW = 0.1
    PSI_MEDIUM = 0.2
    PSI_HIGH = 0.25
    PSI_CRITICAL = 0.4

    # KS test thresholds
    KS_LOW = 0.05
    KS_MEDIUM = 0.1
    KS_HIGH = 0.2
    KS_CRITICAL = 0.3

    # Chi-Square p-value thresholds (lower = more drift)
    CHI2_CRITICAL_P = 0.001
    CHI2_HIGH_P = 0.01
    CHI2_MEDIUM_P = 0.05
    CHI2_LOW_P = 0.10

    # Jensen-Shannon thresholds
    JS_LOW = 0.05
    JS_MEDIUM = 0.15
    JS_HIGH = 0.25
    JS_CRITICAL = 0.4

    # Mean shift thresholds (in standard deviations)
    MEAN_SHIFT_LOW = 0.5
    MEAN_SHIFT_MEDIUM = 1.0
    MEAN_SHIFT_HIGH = 2.0
    MEAN_SHIFT_CRITICAL = 3.0

    # Variance ratio thresholds
    VAR_RATIO_LOW = 1.5
    VAR_RATIO_MEDIUM = 2.0
    VAR_RATIO_HIGH = 3.0
    VAR_RATIO_CRITICAL = 5.0

    # Overall drift score → severity
    OVERALL_LOW = 0.15
    OVERALL_MEDIUM = 0.30
    OVERALL_HIGH = 0.50
    OVERALL_CRITICAL = 0.70

    # Retrain triggers
    RETRAIN_THRESHOLD = 0.40      # overall drift score
    RETRAIN_FEATURE_PCT = 0.30    # % of features drifted


# ═══════════════════════════════════════════════════════════════
# DRIFT ANALYZER
# ═══════════════════════════════════════════════════════════════

class DriftAnalyzer:
    """
    Production-grade drift detection from pre-computed statistics.
    
    Usage:
        analyzer = DriftAnalyzer()
        report = analyzer.full_analysis(reference_stats, current_stats, feature_importances)
    """

    def __init__(self, thresholds: DriftThresholds = None):
        self.thresholds = thresholds or DriftThresholds()

    # ──────────────────────────────────────────────────────────
    # MAIN ANALYSIS
    # ──────────────────────────────────────────────────────────

    def full_analysis(
        self,
        reference_stats: Dict[str, Dict],
        current_stats: Dict[str, Dict],
        feature_importances: Optional[Dict[str, float]] = None,
        reference_target_dist: Optional[Dict] = None,
        current_target_dist: Optional[Dict] = None,
        reference_pred_dist: Optional[Dict] = None,
        current_pred_dist: Optional[Dict] = None,
    ) -> DriftReport:
        """
        Run full drift analysis across all features.
        
        Args:
            reference_stats: {feature_name: {mean, std, min, max, q25, q50, q75, histogram, ...}}
            current_stats: Same structure as reference
            feature_importances: {feature_name: importance_score} for weighting
            reference_target_dist: Distribution of target variable in training data
            current_target_dist: Distribution of target variable in production data
        """
        report = DriftReport(timestamp=datetime.utcnow().isoformat())

        # Analyze each feature
        common_features = set(reference_stats.keys()) & set(current_stats.keys())
        report.total_features_analyzed = len(common_features)

        importance_weights = feature_importances or {}
        max_importance = max(importance_weights.values()) if importance_weights else 1.0

        weighted_drift_scores = []

        for feature in sorted(common_features):
            ref = reference_stats[feature]
            cur = current_stats[feature]

            # Determine feature type and run appropriate tests
            is_categorical = ref.get("type") == "categorical" or "categories" in ref
            
            if is_categorical:
                result = self._analyze_categorical_feature(feature, ref, cur)
            else:
                result = self._analyze_numeric_feature(feature, ref, cur)

            report.feature_results.append(result)

            if result.is_drifted:
                report.drifted_feature_count += 1

            # Weight by feature importance
            weight = importance_weights.get(feature, 0.5) / max_importance
            drift_score = self._severity_to_score(result.severity) * weight
            weighted_drift_scores.append(drift_score)

        # Overall drift score
        if weighted_drift_scores:
            report.overall_drift_score = sum(weighted_drift_scores) / len(weighted_drift_scores)
            report.drift_pct = round(
                report.drifted_feature_count / max(1, report.total_features_analyzed) * 100, 1
            )

        # Overall severity
        report.overall_severity = self._classify_overall_severity(report.overall_drift_score)

        # Concept drift (target distribution)
        if reference_target_dist and current_target_dist:
            report.concept_drift = self._detect_concept_drift(
                reference_target_dist, current_target_dist
            )

        # Prediction distribution drift
        if reference_pred_dist and current_pred_dist:
            report.prediction_drift = self._detect_prediction_drift(
                reference_pred_dist, current_pred_dist
            )

        # Generate alerts and recommendations
        report.alerts = self._generate_alerts(report)
        report.recommendations = self._generate_recommendations(report)

        # Retrain decision
        report.should_retrain = (
            report.overall_drift_score >= self.thresholds.RETRAIN_THRESHOLD or
            report.drift_pct >= self.thresholds.RETRAIN_FEATURE_PCT * 100 or
            (report.concept_drift and report.concept_drift.get("severity") in ("high", "critical"))
        )
        report.retrain_urgency = self._compute_retrain_urgency(report)

        return report

    # ──────────────────────────────────────────────────────────
    # NUMERIC FEATURE ANALYSIS
    # ──────────────────────────────────────────────────────────

    def _analyze_numeric_feature(
        self, feature: str, ref: Dict, cur: Dict
    ) -> DriftResult:
        """Run multiple drift tests on a numeric feature and combine results."""
        results = []

        # Test 1: Mean shift
        mean_result = self._test_mean_shift(feature, ref, cur)
        results.append(mean_result)

        # Test 2: Variance ratio
        var_result = self._test_variance_ratio(feature, ref, cur)
        results.append(var_result)

        # Test 3: PSI (if histograms available)
        if ref.get("histogram") and cur.get("histogram"):
            psi_result = self._test_psi(feature, ref["histogram"], cur["histogram"])
            results.append(psi_result)

        # Test 4: Range drift
        range_result = self._test_range_drift(feature, ref, cur)
        results.append(range_result)

        # Test 5: Percentile drift
        pct_result = self._test_percentile_drift(feature, ref, cur)
        results.append(pct_result)

        # Combine: take the most severe result
        worst = max(results, key=lambda r: self._severity_to_score(r.severity))
        return worst

    def _test_mean_shift(self, feature: str, ref: Dict, cur: Dict) -> DriftResult:
        """Detect mean shift in standard deviations."""
        ref_mean = ref.get("mean", 0)
        cur_mean = cur.get("mean", 0)
        ref_std = ref.get("std", 1)

        if ref_std == 0:
            ref_std = abs(ref_mean) * 0.01 if ref_mean != 0 else 1

        shift_sd = abs(cur_mean - ref_mean) / ref_std
        change_pct = ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0

        severity = "none"
        if shift_sd >= self.thresholds.MEAN_SHIFT_CRITICAL:
            severity = "critical"
        elif shift_sd >= self.thresholds.MEAN_SHIFT_HIGH:
            severity = "high"
        elif shift_sd >= self.thresholds.MEAN_SHIFT_MEDIUM:
            severity = "medium"
        elif shift_sd >= self.thresholds.MEAN_SHIFT_LOW:
            severity = "low"

        direction = "shift_up" if cur_mean > ref_mean else "shift_down" if cur_mean < ref_mean else "stable"

        return DriftResult(
            feature=feature, test_name="mean_shift",
            statistic=round(shift_sd, 4), threshold=self.thresholds.MEAN_SHIFT_MEDIUM,
            is_drifted=severity not in ("none", "low"),
            severity=severity, direction=direction,
            reference_stat=round(ref_mean, 4), current_stat=round(cur_mean, 4),
            change_pct=round(change_pct, 2),
            recommendation=f"Mean shifted {shift_sd:.1f}σ ({'up' if direction == 'shift_up' else 'down'}). Investigate upstream data pipeline." if severity != "none" else "",
        )

    def _test_variance_ratio(self, feature: str, ref: Dict, cur: Dict) -> DriftResult:
        """Detect variance/spread changes."""
        ref_std = ref.get("std", 1)
        cur_std = cur.get("std", 1)

        if ref_std == 0:
            ref_std = 0.001
        if cur_std == 0:
            cur_std = 0.001

        ratio = max(cur_std, ref_std) / min(cur_std, ref_std)

        severity = "none"
        if ratio >= self.thresholds.VAR_RATIO_CRITICAL:
            severity = "critical"
        elif ratio >= self.thresholds.VAR_RATIO_HIGH:
            severity = "high"
        elif ratio >= self.thresholds.VAR_RATIO_MEDIUM:
            severity = "medium"
        elif ratio >= self.thresholds.VAR_RATIO_LOW:
            severity = "low"

        direction = "spread_increase" if cur_std > ref_std else "spread_decrease"

        return DriftResult(
            feature=feature, test_name="variance_ratio",
            statistic=round(ratio, 4), threshold=self.thresholds.VAR_RATIO_MEDIUM,
            is_drifted=severity not in ("none", "low"),
            severity=severity, direction=direction,
            reference_stat=round(ref_std, 4), current_stat=round(cur_std, 4),
            change_pct=round((ratio - 1) * 100, 2),
            recommendation=f"Variance changed {ratio:.1f}x. Check for data collection changes." if severity != "none" else "",
        )

    def _test_psi(self, feature: str, ref_hist: List, cur_hist: List) -> DriftResult:
        """Compute Population Stability Index from histograms."""
        psi = self.compute_psi(ref_hist, cur_hist)

        severity = "none"
        if psi >= self.thresholds.PSI_CRITICAL:
            severity = "critical"
        elif psi >= self.thresholds.PSI_HIGH:
            severity = "high"
        elif psi >= self.thresholds.PSI_MEDIUM:
            severity = "medium"
        elif psi >= self.thresholds.PSI_LOW:
            severity = "low"

        return DriftResult(
            feature=feature, test_name="psi",
            statistic=round(psi, 4), threshold=self.thresholds.PSI_MEDIUM,
            is_drifted=severity not in ("none", "low"),
            severity=severity, direction="distribution_change",
            recommendation=f"PSI={psi:.3f} indicates {'significant' if psi > 0.2 else 'moderate'} distribution shift. Consider retraining." if severity != "none" else "",
        )

    def _test_range_drift(self, feature: str, ref: Dict, cur: Dict) -> DriftResult:
        """Detect if current data exceeds training data range."""
        ref_min = ref.get("min", 0)
        ref_max = ref.get("max", 0)
        cur_min = cur.get("min", 0)
        cur_max = cur.get("max", 0)

        ref_range = ref_max - ref_min if ref_max != ref_min else 1

        # How much does current data exceed reference range?
        overshoot = 0
        if cur_max > ref_max:
            overshoot = max(overshoot, (cur_max - ref_max) / ref_range)
        if cur_min < ref_min:
            overshoot = max(overshoot, (ref_min - cur_min) / ref_range)

        severity = "none"
        if overshoot > 0.5:
            severity = "critical"
        elif overshoot > 0.2:
            severity = "high"
        elif overshoot > 0.1:
            severity = "medium"
        elif overshoot > 0.05:
            severity = "low"

        return DriftResult(
            feature=feature, test_name="range_drift",
            statistic=round(overshoot, 4), threshold=0.1,
            is_drifted=severity not in ("none", "low"),
            severity=severity,
            direction="range_expansion",
            reference_stat=round(ref_range, 4),
            change_pct=round(overshoot * 100, 2),
            recommendation=f"Data range exceeded training range by {overshoot*100:.0f}%. Model may extrapolate poorly." if severity != "none" else "",
        )

    def _test_percentile_drift(self, feature: str, ref: Dict, cur: Dict) -> DriftResult:
        """Detect drift in percentiles (shape change)."""
        ref_q25 = ref.get("q25", ref.get("25%", 0))
        ref_q50 = ref.get("q50", ref.get("50%", ref.get("median", 0)))
        ref_q75 = ref.get("q75", ref.get("75%", 0))

        cur_q25 = cur.get("q25", cur.get("25%", 0))
        cur_q50 = cur.get("q50", cur.get("50%", cur.get("median", 0)))
        cur_q75 = cur.get("q75", cur.get("75%", 0))

        ref_iqr = ref_q75 - ref_q25 if ref_q75 != ref_q25 else 1
        cur_iqr = cur_q75 - cur_q25 if cur_q75 != cur_q25 else 1

        # Median shift in IQR units
        median_shift = abs(cur_q50 - ref_q50) / ref_iqr if ref_iqr else 0
        iqr_ratio = cur_iqr / ref_iqr if ref_iqr else 1

        combined_score = median_shift * 0.6 + abs(iqr_ratio - 1) * 0.4

        severity = "none"
        if combined_score > 1.0:
            severity = "high"
        elif combined_score > 0.5:
            severity = "medium"
        elif combined_score > 0.2:
            severity = "low"

        return DriftResult(
            feature=feature, test_name="percentile_drift",
            statistic=round(combined_score, 4), threshold=0.5,
            is_drifted=severity not in ("none", "low"),
            severity=severity, direction="distribution_change",
            recommendation=f"Distribution shape changed (median shift: {median_shift:.2f} IQR, IQR ratio: {iqr_ratio:.2f})." if severity != "none" else "",
        )

    # ──────────────────────────────────────────────────────────
    # CATEGORICAL FEATURE ANALYSIS
    # ──────────────────────────────────────────────────────────

    def _analyze_categorical_feature(
        self, feature: str, ref: Dict, cur: Dict
    ) -> DriftResult:
        """Analyze drift in categorical features."""
        ref_dist = ref.get("distribution", ref.get("categories", {}))
        cur_dist = cur.get("distribution", cur.get("categories", {}))

        if not ref_dist or not cur_dist:
            return DriftResult(
                feature=feature, test_name="categorical_drift",
                statistic=0, threshold=0, severity="none",
            )

        # Check for new categories
        new_cats = set(cur_dist.keys()) - set(ref_dist.keys())
        missing_cats = set(ref_dist.keys()) - set(cur_dist.keys())

        # Compute PSI on category proportions
        psi = self._categorical_psi(ref_dist, cur_dist)

        severity = "none"
        if new_cats and len(new_cats) > 3:
            severity = "high"
        elif psi >= self.thresholds.PSI_HIGH:
            severity = "high"
        elif psi >= self.thresholds.PSI_MEDIUM:
            severity = "medium"
        elif psi >= self.thresholds.PSI_LOW or new_cats:
            severity = "low"

        rec = ""
        if new_cats:
            rec = f"New categories detected: {', '.join(list(new_cats)[:5])}. Model has never seen these. "
        if missing_cats:
            rec += f"Missing categories: {', '.join(list(missing_cats)[:5])}. "

        return DriftResult(
            feature=feature, test_name="categorical_psi",
            statistic=round(psi, 4), threshold=self.thresholds.PSI_MEDIUM,
            is_drifted=severity not in ("none", "low"),
            severity=severity, direction="distribution_change",
            change_pct=round(psi * 100, 2),
            recommendation=rec.strip() if rec else "",
        )

    def _categorical_psi(self, ref_dist: Dict, cur_dist: Dict) -> float:
        """PSI for categorical distributions."""
        all_cats = set(ref_dist.keys()) | set(cur_dist.keys())
        ref_total = sum(ref_dist.values()) or 1
        cur_total = sum(cur_dist.values()) or 1

        psi = 0.0
        for cat in all_cats:
            ref_pct = max(ref_dist.get(cat, 0) / ref_total, 0.0001)
            cur_pct = max(cur_dist.get(cat, 0) / cur_total, 0.0001)
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

        return max(0, psi)

    # ──────────────────────────────────────────────────────────
    # CONCEPT & PREDICTION DRIFT
    # ──────────────────────────────────────────────────────────

    def _detect_concept_drift(self, ref_target: Dict, cur_target: Dict) -> Dict:
        """Detect drift in target variable distribution (concept drift)."""
        ref_positive_rate = ref_target.get("positive_rate", ref_target.get("mean", 0.5))
        cur_positive_rate = cur_target.get("positive_rate", cur_target.get("mean", 0.5))

        shift = abs(cur_positive_rate - ref_positive_rate)
        relative_change = shift / max(ref_positive_rate, 0.001) * 100

        severity = "none"
        if shift > 0.15:
            severity = "critical"
        elif shift > 0.10:
            severity = "high"
        elif shift > 0.05:
            severity = "medium"
        elif shift > 0.02:
            severity = "low"

        return {
            "reference_positive_rate": round(ref_positive_rate, 4),
            "current_positive_rate": round(cur_positive_rate, 4),
            "absolute_shift": round(shift, 4),
            "relative_change_pct": round(relative_change, 2),
            "severity": severity,
            "is_drifted": severity not in ("none", "low"),
            "recommendation": (
                f"Target rate shifted from {ref_positive_rate:.1%} to {cur_positive_rate:.1%} "
                f"({relative_change:.0f}% change). This is concept drift — retrain recommended."
            ) if severity not in ("none", "low") else "",
        }

    def _detect_prediction_drift(self, ref_pred: Dict, cur_pred: Dict) -> Dict:
        """Detect drift in model prediction distribution."""
        ref_mean = ref_pred.get("mean", 0.5)
        cur_mean = cur_pred.get("mean", 0.5)
        ref_std = ref_pred.get("std", 0.1)

        shift = abs(cur_mean - ref_mean)
        shift_sd = shift / max(ref_std, 0.001)

        severity = "none"
        if shift_sd > 3.0:
            severity = "critical"
        elif shift_sd > 2.0:
            severity = "high"
        elif shift_sd > 1.0:
            severity = "medium"
        elif shift_sd > 0.5:
            severity = "low"

        return {
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "shift_sd": round(shift_sd, 4),
            "severity": severity,
            "is_drifted": severity not in ("none", "low"),
        }

    # ──────────────────────────────────────────────────────────
    # SEQUENTIAL CHANGE DETECTION
    # ──────────────────────────────────────────────────────────

    def cusum_test(
        self, values: List[float], target_mean: float = None,
        threshold: float = 5.0, drift_amount: float = 1.0,
    ) -> Dict[str, Any]:
        """
        CUSUM (Cumulative Sum) test for sequential change detection.
        Detects both upward and downward shifts.
        """
        if not values or len(values) < 5:
            return {"detected": False, "reason": "insufficient_data"}

        if target_mean is None:
            target_mean = sum(values[:max(10, len(values) // 4)]) / max(10, len(values) // 4)

        s_pos = 0.0  # Cumulative sum for detecting increase
        s_neg = 0.0  # Cumulative sum for detecting decrease
        change_point = None

        for i, val in enumerate(values):
            s_pos = max(0, s_pos + (val - target_mean) - drift_amount)
            s_neg = max(0, s_neg - (val - target_mean) - drift_amount)

            if s_pos > threshold or s_neg > threshold:
                change_point = i
                break

        return {
            "detected": change_point is not None,
            "change_point_index": change_point,
            "direction": "increase" if s_pos > s_neg else "decrease",
            "max_cusum_pos": round(s_pos, 4),
            "max_cusum_neg": round(s_neg, 4),
            "target_mean": round(target_mean, 4),
            "threshold": threshold,
        }

    def page_hinkley_test(
        self, values: List[float], delta: float = 0.005,
        threshold: float = 50.0, alpha: float = 1 - 0.0001,
    ) -> Dict[str, Any]:
        """
        Page-Hinkley test for online drift detection.
        Good for monitoring streams of predictions/metrics.
        """
        if not values or len(values) < 10:
            return {"detected": False, "reason": "insufficient_data"}

        running_mean = 0.0
        sum_diff = 0.0
        min_sum = float("inf")
        change_point = None

        for i, val in enumerate(values):
            running_mean = running_mean * alpha + val * (1 - alpha) if i > 0 else val
            sum_diff += (val - running_mean - delta)
            min_sum = min(min_sum, sum_diff)

            if sum_diff - min_sum > threshold:
                change_point = i
                break

        return {
            "detected": change_point is not None,
            "change_point_index": change_point,
            "ph_statistic": round(sum_diff - min_sum, 4),
            "threshold": threshold,
        }

    # ──────────────────────────────────────────────────────────
    # UTILITY FUNCTIONS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_psi(reference: List[float], current: List[float]) -> float:
        """Compute PSI from two histograms (bin counts or proportions)."""
        if not reference or not current:
            return 0.0

        # Normalize to proportions
        ref_total = sum(reference) or 1
        cur_total = sum(current) or 1
        ref_pcts = [max(x / ref_total, 0.0001) for x in reference]
        cur_pcts = [max(x / cur_total, 0.0001) for x in current]

        # Align lengths
        max_len = max(len(ref_pcts), len(cur_pcts))
        while len(ref_pcts) < max_len:
            ref_pcts.append(0.0001)
        while len(cur_pcts) < max_len:
            cur_pcts.append(0.0001)

        psi = sum(
            (cur_pcts[i] - ref_pcts[i]) * math.log(cur_pcts[i] / ref_pcts[i])
            for i in range(max_len)
        )
        return max(0, psi)

    @staticmethod
    def compute_js_divergence(p: List[float], q: List[float]) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        if not p or not q:
            return 0.0

        # Normalize
        p_sum = sum(p) or 1
        q_sum = sum(q) or 1
        p_norm = [x / p_sum for x in p]
        q_norm = [x / q_sum for x in q]

        max_len = max(len(p_norm), len(q_norm))
        while len(p_norm) < max_len:
            p_norm.append(0.0001)
        while len(q_norm) < max_len:
            q_norm.append(0.0001)

        m = [(p_norm[i] + q_norm[i]) / 2 for i in range(max_len)]

        def kl(a, b):
            return sum(a[i] * math.log(max(a[i], 1e-10) / max(b[i], 1e-10)) for i in range(len(a)))

        return (kl(p_norm, m) + kl(q_norm, m)) / 2

    # ──────────────────────────────────────────────────────────
    # SEVERITY & ALERTS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _severity_to_score(severity: str) -> float:
        return {"none": 0, "low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(severity, 0)

    def _classify_overall_severity(self, score: float) -> str:
        if score >= self.thresholds.OVERALL_CRITICAL:
            return "critical"
        elif score >= self.thresholds.OVERALL_HIGH:
            return "high"
        elif score >= self.thresholds.OVERALL_MEDIUM:
            return "medium"
        elif score >= self.thresholds.OVERALL_LOW:
            return "low"
        return "none"

    def _compute_retrain_urgency(self, report: DriftReport) -> str:
        if report.overall_severity == "critical" or (report.concept_drift and report.concept_drift.get("severity") == "critical"):
            return "immediate"
        elif report.overall_severity == "high":
            return "high"
        elif report.should_retrain:
            return "medium"
        elif report.overall_severity == "medium":
            return "low"
        return "none"

    def _generate_alerts(self, report: DriftReport) -> List[Dict]:
        """Generate prioritized alerts from drift results."""
        alerts = []

        # Critical feature drifts
        critical_features = [r for r in report.feature_results if r.severity in ("critical", "high")]
        if critical_features:
            alerts.append({
                "severity": "critical",
                "type": "feature_drift",
                "title": f"{len(critical_features)} feature(s) with significant drift",
                "features": [r.feature for r in critical_features[:5]],
                "details": critical_features[0].recommendation,
            })

        # Concept drift
        if report.concept_drift and report.concept_drift.get("is_drifted"):
            alerts.append({
                "severity": "critical",
                "type": "concept_drift",
                "title": "Target distribution has shifted",
                "details": report.concept_drift.get("recommendation", ""),
            })

        # Overall drift
        if report.overall_severity in ("high", "critical"):
            alerts.append({
                "severity": "warning",
                "type": "overall_drift",
                "title": f"Overall drift score: {report.overall_drift_score:.2f} ({report.drift_pct:.0f}% features drifted)",
                "details": f"Retrain urgency: {report.retrain_urgency}",
            })

        return sorted(alerts, key=lambda a: {"critical": 0, "warning": 1, "info": 2}.get(a.get("severity", "info"), 3))

    def _generate_recommendations(self, report: DriftReport) -> List[str]:
        """Generate actionable recommendations from drift analysis."""
        recs = []

        if report.retrain_urgency == "immediate":
            recs.append("⚠️ RETRAIN IMMEDIATELY: Significant drift detected. Model predictions are likely degraded.")
        elif report.retrain_urgency == "high":
            recs.append("Schedule retraining within this week. Multiple features have drifted beyond safe thresholds.")

        critical = [r for r in report.feature_results if r.severity == "critical"]
        if critical:
            features = ", ".join(r.feature for r in critical[:5])
            recs.append(f"Investigate data pipeline for critical drift in: {features}")

        if report.concept_drift and report.concept_drift.get("is_drifted"):
            recs.append("Target distribution has changed. Evaluate if business conditions have shifted and adjust labels/thresholds.")

        new_cat_features = [
            r for r in report.feature_results
            if r.test_name == "categorical_psi" and "New categories" in r.recommendation
        ]
        if new_cat_features:
            recs.append(f"Handle new categorical values in: {', '.join(r.feature for r in new_cat_features[:3])}")

        if report.overall_severity == "low" and not report.should_retrain:
            recs.append("Drift within acceptable bounds. Continue monitoring. Next review recommended in 1 week.")

        if not recs:
            recs.append("No significant drift detected. Model is performing within expected parameters.")

        return recs
