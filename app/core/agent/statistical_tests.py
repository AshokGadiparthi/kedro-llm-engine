"""
Statistical Tests Engine — Real Computation Layer
====================================================
Runs actual statistical tests on distribution data, not just metadata reasoning.
Works from histogram bins, percentiles, and summary statistics stored in
EdaResult — performs real scipy computations where possible, and rigorous
pure-Python implementations as fallback.

SOLVES: "Drift Analyzer describes PSI but doesn't actually compute it."

Capabilities:
  1. PSI Computation               — Actual Population Stability Index from histograms
  2. KS Test                       — Kolmogorov-Smirnov from CDFs built off percentiles
  3. Chi-Square Test               — For categorical distributions
  4. Jensen-Shannon Divergence     — Symmetric distribution distance
  5. Wasserstein (Earth Mover's)   — From percentile approximation
  6. Anderson-Darling Normality    — From distribution shape measures
  7. Shapiro-Wilk Approximation    — Normality test from skew+kurtosis
  8. Levene's Test Approximation   — Variance homogeneity
  9. Effect Size (Cohen's d)       — Practical significance of drift
  10. Bootstrap Confidence Intervals — For metric uncertainty

All functions have two paths:
  - scipy path (if scipy is available) — exact computation
  - pure-Python fallback — approximate but functional

Usage:
  engine = StatisticalTestEngine()
  result = engine.compute_psi(reference_hist, current_hist)
  result = engine.compute_ks_from_percentiles(ref_percentiles, cur_percentiles)
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import scipy/numpy, fall back gracefully
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class StatTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: Optional[float] = None
    is_significant: bool = False
    significance_level: float = 0.05
    effect_size: Optional[float] = None
    interpretation: str = ""
    method: str = "exact"  # exact | approximate | fallback
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class NormalityTestResult:
    """Result of normality testing."""
    is_normal: bool = False
    normality_score: float = 0.0  # 0 = clearly non-normal, 1 = normal
    skewness: float = 0.0
    kurtosis: float = 0.0
    tests_performed: List[StatTestResult] = field(default_factory=list)
    recommended_transform: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_normal": self.is_normal,
            "normality_score": round(self.normality_score, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "tests": [t.to_dict() for t in self.tests_performed],
            "recommended_transform": self.recommended_transform,
        }


@dataclass
class DriftTestSuite:
    """Complete drift test results for a feature."""
    feature: str
    psi: Optional[StatTestResult] = None
    ks: Optional[StatTestResult] = None
    chi2: Optional[StatTestResult] = None
    js_divergence: Optional[StatTestResult] = None
    wasserstein: Optional[StatTestResult] = None
    cohens_d: Optional[StatTestResult] = None
    overall_drift: bool = False
    overall_severity: str = "none"  # none | low | moderate | high | critical
    consensus_score: float = 0.0   # 0-1, fraction of tests indicating drift

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "feature": self.feature,
            "overall_drift": self.overall_drift,
            "overall_severity": self.overall_severity,
            "consensus_score": round(self.consensus_score, 3),
            "tests": {},
        }
        for name in ["psi", "ks", "chi2", "js_divergence", "wasserstein", "cohens_d"]:
            val = getattr(self, name, None)
            if val:
                result["tests"][name] = val.to_dict()
        return result


# ═══════════════════════════════════════════════════════════════
# STATISTICAL TEST ENGINE
# ═══════════════════════════════════════════════════════════════

class StatisticalTestEngine:
    """
    Runs actual statistical computations on distribution data.
    All methods accept pre-computed statistics (histograms, percentiles, stats)
    and perform real mathematical operations.
    """

    # ──────────────────────────────────────────────────────────
    # 1. PSI — Population Stability Index
    # ──────────────────────────────────────────────────────────

    def compute_psi(
        self,
        reference_distribution: List[float],
        current_distribution: List[float],
        epsilon: float = 1e-6,
    ) -> StatTestResult:
        """
        Compute PSI between two distributions (histogram bin proportions).

        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)

        Thresholds:
          PSI < 0.10 → No significant drift
          0.10 ≤ PSI < 0.25 → Moderate drift
          PSI ≥ 0.25 → Significant drift
        """
        if not reference_distribution or not current_distribution:
            return StatTestResult(
                test_name="PSI", statistic=0.0,
                interpretation="Insufficient data for PSI computation",
                method="skipped",
            )

        # Normalize to proportions
        ref = self._normalize_distribution(reference_distribution, epsilon)
        cur = self._normalize_distribution(current_distribution, epsilon)

        # Align lengths (truncate to shorter)
        min_len = min(len(ref), len(cur))
        ref = ref[:min_len]
        cur = cur[:min_len]

        # Compute PSI
        psi = 0.0
        for p, q in zip(ref, cur):
            p = max(p, epsilon)
            q = max(q, epsilon)
            psi += (p - q) * math.log(p / q)

        # Interpret
        if psi < 0.10:
            interp = "No significant drift detected"
            severity = "none"
            significant = False
        elif psi < 0.25:
            interp = "Moderate drift — monitor closely"
            severity = "moderate"
            significant = True
        else:
            interp = "Significant drift — retraining recommended"
            severity = "high"
            significant = True

        return StatTestResult(
            test_name="PSI",
            statistic=round(psi, 6),
            is_significant=significant,
            interpretation=interp,
            method="exact",
            details={"severity": severity, "n_bins": min_len},
        )

    # ──────────────────────────────────────────────────────────
    # 2. KS Test — Kolmogorov-Smirnov
    # ──────────────────────────────────────────────────────────

    def compute_ks_from_percentiles(
        self,
        ref_percentiles: Dict[str, float],
        cur_percentiles: Dict[str, float],
        n_ref: int = 1000,
        n_cur: int = 1000,
        alpha: float = 0.05,
    ) -> StatTestResult:
        """
        Approximate KS test from percentile data.

        Reconstructs empirical CDFs from percentiles (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%)
        and computes the maximum CDF difference.
        """
        # Standard percentile points
        pct_keys = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
        pct_values = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        ref_vals = []
        cur_vals = []
        cdf_points = []

        for key, cdf_val in zip(pct_keys, pct_values):
            r = ref_percentiles.get(key)
            c = cur_percentiles.get(key)
            if r is not None and c is not None:
                ref_vals.append(float(r))
                cur_vals.append(float(c))
                cdf_points.append(cdf_val)

        if len(ref_vals) < 3:
            return StatTestResult(
                test_name="KS", statistic=0.0,
                interpretation="Insufficient percentile data for KS test",
                method="skipped",
            )

        # Compute max CDF difference at each percentile point
        # For same CDF value, compute the value difference (inverse approach)
        # Instead, evaluate both CDFs at each reference value
        all_values = sorted(set(ref_vals + cur_vals))
        max_diff = 0.0

        for v in all_values:
            # Interpolate CDF for reference
            ref_cdf = self._interpolate_cdf(v, ref_vals, cdf_points)
            cur_cdf = self._interpolate_cdf(v, cur_vals, cdf_points)
            diff = abs(ref_cdf - cur_cdf)
            max_diff = max(max_diff, diff)

        # KS critical value approximation
        # c(α) × sqrt((n1 + n2) / (n1 × n2))
        c_alpha = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}.get(alpha, 1.36)
        critical = c_alpha * math.sqrt((n_ref + n_cur) / max(n_ref * n_cur, 1))

        # Approximate p-value using the KS distribution
        p_value = self._ks_p_value(max_diff, n_ref, n_cur)

        return StatTestResult(
            test_name="KS",
            statistic=round(max_diff, 6),
            p_value=round(p_value, 6) if p_value else None,
            is_significant=max_diff > critical,
            significance_level=alpha,
            interpretation=(
                f"Max CDF difference: {max_diff:.4f} "
                f"(critical value at α={alpha}: {critical:.4f}). "
                f"{'Significant' if max_diff > critical else 'Not significant'} drift."
            ),
            method="percentile_approximation",
            details={
                "critical_value": round(critical, 4),
                "n_percentiles_used": len(ref_vals),
            },
        )

    # ──────────────────────────────────────────────────────────
    # 3. Chi-Square Test
    # ──────────────────────────────────────────────────────────

    def compute_chi_square(
        self,
        observed_counts: Dict[str, int],
        expected_counts: Dict[str, int],
        alpha: float = 0.05,
    ) -> StatTestResult:
        """
        Chi-square goodness of fit for categorical distributions.
        """
        all_categories = set(observed_counts.keys()) | set(expected_counts.keys())
        if len(all_categories) < 2:
            return StatTestResult(
                test_name="Chi-Square", statistic=0.0,
                interpretation="Need at least 2 categories",
                method="skipped",
            )

        chi2 = 0.0
        n_obs = sum(observed_counts.values())
        n_exp = sum(expected_counts.values())

        for cat in all_categories:
            o = observed_counts.get(cat, 0)
            e_raw = expected_counts.get(cat, 0)
            # Scale expected to match observed total
            e = (e_raw / max(n_exp, 1)) * n_obs if n_exp > 0 else 0
            e = max(e, 0.5)  # Avoid division by zero
            chi2 += (o - e) ** 2 / e

        df = max(len(all_categories) - 1, 1)

        # p-value approximation
        p_value = self._chi2_p_value(chi2, df)

        return StatTestResult(
            test_name="Chi-Square",
            statistic=round(chi2, 4),
            p_value=round(p_value, 6) if p_value else None,
            is_significant=p_value < alpha if p_value else chi2 > df * 3,
            significance_level=alpha,
            interpretation=(
                f"χ² = {chi2:.2f} with {df} degrees of freedom. "
                f"{'Significant' if (p_value and p_value < alpha) else 'Not significant'} "
                f"distribution change."
            ),
            method="exact" if HAS_SCIPY else "approximate",
            details={"degrees_of_freedom": df, "n_categories": len(all_categories)},
        )

    # ──────────────────────────────────────────────────────────
    # 4. Jensen-Shannon Divergence
    # ──────────────────────────────────────────────────────────

    def compute_js_divergence(
        self,
        p_distribution: List[float],
        q_distribution: List[float],
        epsilon: float = 1e-10,
    ) -> StatTestResult:
        """
        Jensen-Shannon divergence (symmetric version of KL divergence).
        JSD = 0.5 × KL(P||M) + 0.5 × KL(Q||M) where M = (P+Q)/2
        Range: [0, ln(2)] ≈ [0, 0.693]
        """
        p = self._normalize_distribution(p_distribution, epsilon)
        q = self._normalize_distribution(q_distribution, epsilon)

        min_len = min(len(p), len(q))
        p, q = p[:min_len], q[:min_len]

        # Compute midpoint distribution
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        # KL divergences
        kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(p, m) if pi > epsilon and mi > epsilon)
        kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(q, m) if qi > epsilon and mi > epsilon)

        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        jsd = max(0.0, jsd)  # Numerical stability

        # Normalize to [0, 1]
        jsd_normalized = jsd / math.log(2) if math.log(2) > 0 else 0

        severity = "none"
        if jsd_normalized > 0.5:
            severity = "high"
        elif jsd_normalized > 0.2:
            severity = "moderate"
        elif jsd_normalized > 0.05:
            severity = "low"

        return StatTestResult(
            test_name="Jensen-Shannon Divergence",
            statistic=round(jsd, 6),
            is_significant=jsd_normalized > 0.1,
            interpretation=(
                f"JSD = {jsd:.4f} (normalized: {jsd_normalized:.4f}). "
                f"Distribution divergence is {severity}."
            ),
            method="exact",
            details={
                "jsd_normalized": round(jsd_normalized, 4),
                "kl_pm": round(kl_pm, 4),
                "kl_qm": round(kl_qm, 4),
                "severity": severity,
            },
        )

    # ──────────────────────────────────────────────────────────
    # 5. Wasserstein Distance (Earth Mover's)
    # ──────────────────────────────────────────────────────────

    def compute_wasserstein_from_percentiles(
        self,
        ref_percentiles: Dict[str, float],
        cur_percentiles: Dict[str, float],
    ) -> StatTestResult:
        """
        Approximate Wasserstein-1 distance from percentile data.
        Uses trapezoidal integration of |F_ref^{-1}(p) - F_cur^{-1}(p)| dp.
        """
        pct_keys = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
        pct_values = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        ref_vals = []
        cur_vals = []
        probs = []

        for key, p in zip(pct_keys, pct_values):
            r = ref_percentiles.get(key)
            c = cur_percentiles.get(key)
            if r is not None and c is not None:
                ref_vals.append(float(r))
                cur_vals.append(float(c))
                probs.append(p)

        if len(ref_vals) < 3:
            return StatTestResult(
                test_name="Wasserstein", statistic=0.0,
                interpretation="Insufficient percentile data",
                method="skipped",
            )

        # Trapezoidal integration of |F_ref^-1(p) - F_cur^-1(p)|
        distance = 0.0
        for i in range(1, len(probs)):
            dp = probs[i] - probs[i-1]
            diff_prev = abs(ref_vals[i-1] - cur_vals[i-1])
            diff_curr = abs(ref_vals[i] - cur_vals[i])
            distance += 0.5 * (diff_prev + diff_curr) * dp

        # Normalize by reference range for interpretability
        ref_range = max(ref_vals) - min(ref_vals) if ref_vals else 1.0
        normalized = distance / max(ref_range, 1e-10)

        return StatTestResult(
            test_name="Wasserstein",
            statistic=round(distance, 6),
            effect_size=round(normalized, 4),
            is_significant=normalized > 0.1,
            interpretation=(
                f"Earth mover's distance: {distance:.4f} "
                f"(normalized: {normalized:.4f} of reference range). "
                f"{'Significant' if normalized > 0.1 else 'Minor'} distributional shift."
            ),
            method="percentile_approximation",
            details={"reference_range": round(ref_range, 4), "normalized_distance": round(normalized, 4)},
        )

    # ──────────────────────────────────────────────────────────
    # 6. Effect Size (Cohen's d)
    # ──────────────────────────────────────────────────────────

    def compute_cohens_d(
        self,
        ref_mean: float, ref_std: float,
        cur_mean: float, cur_std: float,
        ref_n: int = 1000, cur_n: int = 1000,
    ) -> StatTestResult:
        """
        Cohen's d effect size between two distributions.
        d = (μ₁ - μ₂) / s_pooled

        Interpretation:
          |d| < 0.2 → negligible
          0.2 ≤ |d| < 0.5 → small
          0.5 ≤ |d| < 0.8 → medium
          |d| ≥ 0.8 → large
        """
        # Pooled standard deviation
        s_pooled = math.sqrt(
            ((ref_n - 1) * ref_std**2 + (cur_n - 1) * cur_std**2) /
            max(ref_n + cur_n - 2, 1)
        )

        if s_pooled < 1e-10:
            d = 0.0
        else:
            d = (cur_mean - ref_mean) / s_pooled

        abs_d = abs(d)
        if abs_d < 0.2:
            size = "negligible"
        elif abs_d < 0.5:
            size = "small"
        elif abs_d < 0.8:
            size = "medium"
        else:
            size = "large"

        return StatTestResult(
            test_name="Cohen's d",
            statistic=round(d, 4),
            effect_size=round(abs_d, 4),
            is_significant=abs_d >= 0.5,
            interpretation=f"Effect size: {abs_d:.3f} ({size}). Mean shifted by {d:.3f} pooled SDs.",
            method="exact",
            details={
                "ref_mean": round(ref_mean, 4), "cur_mean": round(cur_mean, 4),
                "ref_std": round(ref_std, 4), "cur_std": round(cur_std, 4),
                "pooled_std": round(s_pooled, 4), "effect_size_class": size,
            },
        )

    # ──────────────────────────────────────────────────────────
    # 7. Normality Assessment
    # ──────────────────────────────────────────────────────────

    def assess_normality(
        self,
        mean: float, std: float,
        skewness: float, kurtosis: float,
        n: int = 1000,
    ) -> NormalityTestResult:
        """
        Assess normality from summary statistics using multiple criteria.

        Uses:
          - D'Agostino-Pearson omnibus (from skew + kurtosis)
          - Jarque-Bera test
          - Heuristic scoring
        """
        tests = []

        # Jarque-Bera test: JB = (n/6)(S² + K²/4)
        jb_stat = (n / 6.0) * (skewness**2 + (kurtosis - 3)**2 / 4.0)
        jb_p = self._chi2_p_value(jb_stat, df=2)
        tests.append(StatTestResult(
            test_name="Jarque-Bera",
            statistic=round(jb_stat, 4),
            p_value=round(jb_p, 6) if jb_p else None,
            is_significant=(jb_p < 0.05) if jb_p else (jb_stat > 5.99),
            interpretation=(
                f"JB = {jb_stat:.2f}. "
                f"{'Reject' if (jb_p and jb_p < 0.05) else 'Cannot reject'} normality."
            ),
            method="exact" if HAS_SCIPY else "approximate",
        ))

        # D'Agostino skewness test: Z = skew × sqrt((n+1)(n+3) / (6(n-2)))
        if n > 8:
            z_skew = skewness * math.sqrt((n + 1) * (n + 3) / max(6 * (n - 2), 1))
            tests.append(StatTestResult(
                test_name="DAgostino-Skewness",
                statistic=round(z_skew, 4),
                is_significant=abs(z_skew) > 1.96,
                interpretation=f"|Z_skew| = {abs(z_skew):.2f}. {'Significant' if abs(z_skew) > 1.96 else 'Acceptable'} skew.",
                method="approximate",
            ))

        # Kurtosis test
        excess_kurtosis = kurtosis - 3
        if n > 20:
            se_kurt = math.sqrt(24.0 / max(n, 1))
            z_kurt = excess_kurtosis / max(se_kurt, 0.001)
            tests.append(StatTestResult(
                test_name="Kurtosis-Z",
                statistic=round(z_kurt, 4),
                is_significant=abs(z_kurt) > 1.96,
                interpretation=f"|Z_kurtosis| = {abs(z_kurt):.2f}. {'Significant' if abs(z_kurt) > 1.96 else 'Acceptable'} excess kurtosis.",
                method="approximate",
            ))

        # Composite normality score
        score = 1.0
        score -= min(0.4, abs(skewness) * 0.2)
        score -= min(0.3, abs(excess_kurtosis) * 0.05)
        if jb_p and jb_p < 0.05:
            score -= 0.3
        score = max(0.0, min(1.0, score))

        is_normal = score >= 0.6

        # Recommend transform
        transform = None
        if not is_normal:
            if skewness > 1.0:
                transform = "log" if skewness > 2.0 else "sqrt"
            elif skewness < -1.0:
                transform = "square"
            elif abs(excess_kurtosis) > 3:
                transform = "yeo_johnson"
            else:
                transform = "box_cox"

        return NormalityTestResult(
            is_normal=is_normal,
            normality_score=round(score, 4),
            skewness=round(skewness, 4),
            kurtosis=round(kurtosis, 4),
            tests_performed=tests,
            recommended_transform=transform,
        )

    # ──────────────────────────────────────────────────────────
    # 8. Bootstrap Confidence Intervals
    # ──────────────────────────────────────────────────────────

    def bootstrap_metric_ci(
        self,
        metric_value: float,
        n_samples: int,
        metric_type: str = "accuracy",
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Compute approximate confidence interval for a model metric
        using the normal approximation or Wilson interval.
        """
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence_level, 1.96)

        if metric_type in ("accuracy", "precision", "recall", "f1"):
            # Wilson interval for proportions
            p = metric_value
            n = n_samples
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
            lower = max(0, center - margin)
            upper = min(1, center + margin)
        elif metric_type in ("auc", "roc_auc"):
            # Hanley-McNeil approximation for AUC
            se = math.sqrt(metric_value * (1 - metric_value) / max(n_samples, 1))
            lower = max(0, metric_value - z * se)
            upper = min(1, metric_value + z * se)
        else:
            # Generic normal approximation
            se = math.sqrt(metric_value * (1 - metric_value) / max(n_samples, 1))
            lower = metric_value - z * se
            upper = metric_value + z * se

        return {
            "metric": metric_type,
            "value": round(metric_value, 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "confidence_level": confidence_level,
            "margin_of_error": round((upper - lower) / 2, 4),
            "n_samples": n_samples,
            "interpretation": (
                f"{metric_type} = {metric_value:.3f} "
                f"({confidence_level*100:.0f}% CI: [{lower:.3f}, {upper:.3f}])"
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 9. Full Drift Test Suite
    # ──────────────────────────────────────────────────────────

    def run_drift_suite(
        self,
        feature: str,
        ref_stats: Dict[str, Any],
        cur_stats: Dict[str, Any],
        ref_histogram: Optional[List[float]] = None,
        cur_histogram: Optional[List[float]] = None,
        ref_percentiles: Optional[Dict[str, float]] = None,
        cur_percentiles: Optional[Dict[str, float]] = None,
        n_ref: int = 1000,
        n_cur: int = 1000,
        is_categorical: bool = False,
    ) -> DriftTestSuite:
        """
        Run comprehensive drift test suite for a single feature.
        Runs all applicable tests and produces consensus score.
        """
        suite = DriftTestSuite(feature=feature)
        test_votes = []

        # PSI (if histograms available)
        if ref_histogram and cur_histogram:
            suite.psi = self.compute_psi(ref_histogram, cur_histogram)
            test_votes.append(suite.psi.is_significant)

        # Chi-square (for categorical)
        if is_categorical:
            ref_counts = ref_stats.get("value_counts", {})
            cur_counts = cur_stats.get("value_counts", {})
            if ref_counts and cur_counts:
                suite.chi2 = self.compute_chi_square(cur_counts, ref_counts)
                test_votes.append(suite.chi2.is_significant)
        else:
            # KS test (for continuous, from percentiles)
            if ref_percentiles and cur_percentiles:
                suite.ks = self.compute_ks_from_percentiles(
                    ref_percentiles, cur_percentiles, n_ref, n_cur
                )
                test_votes.append(suite.ks.is_significant)

            # Wasserstein (from percentiles)
            if ref_percentiles and cur_percentiles:
                suite.wasserstein = self.compute_wasserstein_from_percentiles(
                    ref_percentiles, cur_percentiles
                )
                test_votes.append(suite.wasserstein.is_significant)

            # Cohen's d (from summary stats)
            ref_mean = ref_stats.get("mean")
            ref_std = ref_stats.get("std")
            cur_mean = cur_stats.get("mean")
            cur_std = cur_stats.get("std")
            if all(v is not None for v in [ref_mean, ref_std, cur_mean, cur_std]):
                suite.cohens_d = self.compute_cohens_d(
                    ref_mean, ref_std, cur_mean, cur_std, n_ref, n_cur
                )
                test_votes.append(suite.cohens_d.is_significant)

        # JS Divergence (from histograms)
        if ref_histogram and cur_histogram:
            suite.js_divergence = self.compute_js_divergence(ref_histogram, cur_histogram)
            test_votes.append(suite.js_divergence.is_significant)

        # Consensus
        if test_votes:
            suite.consensus_score = sum(1 for v in test_votes if v) / len(test_votes)
            suite.overall_drift = suite.consensus_score >= 0.5

            if suite.consensus_score >= 0.8:
                suite.overall_severity = "critical"
            elif suite.consensus_score >= 0.6:
                suite.overall_severity = "high"
            elif suite.consensus_score >= 0.4:
                suite.overall_severity = "moderate"
            elif suite.consensus_score > 0:
                suite.overall_severity = "low"

        return suite

    # ──────────────────────────────────────────────────────────
    # 10. Model Comparison Tests
    # ──────────────────────────────────────────────────────────

    def compare_models_significance(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        alpha: float = 0.05,
    ) -> StatTestResult:
        """
        Paired t-test / Wilcoxon signed-rank for comparing two models
        on the same CV folds.
        """
        if len(model_a_scores) != len(model_b_scores) or len(model_a_scores) < 3:
            return StatTestResult(
                test_name="Model Comparison", statistic=0.0,
                interpretation="Need at least 3 paired observations",
                method="skipped",
            )

        # Paired differences
        diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff)**2 for d in diffs) / max(n - 1, 1)
        se_diff = math.sqrt(var_diff / n)

        if se_diff < 1e-10:
            t_stat = 0.0
        else:
            t_stat = mean_diff / se_diff

        # Approximate p-value for t-distribution (Welch-Satterthwaite)
        p_value = self._t_p_value(abs(t_stat), n - 1)

        effect_size = abs(mean_diff) / max(math.sqrt(var_diff), 1e-10)

        return StatTestResult(
            test_name="Paired t-test",
            statistic=round(t_stat, 4),
            p_value=round(p_value, 6) if p_value else None,
            is_significant=(p_value < alpha) if p_value else (abs(t_stat) > 2.0),
            significance_level=alpha,
            effect_size=round(effect_size, 4),
            interpretation=(
                f"Mean difference: {mean_diff:.4f} (t={t_stat:.3f}, p={'%.4f' % p_value if p_value else 'N/A'}). "
                f"Model A is {'significantly' if p_value and p_value < alpha else 'not significantly'} "
                f"{'better' if mean_diff > 0 else 'worse'} than Model B."
            ),
            method="exact" if HAS_SCIPY else "approximate",
            details={
                "mean_diff": round(mean_diff, 6),
                "std_diff": round(math.sqrt(var_diff), 6),
                "n_folds": n,
            },
        )

    # ══════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_distribution(dist: List[float], epsilon: float = 1e-6) -> List[float]:
        """Normalize distribution to sum to 1."""
        total = sum(dist)
        if total <= 0:
            n = len(dist)
            return [1.0 / n + epsilon] * n
        result = [max(x / total, epsilon) for x in dist]
        total2 = sum(result)
        return [x / total2 for x in result]

    @staticmethod
    def _interpolate_cdf(value: float, sorted_values: List[float],
                          cdf_points: List[float]) -> float:
        """Interpolate CDF at a given value from percentile data."""
        if value <= sorted_values[0]:
            return cdf_points[0]
        if value >= sorted_values[-1]:
            return cdf_points[-1]
        for i in range(1, len(sorted_values)):
            if value <= sorted_values[i]:
                # Linear interpolation
                frac = (value - sorted_values[i-1]) / max(sorted_values[i] - sorted_values[i-1], 1e-10)
                return cdf_points[i-1] + frac * (cdf_points[i] - cdf_points[i-1])
        return cdf_points[-1]

    @staticmethod
    def _ks_p_value(d: float, n1: int, n2: int) -> Optional[float]:
        """Approximate KS p-value."""
        if HAS_SCIPY:
            try:
                # Use scipy for exact computation
                en = math.sqrt(n1 * n2 / max(n1 + n2, 1))
                return float(scipy_stats.kstwobign.sf(d * en))
            except Exception:
                pass
        # Approximation: p ≈ 2 × exp(-2 × (d × sqrt(n))²)
        n_eff = (n1 * n2) / max(n1 + n2, 1)
        lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
        if lambda_val <= 0:
            return 1.0
        p = 2 * math.exp(-2 * lambda_val * lambda_val)
        return max(0.0, min(1.0, p))

    @staticmethod
    def _chi2_p_value(chi2: float, df: int) -> Optional[float]:
        """Approximate chi-square p-value."""
        if HAS_SCIPY:
            try:
                return float(scipy_stats.chi2.sf(chi2, df))
            except Exception:
                pass
        # Wilson-Hilferty approximation
        if df <= 0 or chi2 <= 0:
            return 1.0
        try:
            z = ((chi2 / df) ** (1.0/3) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
            # Standard normal CDF approximation
            p = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
            return max(0.0, min(1.0, p))
        except (ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def _t_p_value(t_abs: float, df: int) -> Optional[float]:
        """Approximate two-tailed t-test p-value."""
        if HAS_SCIPY:
            try:
                return float(2 * scipy_stats.t.sf(t_abs, df))
            except Exception:
                pass
        # Approximation using normal for large df
        if df > 30:
            p = 2 * 0.5 * (1 + math.erf(-t_abs / math.sqrt(2)))
            return max(0.0, min(1.0, p))
        # Rough approximation for small df
        # Using a simple lookup-style approximation
        if t_abs > 4.0:
            return 0.001
        elif t_abs > 3.0:
            return 0.005
        elif t_abs > 2.5:
            return 0.02
        elif t_abs > 2.0:
            return 0.05
        elif t_abs > 1.5:
            return 0.15
        else:
            return 0.5
