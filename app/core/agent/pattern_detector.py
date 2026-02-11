"""
Pattern Detector — Advanced Anti-Pattern & Signal Detection
=============================================================
Detects complex patterns that simple threshold rules miss.
Works entirely from compiled context metadata.

Detections:
  1. Target Leakage Signals     — Features that are proxies for the target
  2. Confounding Variable Signs  — Hidden variables driving correlations
  3. Simpson's Paradox Risk      — Aggregate trend vs subgroup trends
  4. Class Overlap Estimation    — How separable are the classes
  5. Label Noise Signals         — Inconsistencies suggesting labeling errors
  6. Feature Redundancy Groups   — Sets of features providing same info
  7. Data Drift Indicators       — Distribution shift signals (for monitoring)
  8. Concept Drift Indicators    — Relationship change signals
  9. Schema Drift Detection      — Column type/name changes
  10. Curse of Dimensionality    — When feature space is too large
"""

import logging
import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectedPattern:
    """A detected anti-pattern or signal."""
    pattern_type: str           # leakage | confounding | simpson | overlap | noise | redundancy | drift
    severity: str               # critical | warning | info
    title: str
    explanation: str
    evidence: List[str]
    affected_features: List[str] = field(default_factory=list)
    confidence: float = 0.5
    recommended_action: str = ""
    quantitative_detail: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "severity": self.severity,
            "title": self.title,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "affected_features": self.affected_features,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "quantitative_detail": self.quantitative_detail,
        }


class PatternDetector:
    """
    Detects complex patterns from compiled context metadata.
    All methods are pure analytical functions — no database access.
    """

    def detect_all(self, context: Dict[str, Any]) -> List[DetectedPattern]:
        """Run all pattern detectors and return findings."""
        # Bridge target_variable → screen_context (same fix as rule_engine)
        target_info = context.get("target_variable", {})
        if isinstance(target_info, dict) and target_info.get("name"):
            sc = context.get("screen_context")
            if sc is None:
                sc = {}
                context["screen_context"] = sc
            if not sc.get("target_column"):
                sc["target_column"] = target_info["name"]
            fs = context.get("frontend_state")
            if fs is None:
                fs = {}
                context["frontend_state"] = fs
            if not fs.get("target_column"):
                fs["target_column"] = target_info["name"]

        patterns = []
        patterns.extend(self._detect_leakage_signals(context))
        patterns.extend(self._detect_confounding_signs(context))
        patterns.extend(self._detect_class_overlap(context))
        patterns.extend(self._detect_dimensionality_curse(context))
        patterns.extend(self._detect_feature_redundancy(context))
        patterns.extend(self._detect_label_noise_signals(context))
        patterns.extend(self._detect_data_drift(context))
        patterns.extend(self._detect_train_test_contamination(context))
        patterns.extend(self._detect_distribution_pathologies(context))

        # Sort by severity
        order = {"critical": 0, "warning": 1, "info": 2}
        patterns.sort(key=lambda p: (order.get(p.severity, 99), -p.confidence))
        return patterns

    # ──────────────────────────────────────────────────────────
    # 1. TARGET LEAKAGE SIGNALS
    # ──────────────────────────────────────────────────────────

    def _detect_leakage_signals(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect features that may leak the target variable.

        Signals:
        - Perfect or near-perfect correlation with target
        - Feature names suggesting post-hoc derivation
        - Features with suspiciously high predictive power
        """
        patterns = []
        correlations = ctx.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])
        feature_stats = ctx.get("feature_stats", {})
        profile = ctx.get("dataset_profile", {})

        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")

        if not target:
            return patterns

        # Check for high target correlations
        suspect_features = []
        for pair in high_pairs:
            f1, f2 = pair.get("feature1", ""), pair.get("feature2", "")
            corr = abs(pair.get("correlation", 0))

            if target in (f1, f2):
                other = f2 if f1 == target else f1
                if corr >= 0.95:
                    suspect_features.append({"feature": other, "correlation": corr, "risk": "extreme"})
                elif corr >= 0.85:
                    suspect_features.append({"feature": other, "correlation": corr, "risk": "high"})

        if suspect_features:
            extreme = [f for f in suspect_features if f["risk"] == "extreme"]
            if extreme:
                patterns.append(DetectedPattern(
                    pattern_type="leakage",
                    severity="critical",
                    title=f"Probable Target Leakage ({len(extreme)} feature{'s' if len(extreme) > 1 else ''})",
                    explanation=(
                        f"Feature(s) with near-perfect correlation to target '{target}': "
                        f"{', '.join(f['feature'] + ' (r=' + str(round(f['correlation'], 3)) + ')' for f in extreme)}. "
                        f"This almost certainly indicates data leakage — these features are "
                        f"derived from or perfectly predict the target, which means the model "
                        f"will show excellent training/test scores but fail completely on new data."
                    ),
                    evidence=[
                        f"{f['feature']}: r={f['correlation']:.3f} with target"
                        for f in extreme
                    ],
                    affected_features=[f["feature"] for f in extreme],
                    confidence=0.95,
                    recommended_action=(
                        "Remove these features IMMEDIATELY. They are either: "
                        "(1) directly derived from the target, (2) collected after the outcome, "
                        "or (3) perfect proxies. A model using these will not generalize."
                    ),
                    quantitative_detail={
                        "leaking_features": [
                            {"name": f["feature"], "correlation": f["correlation"]}
                            for f in extreme
                        ]
                    },
                ))

        # Check for post-hoc feature names
        # Note: "billing" and "payment" removed — too generic, causes false positives
        # on legitimate features like PaperlessBilling, PaymentMethod
        posthoc_patterns = [
            "result", "outcome", "final", "total_charge",
            "settled", "resolved", "closed", "cancelled", "cancellation",
            "end_date", "close_date", "resolution", "disposition",
        ]
        # Exclusion: if column name contains these, it's likely a feature, not post-hoc
        posthoc_exclude = ["method", "type", "plan", "option", "preference", "category"]
        column_names = profile.get("column_names", [])
        posthoc_suspects = []
        for col in column_names:
            col_lower = col.lower()
            if col == target:
                continue
            if any(ex in col_lower for ex in posthoc_exclude):
                continue
            for pattern in posthoc_patterns:
                if pattern in col_lower:
                    posthoc_suspects.append(col)
                    break

        if posthoc_suspects and not suspect_features:
            patterns.append(DetectedPattern(
                pattern_type="leakage",
                severity="info",
                title=f"Potential Post-Hoc Features Detected ({len(posthoc_suspects)})",
                explanation=(
                    f"Column names suggest these may be derived after the outcome: "
                    f"{', '.join(posthoc_suspects[:5])}. If these represent information "
                    f"available only after the target event, they constitute target leakage."
                ),
                evidence=[f"Suspicious name: {col}" for col in posthoc_suspects[:5]],
                affected_features=posthoc_suspects[:5],
                confidence=0.4,
                recommended_action=(
                    "Verify the causal timeline: were these features known BEFORE the "
                    "target event? If not, exclude them from training."
                ),
            ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 2. CONFOUNDING VARIABLE SIGNS
    # ──────────────────────────────────────────────────────────

    def _detect_confounding_signs(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect potential confounding variables.

        A confounding variable drives both the predictor and the outcome,
        creating a spurious correlation. We detect this via:
        - Two features highly correlated with each other AND both correlated with target
        - Suggests a hidden common cause
        """
        patterns = []
        correlations = ctx.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])

        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        target = screen_ctx.get("target_column") or frontend.get("target_column")

        if not target or len(high_pairs) < 2:
            return patterns

        # Build feature-target correlation map
        target_corr = {}
        for pair in high_pairs:
            f1, f2 = pair.get("feature1", ""), pair.get("feature2", "")
            corr = abs(pair.get("correlation", 0))
            if f1 == target:
                target_corr[f2] = corr
            elif f2 == target:
                target_corr[f1] = corr

        # Find feature pairs that are both correlated with each other AND target
        confound_trios = []
        for pair in high_pairs:
            f1, f2 = pair.get("feature1", ""), pair.get("feature2", "")
            inter_corr = abs(pair.get("correlation", 0))
            if f1 == target or f2 == target:
                continue
            if inter_corr < 0.6:
                continue

            tc1 = target_corr.get(f1, 0)
            tc2 = target_corr.get(f2, 0)

            if tc1 > 0.3 and tc2 > 0.3:
                confound_trios.append({
                    "feature_a": f1, "feature_b": f2,
                    "inter_correlation": inter_corr,
                    "target_corr_a": tc1, "target_corr_b": tc2,
                })

        if confound_trios:
            patterns.append(DetectedPattern(
                pattern_type="confounding",
                severity="warning",
                title=f"Potential Confounding Detected ({len(confound_trios)} trio{'s' if len(confound_trios) > 1 else ''})",
                explanation=(
                    f"Found {len(confound_trios)} feature pair(s) where both features correlate "
                    f"with each other AND with the target. This pattern suggests a hidden "
                    f"confounding variable may be driving the relationship. "
                    f"Example: {confound_trios[0]['feature_a']} and {confound_trios[0]['feature_b']} "
                    f"(inter-r={confound_trios[0]['inter_correlation']:.2f}) both correlate with target "
                    f"(r={confound_trios[0]['target_corr_a']:.2f}, r={confound_trios[0]['target_corr_b']:.2f})."
                ),
                evidence=[
                    f"{t['feature_a']} ↔ {t['feature_b']} (r={t['inter_correlation']:.2f}), "
                    f"both → target (r={t['target_corr_a']:.2f}, {t['target_corr_b']:.2f})"
                    for t in confound_trios[:3]
                ],
                affected_features=list(set(
                    [t["feature_a"] for t in confound_trios] +
                    [t["feature_b"] for t in confound_trios]
                )),
                confidence=0.55,
                recommended_action=(
                    "When features share a confounder: (1) Consider which one has the more "
                    "direct causal relationship with the target. (2) Partial correlation analysis "
                    "can help identify the true driver. (3) For prediction purposes (not causal "
                    "inference), you can keep both — but be cautious about feature importance interpretation."
                ),
                quantitative_detail={"trios": confound_trios[:5]},
            ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 3. CLASS OVERLAP ESTIMATION
    # ──────────────────────────────────────────────────────────

    def _detect_class_overlap(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Estimate class separability from feature statistics.
        High overlap → harder classification, lower AUC ceiling.
        """
        patterns = []
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})

        if not metrics:
            return patterns

        auc = metrics.get("roc_auc") or metrics.get("auc_roc") or metrics.get("auc", 0)
        f1 = metrics.get("f1_score") or metrics.get("f1", 0)
        accuracy = metrics.get("accuracy", 0)

        if not auc and not f1:
            return patterns

        # Low AUC + low F1 + decent accuracy = severe class overlap
        if auc and 0.5 < auc < 0.7 and f1 and f1 < 0.4:
            patterns.append(DetectedPattern(
                pattern_type="overlap",
                severity="warning",
                title="Significant Class Overlap — Problem May Be Inherently Difficult",
                explanation=(
                    f"AUC-ROC of {auc:.3f} with F1 of {f1:.3f} suggests the classes "
                    f"significantly overlap in feature space. This means no model (no matter "
                    f"how complex) can perfectly separate the classes — there's a ceiling "
                    f"on achievable performance. This is a property of the DATA, not the model."
                ),
                evidence=[
                    f"AUC-ROC: {auc:.3f} (ideal: >0.8)",
                    f"F1: {f1:.3f} (suggests poor class separation)",
                ],
                confidence=0.7,
                recommended_action=(
                    "When classes inherently overlap: "
                    "(1) Better feature engineering is your best lever — create features that "
                    "discriminate between classes. "
                    "(2) Consider domain expert features. "
                    "(3) Collect additional data signals. "
                    "(4) Accept that some error rate is inevitable and optimize the threshold "
                    "for your business cost function."
                ),
                quantitative_detail={
                    "auc_roc": auc,
                    "f1": f1,
                    "estimated_overlap": round(1 - (auc - 0.5) * 2, 2),
                },
            ))

        # Suspiciously perfect scores = likely leakage or trivial problem
        if auc and auc > 0.99:
            patterns.append(DetectedPattern(
                pattern_type="leakage",
                severity="critical",
                title=f"Suspiciously Perfect AUC ({auc:.4f}) — Possible Data Leakage",
                explanation=(
                    f"AUC-ROC of {auc:.4f} is nearly perfect. In real-world problems, "
                    f"this almost always indicates data leakage rather than a genuinely "
                    f"perfect model. Check for features that directly encode the target."
                ),
                evidence=[f"AUC-ROC: {auc:.4f}"],
                confidence=0.85,
                recommended_action=(
                    "Investigate immediately: (1) Check if any feature has near-perfect "
                    "correlation with the target. (2) Verify the train/test split is proper. "
                    "(3) Look for features created after the target event."
                ),
            ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 4. CURSE OF DIMENSIONALITY
    # ──────────────────────────────────────────────────────────

    def _detect_dimensionality_curse(self, ctx: Dict) -> List[DetectedPattern]:
        """Detect when the feature space is too large for the sample size."""
        patterns = []
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        if rows == 0 or cols == 0:
            return patterns

        ratio = rows / cols

        if ratio < 2:
            # Skip critical PD-DIM when cols > rows — DQ-008 rule already fires
            # Pattern detector adds value only for marginal cases (ratio 2-10)
            if rows >= cols:
                patterns.append(DetectedPattern(
                    pattern_type="dimensionality",
                    severity="critical",
                    title=f"Curse of Dimensionality: {cols} Features, {rows} Samples (Ratio: {ratio:.1f}:1)",
                    explanation=(
                        f"With {cols} features and only {rows} samples, the feature-to-sample "
                        f"ratio is {ratio:.1f}:1. In this regime:\n"
                        f"• Distance metrics become meaningless (all points equidistant)\n"
                        f"• Every model will overfit without extreme regularization\n"
                        f"• Cross-validation estimates are unreliable\n"
                        f"• Need at minimum 10x samples per feature for stable estimates"
                    ),
                    evidence=[
                        f"Features: {cols}",
                        f"Samples: {rows}",
                        f"Ratio: {ratio:.1f}:1 (need ≥10:1)",
                        f"Recommended max features: {rows // 10}",
                    ],
                    affected_features=[],
                    confidence=1.0,
                    recommended_action=(
                        f"Aggressively reduce to ≤{rows // 10} features using: "
                        f"(1) Variance threshold (drop near-zero variance). "
                        f"(2) Mutual information scoring. "
                        f"(3) L1 regularization (Lasso) for automatic selection. "
                        f"(4) PCA to {min(rows // 15, 20)} components."
                    ),
                    quantitative_detail={
                        "features": cols, "samples": rows,
                        "ratio": round(ratio, 2),
                        "recommended_max_features": rows // 10,
                    },
                ))
        elif ratio < 10:
            patterns.append(DetectedPattern(
                pattern_type="dimensionality",
                severity="warning",
                title=f"Marginal Sample-to-Feature Ratio ({ratio:.1f}:1)",
                explanation=(
                    f"Ratio of {ratio:.1f}:1 is below the recommended 10:1 minimum. "
                    f"Models may produce unstable coefficients and optimistic CV scores."
                ),
                evidence=[f"Ratio: {ratio:.1f}:1", f"Need: ≥10:1 (ideally 20:1)"],
                confidence=0.9,
                recommended_action=(
                    f"Consider feature selection to reduce from {cols} to ~{rows // 15} features. "
                    f"Use regularized models (Ridge, Lasso, ElasticNet) to stabilize estimates."
                ),
            ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 5. FEATURE REDUNDANCY GROUPS
    # ──────────────────────────────────────────────────────────

    def _detect_feature_redundancy(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect groups of features providing redundant information.
        Goes beyond pairwise correlation to find multi-feature redundancy.
        """
        patterns = []
        correlations = ctx.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])

        # Find features that are correlated with many others
        corr_counts = {}
        for pair in high_pairs:
            abs_corr = pair.get("abs_correlation", 0)
            if abs_corr >= 0.65:
                f1, f2 = pair["feature1"], pair["feature2"]
                corr_counts[f1] = corr_counts.get(f1, 0) + 1
                corr_counts[f2] = corr_counts.get(f2, 0) + 1

        # Features correlated with 3+ others are "hub" features
        hub_features = {f: c for f, c in corr_counts.items() if c >= 3}

        if hub_features:
            top_hubs = sorted(hub_features.items(), key=lambda x: -x[1])[:5]
            patterns.append(DetectedPattern(
                pattern_type="redundancy",
                severity="warning",
                title=f"Feature Redundancy Hubs Detected ({len(hub_features)} features)",
                explanation=(
                    f"These features are each correlated (|r| ≥ 0.65) with 3+ other features: "
                    f"{', '.join(f'{f} (→{c} others)' for f, c in top_hubs)}. "
                    f"This creates redundant information that inflates model complexity, "
                    f"dilutes feature importance scores, and can cause instability in "
                    f"coefficient-based models."
                ),
                evidence=[
                    f"{f}: correlated with {c} other features"
                    for f, c in top_hubs
                ],
                affected_features=[f for f, _ in top_hubs],
                confidence=0.8,
                recommended_action=(
                    "For each hub feature: keep it only if it has the strongest individual "
                    "correlation with the target. Otherwise, replace the group with PCA "
                    "components or select the single best representative."
                ),
                quantitative_detail={
                    "hub_features": dict(top_hubs),
                    "total_redundant": len(hub_features),
                },
            ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 6. LABEL NOISE SIGNALS
    # ──────────────────────────────────────────────────────────

    def _detect_label_noise_signals(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect signals that suggest labeling errors in the target.

        Indicators:
        - Very low F1 despite good features (high correlation features exist)
        - High AUC but low accuracy (model separates but labels are noisy)
        - Training accuracy plateaus well below 100% even with high capacity
        """
        patterns = []
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})

        if not metrics:
            return patterns

        train_score = metrics.get("train_score") or metrics.get("training_accuracy", 0)
        test_score = metrics.get("test_score") or metrics.get("test_accuracy", 0)
        auc = metrics.get("roc_auc") or metrics.get("auc_roc", 0)
        f1 = metrics.get("f1_score") or metrics.get("f1", 0)

        # Signal: Model can rank well (high AUC) but can't classify well (low F1)
        if auc and f1 and auc > 0.8 and f1 < 0.4:
            patterns.append(DetectedPattern(
                pattern_type="label_noise",
                severity="info",
                title="Possible Label Noise — High AUC But Low F1",
                explanation=(
                    f"The model has good discrimination ability (AUC={auc:.3f}) but poor "
                    f"classification performance (F1={f1:.3f}). This paradox can indicate "
                    f"label noise: the model learns the general pattern but noisy labels "
                    f"prevent clean class boundaries. Another explanation is severe class "
                    f"imbalance with a suboptimal threshold."
                ),
                evidence=[
                    f"AUC-ROC: {auc:.3f} (good ranking)",
                    f"F1: {f1:.3f} (poor classification)",
                    f"Gap suggests noisy boundary",
                ],
                confidence=0.45,
                recommended_action=(
                    "(1) Optimize the classification threshold using the PR curve. "
                    "(2) If threshold tuning doesn't help, investigate label quality. "
                    "(3) Use confident learning (cleanlab library) to identify mislabeled samples. "
                    "(4) Re-examine the labeling criteria with domain experts."
                ),
            ))

        # Signal: Training accuracy plateaus well below 1.0 for high-capacity model
        if train_score and 0.65 < train_score < 0.85:
            gap = abs(train_score - test_score) if test_score else 0
            if gap < 0.05:  # Not overfitting, but not learning either
                patterns.append(DetectedPattern(
                    pattern_type="label_noise",
                    severity="info",
                    title="Model Plateaus at Moderate Accuracy — Data Ceiling?",
                    explanation=(
                        f"Training accuracy ({train_score:.1%}) is similar to test accuracy "
                        f"({test_score:.1%}) — the model isn't overfitting, but it's also "
                        f"hitting a ceiling. This can mean: (1) the features genuinely can't "
                        f"predict better, (2) there's irreducible noise/label errors, or "
                        f"(3) the model architecture is too simple."
                    ),
                    evidence=[
                        f"Train: {train_score:.1%}",
                        f"Test: {test_score:.1%}",
                        f"Gap: {gap:.1%} (not overfitting)",
                    ],
                    confidence=0.35,
                    recommended_action=(
                        "Try a more complex model (e.g., XGBoost if using linear). "
                        "If the plateau persists, the data itself may have a noise ceiling. "
                        "Consider better feature engineering or collecting additional signals."
                    ),
                ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 7. DATA DRIFT INDICATORS (for monitoring)
    # ──────────────────────────────────────────────────────────

    def _detect_data_drift(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect data drift signals from monitoring context.
        Checks PSI, feature distribution shifts, prediction distribution changes.
        """
        patterns = []
        frontend = ctx.get("frontend_state", {}) or {}
        screen = ctx.get("screen", "")

        if screen not in ("monitoring", "predictions"):
            return patterns

        # Check for drift metrics in frontend state
        drift_metrics = frontend.get("drift_metrics", {})
        feature_drifts = frontend.get("feature_drifts", [])

        # PSI-based drift detection
        for feature_drift in feature_drifts:
            if not isinstance(feature_drift, dict):
                continue
            psi = feature_drift.get("psi", 0)
            feature_name = feature_drift.get("feature", "unknown")

            if psi > 0.25:
                patterns.append(DetectedPattern(
                    pattern_type="data_drift",
                    severity="critical",
                    title=f"Significant Data Drift: {feature_name} (PSI={psi:.3f})",
                    explanation=(
                        f"Feature '{feature_name}' shows a Population Stability Index of {psi:.3f}. "
                        f"PSI > 0.25 indicates a significant distribution shift from the training data. "
                        f"Model predictions may no longer be reliable for this feature's values."
                    ),
                    evidence=[f"PSI: {psi:.3f}", "Threshold: 0.25 (significant shift)"],
                    affected_features=[feature_name],
                    confidence=0.9,
                    recommended_action=(
                        "This feature has drifted significantly. Consider: "
                        "(1) Retraining the model on recent data. "
                        "(2) Investigating why this feature changed (seasonality? data pipeline issue?). "
                        "(3) Adding a fallback prediction for drifted inputs."
                    ),
                ))
            elif psi > 0.1:
                patterns.append(DetectedPattern(
                    pattern_type="data_drift",
                    severity="warning",
                    title=f"Moderate Drift: {feature_name} (PSI={psi:.3f})",
                    explanation=(
                        f"Feature '{feature_name}' PSI of {psi:.3f} indicates moderate distribution shift."
                    ),
                    evidence=[f"PSI: {psi:.3f}", "Threshold: 0.1 (moderate shift)"],
                    affected_features=[feature_name],
                    confidence=0.7,
                    recommended_action="Monitor closely. Schedule retraining if PSI exceeds 0.25.",
                ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 8. TRAIN/TEST CONTAMINATION
    # ──────────────────────────────────────────────────────────

    def _detect_train_test_contamination(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect signs that train and test data are contaminated.

        Signals:
        - Test score higher than train score (impossible without contamination)
        - Test and train scores nearly identical on non-trivial problem
        - Duplicate rows + random split = contamination risk
        """
        patterns = []
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})
        quality = ctx.get("data_quality", {})

        train_score = metrics.get("train_score") or metrics.get("training_accuracy", 0)
        test_score = metrics.get("test_score") or metrics.get("test_accuracy", 0)

        # Test > Train is a strong contamination signal
        if train_score and test_score and test_score > train_score + 0.02:
            patterns.append(DetectedPattern(
                pattern_type="contamination",
                severity="critical",
                title=f"Test Score > Train Score — Possible Data Contamination",
                explanation=(
                    f"Test accuracy ({test_score:.1%}) exceeds training accuracy ({train_score:.1%}) "
                    f"by {(test_score - train_score)*100:.1f}pp. This is statistically unlikely "
                    f"and typically indicates data leakage or contamination between train/test sets."
                ),
                evidence=[
                    f"Train: {train_score:.1%}",
                    f"Test: {test_score:.1%}",
                    f"Gap: {(test_score - train_score)*100:+.1f}pp (should be ≤0)",
                ],
                confidence=0.85,
                recommended_action=(
                    "Verify: (1) No duplicate rows appear in both train and test. "
                    "(2) No feature engineering used test data statistics. "
                    "(3) Scaling/imputation was fit only on training data. "
                    "(4) If using time data, ensure no future leakage."
                ),
            ))

        # Duplicates + random split = contamination risk
        dup_pct = quality.get("duplicate_pct", 0)
        if dup_pct > 5:
            patterns.append(DetectedPattern(
                pattern_type="contamination",
                severity="warning",
                title=f"Duplicate Rows ({dup_pct:.1f}%) + Random Split = Contamination Risk",
                explanation=(
                    f"With {dup_pct:.1f}% duplicate rows, a random train/test split will likely "
                    f"place duplicates in both sets. The model effectively 'sees' test samples "
                    f"during training, inflating evaluation metrics."
                ),
                evidence=[f"Duplicate rows: {dup_pct:.1f}%"],
                confidence=0.75,
                recommended_action=(
                    "Remove duplicates BEFORE splitting, or use group-based splitting "
                    "to ensure no duplicate groups span train/test."
                ),
            ))

        return patterns

    # ──────────────────────────────────────────────────────────
    # 9. DISTRIBUTION PATHOLOGIES
    # ──────────────────────────────────────────────────────────

    def _detect_distribution_pathologies(self, ctx: Dict) -> List[DetectedPattern]:
        """
        Detect pathological distribution issues across all features.
        """
        patterns = []
        feature_stats = ctx.get("feature_stats", {})
        numeric_stats = feature_stats.get("numeric_stats", {})
        profile = ctx.get("dataset_profile", {})
        rows = profile.get("rows", 0)

        extreme_skew_cols = []
        zero_variance_cols = []
        extreme_outlier_cols = []

        for col, stats in numeric_stats.items():
            if not isinstance(stats, dict):
                continue

            skew = abs(stats.get("skewness", 0) or 0)
            std = stats.get("std", 0) or 0
            min_val = stats.get("min", 0) or 0
            max_val = stats.get("max", 0) or 0
            q1 = stats.get("25%", stats.get("q1", 0)) or 0
            q3 = stats.get("75%", stats.get("q3", 0)) or 0
            iqr = q3 - q1

            if std == 0 or (min_val == max_val):
                zero_variance_cols.append(col)
            elif skew > 5:
                extreme_skew_cols.append({"column": col, "skewness": skew})

            if iqr > 0:
                upper = q3 + 3 * iqr
                lower = q1 - 3 * iqr
                if max_val > upper * 2 or min_val < lower * 2:
                    extreme_outlier_cols.append(col)

        # NOTE: Zero-variance detection removed from patterns — FH-003 rule already
        # covers this with identical output, causing duplicate cards in the UI.
        # Pattern detector adds value for distribution issues NOT covered by rules.

        if len(extreme_skew_cols) >= 3:
            patterns.append(DetectedPattern(
                pattern_type="distribution",
                severity="info",
                title=f"{len(extreme_skew_cols)} Features Have Extreme Skewness (|skew| > 5)",
                explanation=(
                    f"Highly skewed features: "
                    f"{', '.join(f['column'] + ' (skew=' + str(round(f['skewness'], 1)) + ')' for f in extreme_skew_cols[:5])}. "
                    f"Extreme skew degrades model performance for algorithms that assume "
                    f"symmetric distributions (Logistic Regression, SVM, KNN)."
                ),
                evidence=[
                    f"{f['column']}: skewness={f['skewness']:.1f}"
                    for f in extreme_skew_cols[:5]
                ],
                affected_features=[f["column"] for f in extreme_skew_cols],
                confidence=0.85,
                recommended_action=(
                    "Apply log(1+x) transform for right-skewed positive features, "
                    "or Yeo-Johnson transform for features with negative values. "
                    "Tree-based models are less affected but transforms still help."
                ),
            ))

        if len(extreme_outlier_cols) >= 3:
            patterns.append(DetectedPattern(
                pattern_type="distribution",
                severity="info",
                title=f"{len(extreme_outlier_cols)} Features Have Extreme Outliers",
                explanation=(
                    f"Features with values far beyond 3×IQR: {', '.join(extreme_outlier_cols[:5])}. "
                    f"Extreme outliers can dominate model fitting, especially for linear models "
                    f"and KNN."
                ),
                evidence=[f"{col}: values beyond 3×IQR" for col in extreme_outlier_cols[:5]],
                affected_features=extreme_outlier_cols,
                confidence=0.7,
                recommended_action=(
                    "Options: (1) RobustScaler instead of StandardScaler. "
                    "(2) Winsorize at 1st/99th percentiles. "
                    "(3) Use tree-based models which are naturally robust to outliers."
                ),
            ))

        return patterns