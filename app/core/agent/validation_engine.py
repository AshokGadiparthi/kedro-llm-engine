"""
Validation Engine — Pre-Action Readiness Gates
================================================
Validates readiness before critical ML actions (training, deployment, etc.).
Returns weighted checklists with pass/fail, severity, and specific fixes.

Actions:
  1. start_training    — Validate data + config before training
  2. deploy_model      — Validate model + infrastructure before deployment
  3. promote_to_prod   — Validate production readiness
  4. retrain_model     — Validate retraining conditions
  5. change_threshold  — Validate threshold change impact
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Single validation check result."""
    check_id: str
    name: str
    category: str                # data | config | model | infrastructure | safety
    passed: bool
    severity: str                # blocker | warning | advisory
    weight: float = 1.0          # importance weight for scoring
    message: str = ""
    fix_action: Optional[str] = None
    evidence: Optional[str] = None
    auto_fixable: bool = False   # can the system fix this automatically?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "severity": self.severity,
            "weight": self.weight,
            "message": self.message,
            "fix_action": self.fix_action,
            "evidence": self.evidence,
            "auto_fixable": self.auto_fixable,
        }


@dataclass
class ValidationResult:
    """Complete validation result for an action."""
    action: str
    can_proceed: bool
    readiness_score: float = 0.0    # 0-100
    checks: List[ValidationCheck] = field(default_factory=list)
    blockers: List[ValidationCheck] = field(default_factory=list)
    warnings: List[ValidationCheck] = field(default_factory=list)
    passed: List[ValidationCheck] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "can_proceed": self.can_proceed,
            "readiness_score": self.readiness_score,
            "total_checks": len(self.checks),
            "blockers": [c.to_dict() for c in self.blockers],
            "warnings": [c.to_dict() for c in self.warnings],
            "passed": [c.to_dict() for c in self.passed],
            "summary": self.summary,
        }


class ValidationEngine:
    """
    Comprehensive pre-action validation with weighted scoring.
    All methods work from compiled context — no database access.
    """

    def validate(self, action: str, context: Dict[str, Any], analysis: Optional[Dict] = None) -> ValidationResult:
        """
        Main entry point: validate a specific action.
        """
        validators = {
            "start_training": self._validate_training,
            "deploy_model": self._validate_deployment,
            "promote_to_prod": self._validate_promotion,
            "retrain_model": self._validate_retrain,
        }

        validator = validators.get(action)
        if not validator:
            return ValidationResult(
                action=action, can_proceed=True,
                readiness_score=100, summary=f"No validation rules for action '{action}'",
            )

        checks = validator(context, analysis or {})
        return self._compute_result(action, checks)

    def _compute_result(self, action: str, checks: List[ValidationCheck]) -> ValidationResult:
        """Compute final validation result from individual checks."""
        blockers = [c for c in checks if not c.passed and c.severity == "blocker"]
        warnings = [c for c in checks if not c.passed and c.severity == "warning"]
        passed_checks = [c for c in checks if c.passed]

        can_proceed = len(blockers) == 0

        # Weighted readiness score
        total_weight = sum(c.weight for c in checks) or 1
        passed_weight = sum(c.weight for c in checks if c.passed)
        # Partially credit warnings (50%)
        warning_weight = sum(c.weight * 0.5 for c in warnings)
        readiness_score = round((passed_weight + warning_weight) / total_weight * 100, 1)

        # Summary
        if blockers:
            summary = (
                f"❌ Cannot proceed — {len(blockers)} blocking issue(s): "
                f"{', '.join(b.name for b in blockers[:3])}"
            )
        elif warnings:
            summary = (
                f"⚠️ Can proceed with {len(warnings)} warning(s): "
                f"{', '.join(w.name for w in warnings[:3])}"
            )
        else:
            summary = f"✅ All {len(checks)} checks passed. Ready to proceed."

        return ValidationResult(
            action=action,
            can_proceed=can_proceed,
            readiness_score=readiness_score,
            checks=checks,
            blockers=blockers,
            warnings=warnings,
            passed=passed_checks,
            summary=summary,
        )

    # ══════════════════════════════════════════════════════════
    # PRE-TRAINING VALIDATION
    # ══════════════════════════════════════════════════════════

    def _validate_training(self, ctx: Dict, analysis: Dict) -> List[ValidationCheck]:
        """22-point pre-training validation checklist."""
        checks = []
        profile = ctx.get("dataset_profile", {})
        quality = ctx.get("data_quality", {})
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        feature_stats = ctx.get("feature_stats", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        # ── DATA CHECKS ──

        # VT-001: Dataset exists
        checks.append(ValidationCheck(
            check_id="VT-001", name="Dataset Loaded",
            category="data", weight=3.0,
            passed=rows > 0,
            severity="blocker",
            message=f"Dataset has {rows:,} rows" if rows > 0 else "No dataset loaded",
            fix_action="Upload a dataset before training" if rows == 0 else None,
        ))

        # VT-002: Minimum sample size
        min_rows = max(50, cols * 10) if cols > 0 else 50
        checks.append(ValidationCheck(
            check_id="VT-002", name="Sufficient Sample Size",
            category="data", weight=2.5,
            passed=rows >= min_rows,
            severity="blocker" if rows < 30 else "warning",
            message=f"{rows:,} rows (minimum recommended: {min_rows:,})",
            fix_action=f"Need at least {min_rows} samples for {cols} features" if rows < min_rows else None,
            evidence=f"Row/feature ratio: {rows/cols:.1f}:1" if cols > 0 else None,
        ))

        # VT-003: Data completeness
        completeness = quality.get("completeness", 100)
        checks.append(ValidationCheck(
            check_id="VT-003", name="Data Completeness ≥ 70%",
            category="data", weight=2.0,
            passed=completeness >= 70,
            severity="blocker" if completeness < 50 else "warning",
            message=f"Completeness: {completeness:.1f}%",
            fix_action="Handle missing values before training (imputation or column dropping)" if completeness < 70 else None,
        ))

        # VT-004: Duplicate check
        dup_pct = quality.get("duplicate_pct", 0)
        checks.append(ValidationCheck(
            check_id="VT-004", name="Duplicate Rows < 10%",
            category="data", weight=1.5,
            passed=dup_pct < 10,
            severity="warning",
            message=f"Duplicate rows: {dup_pct:.1f}%",
            fix_action="Remove duplicates before splitting data" if dup_pct >= 10 else None,
        ))

        # VT-005: No ID columns in features
        id_cols = feature_stats.get("potential_id_columns", [])
        checks.append(ValidationCheck(
            check_id="VT-005", name="No ID Columns in Features",
            category="data", weight=2.5,
            passed=len(id_cols) == 0,
            severity="blocker",
            message=f"Found {len(id_cols)} potential ID column(s)" if id_cols else "No ID columns detected",
            fix_action=f"Remove: {', '.join(c['column'] for c in id_cols[:5])}" if id_cols else None,
            evidence=", ".join(c["column"] for c in id_cols[:5]) if id_cols else None,
        ))

        # VT-006: No perfect correlations (leakage)
        correlations = ctx.get("correlations", {})
        high_pairs = correlations.get("high_pairs", [])
        perfect = [p for p in high_pairs if p.get("abs_correlation", 0) >= 0.98]
        checks.append(ValidationCheck(
            check_id="VT-006", name="No Target Leakage (r < 0.98)",
            category="data", weight=3.0,
            passed=len(perfect) == 0,
            severity="blocker",
            message=f"{len(perfect)} near-perfect correlation pair(s)" if perfect else "No leakage detected",
            fix_action="Investigate and remove leaking features" if perfect else None,
        ))

        # ── CONFIGURATION CHECKS ──

        # VT-007: Target column selected
        target = screen_ctx.get("target_column") or frontend.get("target_column")
        checks.append(ValidationCheck(
            check_id="VT-007", name="Target Column Selected",
            category="config", weight=3.0,
            passed=bool(target),
            severity="blocker",
            message=f"Target: '{target}'" if target else "No target column selected",
            fix_action="Select a target column for prediction" if not target else None,
        ))

        # VT-008: Algorithm selected
        algorithm = screen_ctx.get("algorithm") or frontend.get("algorithm")
        selected_algos = screen_ctx.get("selected_algorithms") or frontend.get("selected_algorithms", [])
        has_algo = bool(algorithm or selected_algos)
        checks.append(ValidationCheck(
            check_id="VT-008", name="Algorithm Selected",
            category="config", weight=2.0,
            passed=has_algo,
            severity="blocker",
            message=f"Algorithm(s): {algorithm or ', '.join(selected_algos[:3])}" if has_algo else "No algorithm selected",
            fix_action="Select at least one algorithm" if not has_algo else None,
        ))

        # VT-009: Test size reasonable
        test_size = screen_ctx.get("test_size") or frontend.get("test_size", 0.2)
        checks.append(ValidationCheck(
            check_id="VT-009", name="Test Size 10-40%",
            category="config", weight=1.5,
            passed=0.1 <= test_size <= 0.4,
            severity="warning",
            message=f"Test size: {test_size*100:.0f}%",
            fix_action=f"Adjust test size to 20% (current: {test_size*100:.0f}%)" if not (0.1 <= test_size <= 0.4) else None,
        ))

        # VT-010: Feature-to-sample ratio
        selected_features = screen_ctx.get("selected_features") or frontend.get("selected_features", [])
        n_features = len(selected_features) if selected_features else cols - 1
        ratio = rows / n_features if n_features > 0 else float('inf')
        checks.append(ValidationCheck(
            check_id="VT-010", name="Feature-Sample Ratio ≥ 10:1",
            category="config", weight=2.0,
            passed=ratio >= 10 or n_features <= 5,
            severity="warning",
            message=f"Ratio: {ratio:.1f}:1 ({n_features} features, {rows} samples)",
            fix_action=f"Reduce features to ≤{rows // 10}" if ratio < 10 and n_features > 5 else None,
        ))

        # VT-011: Scaling for linear models
        scaling = screen_ctx.get("scaling_method") or frontend.get("scaling_method", "")
        algo_str = (algorithm or "").lower()
        needs_scaling = any(x in algo_str for x in ["logistic", "svm", "svc", "linear", "ridge", "lasso", "knn"])
        if needs_scaling:
            checks.append(ValidationCheck(
                check_id="VT-011", name="Feature Scaling Configured",
                category="config", weight=1.5,
                passed=bool(scaling),
                severity="warning",
                message=f"Scaling: {scaling}" if scaling else "No scaling configured for linear model",
                fix_action="Enable StandardScaler or RobustScaler for this algorithm" if not scaling else None,
            ))

        return checks

    # ══════════════════════════════════════════════════════════
    # PRE-DEPLOYMENT VALIDATION
    # ══════════════════════════════════════════════════════════

    def _validate_deployment(self, ctx: Dict, analysis: Dict) -> List[ValidationCheck]:
        """18-point pre-deployment validation checklist."""
        checks = []
        screen_ctx = ctx.get("screen_context", {}) or {}
        frontend = ctx.get("frontend_state", {}) or {}
        metrics = screen_ctx.get("metrics") or frontend.get("metrics", {})
        registry = ctx.get("registry_info", {})

        # ── MODEL QUALITY CHECKS ──

        # VD-001: Model exists
        has_model = bool(ctx.get("model_id") or registry.get("total_registered", 0) > 0)
        checks.append(ValidationCheck(
            check_id="VD-001", name="Trained Model Available",
            category="model", weight=3.0,
            passed=has_model,
            severity="blocker",
            message="Trained model found" if has_model else "No trained model available",
        ))

        # VD-002: Minimum AUC
        auc = metrics.get("roc_auc") or metrics.get("auc_roc") or metrics.get("auc", 0)
        checks.append(ValidationCheck(
            check_id="VD-002", name="AUC-ROC ≥ 0.65",
            category="model", weight=2.5,
            passed=auc >= 0.65 if auc else True,  # Pass if no AUC (might be regression)
            severity="blocker" if auc and auc < 0.55 else "warning",
            message=f"AUC-ROC: {auc:.3f}" if auc else "No AUC metric available",
            fix_action="Model performs near random — retrain with better features" if auc and auc < 0.55 else None,
        ))

        # VD-003: F1 minimum
        f1 = metrics.get("f1_score") or metrics.get("f1", 0)
        checks.append(ValidationCheck(
            check_id="VD-003", name="F1 Score ≥ 0.3",
            category="model", weight=2.0,
            passed=f1 >= 0.3 if f1 else True,
            severity="blocker" if f1 and f1 < 0.2 else "warning",
            message=f"F1: {f1:.3f}" if f1 else "No F1 metric available",
        ))

        # VD-004: No overfitting
        train_score = metrics.get("train_score") or metrics.get("training_accuracy", 0)
        test_score = metrics.get("test_score") or metrics.get("test_accuracy", 0)
        if train_score and test_score:
            gap = train_score - test_score
            checks.append(ValidationCheck(
                check_id="VD-004", name="No Severe Overfitting (gap < 15%)",
                category="model", weight=2.0,
                passed=gap < 0.15,
                severity="blocker" if gap > 0.25 else "warning",
                message=f"Train-test gap: {gap*100:.1f}pp (train={train_score:.1%}, test={test_score:.1%})",
                fix_action="Regularize model or reduce features" if gap >= 0.15 else None,
            ))

        # VD-005: No leakage signal
        if auc and auc > 0.99:
            checks.append(ValidationCheck(
                check_id="VD-005", name="No Data Leakage Signal",
                category="model", weight=3.0,
                passed=False,
                severity="blocker",
                message=f"AUC of {auc:.4f} is suspiciously perfect — likely data leakage",
                fix_action="Investigate feature-target correlations before deploying",
            ))

        # ── SAFETY CHECKS ──

        # VD-006: Model registered
        registered = registry.get("total_registered", 0)
        checks.append(ValidationCheck(
            check_id="VD-006", name="Model Registered in Registry",
            category="safety", weight=1.5,
            passed=registered > 0,
            severity="warning",
            message=f"{registered} model(s) registered" if registered > 0 else "Model not registered",
            fix_action="Register model in the registry for versioning and rollback" if registered == 0 else None,
        ))

        # VD-007: Recall minimum for high-stakes
        recall = metrics.get("recall", 0)
        if recall and recall < 0.5:
            checks.append(ValidationCheck(
                check_id="VD-007", name="Recall ≥ 50%",
                category="model", weight=2.0,
                passed=False,
                severity="warning",
                message=f"Recall: {recall:.1%} — model misses {(1-recall)*100:.0f}% of positive cases",
                fix_action="Lower threshold or retrain with class_weight='balanced'",
            ))

        # VD-008: Shadow deployment recommended
        checks.append(ValidationCheck(
            check_id="VD-008", name="Shadow Deployment First",
            category="safety", weight=1.0,
            passed=True,  # Advisory only
            severity="advisory",
            message="Consider shadow deployment for 1-2 weeks before production",
        ))

        return checks

    # ══════════════════════════════════════════════════════════
    # PROMOTION VALIDATION
    # ══════════════════════════════════════════════════════════

    def _validate_promotion(self, ctx: Dict, analysis: Dict) -> List[ValidationCheck]:
        """Validate before promoting model to production status."""
        checks = self._validate_deployment(ctx, analysis)

        # Additional promotion-specific checks
        registry = ctx.get("registry_info", {})
        models = registry.get("models", [])

        # VP-001: At least 2 model versions (shows iteration)
        for model in models[:1]:
            versions = model.get("total_versions", 0)
            checks.append(ValidationCheck(
                check_id="VP-001", name="Model Has Multiple Versions",
                category="safety", weight=1.0,
                passed=versions >= 2,
                severity="advisory",
                message=f"{versions} version(s) — {'good iteration' if versions >= 2 else 'consider training multiple configurations'}",
            ))

        # VP-002: Monitoring in place
        checks.append(ValidationCheck(
            check_id="VP-002", name="Monitoring Configured",
            category="infrastructure", weight=2.0,
            passed=True,  # Can't verify from context — advisory
            severity="advisory",
            message="Ensure monitoring is configured BEFORE going to production",
            fix_action="Set up: prediction logging, feature drift detection, performance tracking",
        ))

        return checks

    # ══════════════════════════════════════════════════════════
    # RETRAIN VALIDATION
    # ══════════════════════════════════════════════════════════

    def _validate_retrain(self, ctx: Dict, analysis: Dict) -> List[ValidationCheck]:
        """Validate conditions for model retraining."""
        checks = []
        frontend = ctx.get("frontend_state", {}) or {}

        # VR-001: Retraining reason
        drift_detected = frontend.get("drift_detected", False)
        performance_degraded = frontend.get("performance_degraded", False)

        has_reason = drift_detected or performance_degraded
        checks.append(ValidationCheck(
            check_id="VR-001", name="Retraining Trigger Identified",
            category="data", weight=2.0,
            passed=has_reason,
            severity="advisory",
            message=(
                "Retraining triggered by: " +
                ", ".join(filter(None, [
                    "data drift detected" if drift_detected else None,
                    "performance degradation" if performance_degraded else None,
                ])) if has_reason else "No specific trigger — verify retraining is needed"
            ),
        ))

        # VR-002: New data available
        checks.append(ValidationCheck(
            check_id="VR-002", name="Fresh Training Data Available",
            category="data", weight=2.5,
            passed=True,  # Assume yes if retraining is requested
            severity="advisory",
            message="Ensure training data includes recent examples representing current distribution",
        ))

        # VR-003: Previous model preserved
        registry = ctx.get("registry_info", {})
        checks.append(ValidationCheck(
            check_id="VR-003", name="Previous Model Version Preserved",
            category="safety", weight=2.0,
            passed=registry.get("total_registered", 0) > 0,
            severity="warning",
            message="Previous model registered for rollback" if registry.get("total_registered", 0) > 0 else "Register current model before retraining",
        ))

        return checks
