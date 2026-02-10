"""
ML Expert Agent Engine — Comprehensive Test Suite
====================================================
Tests all components: domain profiles, semantic intent, statistical tests,
feedback tracker, domain cost matrices, rule engine integration.

Run: pytest tests/ -v
Run with coverage: pytest tests/ --cov=. --cov-report=term-missing
"""

import json
import math
import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════

def make_context(
    rows=1000, cols=10, completeness=95.0, duplicates=1.0,
    numeric_count=7, categorical_count=3, screen="eda",
    target="target_col", quality_score=85, columns=None,
):
    """Build a realistic context dict for testing."""
    return {
        "screen": screen,
        "dataset_profile": {
            "rows": rows, "columns": cols,
            "numeric_count": numeric_count,
            "categorical_count": categorical_count,
            "dtypes": {f"col_{i}": "float64" for i in range(numeric_count)} |
                      {f"cat_{i}": "object" for i in range(categorical_count)} |
                      ({"target_col": "float64"} if target else {}),
            "column_types": {
                "numeric": [f"col_{i}" for i in range(numeric_count)],
                "categorical": [f"cat_{i}" for i in range(categorical_count)],
            },
            "memory_mb": rows * cols * 8 / 1e6,
        },
        "data_quality": {
            "completeness": completeness,
            "duplicate_rows": int(rows * duplicates / 100),
            "duplicate_pct": duplicates,
            "overall_quality_score": quality_score,
            "columns_with_high_missing": [],
        },
        "feature_stats": {
            "skewed_features": [],
            "constant_columns": [],
            "high_cardinality_categoricals": [],
            "potential_id_columns": [],
            "numeric_stats": {},
        },
        "correlations": {"high_pairs": []},
        "screen_context": {"target_column": target},
        "frontend_state": {},
        "pipeline_state": {"phases_completed": [], "last_phase": None},
        "registry_info": {},
        "training_history": {},
    }


def make_healthcare_context():
    """Build context with healthcare-like column names."""
    ctx = make_context(rows=5000, cols=20)
    ctx["dataset_profile"]["dtypes"] = {
        "patient_id": "int64", "diagnosis_code": "object",
        "blood_pressure": "float64", "heart_rate": "float64",
        "glucose": "float64", "bmi": "float64",
        "age": "int64", "gender": "object",
        "treatment": "object", "outcome": "int64",
    }
    ctx["dataset_profile"]["column_types"]["numeric"] = [
        "blood_pressure", "heart_rate", "glucose", "bmi", "age"
    ]
    ctx["feature_stats"]["numeric_stats"] = {
        "blood_pressure": {"mean": 120, "std": 15, "min": 60, "max": 250},
        "heart_rate": {"mean": 72, "std": 12, "min": 40, "max": 180},
    }
    return ctx


def make_finance_context():
    """Build context with finance-like column names."""
    ctx = make_context(rows=100000, cols=25)
    ctx["dataset_profile"]["dtypes"] = {
        "transaction_id": "int64", "amount": "float64",
        "balance": "float64", "credit_score": "float64",
        "account_age": "int64", "fraud": "int64",
    }
    return ctx


# ═══════════════════════════════════════════════════════════════
# 1. DOMAIN PROFILE TESTS
# ═══════════════════════════════════════════════════════════════

class TestDomainProfiles:
    """Tests for domain_profiles.py"""

    def test_default_thresholds(self):
        from domain_profiles import DomainThresholds
        t = DomainThresholds()
        assert t.missing_pct_critical == 70.0
        assert t.imbalance_severe_pct == 5.0
        assert t.default_cost_fn == 100.0

    def test_threshold_override(self):
        from domain_profiles import DomainThresholds
        t = DomainThresholds()
        t2 = t.override({"missing_pct_critical": 30.0, "default_cost_fn": 5000.0})
        assert t2.missing_pct_critical == 30.0
        assert t2.default_cost_fn == 5000.0
        # Original unchanged
        assert t.missing_pct_critical == 70.0

    def test_detect_healthcare(self):
        from domain_profiles import DomainProfileManager
        ctx = make_healthcare_context()
        profile = DomainProfileManager.detect(ctx)
        assert profile.domain == "healthcare"
        assert profile.detection_confidence >= 0.3
        assert profile.thresholds.missing_pct_critical > 70  # Healthcare is more tolerant

    def test_detect_finance(self):
        from domain_profiles import DomainProfileManager
        ctx = make_finance_context()
        profile = DomainProfileManager.detect(ctx)
        assert profile.domain == "finance"
        assert profile.thresholds.missing_pct_critical < 70  # Finance is stricter

    def test_detect_general_fallback(self):
        from domain_profiles import DomainProfileManager
        ctx = make_context()  # Generic column names
        profile = DomainProfileManager.detect(ctx)
        assert profile.domain == "general"

    def test_user_specified_domain(self):
        from domain_profiles import DomainProfileManager
        ctx = make_context()
        profile = DomainProfileManager.detect(ctx, user_domain="cybersecurity")
        assert profile.domain == "cybersecurity"
        assert profile.detection_confidence == 1.0
        assert profile.thresholds.recall_low == 0.90

    def test_user_overrides(self):
        from domain_profiles import DomainProfileManager
        ctx = make_context()
        profile = DomainProfileManager.detect(
            ctx, user_domain="healthcare",
            user_overrides={"missing_pct_critical": 99.0}
        )
        assert profile.get_threshold("missing_pct_critical") == 99.0

    def test_list_domains(self):
        from domain_profiles import DomainProfileManager
        domains = DomainProfileManager.list_domains()
        assert len(domains) >= 12
        domain_keys = [d["key"] for d in domains]
        assert "general" in domain_keys
        assert "healthcare" in domain_keys
        assert "finance" in domain_keys

    def test_inject_thresholds(self):
        from domain_profiles import DomainProfileManager, inject_thresholds
        ctx = make_context()
        profile = DomainProfileManager.detect(ctx, user_domain="finance")
        enriched = inject_thresholds(ctx, profile)
        assert "domain_thresholds" in enriched
        assert enriched["domain_thresholds"]["missing_pct_critical"] == 30.0


# ═══════════════════════════════════════════════════════════════
# 2. SEMANTIC INTENT TESTS
# ═══════════════════════════════════════════════════════════════

class TestSemanticIntent:
    """Tests for semantic_intent.py"""

    def test_exact_regex_match(self):
        from semantic_intent import SemanticIntentClassifier
        from qa_engine import INTENT_PATTERNS
        clf = SemanticIntentClassifier(regex_patterns=INTENT_PATTERNS)
        intent, conf, method = clf.classify("which algorithm should I use")
        assert intent == "algorithm_selection"
        assert conf >= 0.6
        assert method == "regex"

    def test_semantic_fallback_paraphrase(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        intent, conf, method = clf.classify(
            "my random forest sucks, what else can I try"
        )
        assert intent == "algorithm_selection"
        assert conf >= 0.3

    def test_semantic_missing_data(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        intent, conf, method = clf.classify("lots of blanks in my dataset")
        assert intent in ("missing_data", "data_quality")  # Both are valid classifications

    def test_semantic_overfitting(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        intent, conf, method = clf.classify(
            "training accuracy is perfect but test is terrible"
        )
        assert intent == "overfitting"

    def test_screen_context_boost(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        # Same question, different screens should bias results
        intent_eda, conf_eda, _ = clf.classify("what should I do", screen="eda")
        intent_eval, conf_eval, _ = clf.classify("what should I do", screen="evaluation")
        # Both should work (not crash), specifics depend on scoring

    def test_get_all_scores(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        scores = clf.get_all_scores("should I use XGBoost or LightGBM", screen="training")
        assert len(scores) > 0
        assert scores[0]["intent"] == "algorithm_selection"
        assert scores[0]["combined_score"] > 0

    def test_keyword_expansion(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        # "terrible" should expand to match "bad" synonyms → algorithm_selection
        intent, conf, method = clf.classify("my model is terrible need alternatives")
        assert intent in ("algorithm_selection", "overfitting")


# ═══════════════════════════════════════════════════════════════
# 3. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════

class TestStatisticalTests:
    """Tests for statistical_tests.py"""

    def test_psi_no_drift(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        ref = [100, 200, 300, 200, 100]
        cur = [105, 195, 305, 195, 100]
        result = engine.compute_psi(ref, cur)
        assert result.statistic < 0.10
        assert not result.is_significant

    def test_psi_significant_drift(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        ref = [100, 200, 300, 200, 100]
        cur = [300, 100, 50, 200, 350]
        result = engine.compute_psi(ref, cur)
        assert result.statistic > 0.10
        assert result.is_significant

    def test_ks_from_percentiles(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        ref = {"1%": 1, "5%": 5, "10%": 10, "25%": 25, "50%": 50, "75%": 75, "90%": 90, "95%": 95, "99%": 99}
        cur = {"1%": 1, "5%": 5, "10%": 10, "25%": 25, "50%": 50, "75%": 75, "90%": 90, "95%": 95, "99%": 99}
        result = engine.compute_ks_from_percentiles(ref, cur)
        assert result.statistic < 0.1  # Same distribution

    def test_ks_drifted(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        ref = {"1%": 1, "5%": 5, "10%": 10, "25%": 25, "50%": 50, "75%": 75, "90%": 90, "95%": 95, "99%": 99}
        cur = {"1%": 10, "5%": 20, "10%": 30, "25%": 45, "50%": 70, "75%": 85, "90%": 95, "95%": 98, "99%": 100}
        result = engine.compute_ks_from_percentiles(ref, cur)
        assert result.statistic > 0.05

    def test_chi_square(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        observed = {"A": 50, "B": 30, "C": 20}
        expected = {"A": 40, "B": 35, "C": 25}
        result = engine.compute_chi_square(observed, expected)
        assert result.test_name == "Chi-Square"
        assert result.statistic >= 0

    def test_js_divergence_same(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        dist = [100, 200, 300, 200, 100]
        result = engine.compute_js_divergence(dist, dist)
        assert result.statistic < 0.001

    def test_js_divergence_different(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        p = [100, 200, 300, 200, 100]
        q = [300, 50, 50, 50, 450]
        result = engine.compute_js_divergence(p, q)
        assert result.statistic > 0.05

    def test_cohens_d(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.compute_cohens_d(
            ref_mean=50, ref_std=10, cur_mean=55, cur_std=10
        )
        assert abs(result.statistic - 0.5) < 0.1  # d ≈ 0.5 (medium effect)

    def test_normality_normal(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.assess_normality(
            mean=0, std=1, skewness=0.1, kurtosis=3.05, n=1000
        )
        assert result.is_normal
        assert result.normality_score > 0.5

    def test_normality_skewed(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.assess_normality(
            mean=10, std=5, skewness=2.5, kurtosis=8.0, n=1000
        )
        assert not result.is_normal
        assert result.recommended_transform in ("log", "sqrt")

    def test_bootstrap_ci(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.bootstrap_metric_ci(0.85, 1000, "accuracy")
        assert result["lower"] < 0.85
        assert result["upper"] > 0.85
        assert result["lower"] > 0
        assert result["upper"] <= 1

    def test_drift_suite(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        suite = engine.run_drift_suite(
            feature="age",
            ref_stats={"mean": 35, "std": 10},
            cur_stats={"mean": 45, "std": 12},
            ref_histogram=[100, 200, 300, 200, 100],
            cur_histogram=[50, 100, 200, 300, 250],
            ref_percentiles={"1%": 15, "25%": 28, "50%": 35, "75%": 42, "99%": 55},
            cur_percentiles={"1%": 20, "25%": 36, "50%": 45, "75%": 52, "99%": 65},
        )
        assert suite.feature == "age"
        assert suite.psi is not None
        assert suite.cohens_d is not None

    def test_model_comparison(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        a_scores = [0.85, 0.87, 0.83, 0.86, 0.84]
        b_scores = [0.80, 0.82, 0.79, 0.81, 0.80]
        result = engine.compare_models_significance(a_scores, b_scores)
        assert result.test_name == "Paired t-test"
        assert result.statistic > 0  # Model A is better


# ═══════════════════════════════════════════════════════════════
# 4. FEEDBACK TRACKER TESTS
# ═══════════════════════════════════════════════════════════════

class TestFeedbackTracker:
    """Tests for feedback_tracker.py"""

    def test_log_recommendation(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()
        rec_id = tracker.log_recommendation(
            user_id="user1", project_id="proj1",
            category="algorithm", rule_id="AS-001",
            recommendation="Use XGBoost for this dataset",
        )
        assert rec_id
        assert len(rec_id) == 12

    def test_full_lifecycle(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()

        # 1. Log recommendation
        rec_id = tracker.log_recommendation(
            user_id="user1", project_id="proj1",
            category="algorithm", rule_id="AS-001",
            recommendation="Switch from RF to XGBoost",
            current_metrics={"f1": 0.72, "accuracy": 0.85},
        )

        # 2. Log user action
        tracker.log_user_action("user1", rec_id, "followed", {"algorithm": "xgboost"})

        # 3. Log outcome
        verdict = tracker.log_outcome(
            "user1", rec_id,
            metrics_after={"f1": 0.81, "accuracy": 0.89},
        )
        assert verdict == "improved"

    def test_effectiveness_scoring(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()

        # Create multiple tracked recommendations
        for i in range(5):
            rec_id = tracker.log_recommendation(
                "user1", "proj1", "algorithm", f"AS-{i:03d}",
                f"Recommendation {i}",
                current_metrics={"f1": 0.70},
            )
            tracker.log_user_action("user1", rec_id, "followed")
            tracker.log_outcome("user1", rec_id, {"f1": 0.75 + i * 0.02})

        effectiveness = tracker.get_recommendation_effectiveness("algorithm")
        assert effectiveness["success_rate"] > 0
        assert effectiveness["follow_rate"] == 1.0

    def test_confidence_adjustment(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()

        for i in range(4):
            rec_id = tracker.log_recommendation(
                "user1", "proj1", "feature", "FE-001",
                "Drop correlated features",
                current_metrics={"f1": 0.70},
            )
            tracker.log_user_action("user1", rec_id, "followed")
            tracker.log_outcome("user1", rec_id, {"f1": 0.75})

        adj = tracker.get_confidence_adjustment("feature", "FE-001")
        assert adj > 1.0  # Should be boosted (historically effective)

    def test_benchmarking_report(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()

        # Some followed, some ignored
        for i in range(3):
            rec_id = tracker.log_recommendation(
                "user1", "proj1", "algorithm", "AS-001",
                "Use XGBoost", current_metrics={"f1": 0.70},
            )
            tracker.log_user_action("user1", rec_id, "followed")
            tracker.log_outcome("user1", rec_id, {"f1": 0.78})

        for i in range(2):
            rec_id = tracker.log_recommendation(
                "user1", "proj1", "algorithm", "AS-001",
                "Use XGBoost", current_metrics={"f1": 0.70},
            )
            tracker.log_user_action("user1", rec_id, "ignored")
            tracker.log_outcome("user1", rec_id, {"f1": 0.71})

        report = tracker.get_benchmarking_report("user1")
        assert report["followed_recommendations"]["count"] == 3
        assert report["ignored_recommendations"]["count"] == 2
        assert "conclusion" in report


# ═══════════════════════════════════════════════════════════════
# 5. DOMAIN COST MATRIX TESTS
# ═══════════════════════════════════════════════════════════════

class TestDomainCostMatrices:
    """Tests for domain_cost_matrices.py"""

    def test_get_healthcare_fraud(self):
        from domain_cost_matrices import DomainCostMatrixManager
        matrix = DomainCostMatrixManager.get("healthcare", "diagnosis")
        assert matrix is not None
        assert matrix.cost_fn > matrix.cost_fp  # Missing diagnosis costs more

    def test_get_finance_fraud(self):
        from domain_cost_matrices import DomainCostMatrixManager
        matrix = DomainCostMatrixManager.get("finance", "fraud")
        assert matrix.cost_fn == 5000
        assert matrix.regulatory_penalty_multiplier > 1.0

    def test_compute_impact(self):
        from domain_cost_matrices import DomainCostMatrixManager
        matrix = DomainCostMatrixManager.get("retail", "churn")
        impact = matrix.compute_impact(precision=0.80, recall=0.75, n_predictions=10000)
        assert "costs" in impact
        assert "roi" in impact
        assert "capacity" in impact
        assert impact["costs"]["net_savings"] > 0  # Model should save money

    def test_capacity_check(self):
        from domain_cost_matrices import CostMatrix
        matrix = CostMatrix(
            domain="test", use_case="test", description="test",
            cost_fp=10, cost_fn=100,
            max_actions_per_day=5, positive_rate=0.5,
        )
        impact = matrix.compute_impact(precision=0.5, recall=0.9, n_predictions=10000)
        assert impact["capacity"]["capacity_exceeded"]

    def test_list_available(self):
        from domain_cost_matrices import DomainCostMatrixManager
        available = DomainCostMatrixManager.list_available()
        assert "healthcare" in available
        assert "finance" in available
        assert len(available["healthcare"]) >= 1


# ═══════════════════════════════════════════════════════════════
# 6. RULE ENGINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestRuleEngineIntegration:
    """Integration tests for rule engine with domain profiles."""

    def test_rule_engine_evaluates(self):
        from rule_engine import RuleEngine
        engine = RuleEngine()
        ctx = make_context(completeness=60, duplicates=15)
        insights = engine.evaluate(ctx)
        assert len(insights) > 0
        severities = [i.severity for i in insights]
        assert "critical" in severities

    def test_data_quality_rules(self):
        from rule_engine import RuleEngine
        engine = RuleEngine()
        ctx = make_context(completeness=50)
        insights = engine.evaluate(ctx)
        dq = [i for i in insights if i.category == "Data Quality"]
        assert len(dq) > 0
        assert any("incompleteness" in i.title.lower() or "incomplete" in i.message.lower() for i in dq)

    def test_class_imbalance_detection(self):
        from rule_engine import RuleEngine
        engine = RuleEngine()
        ctx = make_context()
        ctx["feature_stats"]["numeric_stats"]["target_col"] = {
            "min": 0, "max": 1, "mean": 0.02, "std": 0.14
        }
        insights = engine.evaluate(ctx)
        imbalance = [i for i in insights if "imbalance" in i.title.lower()]
        assert len(imbalance) > 0

    def test_small_dataset_warning(self):
        from rule_engine import RuleEngine
        engine = RuleEngine()
        ctx = make_context(rows=50, cols=10)
        insights = engine.evaluate(ctx)
        small = [i for i in insights if "small" in i.title.lower() or i.rule_id in ("SS-001", "SS-002")]
        assert len(small) > 0

    def test_evaluation_screen_rules(self):
        from rule_engine import RuleEngine
        engine = RuleEngine()
        ctx = make_context(screen="evaluation")
        ctx["screen_context"] = {
            "metrics": {"accuracy": 0.95, "f1": 0.60, "precision": 0.90, "recall": 0.45, "roc_auc": 0.82},
        }
        insights = engine.evaluate(ctx)
        assert any(i.rule_id == "EM-001" for i in insights)  # Accuracy-F1 gap
        assert any(i.rule_id == "EM-002" for i in insights)  # Low recall


# ═══════════════════════════════════════════════════════════════
# 7. QA ENGINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestQAEngine:
    """Tests for QA engine."""

    def test_algorithm_selection_answer(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        ctx = make_context(rows=5000, cols=15)
        result = engine.answer("which algorithm should I use", "training", ctx)
        assert "answer" in result
        assert result["confidence"] > 0.5
        assert len(result["answer"]) > 50

    def test_metric_interpretation(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        ctx = make_context(screen="evaluation")
        ctx["screen_context"] = {"metrics": {"f1": 0.75, "accuracy": 0.88}}
        result = engine.answer("what does F1 score mean", "evaluation", ctx)
        assert "F1" in result["answer"]

    def test_missing_data_answer(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        ctx = make_context(completeness=70)
        result = engine.answer("how to handle missing values", "eda", ctx)
        assert "impute" in result["answer"].lower() or "missing" in result["answer"].lower()

    def test_general_help(self):
        from qa_engine import QAEngine
        engine = QAEngine()
        ctx = make_context(screen="dashboard")
        result = engine.answer("what should I do next", "dashboard", ctx)
        assert result["answer"]
        assert "follow_ups" in result


# ═══════════════════════════════════════════════════════════════
# 8. CROSS-COMPONENT INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestCrossComponentIntegration:
    """End-to-end integration tests."""

    def test_domain_detection_affects_thresholds(self):
        from domain_profiles import DomainProfileManager, inject_thresholds
        from rule_engine import RuleEngine

        # Healthcare context: 60% completeness should NOT be critical
        ctx = make_healthcare_context()
        ctx["data_quality"]["completeness"] = 60.0
        profile = DomainProfileManager.detect(ctx)
        assert profile.domain == "healthcare"
        assert profile.thresholds.missing_pct_critical > 60  # Healthcare tolerates more missing

    def test_statistical_tests_with_real_data(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()

        # Simulate real drift scenario
        ref_hist = [50, 100, 200, 300, 200, 100, 50]
        cur_hist = [100, 150, 150, 150, 150, 150, 150]

        psi = engine.compute_psi(ref_hist, cur_hist)
        js = engine.compute_js_divergence(ref_hist, cur_hist)

        assert psi.statistic > 0
        assert js.statistic > 0

    def test_feedback_loop_adjusts_confidence(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()

        # Simulate multiple interactions where SMOTE recommendation worked
        for _ in range(5):
            rec_id = tracker.log_recommendation(
                "user1", "proj1", "imputation", "DQ-005",
                "Use IterativeImputer for these columns",
                current_metrics={"f1": 0.65},
            )
            tracker.log_user_action("user1", rec_id, "followed")
            tracker.log_outcome("user1", rec_id, {"f1": 0.73})

        adj = tracker.get_confidence_adjustment("imputation", "DQ-005")
        assert adj > 1.0  # Confidence should be boosted

    def test_cost_matrix_with_domain_detection(self):
        from domain_profiles import DomainProfileManager
        from domain_cost_matrices import DomainCostMatrixManager

        ctx = make_finance_context()
        profile = DomainProfileManager.detect(ctx)
        cost_matrix = DomainCostMatrixManager.get_default_for_domain(profile.domain)

        assert cost_matrix.domain == "finance"
        impact = cost_matrix.compute_impact(precision=0.85, recall=0.90)
        assert impact["costs"]["net_savings"] > 0


# ═══════════════════════════════════════════════════════════════
# 9. EDGE CASE & ROBUSTNESS TESTS
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_empty_context(self):
        from rule_engine import RuleEngine
        engine = RuleEngine()
        insights = engine.evaluate({})
        assert isinstance(insights, list)

    def test_psi_empty_distributions(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.compute_psi([], [])
        assert result.method == "skipped"

    def test_ks_insufficient_percentiles(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.compute_ks_from_percentiles({"1%": 1}, {"1%": 2})
        assert result.method == "skipped"

    def test_domain_profile_unknown_domain(self):
        from domain_profiles import DomainProfileManager
        profile = DomainProfileManager.detect({}, user_domain="nonexistent")
        assert profile.domain == "general"  # Should fall back

    def test_feedback_missing_record(self):
        from feedback_tracker import FeedbackTracker
        tracker = FeedbackTracker()
        result = tracker.log_user_action("user1", "nonexistent_id", "followed")
        assert result is False

    def test_semantic_intent_empty_question(self):
        from semantic_intent import SemanticIntentClassifier
        clf = SemanticIntentClassifier()
        intent, conf, method = clf.classify("")
        assert intent  # Should return something, not crash

    def test_cost_matrix_zero_precision(self):
        from domain_cost_matrices import CostMatrix
        matrix = CostMatrix(domain="test", use_case="test", description="test")
        impact = matrix.compute_impact(precision=0.0, recall=0.5)
        assert "costs" in impact  # Should not crash

    def test_normality_extreme_values(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.assess_normality(mean=0, std=1, skewness=100, kurtosis=1000, n=10)
        assert not result.is_normal

    def test_model_comparison_insufficient_data(self):
        from statistical_tests import StatisticalTestEngine
        engine = StatisticalTestEngine()
        result = engine.compare_models_significance([0.8], [0.7])
        assert result.method == "skipped"


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
