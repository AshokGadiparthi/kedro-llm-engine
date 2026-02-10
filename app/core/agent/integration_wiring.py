"""
Integration Wiring — Connect New Components to Existing Engine
================================================================
This module provides drop-in integration that wires the 5 new components
into the existing orchestrator WITHOUT modifying any existing files.

New Components:
  1. domain_profiles.py      → Domain-adaptive thresholds (Gap #1)
  2. feedback_tracker.py     → Closed-loop recommendation tracking (Gap #2 + #7)
  3. semantic_intent.py      → Embedding-based intent classification (Gap #3)
  4. statistical_tests.py    → Real statistical computations (Gap #4)
  5. domain_cost_matrices.py → Industry-specific cost models (Gap #6)

Plus: tests/test_engine.py   → Comprehensive test suite (Gap #5)

How to Integrate:
  1. Copy all new .py files into app/core/agent/
  2. Import and use EnhancedOrchestrator instead of AgentOrchestrator
  3. All existing behavior is preserved — enhancements are additive

Usage:
  from .integration_wiring import EnhancedOrchestrator
  orchestrator = EnhancedOrchestrator()  # drop-in replacement
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnhancedOrchestrator:
    """
    Drop-in replacement for AgentOrchestrator that adds:
      - Domain-aware thresholds
      - Semantic intent classification
      - Real statistical tests
      - Feedback loop tracking
      - Domain-specific business impact

    Wraps the existing orchestrator — all original behavior preserved.
    """

    def __init__(self, **kwargs):
        # Import existing orchestrator
        from .orchestrator import AgentOrchestrator

        # Initialize the original orchestrator (preserves ALL existing behavior)
        self._base = AgentOrchestrator(**kwargs)

        # Initialize new components
        self._init_domain_profiles()
        self._init_semantic_intent()
        self._init_statistical_tests()
        self._init_feedback_tracker()
        self._init_domain_costs()

    def _init_domain_profiles(self):
        try:
            from .domain_profiles import DomainProfileManager
            self._domain_manager = DomainProfileManager()
            logger.info("Domain profiles loaded")
        except ImportError:
            self._domain_manager = None
            logger.warning("domain_profiles not available")

    def _init_semantic_intent(self):
        try:
            from .semantic_intent import SemanticIntentClassifier
            from .qa_engine import INTENT_PATTERNS
            self._semantic_classifier = SemanticIntentClassifier(
                regex_patterns=INTENT_PATTERNS
            )
            logger.info("Semantic intent classifier loaded")
        except ImportError:
            self._semantic_classifier = None
            logger.warning("semantic_intent not available")

    def _init_statistical_tests(self):
        try:
            from .statistical_tests import StatisticalTestEngine
            self._stat_tests = StatisticalTestEngine()
            logger.info("Statistical test engine loaded")
        except ImportError:
            self._stat_tests = None
            logger.warning("statistical_tests not available")

    def _init_feedback_tracker(self):
        try:
            from .feedback_tracker import FeedbackTracker
            memory = getattr(self._base, 'memory', None)
            self._feedback = FeedbackTracker(memory_store=memory)
            logger.info("Feedback tracker loaded")
        except ImportError:
            self._feedback = None
            logger.warning("feedback_tracker not available")

    def _init_domain_costs(self):
        try:
            from .domain_cost_matrices import DomainCostMatrixManager
            self._cost_manager = DomainCostMatrixManager()
            logger.info("Domain cost matrices loaded")
        except ImportError:
            self._cost_manager = None
            logger.warning("domain_cost_matrices not available")

    # ──────────────────────────────────────────────────────────
    # ENHANCED: get_insights — now with domain-aware thresholds
    # ──────────────────────────────────────────────────────────

    def get_insights(
        self,
        user_id: str,
        screen: str,
        frontend_state: Optional[Dict] = None,
        extra: Optional[Dict] = None,
        user_domain: Optional[str] = None,
        user_threshold_overrides: Optional[Dict] = None,
    ):
        """
        Enhanced insight generation with domain awareness.
        Passes through to base orchestrator with enriched context.
        """
        # Enrich context with domain profile before passing to base
        if self._domain_manager:
            try:
                from .domain_profiles import inject_thresholds

                # Build minimal context for detection
                detection_ctx = {
                    "dataset_profile": (frontend_state or {}).get("dataset_profile", {}),
                    "feature_stats": (frontend_state or {}).get("feature_stats", {}),
                }
                profile = self._domain_manager.detect(
                    detection_ctx,
                    user_domain=user_domain,
                    user_overrides=user_threshold_overrides,
                )

                # Inject into extra so the context compiler passes it through
                extra = extra or {}
                extra["domain_profile"] = profile.to_dict()
                extra["domain_thresholds"] = profile.thresholds.to_dict()
            except Exception as e:
                logger.warning(f"Domain detection failed: {e}")

        # Call original orchestrator
        result = self._base.get_insights(
            user_id=user_id, screen=screen,
            frontend_state=frontend_state, extra=extra,
        )

        # Log recommendations for feedback tracking
        if self._feedback and result:
            self._track_insights(user_id, result)

        return result

    # ──────────────────────────────────────────────────────────
    # ENHANCED: ask — now with semantic intent classification
    # ──────────────────────────────────────────────────────────

    def ask(
        self,
        user_id: str,
        question: str,
        screen: str,
        frontend_state: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None,
    ):
        """
        Enhanced Q&A with semantic fallback for intent classification.
        """
        # Pre-classify intent semantically for richer routing
        semantic_intent = None
        if self._semantic_classifier:
            try:
                intent, confidence, method = self._semantic_classifier.classify(
                    question, screen=screen
                )
                semantic_intent = {
                    "intent": intent,
                    "confidence": confidence,
                    "method": method,
                }
                # Pass semantic hint to base orchestrator via extra
                extra = extra or {}
                extra["semantic_intent"] = semantic_intent
            except Exception as e:
                logger.warning(f"Semantic classification failed: {e}")

        # Call original orchestrator
        result = self._base.ask(
            user_id=user_id, question=question, screen=screen,
            frontend_state=frontend_state,
            conversation_history=conversation_history,
            extra=extra,
        )

        # Enrich with semantic metadata
        if semantic_intent and isinstance(result, dict):
            result["semantic_intent"] = semantic_intent

        return result

    # ──────────────────────────────────────────────────────────
    # NEW: run_statistical_tests
    # ──────────────────────────────────────────────────────────

    def run_statistical_tests(
        self,
        feature: str,
        ref_stats: Dict[str, Any],
        cur_stats: Dict[str, Any],
        **kwargs,
    ) -> Optional[Dict]:
        """Run real statistical drift tests on a feature."""
        if not self._stat_tests:
            return None
        try:
            suite = self._stat_tests.run_drift_suite(
                feature=feature, ref_stats=ref_stats, cur_stats=cur_stats,
                **kwargs,
            )
            return suite.to_dict()
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # NEW: get_business_impact — domain-aware
    # ──────────────────────────────────────────────────────────

    def get_domain_business_impact(
        self,
        precision: float,
        recall: float,
        domain: Optional[str] = None,
        use_case: Optional[str] = None,
        n_predictions: Optional[int] = None,
    ) -> Dict:
        """Get domain-calibrated business impact."""
        if not self._cost_manager:
            return {"error": "Domain cost matrices not available"}

        try:
            matrix = self._cost_manager.get(domain, use_case) if domain else None
            if not matrix:
                matrix = self._cost_manager.get_default_for_domain(domain or "general")

            return matrix.compute_impact(
                precision=precision, recall=recall,
                n_predictions=n_predictions,
            )
        except Exception as e:
            logger.warning(f"Business impact computation failed: {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    # NEW: Feedback loop endpoints
    # ──────────────────────────────────────────────────────────

    def record_recommendation_feedback(
        self, user_id: str, rec_id: str, action: str, details: Optional[Dict] = None
    ) -> bool:
        """Record whether user followed/ignored a recommendation."""
        if not self._feedback:
            return False
        return self._feedback.log_user_action(user_id, rec_id, action, details)

    def record_outcome(
        self, user_id: str, rec_id: str, metrics_after: Dict[str, float],
        metrics_before: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Record the outcome after user acted on a recommendation."""
        if not self._feedback:
            return None
        return self._feedback.log_outcome(user_id, rec_id, metrics_after, metrics_before)

    def auto_detect_outcomes(
        self, user_id: str, project_id: str, current_metrics: Dict[str, float]
    ) -> List[Dict]:
        """Auto-detect outcomes for pending recommendations."""
        if not self._feedback:
            return []
        return self._feedback.auto_detect_outcomes(user_id, project_id, current_metrics)

    def get_effectiveness_report(
        self, user_id: Optional[str] = None, category: Optional[str] = None
    ) -> Dict:
        """Get recommendation effectiveness report."""
        if not self._feedback:
            return {"error": "Feedback tracking not available"}
        return self._feedback.get_recommendation_effectiveness(category, user_id)

    def get_benchmarking_report(self, user_id: Optional[str] = None) -> Dict:
        """Get A/B benchmarking report."""
        if not self._feedback:
            return {"error": "Feedback tracking not available"}
        return self._feedback.get_benchmarking_report(user_id)

    # ──────────────────────────────────────────────────────────
    # PROXY: all other methods pass through to base
    # ──────────────────────────────────────────────────────────

    def __getattr__(self, name):
        """Proxy all other calls to the base orchestrator."""
        return getattr(self._base, name)

    # ──────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────

    def _track_insights(self, user_id: str, result):
        """Log generated insights as trackable recommendations."""
        try:
            insights = []
            if hasattr(result, 'insights'):
                insights = result.insights
            elif isinstance(result, dict):
                insights = result.get("insights", [])

            for insight in insights[:5]:  # Track top 5
                if isinstance(insight, dict):
                    rule_id = insight.get("rule_id", "")
                    action = insight.get("action", "")
                    category = insight.get("category", "general")
                else:
                    rule_id = getattr(insight, "rule_id", "")
                    action = getattr(insight, "action", "")
                    category = getattr(insight, "category", "general")

                if action:  # Only track actionable insights
                    self._feedback.log_recommendation(
                        user_id=user_id,
                        project_id="default",
                        category=category.lower().replace(" ", "_"),
                        rule_id=rule_id,
                        recommendation=action[:200],
                    )
        except Exception as e:
            logger.debug(f"Insight tracking failed: {e}")
