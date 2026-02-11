"""
Agent Orchestrator — The Central Brain
=======================================
Wires ALL engine components into a single coherent pipeline.
This is the ONLY entry point the API should use.

Pipeline per request type:

  INSIGHTS:
    Context Compiler → Statistical Analyzer → Pattern Detector
    → Rule Engine → Recommendation Engine → Business Impact
    → LLM Reasoner (optional) → Memory Store → Response

  ASK (question):
    Context Compiler → Statistical Analyzer → Rule Engine
    → Q&A Engine (deterministic) → LLM Reasoner (enhancement)
    → Memory Store → Response

  VALIDATE (pre-action):
    Context Compiler → Statistical Analyzer → Validation Engine
    → Rule Engine (supporting evidence) → Response

  RECOMMEND:
    Context Compiler → Statistical Analyzer → Pattern Detector
    → Recommendation Engine → Business Impact → Response

Design:
  - Single entry point per request type
  - Composable: each component feeds the next
  - Graceful degradation: if any component fails, the rest still work
  - Caching: expensive computations cached per (dataset_id, screen) key
  - Timing: each stage is timed for observability
"""

import logging
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .context_compiler import ContextCompiler
from .rule_engine import RuleEngine, Insight
from .statistical_analyzer import StatisticalAnalyzer
from .pattern_detector import PatternDetector
from .recommendation_engine import RecommendationEngine
from .business_impact import BusinessImpactCalculator
from .validation_engine import ValidationEngine
from .knowledge_base import (
    get_screen_knowledge, recommend_algorithms as kb_recommend_algorithms,
    get_algorithm_profile, get_metric_guide, detect_pitfall,
)
from .qa_engine import QAEngine
from .llm_reasoner import LLMReasoner
from .memory_store import MemoryStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# RESPONSE TYPES
# ═══════════════════════════════════════════════════════════════

class InsightBundle:
    """Complete insight response with all layers."""
    def __init__(self):
        self.insights: List[Dict] = []
        self.patterns: List[Dict] = []
        self.recommendations: Dict = {}
        self.business_impact: Dict = {}
        self.advice: Optional[str] = None
        self.source: str = "rules_only"
        self.context_summary: Dict = {}
        self.timing: Dict = {}
        self.memory_applied: List[Dict] = []
        self.critical_count: int = 0
        self.warning_count: int = 0
        self.suggested_questions: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insights": self.insights,
            "patterns": self.patterns,
            "recommendations": self.recommendations,
            "business_impact": self.business_impact,
            "advice": self.advice,
            "source": self.source,
            "context_summary": self.context_summary,
            "timing": self.timing,
            "memory_applied": self.memory_applied,
            "suggested_questions": self.suggested_questions,
            "counts": {
                "critical": self.critical_count,
                "warning": self.warning_count,
                "total": len(self.insights),
                "patterns": len(self.patterns),
            },
        }


class AskBundle:
    """Complete answer response."""
    def __init__(self):
        self.answer: str = ""
        self.supporting_insights: List[Dict] = []
        self.related_recommendations: Dict = {}
        self.source: str = "qa_engine"
        self.confidence: float = 0.0
        self.follow_up_questions: List[str] = []
        self.timing: Dict = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "supporting_insights": self.supporting_insights,
            "related_recommendations": self.related_recommendations,
            "source": self.source,
            "confidence": self.confidence,
            "follow_up_questions": self.follow_up_questions,
            "timing": self.timing,
        }


class ValidateBundle:
    """Complete validation response."""
    def __init__(self):
        self.can_proceed: bool = True
        self.verdict: str = "ready"
        self.blockers: List[Dict] = []
        self.warnings: List[Dict] = []
        self.checks_passed: List[Dict] = []
        self.recommendations: List[str] = []
        self.risk_score: float = 0.0
        self.timing: Dict = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_proceed": self.can_proceed,
            "verdict": self.verdict,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "checks_passed": self.checks_passed,
            "recommendations": self.recommendations,
            "risk_score": self.risk_score,
            "timing": self.timing,
        }


# ═══════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════

class _AnalysisCache:
    """In-memory LRU cache for expensive computations."""

    def __init__(self, max_size: int = 50, ttl_seconds: int = 300):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)

    def _make_key(self, *args) -> str:
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, *key_parts) -> Optional[Any]:
        key = self._make_key(*key_parts)
        if key in self._cache:
            value, ts = self._cache[key]
            if datetime.utcnow() - ts < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, value: Any, *key_parts):
        key = self._make_key(*key_parts)
        if len(self._cache) >= self._max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[key] = (value, datetime.utcnow())

    def invalidate(self, *key_parts):
        key = self._make_key(*key_parts)
        self._cache.pop(key, None)

    def clear(self):
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Central brain. Single entry point for all agent operations.
    Wires Context → Analysis → Detection → Rules → Recommendations → Response.
    """

    def __init__(self, db_session, memory_store: Optional[MemoryStore] = None):
        self.db = db_session

        # Core components
        self.context_compiler = ContextCompiler(db_session)
        self.rule_engine = RuleEngine()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.pattern_detector = PatternDetector()
        self.recommendation_engine = RecommendationEngine()
        self.business_impact = BusinessImpactCalculator()
        self.validation_engine = ValidationEngine()
        self.qa_engine = QAEngine()
        self.llm_reasoner = LLMReasoner()
        self.memory = memory_store

        # Cache
        self._cache = _AnalysisCache(max_size=50, ttl_seconds=300)
        self._ai_cache = _AnalysisCache(max_size=30, ttl_seconds=1800)  # 30 min TTL for AI responses

    # ──────────────────────────────────────────────────────────
    # 1. GET INSIGHTS (main pipeline)
    # ──────────────────────────────────────────────────────────

    async def get_insights(
            self,
            screen: str,
            project_id: Optional[str] = None,
            dataset_id: Optional[str] = None,
            model_id: Optional[str] = None,
            user_id: Optional[str] = None,
            extra: Optional[Dict] = None,
            use_llm: str = "auto",
    ) -> InsightBundle:
        """
        Full insight pipeline:
        Context → Analysis → Patterns → Rules → Recommendations → Impact → LLM → Memory
        """
        bundle = InsightBundle()
        timings = {}

        try:
            # ── Stage 1: Compile Context ──
            t0 = time.time()
            context = await self.context_compiler.compile(
                screen=screen, project_id=project_id,
                dataset_id=dataset_id, model_id=model_id, extra=extra,
            )
            timings["context_compile"] = round(time.time() - t0, 3)

            # ── Stage 2: Statistical Analysis (cached) ──
            t0 = time.time()
            analysis = self._get_cached_analysis(context, dataset_id)
            timings["statistical_analysis"] = round(time.time() - t0, 3)

            # ── Stage 3: Pattern Detection (cached) ──
            t0 = time.time()
            patterns = self._get_cached_patterns(context, dataset_id)
            bundle.patterns = [p.to_dict() for p in patterns]
            timings["pattern_detection"] = round(time.time() - t0, 3)

            # ── Stage 4: Rule Engine ──
            t0 = time.time()
            raw_insights = self.rule_engine.evaluate(context)

            # Merge pattern detector findings as additional insights
            for pattern in patterns:
                raw_insights.append(Insight(
                    severity=pattern.severity,
                    category=f"Pattern: {pattern.pattern_type.replace('_', ' ').title()}",
                    title=pattern.title,
                    message=pattern.explanation,
                    action=pattern.recommended_action,
                    evidence="; ".join(pattern.evidence[:3]),
                    confidence=pattern.confidence,
                    tags=[pattern.pattern_type, "pattern_detector"],
                    rule_id=f"PD-{pattern.pattern_type[:3].upper()}",
                ))

            # Apply memory-based adjustments
            if self.memory and user_id:
                raw_insights = self._apply_memory_adjustments(raw_insights, user_id, bundle)

            # Re-sort after merging
            order = {"critical": 0, "warning": 1, "info": 2, "tip": 3, "success": 4}
            raw_insights.sort(key=lambda x: (order.get(x.severity, 99), -(x.confidence or 0)))

            # ── Deduplicate: same rule_id should appear only once ──
            seen_rules = set()
            deduped = []
            for insight in raw_insights:
                key = insight.rule_id or ""
                if key and key in seen_rules:
                    logger.debug(f"Dedup: skipping duplicate insight {key}")
                    continue
                if key:
                    seen_rules.add(key)
                deduped.append(insight)
            raw_insights = deduped

            bundle.insights = [i.to_dict() for i in raw_insights]
            bundle.critical_count = sum(1 for i in raw_insights if i.severity == "critical")
            bundle.warning_count = sum(1 for i in raw_insights if i.severity == "warning")
            timings["rule_engine"] = round(time.time() - t0, 3)

            # ── Stage 5: Recommendations (screen-appropriate) ──
            t0 = time.time()
            bundle.recommendations = self._get_screen_recommendations(
                screen, context, analysis
            )
            timings["recommendations"] = round(time.time() - t0, 3)

            # ── Stage 6: Business Impact (if evaluation/deployment screen) ──
            t0 = time.time()
            if screen in ("evaluation", "deployment", "monitoring"):
                bundle.business_impact = self._compute_business_impact(context, extra)
            timings["business_impact"] = round(time.time() - t0, 3)

            # ── Stage 7: LLM Synthesis (smart gating — saves API cost) ──
            t0 = time.time()
            should_call_llm = self._should_call_llm(
                use_llm=use_llm,
                critical_count=bundle.critical_count,
                warning_count=bundle.warning_count,
                total_insights=len(raw_insights),
                screen=screen,
            )

            if should_call_llm:
                try:
                    llm_result = await self.llm_reasoner.reason(
                        context, bundle.insights[:10]
                    )
                    bundle.advice = llm_result.advice if hasattr(llm_result, 'advice') else (llm_result.get("advice") if isinstance(llm_result, dict) else None)
                    bundle.source = getattr(llm_result, 'source', 'rules_only') if not isinstance(llm_result, dict) else llm_result.get("source", "rules_only")
                except Exception as e:
                    logger.warning(f"LLM reasoning failed (non-critical): {e}")
                    bundle.advice = self._generate_rules_synthesis(raw_insights, context)
                    bundle.source = "rules_only"
            else:
                bundle.advice = self._generate_rules_synthesis(raw_insights, context)
                bundle.source = "rules_only"
                logger.info(f"LLM skipped (use_llm={use_llm}, critical={bundle.critical_count}, "
                            f"warnings={bundle.warning_count}) — saved ~$0.02")
            timings["llm_reasoning"] = round(time.time() - t0, 3)

            # ── Build context summary ──
            bundle.context_summary = self._build_context_summary(context, analysis)
            bundle.timing = timings

            # ── Generate dynamic suggested questions ──
            bundle.suggested_questions = self._generate_suggested_questions(
                raw_insights, context, screen
            )

        except Exception as e:
            logger.error(f"Orchestrator.get_insights failed: {e}", exc_info=True)
            bundle.timing = timings

        return bundle

    # ──────────────────────────────────────────────────────────
    # 2. ASK EXPERT (question answering)
    # ──────────────────────────────────────────────────────────

    async def ask(
            self,
            screen: str,
            question: str,
            project_id: Optional[str] = None,
            dataset_id: Optional[str] = None,
            model_id: Optional[str] = None,
            user_id: Optional[str] = None,
            extra: Optional[Dict] = None,
            conversation_history: Optional[List[Dict]] = None,
            use_llm: str = "auto",
    ) -> AskBundle:
        """
        Question answering pipeline:
        Context → Analysis → Rules → Q&A Engine → LLM Enhancement → Memory
        """
        bundle = AskBundle()
        timings = {}

        try:
            # ── Stage 1: Compile Context ──
            t0 = time.time()
            context = await self.context_compiler.compile(
                screen=screen, project_id=project_id,
                dataset_id=dataset_id, model_id=model_id, extra=extra,
            )
            timings["context_compile"] = round(time.time() - t0, 3)

            # ── Stage 2: Statistical Analysis ──
            t0 = time.time()
            analysis = self._get_cached_analysis(context, dataset_id)
            timings["statistical_analysis"] = round(time.time() - t0, 3)

            # ── Stage 3: Rule Engine (for supporting evidence) ──
            t0 = time.time()
            raw_insights = self.rule_engine.evaluate(context)
            bundle.supporting_insights = [
                i.to_dict() for i in raw_insights[:5]
            ]
            timings["rule_engine"] = round(time.time() - t0, 3)

            # ── Stage 4: Deterministic Q&A ──
            t0 = time.time()
            qa_result = self.qa_engine.answer(
                question=question,
                screen=screen,
                context=context,
                analysis=analysis,
                insights=[i.to_dict() for i in raw_insights],
                conversation_history=conversation_history,
            )
            bundle.answer = qa_result["answer"]
            bundle.confidence = qa_result["confidence"]
            bundle.follow_up_questions = qa_result.get("follow_ups", [])
            bundle.source = "qa_engine"
            timings["qa_engine"] = round(time.time() - t0, 3)

            # ── Stage 5: LLM Enhancement (smart gating — saves cost) ──
            t0 = time.time()
            should_use = self._should_call_llm_for_ask(
                use_llm=use_llm,
                qa_confidence=qa_result["confidence"],
            )
            if should_use:
                # Check AI cache first (same dataset + question = same answer)
                ai_cache_key = None
                if dataset_id:
                    q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
                    ai_cache_key = ("ai_ask", dataset_id, q_hash)
                    cached_ai = self._ai_cache.get(ai_cache_key)
                    if cached_ai:
                        bundle.answer = cached_ai["answer"]
                        bundle.source = cached_ai["source"] + ":cached"
                        logger.info(f"AI cache hit for {ai_cache_key} — saved ~$0.02")
                        should_use = False  # skip LLM call

                if should_use:
                    try:
                        llm_result = await self.llm_reasoner.reason(
                            context,
                            [i.to_dict() for i in raw_insights[:8]],
                            question=question,
                        )
                        has_llm = getattr(llm_result, 'has_llm', False) if not isinstance(llm_result, dict) else llm_result.get("has_llm", False)
                        if has_llm:
                            answer = getattr(llm_result, 'advice', bundle.answer) if not isinstance(llm_result, dict) else llm_result.get("advice", bundle.answer)
                            src = getattr(llm_result, 'source', '') if not isinstance(llm_result, dict) else llm_result.get("source", "")
                            bundle.answer = answer
                            bundle.source = f"llm_enhanced:{src}"

                            # Store in AI cache
                            if ai_cache_key:
                                self._ai_cache.set(
                                    {"answer": bundle.answer, "source": bundle.source},
                                    ai_cache_key,
                                )
                    except Exception as e:
                        logger.warning(f"LLM enhancement failed: {e}")
            else:
                logger.info(f"LLM skipped for /ask (use_llm={use_llm}, confidence={qa_result['confidence']:.2f}) — saved ~$0.01")
            timings["llm_enhancement"] = round(time.time() - t0, 3)

            # ── Stage 6: Related Recommendations ──
            t0 = time.time()
            bundle.related_recommendations = self._get_question_recommendations(
                question, screen, context, analysis
            )
            timings["recommendations"] = round(time.time() - t0, 3)

            # ── Store in memory ──
            if self.memory and user_id:
                self.memory.remember(
                    user_id=user_id,
                    memory_type="question",
                    key=f"q:{screen}:{question[:80]}",
                    value={"question": question, "screen": screen,
                           "timestamp": datetime.utcnow().isoformat()},
                )

            bundle.timing = timings

        except Exception as e:
            logger.error(f"Orchestrator.ask failed: {e}", exc_info=True)
            bundle.answer = "I encountered an error processing your question. Please try again."
            bundle.source = "error"
            bundle.timing = timings

        return bundle

    # ──────────────────────────────────────────────────────────
    # 3. VALIDATE ACTION (pre-flight checks)
    # ──────────────────────────────────────────────────────────

    async def validate(
            self,
            action: str,
            screen: str,
            project_id: Optional[str] = None,
            dataset_id: Optional[str] = None,
            model_id: Optional[str] = None,
            extra: Optional[Dict] = None,
    ) -> ValidateBundle:
        """
        Validation pipeline:
        Context → Analysis → Validation Engine → Rule Engine (evidence) → Response
        """
        bundle = ValidateBundle()
        timings = {}

        try:
            # ── Stage 1: Compile Context ──
            t0 = time.time()
            context = await self.context_compiler.compile(
                screen=screen, project_id=project_id,
                dataset_id=dataset_id, model_id=model_id, extra=extra,
            )
            timings["context_compile"] = round(time.time() - t0, 3)

            # ── Stage 2: Analysis ──
            t0 = time.time()
            analysis = self._get_cached_analysis(context, dataset_id)
            timings["analysis"] = round(time.time() - t0, 3)

            # ── Stage 3: Validation Engine ──
            t0 = time.time()
            validation_result = self.validation_engine.validate(
                action=action, context=context, analysis=analysis,
            )

            bundle.can_proceed = validation_result.can_proceed
            bundle.verdict = "blocked" if not validation_result.can_proceed else "ready"
            bundle.risk_score = round(100 - validation_result.readiness_score, 1)
            bundle.blockers = [c.to_dict() for c in validation_result.checks if not c.passed and c.severity == "blocker"]
            bundle.warnings = [c.to_dict() for c in validation_result.checks if not c.passed and c.severity == "warning"]
            bundle.checks_passed = [c.to_dict() for c in validation_result.checks if c.passed]
            timings["validation_engine"] = round(time.time() - t0, 3)

            # ── Stage 4: Extract actionable recommendations ──
            t0 = time.time()
            bundle.recommendations = []
            for check in validation_result.checks:
                if not check.passed and check.fix_action:
                    bundle.recommendations.append(check.fix_action)

            # Supplement with rule engine criticals
            raw_insights = self.rule_engine.evaluate(context)
            for insight in raw_insights:
                if insight.severity == "critical" and insight.action:
                    if insight.action not in bundle.recommendations:
                        bundle.recommendations.append(insight.action)
            bundle.recommendations = bundle.recommendations[:8]
            timings["recommendations"] = round(time.time() - t0, 3)

            bundle.timing = timings

        except Exception as e:
            logger.error(f"Orchestrator.validate failed: {e}", exc_info=True)
            bundle.can_proceed = True  # fail open
            bundle.verdict = "error"
            bundle.timing = timings

        return bundle

    # ──────────────────────────────────────────────────────────
    # 4. GET RECOMMENDATIONS (standalone)
    # ──────────────────────────────────────────────────────────

    async def get_recommendations(
            self,
            screen: str,
            project_id: Optional[str] = None,
            dataset_id: Optional[str] = None,
            model_id: Optional[str] = None,
            extra: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Full recommendation pipeline:
        Context → Analysis → Patterns → Recommendations → Impact
        """
        timings = {}

        try:
            t0 = time.time()
            context = await self.context_compiler.compile(
                screen=screen, project_id=project_id,
                dataset_id=dataset_id, model_id=model_id, extra=extra,
            )
            timings["context"] = round(time.time() - t0, 3)

            t0 = time.time()
            analysis = self._get_cached_analysis(context, dataset_id)
            timings["analysis"] = round(time.time() - t0, 3)

            t0 = time.time()
            recs = self.recommendation_engine.full_recommendations(context, analysis)
            timings["recommendations"] = round(time.time() - t0, 3)

            t0 = time.time()
            if screen in ("evaluation", "deployment", "monitoring"):
                recs["business_impact"] = self._compute_business_impact(context, extra)
            timings["business_impact"] = round(time.time() - t0, 3)

            recs["timing"] = timings
            return recs

        except Exception as e:
            logger.error(f"Orchestrator.get_recommendations failed: {e}", exc_info=True)
            return {"error": str(e), "timing": timings}

    # ──────────────────────────────────────────────────────────
    # 5. DATA READINESS REPORT
    # ──────────────────────────────────────────────────────────

    async def get_data_readiness(
            self,
            dataset_id: str,
            project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive data readiness assessment.
        Context → Full Analysis → Patterns → Readiness Score
        """
        try:
            context = await self.context_compiler.compile(
                screen="eda", project_id=project_id, dataset_id=dataset_id,
            )
            analysis = self._get_cached_analysis(context, dataset_id)
            patterns = self._get_cached_patterns(context, dataset_id)
            readiness = analysis.get("data_readiness", {})

            # Enrich with pattern findings
            pattern_issues = [p for p in patterns if p.severity in ("critical", "warning")]
            if readiness and pattern_issues:
                readiness["additional_concerns"] = [
                    {"type": p.pattern_type, "title": p.title, "severity": p.severity}
                    for p in pattern_issues
                ]

            return readiness

        except Exception as e:
            logger.error(f"Data readiness check failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ══════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    def _get_cached_analysis(self, context: Dict, dataset_id: Optional[str]) -> Dict:
        """Run statistical analysis with caching."""
        cache_key = ("analysis", dataset_id, context.get("screen"))
        cached = self._cache.get(cache_key) if dataset_id else None
        if cached:
            return cached

        analysis = self.statistical_analyzer.full_analysis(context)

        if dataset_id:
            self._cache.set(analysis, cache_key)
        return analysis

    def _get_cached_patterns(self, context: Dict, dataset_id: Optional[str]) -> List:
        """Run pattern detection with caching."""
        cache_key = ("patterns", dataset_id)
        cached = self._cache.get(cache_key) if dataset_id else None
        if cached:
            return cached

        patterns = self.pattern_detector.detect_all(context)

        if dataset_id:
            self._cache.set(patterns, cache_key)
        return patterns

    def _apply_memory_adjustments(
            self, insights: List[Insight], user_id: str, bundle: InsightBundle
    ) -> List[Insight]:
        """Adjust insight priority based on user's past feedback."""
        if not self.memory:
            return insights

        try:
            feedbacks = self.memory.recall(user_id, memory_type="feedback")
            if not feedbacks:
                return insights

            feedback_map = {}
            for fb in feedbacks:
                key = fb.get("key", "")
                val = fb.get("value", {})
                if isinstance(val, dict):
                    feedback_map[key] = val.get("helpful", True)

            adjusted = []
            for insight in insights:
                rule_id = insight.rule_id or ""
                fb_key = f"feedback:{rule_id}"
                if fb_key in feedback_map:
                    if not feedback_map[fb_key]:
                        # User marked unhelpful — deprioritize (lower confidence)
                        insight.confidence = max(0.1, (insight.confidence or 1.0) - 0.3)
                        bundle.memory_applied.append({
                            "rule_id": rule_id,
                            "adjustment": "deprioritized",
                            "reason": "previously marked unhelpful",
                        })
                    else:
                        # User liked it — boost confidence slightly
                        insight.confidence = min(1.0, (insight.confidence or 0.5) + 0.1)
                        bundle.memory_applied.append({
                            "rule_id": rule_id,
                            "adjustment": "boosted",
                            "reason": "previously marked helpful",
                        })
                adjusted.append(insight)

            return adjusted

        except Exception as e:
            logger.warning(f"Memory adjustment failed: {e}")
            return insights

    def _get_screen_recommendations(
            self, screen: str, context: Dict, analysis: Dict
    ) -> Dict:
        """Get screen-appropriate recommendations."""
        try:
            recs = {}

            if screen in ("eda", "data"):
                recs["feature_selection"] = self.recommendation_engine.recommend_feature_selection(context, analysis)
                recs["imputation"] = self.recommendation_engine.recommend_imputation(context, analysis)

            elif screen in ("mlflow", "training"):
                recs["algorithms"] = self.recommendation_engine.recommend_algorithms(context, analysis)
                recs["encoding"] = self.recommendation_engine.recommend_encoding(context, analysis)
                recs["scaling"] = self.recommendation_engine.recommend_scaling(context, analysis)
                recs["cv_strategy"] = self.recommendation_engine.recommend_cv_strategy(context, analysis)

            elif screen == "evaluation":
                recs["threshold"] = self.recommendation_engine.recommend_threshold(context)
                recs["ensemble"] = self.recommendation_engine.recommend_ensemble(context)

            elif screen in ("deployment", "registry"):
                recs["ensemble"] = self.recommendation_engine.recommend_ensemble(context)

            # Always add knowledge base context
            kb = get_screen_knowledge(screen)
            if kb:
                recs["knowledge_base"] = kb

            return recs

        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return {}

    def _get_question_recommendations(
            self, question: str, screen: str, context: Dict, analysis: Dict
    ) -> Dict:
        """Extract recommendations relevant to a user's question."""
        q_lower = question.lower()
        recs = {}

        try:
            if any(w in q_lower for w in ["algorithm", "model", "which", "best", "recommend"]):
                recs["algorithms"] = self.recommendation_engine.recommend_algorithms(context, analysis)

            if any(w in q_lower for w in ["feature", "select", "drop", "remove", "important"]):
                recs["feature_selection"] = self.recommendation_engine.recommend_feature_selection(context, analysis)

            if any(w in q_lower for w in ["threshold", "cutoff", "precision", "recall"]):
                recs["threshold"] = self.recommendation_engine.recommend_threshold(context)

            if any(w in q_lower for w in ["hyperparameter", "tune", "grid", "search"]):
                algo = context.get("screen_context", {}).get("algorithm", "xgboost")
                recs["hyperparameters"] = self.recommendation_engine.recommend_hyperparameter_space(algo, context)

            if any(w in q_lower for w in ["encode", "encoding", "categorical", "one-hot"]):
                recs["encoding"] = self.recommendation_engine.recommend_encoding(context, analysis)

            if any(w in q_lower for w in ["scale", "scaling", "normalize", "standardize"]):
                recs["scaling"] = self.recommendation_engine.recommend_scaling(context, analysis)

            if any(w in q_lower for w in ["missing", "impute", "null", "nan"]):
                recs["imputation"] = self.recommendation_engine.recommend_imputation(context, analysis)

            if any(w in q_lower for w in ["ensemble", "stack", "blend", "combine"]):
                recs["ensemble"] = self.recommendation_engine.recommend_ensemble(context)

            if any(w in q_lower for w in ["cv", "cross-validation", "fold", "validation"]):
                recs["cv_strategy"] = self.recommendation_engine.recommend_cv_strategy(context, analysis)

        except Exception as e:
            logger.warning(f"Question recommendation extraction failed: {e}")

        return recs

    def _compute_business_impact(self, context: Dict, extra: Optional[Dict]) -> Dict:
        """Compute business impact metrics from evaluation data."""
        try:
            metrics = {}
            if extra:
                metrics = extra.get("metrics", {})

            if not metrics:
                screen_ctx = context.get("screen_context", {}) or {}
                metrics = screen_ctx.get("metrics", {})

            if not metrics:
                return {}

            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            profile = context.get("dataset_profile", {})
            rows = profile.get("rows", 0)

            # Cost matrix analysis
            cost_result = self.business_impact.cost_matrix_analysis(
                precision=precision, recall=recall,
                total_samples=rows,
                cost_fp=50, cost_fn=200,
            )

            # ROI estimate
            roi_result = self.business_impact.calculate_roi(
                precision=precision, recall=recall,
                total_predictions_per_month=rows * 2,
                cost_per_prediction=0.01,
                value_per_correct=10.0,
                cost_per_error=50.0,
            )

            return {
                "cost_matrix": cost_result,
                "roi": roi_result,
            }

        except Exception as e:
            logger.warning(f"Business impact computation failed: {e}")
            return {}

    def _build_context_summary(self, context: Dict, analysis: Dict) -> Dict:
        """Build a UI-friendly context summary."""
        profile = context.get("dataset_profile", {})
        quality = context.get("data_quality", {})
        pipeline = context.get("pipeline_state", {})
        screen = context.get("screen", "")

        readiness = analysis.get("data_readiness", {})
        readiness_score = 0
        if isinstance(readiness, dict):
            readiness_score = readiness.get("overall_score", 0)
            if hasattr(readiness, "overall_score"):
                readiness_score = readiness.overall_score

        # Dashboard: show project-level summary, not dataset-level
        if screen == "dashboard":
            registry = context.get("registry_info", {}) or {}
            training = context.get("training_history", {}) or {}
            frontend = context.get("frontend_state", {}) or {}
            return {
                "screen": "dashboard",
                "project_id": context.get("project_id"),
                "total_models": registry.get("total_registered", 0) or frontend.get("total_models", 0),
                "total_datasets": frontend.get("total_datasets", 0),
                "avg_accuracy": frontend.get("avg_accuracy", 0),
                "deployed_models": frontend.get("deployed_models", 0),
                "total_predictions": frontend.get("total_predictions", 0),
                "completed_jobs": training.get("completed_jobs", 0),
                "failed_jobs": training.get("failed_jobs", 0),
                "quality_score": quality.get("overall_quality_score", 0),
                "data_readiness_score": readiness_score,
                "phases_completed": pipeline.get("phases_completed", []),
                "last_phase": pipeline.get("last_phase"),
                "next_phase": pipeline.get("next_recommended_phase"),
            }

        # AI Insights hub: merged project + dataset summary
        if screen == "ai_insights":
            registry = context.get("registry_info", {}) or {}
            training = context.get("training_history", {}) or {}
            frontend = context.get("frontend_state", {}) or {}
            versions = context.get("model_versions", {}) or {}
            return {
                "screen": "ai_insights",
                "project_id": context.get("project_id"),
                "dataset_id": context.get("dataset_id"),
                # Data info
                "rows": profile.get("rows", 0),
                "columns": profile.get("columns", 0),
                "numeric_count": profile.get("numeric_count", 0),
                "categorical_count": profile.get("categorical_count", 0),
                "completeness": quality.get("completeness", 100),
                "quality_score": quality.get("overall_quality_score", 0),
                "duplicate_pct": quality.get("duplicate_pct", 0),
                "data_readiness_score": readiness_score,
                # Model info
                "total_models": registry.get("total_registered", 0) or frontend.get("total_models", 0),
                "total_datasets": frontend.get("total_datasets", 0),
                "deployed_models": frontend.get("deployed_models", 0),
                "total_versions": len((versions.get("versions") or [])),
                "avg_accuracy": frontend.get("avg_accuracy", 0),
                "total_predictions": frontend.get("total_predictions", 0),
                # Pipeline
                "completed_jobs": training.get("completed_jobs", 0),
                "failed_jobs": training.get("failed_jobs", 0),
                "phases_completed": pipeline.get("phases_completed", []),
                "next_phase": pipeline.get("next_recommended_phase"),
            }

        # ── Auto-detected target info ──
        target_info = context.get("target_variable", {}) or {}
        target_name = target_info.get("name")
        target_type = target_info.get("type")
        minority_pct = target_info.get("minority_class_pct")

        # ── Readiness breakdown ──
        readiness_breakdown = {}
        if isinstance(readiness, dict):
            readiness_breakdown = {
                "overall": readiness.get("overall_score", readiness_score),
                "completeness": readiness.get("completeness_score", 0),
                "quality": readiness.get("quality_score", 0),
                "features": readiness.get("feature_score", 0),
                "target": readiness.get("target_score", 0),
                "complexity": readiness.get("complexity_score", 0),
                "difficulty": readiness.get("estimated_difficulty", "unknown"),
                "strengths": readiness.get("strengths", []),
                "bottlenecks": readiness.get("bottlenecks", []),
            }

        return {
            "screen": screen,
            "rows": profile.get("rows", 0),
            "columns": profile.get("columns", 0),
            "numeric_count": profile.get("numeric_count", 0),
            "categorical_count": profile.get("categorical_count", 0),
            "completeness": quality.get("completeness", 100),
            "quality_score": quality.get("overall_quality_score", 100),
            "duplicate_pct": quality.get("duplicate_pct", 0),
            "data_readiness_score": readiness_score,
            "readiness": readiness_breakdown,
            "target_column": target_name,
            "target_type": target_type,
            "target_minority_pct": minority_pct,
            "phases_completed": pipeline.get("phases_completed", []),
            "last_phase": pipeline.get("last_phase"),
            "next_phase": pipeline.get("next_recommended_phase"),
        }

    # ──────────────────────────────────────────────────────────
    # LLM SMART GATING (cost optimization)
    # ──────────────────────────────────────────────────────────

    def _should_call_llm(
            self,
            use_llm: str,
            critical_count: int,
            warning_count: int,
            total_insights: int,
            screen: str,
    ) -> bool:
        """
        Smart gating for /insights — decides if LLM is worth the cost.

        use_llm modes:
          'never'  → Always skip LLM (rules only, $0)
          'always' → Always call LLM (~$0.02 per call)
          'auto'   → Smart: only call when rules need synthesis

        Auto logic:
          - Call LLM when 4+ critical issues (complex, needs prioritized synthesis)
          - Call LLM when 15+ total insights (too many for user to parse alone)
          - Call LLM for evaluation/deployment screens (high-stakes decisions)
          - Skip for simple datasets where rules give clear actionable answers
        """
        if not self.llm_reasoner.enabled:
            return False

        if use_llm == "never":
            return False
        if use_llm == "always":
            return True

        # Auto mode — smart gating
        # High-stakes screens always get LLM synthesis
        if screen in ("evaluation", "deployment", "monitoring"):
            return True

        # Complex situation — many critical issues need prioritized expert guidance
        if critical_count >= 4:
            return True

        # Information overload — too many findings for user to parse alone
        if total_insights >= 15:
            return True

        # Mixed severity with substantial warnings — needs nuanced advice
        if critical_count >= 2 and warning_count >= 5:
            return True

        # Simple situation — rules are clear enough
        return False

    def _should_call_llm_for_ask(
            self,
            use_llm: str,
            qa_confidence: float,
    ) -> bool:
        """
        Smart gating for /ask — decides if LLM should enhance the QA answer.

        Auto logic:
          - Call LLM only when QA engine confidence < 0.7
          - This means the deterministic engine couldn't find a confident match
        """
        if not self.llm_reasoner.enabled:
            return False

        if use_llm == "never":
            return False
        if use_llm == "always":
            return True

        # Auto mode — only enhance low-confidence answers
        return qa_confidence < 0.7

    def _generate_suggested_questions(
            self,
            insights: List,
            context: Dict,
            screen: str,
    ) -> List[str]:
        """
        Generate dynamic suggested questions based on fired insights.
        Returns 3-5 dataset-specific questions for the frontend chips.
        """
        questions = []
        rule_ids = {getattr(i, "rule_id", "") or "" for i in insights}
        severities = {getattr(i, "severity", "") or "" for i in insights}
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        num_count = profile.get("numeric_count", 0)
        cat_count = profile.get("categorical_count", 0)

        # Critical issues
        critical = [i for i in insights if getattr(i, "severity", "") == "critical"]
        if critical:
            questions.append(
                f"How do I fix the {len(critical)} critical issues before training?"
            )

        # Data leakage
        if any(r.startswith("DL-") for r in rule_ids):
            questions.append("Explain the data leakage risk and how to fix it")

        # Feature engineering
        fe_rules = [r for r in rule_ids if r.startswith("FE-")]
        if fe_rules:
            questions.append(
                f"Which feature engineering step will improve accuracy the most?"
            )

        # High cardinality / encoding
        if "FH-004" in rule_ids or cat_count > 10:
            questions.append(
                f"What's the best encoding for my {cat_count} categorical features?"
            )

        # Type mismatch (DQ-010)
        if "DQ-010" in rule_ids:
            questions.append(
                "How do I fix columns stored as wrong data type?"
            )

        # Small dataset
        if rows < 5000:
            questions.append(
                f"Is {rows:,} rows enough? What techniques work for small datasets?"
            )

        # Target variable
        target_rules = [r for r in rule_ids if r.startswith("TV-")]
        if not target_rules:
            questions.append(
                "What accuracy should I expect with this data?"
            )

        # Algorithm selection
        if "FH-006" in rule_ids:
            questions.append(
                "Which algorithm handles my mixed feature types best?"
            )

        # Trim to 5 and ensure no duplicates
        seen = set()
        unique_questions = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique_questions.append(q)
        return unique_questions[:5]

    def _generate_rules_synthesis(
            self,
            insights: List,
            context: Dict,
    ) -> str:
        """
        Generate a structured summary from rules alone (no LLM cost).
        Provides actionable advice by organizing insights by priority.
        """
        profile = context.get("dataset_profile", {})
        rows = profile.get("rows", 0)
        cols = profile.get("columns", 0)

        criticals = [i for i in insights if i.severity == "critical"]
        warnings = [i for i in insights if i.severity == "warning"]
        tips = [i for i in insights if i.severity in ("tip", "info")]
        successes = [i for i in insights if i.severity == "success"]

        parts = []

        # Header
        if rows > 0:
            parts.append(f"**Dataset: {rows:,} rows × {cols} columns**\n")

        # Critical issues
        if criticals:
            parts.append(f"## 🔴 {len(criticals)} Critical Issue(s) — Fix Before Training\n")
            for i, c in enumerate(criticals[:5], 1):
                parts.append(f"{i}. **{c.title}**")
                if c.action:
                    parts.append(f"   → {c.action}\n")
                else:
                    parts.append("")

        # Warnings
        if warnings:
            parts.append(f"\n## ⚠️ {len(warnings)} Warning(s)\n")
            for w in warnings[:5]:
                parts.append(f"- **{w.title}**: {w.message[:120]}{'...' if len(w.message) > 120 else ''}")

        # Positive findings
        if successes:
            parts.append(f"\n## ✅ What's Good\n")
            for s in successes[:3]:
                parts.append(f"- {s.title}")

        # Tips
        if tips:
            parts.append(f"\n## 💡 Suggestions\n")
            for t in tips[:3]:
                parts.append(f"- **{t.title}**: {t.action or t.message[:100]}")

        if not parts:
            parts.append("No significant issues detected. Dataset appears ready for modeling.")

        parts.append(f"\n\n*Analysis: {len(criticals)} critical, {len(warnings)} warnings, "
                     f"{len(tips)} tips | Source: rules_only (LLM not used, $0 cost)*")

        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────
    # FEEDBACK & MEMORY
    # ──────────────────────────────────────────────────────────

    def record_feedback(self, user_id: str, rule_id: str, helpful: bool):
        """Record user feedback on an insight."""
        if self.memory:
            self.memory.learn_from_feedback(user_id, rule_id, helpful)
            # Track feedback patterns for the rule
            self.memory.remember(
                user_id=user_id,
                memory_type="feedback_pattern",
                key=f"rule_feedback_count:{rule_id}",
                value={"helpful_count" if helpful else "unhelpful_count": 1},
            )

    def invalidate_cache(self, dataset_id: str):
        """Invalidate cached analysis for a dataset."""
        self._cache.invalidate("analysis", dataset_id)
        self._cache.invalidate("patterns", dataset_id)