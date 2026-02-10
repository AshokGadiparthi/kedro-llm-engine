"""
ML Expert Agent — Core Module
==============================
Context-aware ML guidance system with 200+ rules, statistical analysis,
pattern detection, business impact quantification, and optional LLM synthesis.

Components:
  ┌──────────────────────────────────────────────────────┐
  │ AgentOrchestrator   — Central brain, single entry    │
  │ ContextCompiler     — Metadata extraction (no raw)   │
  │ RuleEngine          — 200+ deterministic ML rules    │
  │ StatisticalAnalyzer — Deep metadata-driven stats     │
  │ PatternDetector     — Anti-pattern & signal detect   │
  │ RecommendationEng.  — Actionable ML strategies       │
  │ BusinessImpactCalc. — Metrics → dollar value         │
  │ ValidationEngine    — Pre-action readiness gates     │
  │ QAEngine            — Deterministic Q&A (18 intents) │
  │ DriftAnalyzer       — Production drift detection     │
  │ ThresholdOptimizer  — Business-optimal thresholds    │
  │ KnowledgeBase       — Curated ML decision frameworks │
  │ LLMReasoner         — Optional LLM synthesis layer   │
  │ MemoryStore         — Persistent learning & prefs    │
  └──────────────────────────────────────────────────────┘

Usage:
  # Full pipeline (recommended):
  from app.core.agent import AgentOrchestrator
  orchestrator = AgentOrchestrator(db_session)
  bundle = await orchestrator.get_insights(screen="eda", dataset_id="abc")

  # Individual components:
  from app.core.agent import ContextCompiler, RuleEngine
"""

# Core pipeline components
from .context_compiler import ContextCompiler
from .rule_engine import RuleEngine, Insight
from .statistical_analyzer import StatisticalAnalyzer
from .pattern_detector import PatternDetector
from .recommendation_engine import RecommendationEngine
from .business_impact import BusinessImpactCalculator
from .validation_engine import ValidationEngine, ValidationCheck, ValidationResult
from .qa_engine import QAEngine
from .drift_analyzer import DriftAnalyzer
from .threshold_optimizer import ThresholdOptimizer

# Knowledge & reasoning
from .knowledge_base import (
    ALGORITHM_PROFILES,
    METRIC_GUIDE,
    PITFALL_PATTERNS,
    SCREEN_KNOWLEDGE,
    recommend_algorithms,
    get_algorithm_profile,
    get_screen_knowledge,
    get_metric_guide,
    detect_pitfall,
)
from .llm_reasoner import LLMReasoner

# Memory & learning
from .memory_store import MemoryStore

# Orchestrator (imports all above internally)
from .orchestrator import AgentOrchestrator

# Extended rules (auto-registered)
from . import rules_extended

# ── Enhancement Components (v4.0) ──
from .domain_profiles import DomainProfileManager, DomainThresholds, DomainProfile
from .semantic_intent import SemanticIntentClassifier
from .statistical_tests import StatisticalTestEngine
from .feedback_tracker import FeedbackTracker
from .domain_cost_matrices import DomainCostMatrixManager
from .integration_wiring import EnhancedOrchestrator

__all__ = [
    # ── Original v3 ──
    "AgentOrchestrator",
    "ContextCompiler",
    "RuleEngine", "Insight",
    "StatisticalAnalyzer",
    "PatternDetector",
    "RecommendationEngine",
    "BusinessImpactCalculator",
    "ValidationEngine", "ValidationCheck", "ValidationResult",
    "QAEngine",
    "DriftAnalyzer",
    "ThresholdOptimizer",
    "LLMReasoner",
    "MemoryStore",
    "ALGORITHM_PROFILES",
    "METRIC_GUIDE",
    # ── Enhancement v4 ──
    "EnhancedOrchestrator",
    "DomainProfileManager",
    "DomainThresholds",
    "DomainProfile",
    "SemanticIntentClassifier",
    "StatisticalTestEngine",
    "FeedbackTracker",
    "DomainCostMatrixManager",
]
