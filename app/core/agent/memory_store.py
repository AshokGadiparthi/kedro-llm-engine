"""
Memory Store — Persistent Agent Learning & Personalization
============================================================
Production-grade memory system for the ML Expert Agent.

Capabilities:
  1. Feedback Learning      — Track helpful/unhelpful insight ratings
  2. Preference Detection   — Learn user's preferred detail level, algorithms, etc.
  3. Interaction Patterns   — Track which screens/features user spends time on
  4. Decision History       — Record what user decided after agent advice
  5. Temporal Decay         — Old memories lose weight over time
  6. Memory Consolidation   — Merge redundant memories into patterns
  7. Cross-Session Context  — Remember across conversations
  8. User Profiling         — Build ML expertise level profile
  9. Project-Specific Memory — Different memories per project
  10. Insight Effectiveness  — Track which insights led to better models

Storage: SQLAlchemy ORM (PostgreSQL/SQLite)
Design: Write-heavy, read-optimized with in-memory cache layer
"""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from collections import defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATABASE MODEL
# ═══════════════════════════════════════════════════════════════

def create_agent_memory_table(Base):
    """Create SQLAlchemy models for agent memory. Call once at startup."""
    from sqlalchemy import Column, String, DateTime, Text, Float, Integer, Boolean, Index
    from sqlalchemy.sql import func

    class AgentMemory(Base):
        __tablename__ = "agent_memory"
        __table_args__ = (
            Index("ix_agent_memory_user_type", "user_id", "memory_type"),
            Index("ix_agent_memory_user_key", "user_id", "key"),
            Index("ix_agent_memory_project", "user_id", "project_id"),
        )

        id = Column(String, primary_key=True, default=lambda: str(uuid4()))
        user_id = Column(String(100), nullable=False, index=True)
        project_id = Column(String(100), nullable=True, index=True)
        memory_type = Column(String(50), nullable=False, index=True)
        key = Column(String(500), nullable=False, index=True)
        value = Column(Text, nullable=False, default="{}")
        confidence = Column(Float, default=1.0)
        access_count = Column(Integer, default=0)
        reinforcement_count = Column(Integer, default=0)  # times same memory was reinforced
        decay_factor = Column(Float, default=1.0)  # temporal decay (0-1)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
        expires_at = Column(DateTime(timezone=True), nullable=True)

    class AgentFeedbackLog(Base):
        __tablename__ = "agent_feedback_log"
        __table_args__ = (
            Index("ix_feedback_log_user_rule", "user_id", "rule_id"),
        )

        id = Column(String, primary_key=True, default=lambda: str(uuid4()))
        user_id = Column(String(100), nullable=False, index=True)
        project_id = Column(String(100), nullable=True)
        rule_id = Column(String(100), nullable=False, index=True)
        insight_title = Column(String(500), nullable=True)
        screen = Column(String(50), nullable=True)
        helpful = Column(Boolean, nullable=False)
        comment = Column(Text, nullable=True)
        context_snapshot = Column(Text, nullable=True)  # JSON snapshot of context at feedback time
        created_at = Column(DateTime(timezone=True), server_default=func.now())

    class AgentDecisionLog(Base):
        __tablename__ = "agent_decision_log"

        id = Column(String, primary_key=True, default=lambda: str(uuid4()))
        user_id = Column(String(100), nullable=False, index=True)
        project_id = Column(String(100), nullable=True)
        decision_type = Column(String(50), nullable=False)  # algorithm_choice | threshold_change | feature_drop | deploy
        agent_recommendation = Column(Text, nullable=True)
        user_choice = Column(Text, nullable=True)
        followed_recommendation = Column(Boolean, nullable=True)
        outcome_metric = Column(String(50), nullable=True)
        outcome_value = Column(Float, nullable=True)
        created_at = Column(DateTime(timezone=True), server_default=func.now())

    return AgentMemory, AgentFeedbackLog, AgentDecisionLog


# ═══════════════════════════════════════════════════════════════
# IN-MEMORY CACHE
# ═══════════════════════════════════════════════════════════════

class _MemoryCache:
    """Fast in-memory cache for frequently accessed memories."""

    def __init__(self, max_per_user: int = 200):
        self._store: Dict[str, Dict[str, Any]] = {}  # user_id -> {key -> value}
        self._max_per_user = max_per_user

    def get(self, user_id: str, key: str) -> Optional[Any]:
        user_cache = self._store.get(user_id, {})
        return user_cache.get(key)

    def set(self, user_id: str, key: str, value: Any):
        if user_id not in self._store:
            self._store[user_id] = {}
        cache = self._store[user_id]
        if len(cache) >= self._max_per_user:
            # Evict oldest entry
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[key] = value

    def invalidate(self, user_id: str, key: str = None):
        if key:
            self._store.get(user_id, {}).pop(key, None)
        else:
            self._store.pop(user_id, None)

    def clear(self):
        self._store.clear()


# ═══════════════════════════════════════════════════════════════
# MEMORY TYPES
# ═══════════════════════════════════════════════════════════════

class MemoryType:
    """Enumeration of memory types with their decay rates."""
    FEEDBACK = "feedback"              # insight helpful/unhelpful
    PREFERENCE = "preference"          # user preferences (detail level, etc.)
    INTERACTION = "interaction"        # screen visits, feature usage
    DECISION = "decision"              # user decisions after advice
    EXPERTISE_LEVEL = "expertise"      # estimated ML knowledge level
    PROJECT_CONTEXT = "project_ctx"    # project-specific knowledge
    QUESTION_HISTORY = "question"      # past questions asked
    MODEL_HISTORY = "model_history"    # models trained and their outcomes
    PATTERN = "pattern"                # consolidated behavioral patterns

    # Decay half-life in days (how long before memory loses half its weight)
    DECAY_RATES = {
        "feedback": 90,        # feedback stays relevant for ~3 months
        "preference": 180,     # preferences are long-lived
        "interaction": 30,     # recent interactions matter most
        "decision": 60,        # decisions relevant for ~2 months
        "expertise": 365,      # expertise changes slowly
        "project_ctx": 120,    # project context lasts a season
        "question": 14,        # questions are ephemeral
        "model_history": 180,  # model history is long-lived
        "pattern": 365,        # patterns are durable
    }


# ═══════════════════════════════════════════════════════════════
# MAIN MEMORY STORE
# ═══════════════════════════════════════════════════════════════

class MemoryStore:
    """
    Production memory system with learning, decay, and personalization.
    
    Usage:
        store = MemoryStore(db_session, AgentMemory, AgentFeedbackLog, AgentDecisionLog)
        store.remember(user_id, "feedback", "rule:DQ-001", {"helpful": True})
        memories = store.recall(user_id, memory_type="feedback")
        profile = store.get_user_profile(user_id)
    """

    def __init__(self, db_session, memory_model, feedback_model=None, decision_model=None):
        self.db = db_session
        self.Memory = memory_model
        self.FeedbackLog = feedback_model
        self.DecisionLog = decision_model
        self._cache = _MemoryCache(max_per_user=200)

    # ──────────────────────────────────────────────────────────
    # CORE OPERATIONS
    # ──────────────────────────────────────────────────────────

    def remember(
        self, user_id: str, memory_type: str, key: str, value: Any,
        confidence: float = 1.0, project_id: str = None,
        ttl_days: int = None,
    ):
        """Store or update a memory. Reinforces existing memories."""
        try:
            value_json = json.dumps(value, default=str) if not isinstance(value, str) else value

            existing = self.db.query(self.Memory).filter(
                self.Memory.user_id == user_id,
                self.Memory.key == key,
            ).first()

            if existing:
                # Reinforce existing memory
                old_value = json.loads(existing.value) if existing.value else {}
                if isinstance(old_value, dict) and isinstance(value, dict):
                    old_value.update(value)
                    existing.value = json.dumps(old_value, default=str)
                else:
                    existing.value = value_json

                existing.reinforcement_count = (existing.reinforcement_count or 0) + 1
                existing.access_count = (existing.access_count or 0) + 1
                existing.confidence = min(1.0, confidence + 0.05 * existing.reinforcement_count)
                existing.decay_factor = 1.0  # Reset decay on reinforcement
                existing.updated_at = datetime.utcnow()
                existing.is_active = True

                if ttl_days:
                    existing.expires_at = datetime.utcnow() + timedelta(days=ttl_days)
            else:
                # Create new memory
                mem = self.Memory(
                    id=str(uuid4()),
                    user_id=user_id,
                    project_id=project_id,
                    memory_type=memory_type,
                    key=key,
                    value=value_json,
                    confidence=confidence,
                    access_count=1,
                    reinforcement_count=1,
                    decay_factor=1.0,
                    is_active=True,
                    expires_at=datetime.utcnow() + timedelta(days=ttl_days) if ttl_days else None,
                )
                self.db.add(mem)

            self.db.commit()
            self._cache.set(user_id, key, value)

        except Exception as e:
            logger.warning(f"Memory store failed: {e}")
            self.db.rollback()

    def recall(
        self, user_id: str, memory_type: str = None, key: str = None,
        project_id: str = None, limit: int = 50,
        min_confidence: float = 0.0, include_decayed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with optional filtering and decay application."""
        try:
            query = self.db.query(self.Memory).filter(
                self.Memory.user_id == user_id,
                self.Memory.is_active == True,
            )

            if memory_type:
                query = query.filter(self.Memory.memory_type == memory_type)
            if key:
                query = query.filter(self.Memory.key == key)
            if project_id:
                query = query.filter(
                    (self.Memory.project_id == project_id) | (self.Memory.project_id.is_(None))
                )
            if min_confidence > 0:
                query = query.filter(self.Memory.confidence >= min_confidence)

            memories = query.order_by(self.Memory.updated_at.desc()).limit(limit).all()

            results = []
            now = datetime.utcnow()

            for mem in memories:
                # Check expiration
                if mem.expires_at and mem.expires_at < now:
                    mem.is_active = False
                    continue

                # Apply temporal decay
                decay = self._compute_decay(mem, now)
                effective_confidence = (mem.confidence or 1.0) * decay

                if not include_decayed and effective_confidence < 0.1:
                    continue

                # Update access count
                mem.access_count = (mem.access_count or 0) + 1

                try:
                    value = json.loads(mem.value) if isinstance(mem.value, str) else mem.value
                except (json.JSONDecodeError, TypeError):
                    value = mem.value

                results.append({
                    "key": mem.key,
                    "value": value,
                    "memory_type": mem.memory_type,
                    "confidence": round(effective_confidence, 3),
                    "raw_confidence": mem.confidence,
                    "decay_factor": round(decay, 3),
                    "reinforcement_count": mem.reinforcement_count or 0,
                    "access_count": mem.access_count or 0,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None,
                    "updated_at": mem.updated_at.isoformat() if mem.updated_at else None,
                    "project_id": mem.project_id,
                })

            try:
                self.db.commit()
            except Exception:
                self.db.rollback()

            return results

        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return []

    def forget(self, user_id: str, key: str = None, memory_type: str = None):
        """Soft-delete memories."""
        try:
            query = self.db.query(self.Memory).filter(self.Memory.user_id == user_id)
            if key:
                query = query.filter(self.Memory.key == key)
            if memory_type:
                query = query.filter(self.Memory.memory_type == memory_type)
            query.update({"is_active": False})
            self.db.commit()
            self._cache.invalidate(user_id, key)
        except Exception as e:
            logger.warning(f"Memory forget failed: {e}")
            self.db.rollback()

    # ──────────────────────────────────────────────────────────
    # FEEDBACK LEARNING
    # ──────────────────────────────────────────────────────────

    def learn_from_feedback(
        self, user_id: str, rule_id: str, helpful: bool,
        insight_title: str = None, screen: str = None,
        comment: str = None, context_snapshot: Dict = None,
        project_id: str = None,
    ):
        """
        Learn from user's thumbs up/down on insights.
        Updates both the feedback log and the memory store.
        """
        # Log the feedback
        if self.FeedbackLog:
            try:
                log = self.FeedbackLog(
                    id=str(uuid4()),
                    user_id=user_id,
                    project_id=project_id,
                    rule_id=rule_id,
                    insight_title=insight_title,
                    screen=screen,
                    helpful=helpful,
                    comment=comment,
                    context_snapshot=json.dumps(context_snapshot, default=str) if context_snapshot else None,
                )
                self.db.add(log)
                self.db.commit()
            except Exception as e:
                logger.warning(f"Feedback log failed: {e}")
                self.db.rollback()

        # Update memory with feedback aggregation
        feedback_key = f"feedback:{rule_id}"
        existing = self.recall(user_id, memory_type=MemoryType.FEEDBACK, key=feedback_key, limit=1)

        if existing:
            old_val = existing[0].get("value", {})
            if isinstance(old_val, dict):
                helpful_count = old_val.get("helpful_count", 0)
                unhelpful_count = old_val.get("unhelpful_count", 0)
                if helpful:
                    helpful_count += 1
                else:
                    unhelpful_count += 1
                total = helpful_count + unhelpful_count
                new_val = {
                    "helpful_count": helpful_count,
                    "unhelpful_count": unhelpful_count,
                    "helpful_ratio": round(helpful_count / max(1, total), 3),
                    "total_feedback": total,
                    "last_screen": screen,
                    "last_feedback": "helpful" if helpful else "unhelpful",
                }
                self.remember(user_id, MemoryType.FEEDBACK, feedback_key, new_val)
        else:
            self.remember(user_id, MemoryType.FEEDBACK, feedback_key, {
                "helpful_count": 1 if helpful else 0,
                "unhelpful_count": 0 if helpful else 1,
                "helpful_ratio": 1.0 if helpful else 0.0,
                "total_feedback": 1,
                "last_screen": screen,
                "last_feedback": "helpful" if helpful else "unhelpful",
            })

    def get_feedback_adjustments(self, user_id: str) -> Dict[str, float]:
        """
        Get confidence adjustments based on accumulated feedback.
        Returns {rule_id: adjustment_factor} where >1.0 = boost, <1.0 = suppress.
        """
        feedbacks = self.recall(user_id, memory_type=MemoryType.FEEDBACK, limit=100)
        adjustments = {}

        for fb in feedbacks:
            val = fb.get("value", {})
            if not isinstance(val, dict):
                continue

            rule_id = fb["key"].replace("feedback:", "")
            helpful_ratio = val.get("helpful_ratio", 0.5)
            total = val.get("total_feedback", 0)

            if total >= 2:
                # Bayesian adjustment: more feedback = more extreme adjustment
                # Base: 1.0, range: [0.3, 1.3]
                weight = min(1.0, total / 10)  # caps at 10 feedbacks
                adjustment = 0.3 + helpful_ratio * 1.0  # range [0.3, 1.3]
                adjustments[rule_id] = round(1.0 + (adjustment - 1.0) * weight, 3)

        return adjustments

    # ──────────────────────────────────────────────────────────
    # DECISION TRACKING
    # ──────────────────────────────────────────────────────────

    def log_decision(
        self, user_id: str, decision_type: str,
        agent_recommendation: str = None, user_choice: str = None,
        followed: bool = None, outcome_metric: str = None,
        outcome_value: float = None, project_id: str = None,
    ):
        """Log a user decision and whether they followed the agent's recommendation."""
        if self.DecisionLog:
            try:
                log = self.DecisionLog(
                    id=str(uuid4()),
                    user_id=user_id,
                    project_id=project_id,
                    decision_type=decision_type,
                    agent_recommendation=agent_recommendation,
                    user_choice=user_choice,
                    followed_recommendation=followed,
                    outcome_metric=outcome_metric,
                    outcome_value=outcome_value,
                )
                self.db.add(log)
                self.db.commit()
            except Exception as e:
                logger.warning(f"Decision log failed: {e}")
                self.db.rollback()

        # Also store as memory for faster access
        self.remember(user_id, MemoryType.DECISION, f"decision:{decision_type}:{datetime.utcnow().strftime('%Y%m%d%H%M')}", {
            "type": decision_type,
            "recommendation": agent_recommendation,
            "choice": user_choice,
            "followed": followed,
            "outcome": outcome_value,
        }, project_id=project_id, ttl_days=180)

    def get_recommendation_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze how often the user follows recommendations and the outcomes.
        Powerful signal for calibrating agent confidence.
        """
        if not self.DecisionLog:
            return {"available": False}

        try:
            decisions = self.db.query(self.DecisionLog).filter(
                self.DecisionLog.user_id == user_id,
                self.DecisionLog.followed_recommendation.isnot(None),
            ).all()

            if not decisions:
                return {"available": True, "total_decisions": 0}

            total = len(decisions)
            followed = sum(1 for d in decisions if d.followed_recommendation)
            follow_rate = followed / max(1, total)

            # Compare outcomes when followed vs not followed
            followed_outcomes = [d.outcome_value for d in decisions if d.followed_recommendation and d.outcome_value is not None]
            ignored_outcomes = [d.outcome_value for d in decisions if not d.followed_recommendation and d.outcome_value is not None]

            avg_followed = sum(followed_outcomes) / max(1, len(followed_outcomes)) if followed_outcomes else None
            avg_ignored = sum(ignored_outcomes) / max(1, len(ignored_outcomes)) if ignored_outcomes else None

            # By decision type
            by_type = defaultdict(lambda: {"followed": 0, "ignored": 0, "total": 0})
            for d in decisions:
                dt = d.decision_type or "unknown"
                by_type[dt]["total"] += 1
                if d.followed_recommendation:
                    by_type[dt]["followed"] += 1
                else:
                    by_type[dt]["ignored"] += 1

            return {
                "available": True,
                "total_decisions": total,
                "follow_rate": round(follow_rate, 3),
                "followed_count": followed,
                "ignored_count": total - followed,
                "avg_outcome_when_followed": round(avg_followed, 4) if avg_followed is not None else None,
                "avg_outcome_when_ignored": round(avg_ignored, 4) if avg_ignored is not None else None,
                "by_type": dict(by_type),
            }

        except Exception as e:
            logger.warning(f"Effectiveness analysis failed: {e}")
            return {"available": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────
    # USER PROFILING
    # ──────────────────────────────────────────────────────────

    def update_expertise_signal(self, user_id: str, signal_type: str, signal_value: Any):
        """
        Track signals that indicate user's ML expertise level.
        Signals: question_complexity, config_sophistication, terminology_use, etc.
        """
        key = f"expertise_signal:{signal_type}"
        existing = self.recall(user_id, memory_type=MemoryType.EXPERTISE_LEVEL, key=key, limit=1)

        if existing:
            old_val = existing[0].get("value", {})
            signals = old_val.get("signals", [])
            signals.append({"value": signal_value, "ts": datetime.utcnow().isoformat()})
            signals = signals[-20:]  # Keep last 20 signals
            old_val["signals"] = signals
            old_val["latest"] = signal_value
            self.remember(user_id, MemoryType.EXPERTISE_LEVEL, key, old_val)
        else:
            self.remember(user_id, MemoryType.EXPERTISE_LEVEL, key, {
                "signals": [{"value": signal_value, "ts": datetime.utcnow().isoformat()}],
                "latest": signal_value,
            })

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Build a comprehensive user profile from all memory types.
        Used to personalize insight detail level, terminology, etc.
        """
        profile = {
            "user_id": user_id,
            "expertise_level": "intermediate",  # beginner | intermediate | advanced | expert
            "detail_preference": "medium",       # brief | medium | detailed
            "follow_rate": 0.5,
            "interaction_count": 0,
            "favorite_screens": [],
            "preferred_algorithms": [],
            "common_questions": [],
            "feedback_summary": {"total": 0, "helpful_ratio": 0.5},
        }

        try:
            # Expertise signals
            expertise_memories = self.recall(user_id, memory_type=MemoryType.EXPERTISE_LEVEL, limit=20)
            if expertise_memories:
                profile["expertise_level"] = self._estimate_expertise(expertise_memories)

            # Feedback summary
            feedbacks = self.recall(user_id, memory_type=MemoryType.FEEDBACK, limit=100)
            if feedbacks:
                total_fb = sum(f.get("value", {}).get("total_feedback", 0) for f in feedbacks if isinstance(f.get("value"), dict))
                helpful_total = sum(f.get("value", {}).get("helpful_count", 0) for f in feedbacks if isinstance(f.get("value"), dict))
                profile["feedback_summary"] = {
                    "total": total_fb,
                    "helpful_ratio": round(helpful_total / max(1, total_fb), 3),
                }

            # Interaction patterns
            interactions = self.recall(user_id, memory_type=MemoryType.INTERACTION, limit=50)
            profile["interaction_count"] = len(interactions)
            if interactions:
                screen_counts = defaultdict(int)
                for inter in interactions:
                    val = inter.get("value", {})
                    if isinstance(val, dict) and val.get("screen"):
                        screen_counts[val["screen"]] += 1
                profile["favorite_screens"] = sorted(screen_counts, key=screen_counts.get, reverse=True)[:5]

            # Question history
            questions = self.recall(user_id, memory_type=MemoryType.QUESTION_HISTORY, limit=20)
            profile["common_questions"] = [q.get("value", {}).get("question", "") for q in questions[:5]]

            # Preference for detail level (based on question length and feedback patterns)
            if expertise_memories:
                detail_signals = [e.get("value", {}).get("latest") for e in expertise_memories if e.get("key", "").endswith("question_complexity")]
                if detail_signals:
                    avg_complexity = sum(s for s in detail_signals if isinstance(s, (int, float))) / max(1, len(detail_signals))
                    profile["detail_preference"] = "brief" if avg_complexity < 0.3 else "detailed" if avg_complexity > 0.7 else "medium"

            # Recommendation effectiveness
            effectiveness = self.get_recommendation_effectiveness(user_id)
            if effectiveness.get("available"):
                profile["follow_rate"] = effectiveness.get("follow_rate", 0.5)

        except Exception as e:
            logger.warning(f"Profile building failed: {e}")

        return profile

    # ──────────────────────────────────────────────────────────
    # MEMORY CONSOLIDATION
    # ──────────────────────────────────────────────────────────

    def consolidate(self, user_id: str):
        """
        Merge redundant memories into higher-level patterns.
        Run periodically (e.g., nightly) to keep memory store efficient.
        """
        try:
            # Consolidate interaction patterns
            interactions = self.recall(user_id, memory_type=MemoryType.INTERACTION, limit=200, include_decayed=True)
            if len(interactions) > 20:
                screen_pattern = defaultdict(int)
                time_pattern = defaultdict(int)

                for inter in interactions:
                    val = inter.get("value", {})
                    if isinstance(val, dict):
                        screen = val.get("screen", "unknown")
                        screen_pattern[screen] += 1
                        hour = val.get("hour")
                        if hour is not None:
                            time_pattern[hour] += 1

                self.remember(user_id, MemoryType.PATTERN, "pattern:screen_usage", {
                    "screen_frequency": dict(screen_pattern),
                    "peak_hours": sorted(time_pattern, key=time_pattern.get, reverse=True)[:5],
                    "total_interactions": len(interactions),
                    "consolidated_at": datetime.utcnow().isoformat(),
                })

            # Consolidate question patterns
            questions = self.recall(user_id, memory_type=MemoryType.QUESTION_HISTORY, limit=100, include_decayed=True)
            if len(questions) > 10:
                topic_counts = defaultdict(int)
                for q in questions:
                    val = q.get("value", {})
                    if isinstance(val, dict):
                        screen = val.get("screen", "general")
                        topic_counts[screen] += 1

                self.remember(user_id, MemoryType.PATTERN, "pattern:question_topics", {
                    "by_screen": dict(topic_counts),
                    "total_questions": len(questions),
                    "consolidated_at": datetime.utcnow().isoformat(),
                })

            # Clean up expired and deeply decayed memories
            self._cleanup_expired(user_id)

        except Exception as e:
            logger.warning(f"Memory consolidation failed: {e}")

    def _cleanup_expired(self, user_id: str):
        """Remove expired and deeply decayed memories."""
        try:
            now = datetime.utcnow()

            # Remove expired
            self.db.query(self.Memory).filter(
                self.Memory.user_id == user_id,
                self.Memory.expires_at.isnot(None),
                self.Memory.expires_at < now,
            ).update({"is_active": False})

            # Deactivate very old, unreinforced memories
            stale_cutoff = now - timedelta(days=365)
            self.db.query(self.Memory).filter(
                self.Memory.user_id == user_id,
                self.Memory.is_active == True,
                self.Memory.reinforcement_count <= 1,
                self.Memory.updated_at < stale_cutoff,
            ).update({"is_active": False})

            self.db.commit()

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
            self.db.rollback()

    # ──────────────────────────────────────────────────────────
    # PROJECT-SCOPED MEMORY
    # ──────────────────────────────────────────────────────────

    def get_project_context(self, user_id: str, project_id: str) -> Dict[str, Any]:
        """Get all agent memories for a specific project."""
        memories = self.recall(user_id, project_id=project_id, limit=100)
        context = {
            "decisions": [],
            "feedback": [],
            "preferences": {},
            "model_history": [],
        }

        for mem in memories:
            mtype = mem.get("memory_type", "")
            val = mem.get("value", {})
            if mtype == MemoryType.DECISION:
                context["decisions"].append(val)
            elif mtype == MemoryType.FEEDBACK:
                context["feedback"].append(val)
            elif mtype == MemoryType.PREFERENCE:
                if isinstance(val, dict):
                    context["preferences"].update(val)
            elif mtype == MemoryType.MODEL_HISTORY:
                context["model_history"].append(val)

        return context

    # ──────────────────────────────────────────────────────────
    # TEMPORAL DECAY
    # ──────────────────────────────────────────────────────────

    def _compute_decay(self, memory, now: datetime) -> float:
        """
        Compute temporal decay factor using exponential decay.
        decay = e^(-λt) where λ = ln(2)/half_life and t = days since update.
        Reinforced memories decay slower.
        """
        updated = memory.updated_at or memory.created_at or now
        if isinstance(updated, str):
            try:
                updated = datetime.fromisoformat(updated)
            except (ValueError, TypeError):
                return 1.0

        days_old = (now - updated).total_seconds() / 86400
        if days_old <= 0:
            return 1.0

        # Get half-life for this memory type
        half_life = MemoryType.DECAY_RATES.get(memory.memory_type, 60)

        # Reinforced memories decay slower
        reinforcement = memory.reinforcement_count or 1
        effective_half_life = half_life * (1 + 0.2 * min(reinforcement, 10))

        # Exponential decay
        decay_rate = math.log(2) / effective_half_life
        decay = math.exp(-decay_rate * days_old)

        return max(0.01, min(1.0, decay))

    # ──────────────────────────────────────────────────────────
    # EXPERTISE ESTIMATION
    # ──────────────────────────────────────────────────────────

    def _estimate_expertise(self, expertise_memories: List[Dict]) -> str:
        """
        Estimate user's ML expertise from accumulated signals.
        
        Signals include:
        - Question complexity (simple → advanced terminology)
        - Config sophistication (default params → custom hyperparams)
        - Feature usage (basic EDA → advanced feature engineering)
        - Correct metric choice (accuracy vs F1 for imbalanced data)
        """
        scores = []

        for mem in expertise_memories:
            val = mem.get("value", {})
            if isinstance(val, dict):
                signals = val.get("signals", [])
                for s in signals:
                    sig_val = s.get("value")
                    if isinstance(sig_val, (int, float)):
                        scores.append(sig_val)

        if not scores:
            return "intermediate"

        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.8:
            return "expert"
        elif avg_score >= 0.6:
            return "advanced"
        elif avg_score >= 0.3:
            return "intermediate"
        else:
            return "beginner"

    # ──────────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────────

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        try:
            total = self.db.query(self.Memory).filter(
                self.Memory.user_id == user_id,
                self.Memory.is_active == True,
            ).count()

            by_type = {}
            for mtype in [MemoryType.FEEDBACK, MemoryType.PREFERENCE, MemoryType.INTERACTION,
                          MemoryType.DECISION, MemoryType.EXPERTISE_LEVEL, MemoryType.PATTERN]:
                count = self.db.query(self.Memory).filter(
                    self.Memory.user_id == user_id,
                    self.Memory.memory_type == mtype,
                    self.Memory.is_active == True,
                ).count()
                if count:
                    by_type[mtype] = count

            return {
                "user_id": user_id,
                "total_memories": total,
                "by_type": by_type,
                "cache_size": len(self._cache._store.get(user_id, {})),
            }

        except Exception as e:
            logger.warning(f"Memory stats failed: {e}")
            return {"user_id": user_id, "error": str(e)}
