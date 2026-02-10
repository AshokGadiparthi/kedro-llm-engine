"""
Feedback Tracker — Closed-Loop Recommendation → Outcome System
================================================================
Tracks whether agent recommendations actually improved outcomes.
This is the #1 differentiator of world-class ML platforms.

SOLVES: "The agent says 'use SMOTE' but never checks if SMOTE actually helped."

Capabilities:
  1. Recommendation Tracking  — Log what the agent recommended
  2. Action Tracking          — Log what the user actually did
  3. Outcome Measurement      — Compare metrics before/after the action
  4. Effectiveness Scoring    — Score each recommendation type's track record
  5. Adaptive Confidence      — Adjust future recommendation confidence based on history
  6. A/B Benchmarking         — Compare agent-guided vs unguided outcomes
  7. Rule Effectiveness       — Track which rules led to better models
  8. Recommendation Ranking   — Rank recommendations by proven effectiveness

Architecture:
  - Write-ahead log: every recommendation is logged BEFORE user sees it
  - Outcome linked by (user_id, project_id, recommendation_id)
  - Effectiveness scores are computed periodically and cached
  - All storage via the MemoryStore (no new DB tables required)

Usage:
  tracker = FeedbackTracker(memory_store)
  rec_id = tracker.log_recommendation(user_id, "algorithm", "Use XGBoost", context_snapshot)
  tracker.log_user_action(user_id, rec_id, "followed", {"algorithm": "xgboost"})
  tracker.log_outcome(user_id, rec_id, {"f1_before": 0.72, "f1_after": 0.81})
  effectiveness = tracker.get_recommendation_effectiveness("algorithm")
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from collections import defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# RECOMMENDATION LOG ENTRY
# ═══════════════════════════════════════════════════════════════

class RecommendationRecord:
    """A single recommendation and its lifecycle."""

    def __init__(
        self,
        rec_id: str,
        user_id: str,
        project_id: str,
        category: str,        # algorithm | feature | threshold | encoding | scaling | imputation | ...
        rule_id: str,
        recommendation: str,
        context_snapshot: Dict[str, Any],
        timestamp: Optional[str] = None,
    ):
        self.rec_id = rec_id
        self.user_id = user_id
        self.project_id = project_id
        self.category = category
        self.rule_id = rule_id
        self.recommendation = recommendation
        self.context_snapshot = context_snapshot
        self.timestamp = timestamp or datetime.utcnow().isoformat()

        # Lifecycle tracking
        self.user_action: Optional[str] = None        # followed | ignored | modified | rejected
        self.user_action_details: Dict[str, Any] = {}
        self.action_timestamp: Optional[str] = None

        # Outcome tracking
        self.metrics_before: Dict[str, float] = {}
        self.metrics_after: Dict[str, float] = {}
        self.outcome_delta: Dict[str, float] = {}
        self.outcome_verdict: Optional[str] = None    # improved | unchanged | degraded
        self.outcome_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rec_id": self.rec_id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "category": self.category,
            "rule_id": self.rule_id,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "user_action": self.user_action,
            "user_action_details": self.user_action_details,
            "action_timestamp": self.action_timestamp,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "outcome_delta": self.outcome_delta,
            "outcome_verdict": self.outcome_verdict,
            "outcome_timestamp": self.outcome_timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'RecommendationRecord':
        rec = cls(
            rec_id=d["rec_id"], user_id=d["user_id"],
            project_id=d.get("project_id", ""),
            category=d["category"], rule_id=d.get("rule_id", ""),
            recommendation=d["recommendation"],
            context_snapshot=d.get("context_snapshot", {}),
            timestamp=d.get("timestamp"),
        )
        rec.user_action = d.get("user_action")
        rec.user_action_details = d.get("user_action_details", {})
        rec.action_timestamp = d.get("action_timestamp")
        rec.metrics_before = d.get("metrics_before", {})
        rec.metrics_after = d.get("metrics_after", {})
        rec.outcome_delta = d.get("outcome_delta", {})
        rec.outcome_verdict = d.get("outcome_verdict")
        rec.outcome_timestamp = d.get("outcome_timestamp")
        return rec


# ═══════════════════════════════════════════════════════════════
# EFFECTIVENESS SCORE
# ═══════════════════════════════════════════════════════════════

class EffectivenessScore:
    """Aggregated effectiveness of a recommendation category or rule."""

    def __init__(self, category: str):
        self.category = category
        self.total_recommendations = 0
        self.followed_count = 0
        self.ignored_count = 0
        self.improved_count = 0
        self.unchanged_count = 0
        self.degraded_count = 0
        self.avg_improvement: Dict[str, float] = {}
        self.follow_rate = 0.0
        self.success_rate = 0.0  # improved / (improved + degraded)
        self.effectiveness_score = 0.0  # composite 0-1

    def compute(self):
        """Compute derived scores."""
        if self.total_recommendations > 0:
            self.follow_rate = self.followed_count / self.total_recommendations
        outcomes = self.improved_count + self.degraded_count
        if outcomes > 0:
            self.success_rate = self.improved_count / outcomes
        # Composite: weighted combination of follow rate and success rate
        self.effectiveness_score = (
            self.follow_rate * 0.3 +
            self.success_rate * 0.7
        )

    def to_dict(self) -> Dict[str, Any]:
        self.compute()
        return {
            "category": self.category,
            "total_recommendations": self.total_recommendations,
            "followed_count": self.followed_count,
            "ignored_count": self.ignored_count,
            "improved_count": self.improved_count,
            "unchanged_count": self.unchanged_count,
            "degraded_count": self.degraded_count,
            "avg_improvement": self.avg_improvement,
            "follow_rate": round(self.follow_rate, 3),
            "success_rate": round(self.success_rate, 3),
            "effectiveness_score": round(self.effectiveness_score, 3),
        }


# ═══════════════════════════════════════════════════════════════
# FEEDBACK TRACKER
# ═══════════════════════════════════════════════════════════════

class FeedbackTracker:
    """
    Closed-loop system that tracks recommendations → actions → outcomes.
    Learns which recommendations are most effective over time.
    """

    def __init__(self, memory_store=None):
        """
        Args:
            memory_store: MemoryStore instance for persistence.
                         If None, uses in-memory storage only.
        """
        self._memory = memory_store
        self._records: Dict[str, RecommendationRecord] = {}  # rec_id → record
        self._effectiveness_cache: Dict[str, EffectivenessScore] = {}
        self._cache_dirty = True

    # ──────────────────────────────────────────────────────────
    # 1. LOG RECOMMENDATION
    # ──────────────────────────────────────────────────────────

    def log_recommendation(
        self,
        user_id: str,
        project_id: str,
        category: str,
        rule_id: str,
        recommendation: str,
        context_snapshot: Optional[Dict] = None,
        current_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Log a recommendation BEFORE showing it to the user.
        Returns: recommendation ID for future tracking.
        """
        rec_id = str(uuid4())[:12]
        record = RecommendationRecord(
            rec_id=rec_id,
            user_id=user_id,
            project_id=project_id,
            category=category,
            rule_id=rule_id,
            recommendation=recommendation,
            context_snapshot=context_snapshot or {},
        )
        if current_metrics:
            record.metrics_before = current_metrics

        self._records[rec_id] = record
        self._persist_record(record)
        self._cache_dirty = True

        logger.info(f"Logged recommendation {rec_id}: [{category}] {recommendation[:80]}...")
        return rec_id

    # ──────────────────────────────────────────────────────────
    # 2. LOG USER ACTION
    # ──────────────────────────────────────────────────────────

    def log_user_action(
        self,
        user_id: str,
        rec_id: str,
        action: str,  # followed | ignored | modified | rejected
        details: Optional[Dict] = None,
    ) -> bool:
        """
        Log what the user actually did after seeing the recommendation.
        """
        record = self._get_record(rec_id)
        if not record:
            logger.warning(f"Recommendation {rec_id} not found")
            return False

        record.user_action = action
        record.user_action_details = details or {}
        record.action_timestamp = datetime.utcnow().isoformat()

        self._persist_record(record)
        self._cache_dirty = True

        logger.info(f"Logged action for {rec_id}: {action}")
        return True

    # ──────────────────────────────────────────────────────────
    # 3. LOG OUTCOME
    # ──────────────────────────────────────────────────────────

    def log_outcome(
        self,
        user_id: str,
        rec_id: str,
        metrics_after: Dict[str, float],
        metrics_before: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        Log the outcome metrics after the user's action.
        Computes delta and verdict automatically.

        Returns: outcome verdict (improved | unchanged | degraded)
        """
        record = self._get_record(rec_id)
        if not record:
            logger.warning(f"Recommendation {rec_id} not found")
            return None

        if metrics_before:
            record.metrics_before = metrics_before

        record.metrics_after = metrics_after
        record.outcome_timestamp = datetime.utcnow().isoformat()

        # Compute deltas
        for key in set(record.metrics_before.keys()) | set(metrics_after.keys()):
            before = record.metrics_before.get(key)
            after = metrics_after.get(key)
            if before is not None and after is not None:
                record.outcome_delta[key] = after - before

        # Determine verdict based on primary metric
        verdict = self._compute_verdict(record.outcome_delta, record.category)
        record.outcome_verdict = verdict

        self._persist_record(record)
        self._cache_dirty = True

        logger.info(f"Logged outcome for {rec_id}: {verdict} "
                    f"(deltas: {record.outcome_delta})")
        return verdict

    # ──────────────────────────────────────────────────────────
    # 4. AUTO-DETECT OUTCOME
    # ──────────────────────────────────────────────────────────

    def auto_detect_outcomes(
        self,
        user_id: str,
        project_id: str,
        current_metrics: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Automatically check if any pending recommendations can be resolved
        by comparing current metrics with stored before-metrics.

        Called after each training/evaluation cycle.
        """
        resolved = []

        for rec_id, record in self._records.items():
            if (record.user_id != user_id or record.project_id != project_id):
                continue
            if record.outcome_verdict is not None:
                continue  # Already resolved
            if not record.metrics_before:
                continue  # No baseline

            # Check if user has new metrics (implying they did something)
            has_overlap = any(k in current_metrics for k in record.metrics_before)
            if not has_overlap:
                continue

            # Auto-log outcome
            verdict = self.log_outcome(user_id, rec_id, current_metrics)
            if verdict:
                resolved.append({
                    "rec_id": rec_id,
                    "category": record.category,
                    "recommendation": record.recommendation[:100],
                    "verdict": verdict,
                    "delta": record.outcome_delta,
                })

        return resolved

    # ──────────────────────────────────────────────────────────
    # 5. EFFECTIVENESS ANALYSIS
    # ──────────────────────────────────────────────────────────

    def get_recommendation_effectiveness(
        self,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        min_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Get effectiveness scores for recommendation categories.
        """
        if not self._cache_dirty and category and category in self._effectiveness_cache:
            return self._effectiveness_cache[category].to_dict()

        # Aggregate by category
        cat_scores: Dict[str, EffectivenessScore] = defaultdict(lambda: EffectivenessScore(""))

        for rec_id, record in self._records.items():
            if user_id and record.user_id != user_id:
                continue

            cat = record.category
            if cat not in cat_scores:
                cat_scores[cat] = EffectivenessScore(cat)

            score = cat_scores[cat]
            score.total_recommendations += 1

            if record.user_action in ("followed", "modified"):
                score.followed_count += 1
            elif record.user_action in ("ignored", "rejected"):
                score.ignored_count += 1

            if record.outcome_verdict == "improved":
                score.improved_count += 1
            elif record.outcome_verdict == "unchanged":
                score.unchanged_count += 1
            elif record.outcome_verdict == "degraded":
                score.degraded_count += 1

            # Track average improvement per metric
            for metric, delta in record.outcome_delta.items():
                if metric not in score.avg_improvement:
                    score.avg_improvement[metric] = []
                score.avg_improvement[metric].append(delta)

        # Compute averages
        for score in cat_scores.values():
            for metric, deltas in list(score.avg_improvement.items()):
                if isinstance(deltas, list) and deltas:
                    score.avg_improvement[metric] = sum(deltas) / len(deltas)
                else:
                    score.avg_improvement[metric] = 0.0
            score.compute()

        self._effectiveness_cache = cat_scores
        self._cache_dirty = False

        if category:
            if category in cat_scores and cat_scores[category].total_recommendations >= min_samples:
                return cat_scores[category].to_dict()
            return {"category": category, "message": "Insufficient data", "total": 0}

        # Return all categories with enough data
        return {
            "categories": {
                k: v.to_dict() for k, v in cat_scores.items()
                if v.total_recommendations >= min_samples
            },
            "total_recommendations": sum(v.total_recommendations for v in cat_scores.values()),
            "overall_follow_rate": (
                sum(v.followed_count for v in cat_scores.values()) /
                max(sum(v.total_recommendations for v in cat_scores.values()), 1)
            ),
            "overall_success_rate": (
                sum(v.improved_count for v in cat_scores.values()) /
                max(sum(v.improved_count + v.degraded_count for v in cat_scores.values()), 1)
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 6. CONFIDENCE ADJUSTMENT
    # ──────────────────────────────────────────────────────────

    def get_confidence_adjustment(self, category: str, rule_id: str = "") -> float:
        """
        Get confidence multiplier for a recommendation category based on
        historical effectiveness. Returns 0.5 to 1.5.

        > 1.0 → historically effective, boost confidence
        < 1.0 → historically ineffective, reduce confidence
        = 1.0 → no data or neutral
        """
        # Check rule-level first
        if rule_id:
            rule_records = [
                r for r in self._records.values()
                if r.rule_id == rule_id and r.outcome_verdict
            ]
            if len(rule_records) >= 3:
                improved = sum(1 for r in rule_records if r.outcome_verdict == "improved")
                degraded = sum(1 for r in rule_records if r.outcome_verdict == "degraded")
                total = improved + degraded
                if total > 0:
                    rate = improved / total
                    return 0.5 + rate  # 0.5 to 1.5

        # Category-level
        effectiveness = self.get_recommendation_effectiveness(category)
        if isinstance(effectiveness, dict) and "success_rate" in effectiveness:
            return 0.5 + effectiveness["success_rate"]

        return 1.0  # Neutral

    # ──────────────────────────────────────────────────────────
    # 7. BENCHMARKING
    # ──────────────────────────────────────────────────────────

    def get_benchmarking_report(
        self,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate A/B benchmarking report: recommendations followed vs ignored.
        """
        followed_outcomes = []
        ignored_outcomes = []

        for record in self._records.values():
            if user_id and record.user_id != user_id:
                continue
            if not record.outcome_verdict:
                continue

            if record.user_action in ("followed", "modified"):
                followed_outcomes.append(record)
            elif record.user_action in ("ignored", "rejected"):
                ignored_outcomes.append(record)

        def _summarize(records: List[RecommendationRecord]) -> Dict:
            if not records:
                return {"count": 0}
            improved = sum(1 for r in records if r.outcome_verdict == "improved")
            degraded = sum(1 for r in records if r.outcome_verdict == "degraded")
            n = len(records)
            avg_deltas: Dict[str, List[float]] = defaultdict(list)
            for r in records:
                for k, v in r.outcome_delta.items():
                    avg_deltas[k].append(v)
            return {
                "count": n,
                "improved_pct": round(improved / n * 100, 1),
                "degraded_pct": round(degraded / n * 100, 1),
                "avg_metric_changes": {
                    k: round(sum(v) / len(v), 4) for k, v in avg_deltas.items()
                },
            }

        return {
            "followed_recommendations": _summarize(followed_outcomes),
            "ignored_recommendations": _summarize(ignored_outcomes),
            "conclusion": self._benchmarking_conclusion(
                followed_outcomes, ignored_outcomes
            ),
        }

    # ──────────────────────────────────────────────────────────
    # 8. TOP RULES BY EFFECTIVENESS
    # ──────────────────────────────────────────────────────────

    def get_top_rules(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top-performing rules by outcome improvement."""
        rule_stats: Dict[str, Dict] = defaultdict(lambda: {
            "rule_id": "", "total": 0, "improved": 0, "degraded": 0, "avg_delta": []
        })

        for record in self._records.values():
            if not record.rule_id or not record.outcome_verdict:
                continue
            stats = rule_stats[record.rule_id]
            stats["rule_id"] = record.rule_id
            stats["total"] += 1
            if record.outcome_verdict == "improved":
                stats["improved"] += 1
            elif record.outcome_verdict == "degraded":
                stats["degraded"] += 1
            for v in record.outcome_delta.values():
                stats["avg_delta"].append(v)

        results = []
        for rule_id, stats in rule_stats.items():
            if stats["total"] < 2:
                continue
            avg_delta = sum(stats["avg_delta"]) / len(stats["avg_delta"]) if stats["avg_delta"] else 0
            success_rate = stats["improved"] / max(stats["improved"] + stats["degraded"], 1)
            results.append({
                "rule_id": rule_id,
                "total_uses": stats["total"],
                "success_rate": round(success_rate, 3),
                "avg_delta": round(avg_delta, 4),
                "improved": stats["improved"],
                "degraded": stats["degraded"],
            })

        results.sort(key=lambda x: x["success_rate"], reverse=True)
        return results[:n]

    # ══════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    def _compute_verdict(self, deltas: Dict[str, float], category: str) -> str:
        """Compute whether the outcome improved, using smart thresholds."""
        if not deltas:
            return "unchanged"

        # Priority metrics by category
        priority_metrics = {
            "algorithm": ["f1", "f1_score", "roc_auc", "auc", "accuracy"],
            "feature": ["f1", "f1_score", "accuracy", "roc_auc"],
            "threshold": ["f1", "f1_score", "precision", "recall"],
            "encoding": ["f1", "accuracy", "roc_auc"],
            "scaling": ["f1", "accuracy"],
            "imputation": ["f1", "accuracy", "roc_auc"],
            "hyperparameter": ["f1", "roc_auc", "accuracy"],
        }

        metrics_to_check = priority_metrics.get(category, ["f1", "accuracy", "roc_auc"])

        # Check primary metric first
        for metric in metrics_to_check:
            if metric in deltas:
                delta = deltas[metric]
                if delta > 0.005:  # >0.5% improvement is meaningful
                    return "improved"
                elif delta < -0.005:
                    return "degraded"
                return "unchanged"

        # If no priority metric found, check any available
        positive = sum(1 for v in deltas.values() if v > 0.005)
        negative = sum(1 for v in deltas.values() if v < -0.005)

        if positive > negative:
            return "improved"
        elif negative > positive:
            return "degraded"
        return "unchanged"

    def _benchmarking_conclusion(
        self,
        followed: List[RecommendationRecord],
        ignored: List[RecommendationRecord],
    ) -> str:
        """Generate human-readable benchmarking conclusion."""
        if not followed and not ignored:
            return "Insufficient data for benchmarking. Continue using the agent to build a track record."

        followed_improved = sum(1 for r in followed if r.outcome_verdict == "improved") if followed else 0
        ignored_improved = sum(1 for r in ignored if r.outcome_verdict == "improved") if ignored else 0

        f_rate = followed_improved / len(followed) if followed else 0
        i_rate = ignored_improved / len(ignored) if ignored else 0

        if f_rate > i_rate + 0.1:
            return (
                f"Following agent recommendations improved outcomes "
                f"{f_rate*100:.0f}% of the time vs {i_rate*100:.0f}% when ignored. "
                f"The agent's advice is adding measurable value."
            )
        elif i_rate > f_rate + 0.1:
            return (
                f"Ignoring recommendations led to better outcomes ({i_rate*100:.0f}% vs {f_rate*100:.0f}%). "
                f"The agent may need recalibration for your use case."
            )
        else:
            return (
                f"Follow rate: {f_rate*100:.0f}% improved, ignore rate: {i_rate*100:.0f}% improved. "
                f"Outcomes are similar — more data needed for conclusive benchmarking."
            )

    def _get_record(self, rec_id: str) -> Optional[RecommendationRecord]:
        """Get a record from cache or memory store."""
        if rec_id in self._records:
            return self._records[rec_id]

        # Try loading from memory store
        if self._memory:
            try:
                data = self._memory.recall(
                    user_id="system",
                    memory_type="recommendation_log",
                    key=f"rec:{rec_id}",
                )
                if data and isinstance(data, dict):
                    record = RecommendationRecord.from_dict(data)
                    self._records[rec_id] = record
                    return record
            except Exception as e:
                logger.debug(f"Could not load record {rec_id}: {e}")

        return None

    def _persist_record(self, record: RecommendationRecord):
        """Save record to memory store for persistence."""
        if self._memory:
            try:
                self._memory.remember(
                    user_id=record.user_id,
                    memory_type="recommendation_log",
                    key=f"rec:{record.rec_id}",
                    value=record.to_dict(),
                )
            except Exception as e:
                logger.debug(f"Could not persist record {record.rec_id}: {e}")
