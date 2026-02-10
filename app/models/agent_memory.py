"""
Agent Memory — Database Model
================================
SQLAlchemy model for persisting agent memory across sessions.
Stores: feedback, preferences, interaction patterns, decision history.

Add to models/__init__.py or import directly:
  from app.models.agent_memory import AgentMemory

Table auto-created by Base.metadata.create_all(engine).
"""

from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    Column, String, Text, Float, Boolean, Integer,
    DateTime, Index,
)
from app.core.database import Base


class AgentMemory(Base):
    """
    Persistent memory for the ML Expert Agent.
    Each row = one memory entry (feedback, preference, interaction, decision).
    Supports: reinforcement, temporal decay, confidence scoring, expiration.
    """
    __tablename__ = "agent_memory"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(100), nullable=False, index=True)

    # Memory classification
    memory_type = Column(String(50), nullable=False, index=True)
    # Types: feedback, preference, interaction, decision, question, pattern, expertise

    # Key-value storage
    key = Column(String(500), nullable=False)
    value = Column(Text, nullable=True)  # JSON text

    # Scoring & learning
    confidence = Column(Float, default=1.0)
    decay_factor = Column(Float, default=1.0)
    reinforcement_count = Column(Integer, default=1)
    access_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    # Scoping
    project_id = Column(String(36), nullable=True, index=True)
    dataset_id = Column(String(36), nullable=True)
    screen = Column(String(50), nullable=True)

    # Expiration
    expires_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_accessed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_agent_memory_user_type", "user_id", "memory_type"),
        Index("ix_agent_memory_user_key", "user_id", "key"),
        Index("ix_agent_memory_active", "user_id", "is_active"),
    )

    def __repr__(self):
        return f"<AgentMemory(user={self.user_id}, type={self.memory_type}, key={self.key[:40]})>"
