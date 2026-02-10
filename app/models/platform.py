"""
Platform Models — Database tables for ML Platform entities.

These match the interface that ContextCompiler queries via SQLAlchemy.
When merging into your existing app: if you already have these models
(Dataset, EdaResult, etc.), delete this file — keep your real models.
The context_compiler uses try/except imports so it auto-discovers yours.
"""

from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    Column, String, Text, Float, Boolean, Integer,
    DateTime, ForeignKey, Index,
)
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(500), nullable=True)
    file_name = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    file_format = Column(String(50), nullable=True)
    schema = Column(Text, nullable=True)
    collection_id = Column(String(36), ForeignKey("dataset_collections.id"), nullable=True)
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatasetCollection(Base):
    __tablename__ = "dataset_collections"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CollectionTable(Base):
    __tablename__ = "collection_tables"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_id = Column(String(36), ForeignKey("dataset_collections.id"), nullable=True, index=True)
    name = Column(String(500), nullable=True)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns_metadata = Column(Text, nullable=True)


class EdaResult(Base):
    __tablename__ = "eda_results"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    summary = Column(Text, nullable=True)
    quality = Column(Text, nullable=True)
    correlations = Column(Text, nullable=True)
    column_stats = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RegisteredModel(Base):
    __tablename__ = "registered_models"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String(36), nullable=True, index=True)
    name = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    algorithm = Column(String(200), nullable=True)
    framework = Column(String(100), nullable=True)
    current_stage = Column(String(50), nullable=True)
    metrics = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    model_id = Column(String(36), ForeignKey("registered_models.id"), nullable=True, index=True)
    version = Column(Integer, nullable=True)
    stage = Column(String(50), nullable=True)
    metrics = Column(Text, nullable=True)
    parameters = Column(Text, nullable=True)
    run_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(100), nullable=True, index=True)
    project_id = Column(String(36), nullable=True, index=True)
    job_type = Column(String(50), nullable=True)
    status = Column(String(50), nullable=True)
    algorithm = Column(String(200), nullable=True)
    metrics = Column(Text, nullable=True)
    config = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    __table_args__ = (Index("ix_jobs_user_created", "user_id", "created_at"),)
