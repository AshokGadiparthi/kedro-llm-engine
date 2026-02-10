"""
Platform Models — Matches the REAL kedro-engine-dynamic database.
===================================================================
These models MUST match your existing app's table schemas exactly.
SQLAlchemy generates SELECT queries based on these column declarations —
any mismatch causes silent failures.

If your existing app's schema changes, update these models to match.
"""

from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    Column, String, Text, Float, Boolean, Integer,
    DateTime, ForeignKey, Index,
)
from app.core.database import Base


# ═══════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=True)
    project_id = Column(String, nullable=True, index=True)
    description = Column(Text, nullable=True)
    file_name = Column(String(255), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source_type = Column(String(30), nullable=True)
    collection_id = Column(String(36), nullable=True, index=True)


class DatasetCollection(Base):
    __tablename__ = "dataset_collections"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CollectionTable(Base):
    __tablename__ = "collection_tables"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    collection_id = Column(String(36), nullable=True, index=True)
    name = Column(String, nullable=True)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns_metadata = Column(Text, nullable=True)


# ═══════════════════════════════════════════════════════════════
# EDA RESULTS — Main table with JSON columns
# ═══════════════════════════════════════════════════════════════

class EdaResult(Base):
    __tablename__ = "eda_results"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    summary = Column(Text, nullable=True)        # JSON: {shape, columns, dtypes, column_types, memory_usage}
    statistics = Column(Text, nullable=True)      # JSON: {basic_stats, missing_values, duplicates}
    quality = Column(Text, nullable=True)         # JSON: {completeness, uniqueness, validity, consistency}
    correlations = Column(Text, nullable=True)    # JSON: {correlations, threshold, high_correlation_pairs}
    analysis_status = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


# ═══════════════════════════════════════════════════════════════
# MODELS & VERSIONS
# ═══════════════════════════════════════════════════════════════

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
    updated_at = Column(DateTime, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    model_id = Column(String(36), ForeignKey("registered_models.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    version_number = Column(Integer, nullable=False)
    is_current = Column(Boolean, nullable=True)
    status = Column(String(20), nullable=True)
    algorithm = Column(String(100), nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    train_score = Column(Float, nullable=True)
    test_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    job_id = Column(String(100), nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    model_size_mb = Column(Float, nullable=True)
    hyperparameters = Column(Text, nullable=True)
    feature_names = Column(Text, nullable=True)
    feature_importances = Column(Text, nullable=True)
    confusion_matrix = Column(Text, nullable=True)
    training_config = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════
# JOBS
# ═══════════════════════════════════════════════════════════════

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