"""
Platform Models — Auto-Adapting to Real Database Schema
=========================================================
Declares ALL possible columns, then at startup removes any that
don't actually exist in the real database. This prevents
"no such column" errors while supporting different DB versions.

Call `adapt_models_to_schema(engine)` once at app startup.
"""

import logging
from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    Column, String, Text, Float, Boolean, Integer,
    DateTime, ForeignKey, inspect as sa_inspect,
)
from app.core.database import Base

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATASETS (stable schema — unlikely to change)
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
# EDA RESULTS (stable schema)
# ═══════════════════════════════════════════════════════════════

class EdaResult(Base):
    __tablename__ = "eda_results"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    summary = Column(Text, nullable=True)
    statistics = Column(Text, nullable=True)
    quality = Column(Text, nullable=True)
    correlations = Column(Text, nullable=True)
    analysis_status = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


# ═══════════════════════════════════════════════════════════════
# REGISTERED MODELS — Schema may vary across deployments
# ═══════════════════════════════════════════════════════════════

class RegisteredModel(Base):
    __tablename__ = "registered_models"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String(36), nullable=True, index=True)
    name = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    # These columns may not exist in all DB versions — adapter removes if missing
    algorithm = Column(String(200), nullable=True)
    framework = Column(String(100), nullable=True)
    current_stage = Column(String(50), nullable=True)
    metrics = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


# ═══════════════════════════════════════════════════════════════
# MODEL VERSIONS — Rich schema, usually stable
# ═══════════════════════════════════════════════════════════════

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
# JOBS — Schema varies significantly across deployments
# ═══════════════════════════════════════════════════════════════

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(100), nullable=True, index=True)
    # These columns may not exist in all DB versions — adapter removes if missing
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


# ═══════════════════════════════════════════════════════════════
# SCHEMA ADAPTER — Call once at startup
# ═══════════════════════════════════════════════════════════════

_ADAPTABLE_MODELS = [RegisteredModel, ModelVersion, Job]


def adapt_models_to_schema(engine):
    """
    Inspect the real database and remove any declared columns that
    don't actually exist. Prevents 'no such column' SQL errors.

    Call this ONCE at app startup, after engine is created:

        from app.models.platform import adapt_models_to_schema
        adapt_models_to_schema(engine)
    """
    try:
        inspector = sa_inspect(engine)
        available_tables = set(inspector.get_table_names())
    except Exception as e:
        logger.warning(f"Schema adapter: could not inspect database: {e}")
        return

    adapted_count = 0

    for model_cls in _ADAPTABLE_MODELS:
        table_name = model_cls.__tablename__

        if table_name not in available_tables:
            logger.warning(f"Schema adapter: table '{table_name}' not found — skipping")
            continue

        try:
            real_columns = {col["name"] for col in inspector.get_columns(table_name)}
        except Exception as e:
            logger.warning(f"Schema adapter: could not read columns for '{table_name}': {e}")
            continue

        mapper_table = model_cls.__table__
        declared_columns = {col.name for col in mapper_table.columns}

        missing = declared_columns - real_columns
        extra_in_db = real_columns - declared_columns

        if missing:
            logger.info(
                f"Schema adapter: {table_name} — removing {len(missing)} "
                f"non-existent column(s): {sorted(missing)}"
            )
            for col_name in missing:
                try:
                    col_obj = mapper_table.c[col_name]
                    mapper_table._columns.remove(col_obj)
                    adapted_count += 1
                except Exception as e:
                    logger.warning(f"Schema adapter: could not remove {table_name}.{col_name}: {e}")

        if extra_in_db:
            logger.info(
                f"Schema adapter: {table_name} has extra DB column(s) "
                f"not in model: {sorted(extra_in_db)}"
            )

    if adapted_count > 0:
        logger.info(f"Schema adapter: removed {adapted_count} non-existent column(s) total")
    else:
        logger.info("Schema adapter: all model columns match database — no changes needed")