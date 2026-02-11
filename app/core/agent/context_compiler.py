"""
Context Compiler — Production-Grade Metadata Extraction
=========================================================
Extracts metadata from EVERY platform source to build rich context.
NEVER accesses raw data — only reads database records, EDA results,
model metadata, and Kedro pipeline artifacts.

Data Sources (all read via SQLAlchemy ORM):
  EdaResult       → summary, statistics, quality, correlations
  Dataset         → name, file_info, schema, row_count, quality_score
  DataProfile     → full_report (if available from data_management)
  Job             → pipeline runs, status, params, execution time
  RegisteredModel → registry metadata, deployment status
  ModelVersion    → accuracy, precision, recall, f1, roc_auc,
                    hyperparams, feature_importances, confusion_matrix
  DatasetCollect. → multi-table collections
  CollectionTable → individual table metadata
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _safe_json(raw, default=None):
    """Safely parse a JSON text column."""
    if raw is None:
        return default if default is not None else {}
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default if default is not None else {}
    return default if default is not None else {}


def _sf(val, default=0.0) -> float:
    """Safe float."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _si(val, default=0) -> int:
    """Safe int."""
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _parse_memory(mem_str) -> float:
    """Parse '12.34 MB' → float."""
    if isinstance(mem_str, (int, float)):
        return float(mem_str)
    if isinstance(mem_str, str):
        s = mem_str.lower()
        try:
            num = float(s.replace("mb", "").replace("kb", "").replace("gb", "").strip())
            if "gb" in s:
                return num * 1024
            if "kb" in s:
                return num / 1024
            return num
        except ValueError:
            return 0.0
    return 0.0


class ContextCompiler:
    """
    Compiles rich context from platform metadata for the Agent.
    Every method reads ONLY metadata — never raw data rows.
    """

    def __init__(self, db_session):
        self.db = db_session
        self._table_columns: Dict[str, set] = {}   # Cache of real DB columns
        self._schema_checked = False

    # ── Schema Discovery ──────────────────────────────────────
    def _discover_table_columns(self, table_name: str) -> set:
        """Discover which columns actually exist in a DB table (cached)."""
        if table_name in self._table_columns:
            return self._table_columns[table_name]
        try:
            from sqlalchemy import inspect as sa_inspect
            inspector = sa_inspect(self.db.bind)
            cols = {c["name"] for c in inspector.get_columns(table_name)}
            self._table_columns[table_name] = cols
            if not self._schema_checked:
                logger.info(f"Schema discovery: {table_name} has columns: {sorted(cols)}")
            return cols
        except Exception as e:
            logger.warning(f"Schema discovery failed for {table_name}: {e}")
            return set()

    def _has_column(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in the actual database table."""
        return column_name in self._discover_table_columns(table_name)

    @staticmethod
    def _ga(obj, *attrs, default=None):
        """Get first available attribute from an object. Handles missing columns safely."""
        for attr in attrs:
            val = getattr(obj, attr, None)
            if val is not None:
                return val
        return default

    async def compile(
            self,
            screen: str,
            project_id: Optional[str] = None,
            dataset_id: Optional[str] = None,
            model_id: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile complete context for a given screen."""
        ctx: Dict[str, Any] = {
            "screen": screen,
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
        }

        if dataset_id:
            ctx["dataset_profile"] = await self._get_dataset_profile(dataset_id)
            ctx["data_quality"] = await self._get_data_quality(dataset_id)
            ctx["correlations"] = await self._get_correlations(dataset_id)
            ctx["feature_stats"] = await self._get_feature_stats(dataset_id)
            ctx["target_variable"] = await self._detect_target(dataset_id, extra)
            ctx["collection_info"] = await self._get_collection_info(dataset_id)

        if project_id:
            ctx["training_history"] = await self._get_training_history(project_id)
            ctx["registry_info"] = await self._get_registry_info(project_id, model_id)
            ctx["pipeline_state"] = await self._get_pipeline_state(project_id)
            ctx["model_versions"] = await self._get_model_versions(project_id, model_id)

        enricher = {
            "dashboard": self._enrich_dashboard,
            "data": self._enrich_data,
            "eda": self._enrich_eda,
            "mlflow": self._enrich_training,
            "training": self._enrich_training,
            "evaluation": self._enrich_evaluation,
            "registry": self._enrich_registry,
            "deployment": self._enrich_deployment,
            "predictions": self._enrich_predictions,
            "monitoring": self._enrich_monitoring,
            "ai_insights": self._enrich_ai_insights,
        }.get(screen)

        if enricher:
            ctx["screen_context"] = await enricher(ctx, extra or {})

        if extra:
            ctx["frontend_state"] = extra

        return ctx

    # ══════════════════════════════════════════════════════════
    # CORE EXTRACTORS
    # ══════════════════════════════════════════════════════════

    async def _get_dataset_profile(self, dataset_id: str) -> Dict[str, Any]:
        """Extract dataset shape, dtypes, memory from EdaResult.summary."""
        profile = {
            "rows": 0, "columns": 0, "memory_mb": 0.0,
            "numeric_count": 0, "categorical_count": 0, "datetime_count": 0,
            "column_names": [], "dtypes": {}, "column_types": {},
            "numeric_columns": [], "categorical_columns": [], "datetime_columns": [],
            "file_name": None, "file_size_bytes": None, "file_format": None,
            "schema": None,
        }
        try:
            from app.models.models import EdaResult
            Dataset = self._get_dataset_model()

            if Dataset:
                ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
                if ds:
                    profile["file_name"] = getattr(ds, "file_name", None) or getattr(ds, "name", None)
                    profile["file_size_bytes"] = getattr(ds, "file_size_bytes", None)
                    profile["file_format"] = getattr(ds, "file_format", None)
                    schema = getattr(ds, "schema", None)
                    if schema:
                        profile["schema"] = _safe_json(schema, [])
                    # Fallback row/col counts
                    rc = getattr(ds, "row_count", None)
                    cc = getattr(ds, "column_count", None)
                    if rc:
                        profile["rows"] = _si(rc)
                    if cc:
                        profile["columns"] = _si(cc)

            eda = self.db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
            if eda and eda.summary:
                summary = _safe_json(eda.summary)
                shape = summary.get("shape", [0, 0])
                if isinstance(shape, list) and len(shape) >= 2:
                    profile["rows"] = _si(shape[0]) or profile["rows"]
                    profile["columns"] = _si(shape[1]) or profile["columns"]

                profile["column_names"] = summary.get("columns", [])
                profile["dtypes"] = summary.get("dtypes", {})
                profile["memory_mb"] = _parse_memory(summary.get("memory_usage", "0"))

                ct = summary.get("column_types", {})
                profile["numeric_columns"] = ct.get("numeric", [])
                profile["categorical_columns"] = ct.get("categorical", [])
                profile["datetime_columns"] = ct.get("datetime", [])
                profile["numeric_count"] = len(profile["numeric_columns"])
                profile["categorical_count"] = len(profile["categorical_columns"])
                profile["datetime_count"] = len(profile["datetime_columns"])
                profile["column_types"] = ct

        except Exception as e:
            logger.warning(f"dataset profile error [{dataset_id}]: {e}")
        return profile

    async def _get_data_quality(self, dataset_id: str) -> Dict[str, Any]:
        """Extract quality metrics from EdaResult.quality and statistics."""
        from app.models.models import EdaResult
        quality = {
            "completeness": 100.0, "uniqueness": 100.0,
            "validity": 100.0, "consistency": 100.0,
            "duplicate_rows": 0, "duplicate_pct": 0.0,
            "missing_cells": 0, "total_cells": 0,
            "missing_by_column": {},
            "columns_with_high_missing": [],
            "overall_quality_score": 100.0,
            "quality_checks": [],
        }
        try:
            eda = self.db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
            if not eda:
                return quality
            if eda.quality:
                q = _safe_json(eda.quality)
                quality["completeness"] = _sf(q.get("completeness"), 100.0)
                quality["uniqueness"] = _sf(q.get("uniqueness"), 100.0)
                quality["validity"] = _sf(q.get("validity"), 100.0)
                quality["consistency"] = _sf(q.get("consistency"), 100.0)
                quality["duplicate_rows"] = _si(q.get("duplicate_rows", 0))
                quality["missing_cells"] = _si(q.get("missing_values_count", 0))
                quality["total_cells"] = _si(q.get("total_cells", 0))
                quality["overall_quality_score"] = _sf(q.get("overall_quality_score"), 100.0)
                quality["quality_checks"] = q.get("quality_checks", [])
            if eda.statistics:
                stats = _safe_json(eda.statistics)
                missing = stats.get("missing_values", {})
                quality["missing_by_column"] = missing
                quality["duplicate_rows"] = max(quality["duplicate_rows"], _si(stats.get("duplicates", 0)))
                for col, info in missing.items():
                    pct = _sf(info.get("percent", 0) if isinstance(info, dict) else 0)
                    cnt = _si(info.get("count", 0) if isinstance(info, dict) else 0)
                    if pct > 20:
                        quality["columns_with_high_missing"].append({"column": col, "missing_pct": pct, "missing_count": cnt})
            if eda.summary:
                summary = _safe_json(eda.summary)
                shape = summary.get("shape", [0, 0])
                rows = _si(shape[0] if isinstance(shape, list) else 0)
                if rows > 0:
                    quality["duplicate_pct"] = round(quality["duplicate_rows"] / rows * 100, 2)
        except Exception as e:
            logger.warning(f"data quality error [{dataset_id}]: {e}")
        return quality

    async def _get_correlations(self, dataset_id: str) -> Dict[str, Any]:
        """Extract correlation pairs from EdaResult.correlations."""
        from app.models.models import EdaResult
        result = {"high_pairs": [], "max_correlation": 0.0, "pair_count": 0,
                  "numeric_columns_analyzed": 0, "all_pairs": {}, "correlation_clusters": []}
        try:
            eda = self.db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
            if eda and eda.correlations:
                corr = _safe_json(eda.correlations)
                pairs = corr.get("correlations", {})
                result["numeric_columns_analyzed"] = _si(corr.get("numeric_columns_analyzed", 0))
                result["all_pairs"] = pairs
                for pk, val in pairs.items():
                    av = abs(_sf(val))
                    if av >= 0.5:
                        parts = pk.split("-", 1)
                        result["high_pairs"].append({
                            "feature1": parts[0].strip() if len(parts) > 0 else pk,
                            "feature2": parts[1].strip() if len(parts) > 1 else "",
                            "correlation": round(_sf(val), 3),
                            "abs_correlation": round(av, 3),
                            "severity": "critical" if av >= 0.95 else "warning" if av >= 0.8 else "moderate" if av >= 0.65 else "info",
                        })
                result["high_pairs"].sort(key=lambda x: x["abs_correlation"], reverse=True)
                result["pair_count"] = len(result["high_pairs"])
                if result["high_pairs"]:
                    result["max_correlation"] = result["high_pairs"][0]["abs_correlation"]
                result["correlation_clusters"] = self._build_corr_clusters(pairs)
        except Exception as e:
            logger.warning(f"correlations error [{dataset_id}]: {e}")
        return result

    async def _get_feature_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Per-feature statistics for advanced analysis."""
        from app.models.models import EdaResult
        result = {
            "numeric_stats": {}, "categorical_stats": {},
            "skewed_features": [], "high_cardinality_categoricals": [],
            "low_cardinality_numerics": [], "potential_id_columns": [],
            "constant_columns": [], "zero_variance_columns": [],
            "binary_columns": [], "potential_target_columns": [],
        }
        try:
            eda = self.db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
            if not eda:
                return result
            summary = _safe_json(eda.summary) if eda.summary else {}
            stats = _safe_json(eda.statistics) if eda.statistics else {}
            columns = summary.get("columns", [])
            shape = summary.get("shape", [0, 0])
            rows = _si(shape[0] if isinstance(shape, list) else 0)

            # Numeric
            for col, cs in stats.get("numeric_statistics", {}).items():
                if not isinstance(cs, dict):
                    continue
                mean = _sf(cs.get("mean", 0))
                std = _sf(cs.get("std", 0))
                mn = _sf(cs.get("min", 0))
                mx = _sf(cs.get("max", 0))
                q25 = _sf(cs.get("q25", cs.get("25%", 0)))
                q75 = _sf(cs.get("q75", cs.get("75%", 0)))
                med = _sf(cs.get("median", cs.get("50%", mean)))
                cnt = _si(cs.get("count", 0))
                iqr = q75 - q25
                result["numeric_stats"][col] = {
                    "mean": mean, "std": std, "min": mn, "max": mx,
                    "q25": q25, "q75": q75, "median": med, "count": cnt,
                    "range": mx - mn, "iqr": iqr,
                    "cv": round(std / mean, 4) if mean != 0 else 0,
                }
                if iqr > 0:
                    skew = (mean - med) / iqr
                    if abs(skew) > 0.3:
                        result["skewed_features"].append({
                            "column": col,
                            "direction": "right" if skew > 0 else "left",
                            "magnitude": round(abs(skew), 3),
                        })
                if std == 0 or (mx == mn and cnt > 1):
                    result["constant_columns"].append(col)
                    result["zero_variance_columns"].append(col)
                if mn == 0 and mx == 1 and std > 0:
                    result["binary_columns"].append(col)
                unique_est = min(cnt, int(mx - mn + 1)) if cnt > 0 else 0
                if 2 <= unique_est <= 10 and cnt > 20:
                    result["low_cardinality_numerics"].append({"column": col, "estimated_unique": unique_est})

            # Categorical
            for col, cs in stats.get("categorical_statistics", {}).items():
                if not isinstance(cs, dict):
                    continue
                uniq = _si(cs.get("unique", 0))
                freq = _si(cs.get("freq", 0))
                miss = _si(cs.get("missing", 0))
                result["categorical_stats"][col] = {
                    "unique": uniq, "top_value": cs.get("top"),
                    "top_freq": freq, "missing": miss,
                    "unique_ratio": round(uniq / rows, 4) if rows > 0 else 0,
                }
                if uniq > 50 or (rows > 0 and uniq / rows > 0.5):
                    result["high_cardinality_categoricals"].append({"column": col, "unique_count": uniq, "unique_ratio": round(uniq / rows, 3) if rows else 0})
                if rows > 0 and uniq / rows > 0.95:
                    result["potential_id_columns"].append({"column": col, "unique_count": uniq, "unique_ratio": round(uniq / rows, 3), "detected_by": "cardinality"})
                if uniq == 2:
                    result["potential_target_columns"].append({"column": col, "type": "binary_categorical", "values": uniq})

            # Name patterns
            id_pats = ["id", "uuid", "key", "index", "pk", "sk_id", "_id"]
            tgt_pats = ["target", "label", "class", "churn", "fraud", "outcome", "default", "survived", "is_", "has_"]
            for c in columns:
                cl = c.lower().strip()
                if any(cl == p or cl.endswith(f"_{p}") or cl.startswith(f"{p}_") for p in id_pats):
                    if not any(x["column"] == c for x in result["potential_id_columns"]):
                        result["potential_id_columns"].append({"column": c, "detected_by": "name_pattern"})
                if any(p in cl for p in tgt_pats):
                    if not any(x["column"] == c for x in result["potential_target_columns"]):
                        result["potential_target_columns"].append({"column": c, "type": "name_pattern"})
            for c in result["binary_columns"]:
                if not any(x["column"] == c for x in result["potential_target_columns"]):
                    result["potential_target_columns"].append({"column": c, "type": "binary_numeric"})
        except Exception as e:
            logger.warning(f"feature stats error [{dataset_id}]: {e}")
        return result

    async def _detect_target(self, dataset_id: str, extra: Optional[Dict] = None) -> Dict:
        """Detect or use the specified target variable."""
        result = {"name": None, "detected": False, "type": None, "class_distribution": None, "imbalance_ratio": None, "minority_class_pct": None}

        # Priority 1: User-specified target
        if extra and extra.get("target_column"):
            result["name"] = extra["target_column"]
            result["detected"] = True

        try:
            from app.models.models import EdaResult
            eda = self.db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
            if not eda:
                return result

            summary = _safe_json(eda.summary) if eda.summary else {}
            stats = _safe_json(eda.statistics) if eda.statistics else {}
            columns = summary.get("columns", [])

            # Priority 2: Auto-detect from column name patterns
            if not result["name"]:
                target_patterns = [
                    "target", "label", "class", "churn", "fraud", "outcome",
                    "default", "survived", "y", "attrition", "response",
                    "result", "diagnosis", "is_", "has_",
                ]
                cat_stats = stats.get("categorical_statistics", {})
                num_stats = stats.get("numeric_statistics", {})

                best_candidate = None
                best_score = 0

                for col in columns:
                    cl = col.lower().strip()
                    score = 0

                    # Name pattern matching
                    if any(p in cl for p in target_patterns):
                        score += 3

                    # Binary categorical (exactly 2 unique values)
                    cs = cat_stats.get(col, {})
                    if isinstance(cs, dict) and _si(cs.get("unique", 0)) == 2:
                        score += 2

                    # Binary numeric (0/1)
                    ns = num_stats.get(col, {})
                    if isinstance(ns, dict):
                        mn = _sf(ns.get("min", -1))
                        mx = _sf(ns.get("max", -1))
                        if mn == 0 and mx == 1:
                            score += 2

                    if score > best_score:
                        best_score = score
                        best_candidate = col

                if best_candidate and best_score >= 2:
                    result["name"] = best_candidate
                    result["detected"] = True
                    logger.info(f"Auto-detected target column: '{best_candidate}' (score={best_score})")

            # Enrich with type information
            if result["name"] and eda.statistics:
                cs = stats.get("categorical_statistics", {}).get(result["name"])
                if cs and isinstance(cs, dict):
                    u = _si(cs.get("unique", 0))
                    result["type"] = "binary" if u == 2 else "multiclass" if u <= 20 else "high_cardinality"
                ns = stats.get("numeric_statistics", {}).get(result["name"])
                if ns and isinstance(ns, dict):
                    mn = _sf(ns.get("min", -1))
                    mx = _sf(ns.get("max", -1))
                    if mn == 0 and mx == 1:
                        result["type"] = "binary"
                        mean_val = _sf(ns.get("mean", 0.5))
                        result["minority_class_pct"] = round(min(mean_val, 1 - mean_val) * 100, 1)
                        result["imbalance_ratio"] = round(min(mean_val, 1 - mean_val) / max(mean_val, 1 - mean_val), 3) if max(mean_val, 1 - mean_val) > 0 else 0

        except Exception as e:
            logger.warning(f"Target detection error [{dataset_id}]: {e}")
        return result

    async def _get_collection_info(self, dataset_id: str) -> Optional[Dict]:
        """Check if dataset belongs to a multi-table collection."""
        try:
            from app.models.models import DatasetCollection, CollectionTable
            Dataset = self._get_dataset_model()
            if not Dataset:
                return None
            ds = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not ds or not getattr(ds, "collection_id", None):
                return None
            coll = self.db.query(DatasetCollection).filter(DatasetCollection.id == ds.collection_id).first()
            if not coll:
                return None
            tables = self.db.query(CollectionTable).filter(CollectionTable.collection_id == coll.id).all()
            return {
                "collection_id": coll.id, "collection_name": getattr(coll, "name", ""),
                "total_tables": getattr(coll, "total_tables", len(tables)),
                "tables": [{"name": getattr(t, "table_name", ""), "role": getattr(t, "role", ""), "rows": getattr(t, "row_count", None)} for t in tables],
            }
        except (ImportError, Exception):
            return None

    async def _get_training_history(self, project_id: str) -> Dict[str, Any]:
        """Get history of pipeline training jobs — resilient to schema differences."""
        from app.models.models import Job
        result = {"total_jobs": 0, "completed_jobs": 0, "failed_jobs": 0, "running_jobs": 0,
                  "recent_jobs": [], "avg_execution_time": 0.0, "pipelines_run": []}
        try:
            # Discover which columns the jobs table actually has
            job_cols = self._discover_table_columns("jobs")

            # Build query — only filter by project_id if column exists
            query = self.db.query(Job)
            if "project_id" in job_cols and project_id:
                query = query.filter(Job.project_id == project_id)
            elif "user_id" in job_cols:
                query = query.filter(Job.user_id.isnot(None))

            if "created_at" in job_cols:
                query = query.order_by(Job.created_at.desc())
            jobs = query.limit(50).all()

            result["total_jobs"] = len(jobs)
            total_time = 0.0
            pipes = set()
            for job in jobs:
                st = (self._ga(job, "status") or "unknown").lower()
                if st == "completed":
                    result["completed_jobs"] += 1
                elif st == "failed":
                    result["failed_jobs"] += 1
                elif st in ("running", "pending"):
                    result["running_jobs"] += 1

                # pipeline_name may be stored as pipeline_name, job_type, or type
                pipe = self._ga(job, "pipeline_name", "job_type", "type", default="unknown")
                pipes.add(pipe)

                # execution_time may be stored as execution_time, duration_seconds, or duration
                et = _sf(self._ga(job, "execution_time", "duration_seconds", "duration"), 0)
                if et > 0:
                    total_time += et

                if len(result["recent_jobs"]) < 10:
                    result["recent_jobs"].append({
                        "id": job.id,
                        "pipeline": pipe,
                        "status": st,
                        "algorithm": self._ga(job, "algorithm"),
                        "execution_time": et,
                        "created_at": job.created_at.isoformat() if getattr(job, "created_at", None) else None,
                        "parameters": _safe_json(self._ga(job, "parameters", "config", "configuration"), {}),
                        "metrics": _safe_json(self._ga(job, "metrics"), {}),
                        "error": self._ga(job, "error_message", "error") if st == "failed" else None,
                    })
            result["pipelines_run"] = sorted(pipes)
            if result["completed_jobs"] > 0:
                result["avg_execution_time"] = round(total_time / result["completed_jobs"], 1)
        except Exception as e:
            logger.warning(f"training history error: {e}")
        return result

    async def _get_registry_info(self, project_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get model registry metadata — resilient to schema differences."""
        from app.models.models import RegisteredModel, ModelVersion
        result = {"total_registered": 0, "production_models": 0, "deployed_models": 0, "models": [], "selected_model": None}
        try:
            # Discover which columns actually exist
            rm_cols = self._discover_table_columns("registered_models")
            mv_cols = self._discover_table_columns("model_versions")

            # Build query — only filter by project_id if column exists
            query = self.db.query(RegisteredModel)
            if "project_id" in rm_cols and project_id:
                query = query.filter(RegisteredModel.project_id == project_id)
            models = query.all()

            result["total_registered"] = len(models)

            for m in models[:20]:
                # Get status — could be current_stage, status, or stage
                status = self._ga(m, "current_stage", "status", "stage", default="draft")

                # Get algorithm — could be on model itself or derived from versions
                algo = self._ga(m, "algorithm", "best_algorithm")

                # Get metrics from JSON column if available
                metrics_json = _safe_json(self._ga(m, "metrics"), {})

                # Derive accuracy from metrics JSON or direct column
                best_acc = _sf(
                    self._ga(m, "best_accuracy")
                    or metrics_json.get("accuracy")
                    or metrics_json.get("best_accuracy")
                    or metrics_json.get("f1_score"),
                    0.0
                )

                # Check deployment — could be is_deployed flag, or stage == "production"
                is_deployed = bool(self._ga(m, "is_deployed", default=False))
                if not is_deployed and status and status.lower() in ("production", "deployed", "serving"):
                    is_deployed = True

                # Total versions — could be a column or needs count query
                total_v = _si(self._ga(m, "total_versions"), 0)
                if total_v == 0 and mv_cols:
                    try:
                        total_v = self.db.query(ModelVersion).filter(ModelVersion.model_id == m.id).count()
                    except Exception:
                        total_v = 0

                info = {
                    "id": m.id,
                    "name": self._ga(m, "name", default="unnamed"),
                    "status": status,
                    "best_accuracy": best_acc,
                    "best_algorithm": algo or metrics_json.get("algorithm"),
                    "is_deployed": is_deployed,
                    "total_versions": total_v,
                    "current_version": self._ga(m, "current_version"),
                    "problem_type": self._ga(m, "problem_type"),
                    "framework": self._ga(m, "framework"),
                    "source_dataset_name": self._ga(m, "source_dataset_name"),
                    "description": self._ga(m, "description"),
                }
                result["models"].append(info)

                if status and status.lower() == "production":
                    result["production_models"] += 1
                if is_deployed:
                    result["deployed_models"] += 1
                if model_id and m.id == model_id:
                    result["selected_model"] = info

        except Exception as e:
            logger.warning(f"registry info error: {e}")
        return result

    async def _get_model_versions(self, project_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get model version details — richest source of model performance data."""
        from app.models.models import ModelVersion, RegisteredModel
        result = {"total_versions": 0, "versions": [], "best_version": None, "algorithm_comparison": {}, "metrics_range": {}}
        try:
            query = self.db.query(ModelVersion)
            if model_id:
                query = query.filter(ModelVersion.model_id == model_id)
            else:
                # Query only RegisteredModel.id — safe even if other columns removed
                rm_cols = self._discover_table_columns("registered_models")
                if "project_id" in rm_cols and project_id:
                    mids = [m.id for m in self.db.query(RegisteredModel.id).filter(RegisteredModel.project_id == project_id).all()]
                else:
                    mids = [m.id for m in self.db.query(RegisteredModel.id).all()]
                if not mids:
                    return result
                query = query.filter(ModelVersion.model_id.in_(mids))

            mv_cols = self._discover_table_columns("model_versions")
            if "created_at" in mv_cols:
                query = query.order_by(ModelVersion.created_at.desc())
            versions = query.limit(30).all()

            result["total_versions"] = len(versions)
            best_f1 = -1
            algo_met = {}
            for v in versions:
                vi = {
                    "id": v.id,
                    "model_id": getattr(v, "model_id", None),
                    "version": getattr(v, "version", None),
                    "is_current": bool(getattr(v, "is_current", False)),
                    "status": getattr(v, "status", None) or "draft",
                    "algorithm": getattr(v, "algorithm", None),
                    "metrics": {
                        "accuracy": _sf(getattr(v, "accuracy", None)),
                        "precision": _sf(getattr(v, "precision", None)),
                        "recall": _sf(getattr(v, "recall", None)),
                        "f1_score": _sf(getattr(v, "f1_score", None)),
                        "train_score": _sf(getattr(v, "train_score", None)),
                        "test_score": _sf(getattr(v, "test_score", None)),
                        "roc_auc": _sf(getattr(v, "roc_auc", None)),
                    },
                    "training_time_seconds": _sf(getattr(v, "training_time_seconds", None)),
                    "model_size_mb": _sf(getattr(v, "model_size_mb", None)),
                    "hyperparameters": _safe_json(getattr(v, "hyperparameters", None), {}),
                    "feature_names": _safe_json(getattr(v, "feature_names", None), []),
                    "feature_importances": _safe_json(getattr(v, "feature_importances", None), {}),
                    "confusion_matrix": _safe_json(getattr(v, "confusion_matrix", None), None),
                    "training_config": _safe_json(getattr(v, "training_config", None), {}),
                }
                result["versions"].append(vi)
                f1 = _sf(getattr(v, "f1_score", None), -1)
                if f1 > best_f1:
                    best_f1 = f1
                    result["best_version"] = vi
                algo = getattr(v, "algorithm", None) or "unknown"
                algo_met.setdefault(algo, []).append(vi["metrics"])
            for algo, ml in algo_met.items():
                result["algorithm_comparison"][algo] = {
                    "count": len(ml),
                    "avg_accuracy": round(sum(m["accuracy"] for m in ml) / len(ml), 4),
                    "avg_f1": round(sum(m["f1_score"] for m in ml) / len(ml), 4),
                    "best_f1": round(max(m["f1_score"] for m in ml), 4),
                }
            if result["versions"]:
                for mk in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                    vals = [v["metrics"][mk] for v in result["versions"] if v["metrics"][mk] > 0]
                    if vals:
                        result["metrics_range"][mk] = {"min": round(min(vals), 4), "max": round(max(vals), 4), "spread": round(max(vals) - min(vals), 4)}
        except Exception as e:
            logger.warning(f"model versions error: {e}")
        return result

    async def _get_pipeline_state(self, project_id: str) -> Dict[str, Any]:
        """Determine which pipeline phases have been executed — resilient to schema differences."""
        from app.models.models import Job
        phase_map = {
            "data_loading": ("Phase 1: Data Loading", 1),
            "feature_engineering": ("Phase 2: Feature Engineering", 2),
            "model_training": ("Phase 3: Model Training", 3),
            "training": ("Phase 3: Model Training", 3),
            "phase4": ("Phase 4: Algorithm Comparison", 4),
            "multi_algorithm": ("Phase 4: Algorithm Comparison", 4),
            "phase5": ("Phase 5: Deep Evaluation", 5),
            "analysis": ("Phase 5: Deep Evaluation", 5),
            "evaluation": ("Phase 5: Deep Evaluation", 5),
            "phase6": ("Phase 6: Ensemble", 6),
            "ensemble": ("Phase 6: Ensemble", 6),
            "end_to_end": ("End-to-End Pipeline", 0),
        }
        result = {"phases_completed": [], "phases_failed": [], "last_phase": None, "next_recommended_phase": None}
        try:
            # Discover which columns the jobs table actually has
            job_cols = self._discover_table_columns("jobs")

            query = self.db.query(Job)
            if "project_id" in job_cols and project_id:
                query = query.filter(Job.project_id == project_id)

            if "created_at" in job_cols:
                query = query.order_by(Job.created_at.desc())
            jobs = query.limit(100).all()

            completed, failed = set(), set()
            for job in jobs:
                # pipeline_name may be stored as pipeline_name, job_type, or type
                pl = (self._ga(job, "pipeline_name", "job_type", "type", default="") or "").lower()
                st = (self._ga(job, "status", default="") or "").lower()
                for key, (label, _) in phase_map.items():
                    if key in pl:
                        (completed if st == "completed" else failed).add(label)
            result["phases_completed"] = sorted(completed)
            result["phases_failed"] = sorted(failed - completed)
            ordered = sorted(set(l for l, o in phase_map.values() if o > 0), key=lambda l: next(o for ll, o in phase_map.values() if ll == l))
            for pl in ordered:
                if pl not in completed:
                    result["next_recommended_phase"] = pl
                    break
            if completed:
                result["last_phase"] = sorted(completed, key=lambda l: next((o for ll, o in phase_map.values() if ll == l), 99))[-1]
        except Exception as e:
            logger.warning(f"pipeline state error: {e}")
        return result

    # ══════════════════════════════════════════════════════════
    # SCREEN ENRICHERS
    # ══════════════════════════════════════════════════════════

    async def _enrich_dashboard(self, ctx, extra):
        t = ctx.get("training_history", {})
        r = ctx.get("registry_info", {})
        p = ctx.get("pipeline_state", {})
        return {"view": "overview", "needs_guidance": not t.get("completed_jobs"),
                "has_models": r.get("total_registered", 0) > 0, "next_step": p.get("next_recommended_phase")}

    async def _enrich_ai_insights(self, ctx, extra):
        """Comprehensive enrichment for dedicated AI Insights screen — merges all screen contexts."""
        t = ctx.get("training_history", {}) or {}
        r = ctx.get("registry_info", {}) or {}
        p = ctx.get("pipeline_state", {}) or {}
        q = ctx.get("data_quality", {}) or {}
        v = ctx.get("model_versions", {}) or {}
        frontend = extra or {}

        return {
            "view": "ai_insights_hub",
            # Dashboard-level context
            "has_models": r.get("total_registered", 0) > 0,
            "needs_guidance": not t.get("completed_jobs"),
            "next_step": p.get("next_recommended_phase"),
            # Training context
            "target_column": frontend.get("target_column"),
            "algorithm": frontend.get("algorithm"),
            "selected_features": frontend.get("selected_features", []),
            "problem_type": frontend.get("problem_type", "classification"),
            "test_size": frontend.get("test_size", 0.2),
            "scaling_method": frontend.get("scaling_method", "standard"),
            "cv_folds": frontend.get("cv_folds", 5),
            "stratified": frontend.get("stratified", True),
            # Evaluation context
            "metrics": frontend.get("metrics", {}),
            "threshold": frontend.get("threshold", 0.5),
            "confusion_matrix": frontend.get("confusion_matrix"),
            "feature_importances": frontend.get("feature_importances"),
            "overall_score": frontend.get("overall_score"),
            "model_name": frontend.get("model_name"),
            # Frontend stats for dashboard rules
            "total_models": frontend.get("total_models", 0),
            "deployed_models": frontend.get("deployed_models", 0),
            "total_datasets": frontend.get("total_datasets", 0),
            "avg_accuracy": frontend.get("avg_accuracy"),
            "total_predictions": frontend.get("total_predictions", 0),
        }

    async def _enrich_data(self, ctx, extra):
        return {"view": "data_management", "uploaded_datasets": bool(ctx.get("dataset_id")), "active_tab": extra.get("active_tab", "overview")}

    async def _enrich_eda(self, ctx, extra):
        # Forward auto-detected target so rules/patterns can use it
        target = ctx.get("target_variable", {})
        target_name = target.get("name") if isinstance(target, dict) else None
        target_type = target.get("type") if isinstance(target, dict) else None

        # Also auto-detect from feature_stats if not already found
        if not target_name:
            fs = ctx.get("feature_stats", {})
            candidates = fs.get("potential_target_columns", [])
            if candidates:
                # Pick best candidate (binary > name pattern > last column)
                for c in candidates:
                    if c.get("type") == "binary_categorical" or c.get("type") == "name_pattern":
                        target_name = c.get("column")
                        break
                if not target_name:
                    target_name = candidates[0].get("column")

        return {
            "view": "exploratory_analysis",
            "target_column": extra.get("target_column") or target_name,
            "target_type": target_type,
            "target_detected": bool(target_name),
            "selected_feature": extra.get("selected_feature"),
            "active_tab": extra.get("active_tab", "overview"),
            "problem_type": extra.get("problem_type",
                                      "classification" if target_type in ("binary", "multiclass") else None),
        }

    async def _enrich_training(self, ctx, extra):
        return {
            "view": "model_training", "algorithm": extra.get("algorithm"),
            "target_column": extra.get("target_column"),
            "selected_features": extra.get("selected_features", []),
            "test_size": _sf(extra.get("test_size", 0.2), 0.2),
            "scaling_method": extra.get("scaling_method", "standard"),
            "problem_type": extra.get("problem_type", "classification"),
            "cv_folds": _si(extra.get("cv_folds", 5), 5),
            "selected_algorithms": extra.get("selected_algorithms", []),
            "stratified": extra.get("stratified", True),
        }

    async def _enrich_evaluation(self, ctx, extra):
        return {
            "view": "model_evaluation", "threshold": _sf(extra.get("threshold", 0.5), 0.5),
            "model_name": extra.get("model_name"), "algorithm": extra.get("algorithm"),
            "problem_type": extra.get("problem_type", "classification"),
            "metrics": extra.get("metrics", {}),
            "confusion_matrix": extra.get("confusion_matrix"),
            "overall_score": extra.get("overall_score"),
            "production_readiness": extra.get("production_readiness"),
            "feature_importances": extra.get("feature_importances"),
        }

    async def _enrich_registry(self, ctx, extra):
        return {"view": "model_registry", "selected_model_id": extra.get("selected_model_id")}

    async def _enrich_deployment(self, ctx, extra):
        return {"view": "deployment", "deployment_target": extra.get("deployment_target"), "deployment_mode": extra.get("deployment_mode")}

    async def _enrich_predictions(self, ctx, extra):
        return {"view": "predictions", "prediction_mode": extra.get("mode", "single")}

    async def _enrich_monitoring(self, ctx, extra):
        return {"view": "monitoring", "total_predictions": _si(extra.get("total_predictions", 0)),
                "error_rate": _sf(extra.get("error_rate", 0)), "avg_latency": _sf(extra.get("avg_latency", 0)),
                "drift_scores": extra.get("drift_scores"), "alerts": extra.get("alerts")}

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _get_dataset_model(self):
        try:
            from app.models.data_management import Dataset
            return Dataset
        except ImportError:
            pass
        try:
            from app.models.models import Dataset
            return Dataset
        except ImportError:
            return None

    @staticmethod
    def _build_corr_clusters(pairs: Dict[str, float]) -> List[Dict]:
        """Build clusters of mutually correlated features via BFS."""
        if not pairs:
            return []
        adj = {}
        for pk, val in pairs.items():
            if abs(_sf(val)) < 0.65:
                continue
            parts = pk.split("-", 1)
            if len(parts) != 2:
                continue
            f1, f2 = parts[0].strip(), parts[1].strip()
            adj.setdefault(f1, set()).add(f2)
            adj.setdefault(f2, set()).add(f1)
        if not adj:
            return []
        visited, clusters = set(), []
        for start in adj:
            if start in visited:
                continue
            cluster, queue = [], [start]
            while queue:
                n = queue.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                cluster.append(n)
                queue.extend(nb for nb in adj.get(n, []) if nb not in visited)
            if len(cluster) >= 2:
                cv = []
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        for k in [f"{cluster[i]}-{cluster[j]}", f"{cluster[j]}-{cluster[i]}"]:
                            if k in pairs:
                                cv.append(abs(_sf(pairs[k])))
                                break
                clusters.append({"features": sorted(cluster), "size": len(cluster), "avg_correlation": round(sum(cv) / len(cv), 3) if cv else 0})
        return sorted(clusters, key=lambda c: c["size"], reverse=True)