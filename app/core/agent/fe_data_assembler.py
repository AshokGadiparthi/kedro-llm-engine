"""
FE Data Assembler — Builds Analyzer Inputs from DB Data (No Logs Required)
===========================================================================

The FE Intelligence system's analyzers need two dicts: `config` and `results`.
Previously these ONLY came from parsing Kedro pipeline logs.

This module assembles them from 3 DB sources:
  1. EDA Results  → column names, types, statistics, cardinality, correlations
  2. Job History  → pipeline parameters (scaling_method, etc.)
  3. Model Versions → selected feature names, importances

It also INFERS the 5 fields that are normally only in logs:
  - encoding_details     → from EDA cardinality + pipeline config
  - variance_removed     → from comparing original vs final columns
  - id_columns_detected  → from cardinality + name patterns
  - intermediate shapes  → from column counts + encoding estimates
  - n_selected           → from model version feature count

Data Flow:
  DB Tables → FEDataAssembler.assemble() → (config, results, eda_data)
                                                    ↓
                                         Same analyzers as log path
                                                    ↓
                                         Full FE Intelligence output
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── ID Column Detection Patterns ───
_ID_PATTERNS = re.compile(
    r"(?:^id$|_id$|^id_|_id_|^pk$|_pk$|"
    r"^index$|_index$|^key$|_key$|^uuid$|"
    r"customerid|customer_id|user_id|userid|"
    r"record_id|row_id|transaction_id|order_id)",
    re.IGNORECASE,
)

# ─── Numeric Column Name Patterns (often misclassified as categorical) ───
_NUMERIC_NAME_PATTERNS = re.compile(
    r"(?:amount|price|cost|charge|fee|rate|salary|income|revenue|"
    r"score|count|total|sum|avg|mean|weight|height|age|"
    r"balance|payment|quantity|volume|distance|duration|"
    r"latitude|longitude|temp|temperature)",
    re.IGNORECASE,
)


def _safe_json(raw, default=None):
    """Parse JSON if string, return as-is if dict/list."""
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


class FEDataAssembler:
    """
    Assembles FE analyzer inputs (config, results, eda_data) from DB data.

    Usage:
        assembler = FEDataAssembler(db)
        config, results, eda_data = await assembler.assemble(
            dataset_id="...",
            project_id="...",
            job_id="...",
            model_id="...",
        )
        # Now pass to _run_full_analysis(config, results, eda_data, ...)
    """

    def __init__(self, db):
        self.db = db

    async def assemble(
            self,
            dataset_id: Optional[str] = None,
            project_id: Optional[str] = None,
            job_id: Optional[str] = None,
            model_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Assemble config, results, and eda_data from DB sources.

        Returns:
            (config, results, eda_data) — ready for _run_full_analysis()
        """

        # ── 1. Load raw DB data ──
        eda_raw = await self._load_eda(dataset_id)
        job_raw = await self._load_job(job_id, project_id)
        model_raw = await self._load_model_versions(project_id, model_id)

        # ── 2. Build config from Job parameters ──
        config = self._build_config(job_raw)

        # ── 3. Build results from EDA + Model Versions + inference ──
        results = self._build_results(eda_raw, model_raw, config)

        # ── 4. Build eda_data (pass-through for analyzers) ──
        eda_data = self._build_eda_data(eda_raw)

        # ── 5. Infer missing fields ──
        self._infer_encoding_details(results, eda_raw, config)
        self._infer_id_columns(results, eda_raw)
        self._infer_variance_removed(results, model_raw, eda_raw)
        self._infer_shapes(results, eda_raw, config)

        data_sources = {
            "eda_available": bool(eda_raw),
            "job_available": bool(job_raw),
            "model_versions_available": bool(model_raw.get("versions")),
            "assembly_mode": "db_inferred",
        }
        results["_data_sources"] = data_sources

        logger.info(
            f"[FE-Assembler] Assembled from DB: "
            f"cols={len(results.get('original_columns', []))}, "
            f"numeric={len(results.get('numeric_features', []))}, "
            f"categorical={len(results.get('categorical_features', []))}, "
            f"selected={len(results.get('selected_features', []))}, "
            f"sources={data_sources}"
        )

        return config, results, eda_data

    # ═══════════════════════════════════════════════════════════════
    # DB LOADERS
    # ═══════════════════════════════════════════════════════════════

    async def _load_eda(self, dataset_id: Optional[str]) -> Dict[str, Any]:
        """Load EDA results from database."""
        if not dataset_id or not self.db:
            return {}
        try:
            from app.models.models import EdaResult
            eda = self.db.query(EdaResult).filter(
                EdaResult.dataset_id == dataset_id
            ).first()
            if not eda:
                return {}
            result = {}
            for field in ["summary", "statistics", "quality", "correlations"]:
                raw = getattr(eda, field, None)
                if raw:
                    result[field] = _safe_json(raw, {})
            return result
        except Exception as e:
            logger.warning(f"[FE-Assembler] EDA load failed: {e}")
            return {}

    async def _load_job(
            self, job_id: Optional[str], project_id: Optional[str]
    ) -> Dict[str, Any]:
        """Load the most relevant FE job from database."""
        if not self.db:
            return {}
        try:
            from app.models.models import Job
            job = None

            # Priority 1: specific job_id
            if job_id:
                job = self.db.query(Job).filter(Job.id == job_id).first()

            # Priority 2: latest FE job for project
            if not job and project_id:
                job = (
                    self.db.query(Job)
                    .filter(
                        Job.pipeline_name == "feature_engineering",
                        Job.status == "completed",
                        )
                    .order_by(Job.completed_at.desc())
                    .first()
                )

            # Priority 3: latest FE job globally
            if not job:
                job = (
                    self.db.query(Job)
                    .filter(
                        Job.pipeline_name == "feature_engineering",
                        Job.status == "completed",
                        )
                    .order_by(Job.completed_at.desc())
                    .first()
                )

            if not job:
                return {}

            params = _safe_json(getattr(job, "parameters", None), {})
            results = _safe_json(getattr(job, "results", None), {})

            return {
                "job_id": str(getattr(job, "id", "")),
                "parameters": params,
                "results": results,
                "execution_time": getattr(job, "execution_time", None),
                "status": getattr(job, "status", ""),
                "completed_at": str(getattr(job, "completed_at", "")),
            }
        except Exception as e:
            logger.warning(f"[FE-Assembler] Job load failed: {e}")
            return {}

    async def _load_model_versions(
            self, project_id: Optional[str], model_id: Optional[str]
    ) -> Dict[str, Any]:
        """Load model versions for feature names and importances."""
        if not self.db:
            return {"versions": []}
        try:
            from app.models.models import ModelVersion
            query = self.db.query(ModelVersion)

            if model_id:
                query = query.filter(ModelVersion.model_id == model_id)

            versions = query.order_by(ModelVersion.created_at.desc()).limit(5).all()

            version_list = []
            for v in versions:
                version_list.append({
                    "feature_names": _safe_json(getattr(v, "feature_names", None), []),
                    "feature_importances": _safe_json(
                        getattr(v, "feature_importances", None), {}
                    ),
                    "algorithm": getattr(v, "algorithm", ""),
                    "accuracy": getattr(v, "accuracy", None),
                    "version": getattr(v, "version", ""),
                })

            return {"versions": version_list}
        except Exception as e:
            logger.warning(f"[FE-Assembler] Model versions load failed: {e}")
            return {"versions": []}

    # ═══════════════════════════════════════════════════════════════
    # BUILDERS
    # ═══════════════════════════════════════════════════════════════

    def _build_config(self, job_raw: Dict) -> Dict[str, Any]:
        """Build config dict from job parameters."""
        params = job_raw.get("parameters", {})
        # Kedro stores params nested: {feature_engineering: {scaling_method: ...}}
        fe_params = params.get("feature_engineering", params)

        return {
            "scaling_method": fe_params.get("scaling_method", "standard"),
            "handle_missing_values": fe_params.get("handle_missing_values", True),
            "handle_outliers": fe_params.get("handle_outliers", True),
            "encode_categories": fe_params.get("encode_categories", True),
            "create_polynomial_features": fe_params.get(
                "create_polynomial_features", False
            ),
            "create_interactions": fe_params.get("create_interactions", False),
            "variance_threshold": fe_params.get("variance_threshold", 0.01),
            "n_features_to_select": fe_params.get("n_features_to_select"),
        }

    def _build_results(
            self, eda_raw: Dict, model_raw: Dict, config: Dict
    ) -> Dict[str, Any]:
        """Build results dict from EDA + model data."""
        summary = eda_raw.get("summary", {})
        statistics = eda_raw.get("statistics", {})
        column_types = summary.get("column_types", {})

        # Original columns from EDA
        original_columns = summary.get("columns", [])
        if not original_columns and statistics:
            original_columns = list(statistics.keys())

        # Column types from EDA
        numeric_features = column_types.get("numeric", [])
        categorical_features = column_types.get("categorical", [])
        datetime_features = column_types.get("datetime", [])
        boolean_features = column_types.get("boolean", [])

        # If column_types is empty, infer from statistics dtypes
        if not numeric_features and not categorical_features:
            dtypes = summary.get("dtypes", {})
            for col, dtype in dtypes.items():
                dtype_str = str(dtype).lower()
                if any(t in dtype_str for t in ["int", "float", "number"]):
                    numeric_features.append(col)
                elif any(t in dtype_str for t in ["object", "str", "category", "bool"]):
                    categorical_features.append(col)

        # Selected features from model versions
        selected_features = []
        versions = model_raw.get("versions", [])
        if versions:
            # Use the latest version's feature names
            for v in versions:
                feat_names = v.get("feature_names", [])
                if feat_names:
                    selected_features = feat_names
                    break

        # Execution time from job
        execution_time = None

        return {
            "original_columns": original_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "datetime_features": datetime_features,
            "boolean_features": boolean_features,
            "selected_features": selected_features,
            "id_columns_detected": [],  # Will be inferred
            "variance_removed": [],  # Will be inferred
            "encoding_details": {},  # Will be inferred
            "original_shape": summary.get("shape"),
            "train_shape": None,  # Will be inferred
            "test_shape": None,  # Will be inferred
            "n_rows": summary.get("shape", [0])[0] if summary.get("shape") else None,
            "n_selected": len(selected_features) if selected_features else None,
            "execution_time_seconds": execution_time,
        }

    def _build_eda_data(self, eda_raw: Dict) -> Dict[str, Any]:
        """Build eda_data dict for the analyzers."""
        return {
            "statistics": eda_raw.get("statistics", {}),
            "correlations": eda_raw.get("correlations", {}),
            "quality": eda_raw.get("quality", {}),
            "summary": eda_raw.get("summary", {}),
        }

    # ═══════════════════════════════════════════════════════════════
    # INFERENCE ENGINES (the magic — fills gaps without logs)
    # ═══════════════════════════════════════════════════════════════

    def _infer_id_columns(self, results: Dict, eda_raw: Dict):
        """
        Detect ID columns from EDA statistics.

        Logic:
          - Column name matches ID patterns (customerID, user_id, pk, etc.)
          - High cardinality (>80% unique values)
          - Both conditions → definitely an ID
          - Name pattern alone → likely an ID
        """
        statistics = eda_raw.get("statistics", {})
        summary = eda_raw.get("summary", {})
        n_rows = (summary.get("shape", [0]) or [0])[0]
        original_cols = results.get("original_columns", [])

        id_columns = []
        for col in original_cols:
            is_id = False
            reasons = []

            # Check name pattern
            if _ID_PATTERNS.search(col):
                reasons.append("id_by_name")
                is_id = True

            # Check cardinality
            col_stats = statistics.get(col, {})
            unique_count = _safe_int(
                col_stats.get("unique")
                or col_stats.get("unique_count")
                or col_stats.get("nunique"),
                0,
                )
            if n_rows > 0 and unique_count > 0:
                cardinality_pct = unique_count / n_rows
                if cardinality_pct > 0.8:
                    reasons.append(f"high_cardinality_{cardinality_pct:.0%}")
                    if _ID_PATTERNS.search(col):
                        is_id = True  # Name + high cardinality = definitely ID
                    elif cardinality_pct > 0.95:
                        is_id = True  # Very high cardinality alone = likely ID

            if is_id:
                id_columns.append(col)

        results["id_columns_detected"] = id_columns
        if id_columns:
            logger.info(f"[FE-Assembler] Inferred ID columns: {id_columns}")

    def _infer_encoding_details(
            self, results: Dict, eda_raw: Dict, config: Dict
    ):
        """
        Infer encoding decisions from EDA statistics.

        Simulates the pipeline's encoding logic:
          - <10 unique → one-hot encoding
          - 10-1000 unique → label encoding
          - >1000 unique → dropped or hashed
          - Rare category grouping if <1% frequency
        """
        if not config.get("encode_categories", True):
            return

        statistics = eda_raw.get("statistics", {})
        summary = eda_raw.get("summary", {})
        n_rows = (summary.get("shape", [0]) or [0])[0]
        categorical_cols = results.get("categorical_features", [])
        id_columns = results.get("id_columns_detected", [])

        encoding_details = {}
        total_encoding_features = 0

        for col in categorical_cols:
            if col in id_columns:
                continue  # Skip ID columns

            col_stats = statistics.get(col, {})
            unique_count = _safe_int(
                col_stats.get("unique")
                or col_stats.get("unique_count")
                or col_stats.get("nunique"),
                0,
                )

            # Detect possible type misclassification
            is_numeric_name = bool(_NUMERIC_NAME_PATTERNS.search(col))

            # Simulate rare category grouping
            rare_threshold = 0.01  # 1%
            min_frequency = n_rows * rare_threshold if n_rows > 0 else 0
            # Estimate: most categories in high-cardinality columns are rare
            estimated_after_rare = min(unique_count, max(1, int(n_rows * rare_threshold))) if unique_count > 10 else unique_count

            # Determine encoding strategy
            if unique_count <= 10:
                strategy = "one_hot"
                n_features = unique_count - 1  # drop_first
                rare_grouped = 0
            elif unique_count <= 1000:
                strategy = "label_encoding"
                n_features = 1
                rare_grouped = max(0, unique_count - estimated_after_rare)
            else:
                # High cardinality — would be collapsed
                strategy = "one_hot_with_rare_grouping"
                rare_grouped = unique_count - estimated_after_rare
                n_features = max(1, estimated_after_rare)

            total_encoding_features += n_features

            encoding_details[col] = {
                "strategy": strategy,
                "unique_values": unique_count,
                "n_features_created": n_features,
                "rare_grouped": rare_grouped,
                "is_numeric_by_name": is_numeric_name,
                "inferred": True,  # Flag that this was inferred, not from logs
            }

        results["encoding_details"] = encoding_details
        results["total_encoding_features"] = total_encoding_features

    def _infer_variance_removed(
            self, results: Dict, model_raw: Dict, eda_raw: Dict
    ):
        """
        Infer which features were removed by variance filter.

        Logic:
          - Binary/boolean columns scaled to ~0/1 have variance ~0.25
          - After StandardScaler, binary variance can drop below threshold
          - If original column is binary AND not in selected_features → likely removed
        """
        statistics = eda_raw.get("statistics", {})
        original_cols = results.get("original_columns", [])
        selected_features = results.get("selected_features", [])

        if not selected_features:
            return  # Can't infer without knowing final features

        # Build set of base names from selected features
        # Selected features may have suffixes like _scaled, _Yes, _One year
        selected_bases = set()
        for f in selected_features:
            selected_bases.add(f)
            # Strip common suffixes to get base name
            for suffix in ["_scaled", "_encoded", "_binary"]:
                if f.endswith(suffix):
                    selected_bases.add(f[: -len(suffix)])

        variance_removed = []
        for col in original_cols:
            col_stats = statistics.get(col, {})
            unique = _safe_int(
                col_stats.get("unique") or col_stats.get("nunique"), 0
            )
            dtype = str(col_stats.get("dtype", "")).lower()

            # Check if binary column
            is_binary = unique == 2 or (unique <= 2 and "int" in dtype)

            # Check if this column (or its scaled version) appears in selected
            col_lower = col.lower()
            in_selected = any(
                col_lower in s.lower() or s.lower().startswith(col_lower)
                for s in selected_features
            )

            if is_binary and not in_selected:
                variance_removed.append(f"{col}_scaled")

        results["variance_removed"] = variance_removed
        if variance_removed:
            results["features_before_variance"] = (
                    len(results.get("numeric_features", []))
                    + results.get("total_encoding_features", 0)
            )
            results["features_after_variance"] = (
                    results["features_before_variance"] - len(variance_removed)
            )

    def _infer_shapes(self, results: Dict, eda_raw: Dict, config: Dict):
        """Infer training/test shapes from EDA shape + standard split ratio."""
        summary = eda_raw.get("summary", {})
        shape = summary.get("shape", [])

        if not shape or len(shape) < 2:
            return

        n_rows, n_cols = shape[0], shape[1]

        # Standard 80/20 train/test split
        train_rows = int(n_rows * 0.8)
        test_rows = n_rows - train_rows

        # Estimate final feature count
        n_numeric = len(results.get("numeric_features", []))
        n_encoding = results.get("total_encoding_features", 0)
        n_variance_removed = len(results.get("variance_removed", []))
        n_id = len(results.get("id_columns_detected", []))

        estimated_features = n_numeric + n_encoding - n_variance_removed - n_id
        estimated_features = max(estimated_features, 1)

        # Use selected features count if available
        selected = results.get("selected_features", [])
        if selected:
            final_features = len(selected)
        else:
            final_features = estimated_features

        results["original_shape"] = results.get("original_shape") or [n_rows, n_cols]
        results["train_shape"] = results.get("train_shape") or [train_rows, final_features]
        results["test_shape"] = results.get("test_shape") or [test_rows, final_features]
        results["n_rows"] = results.get("n_rows") or n_rows
        if not results.get("features_before_variance"):
            results["features_before_variance"] = n_numeric + n_encoding
        if not results.get("features_after_variance"):
            results["features_after_variance"] = (
                    results["features_before_variance"] - n_variance_removed
            )
        if not results.get("n_selected"):
            results["n_selected"] = final_features
        if not results.get("features_input_to_selection"):
            results["features_input_to_selection"] = results.get(
                "features_after_variance", estimated_features
            )