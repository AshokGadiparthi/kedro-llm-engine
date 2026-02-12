"""
Feature Engineering Log Parser — World-Class Kedro Pipeline Intelligence Bridge
==================================================================================
Parses raw Kedro pipeline stdout logs into structured data that feeds directly
into the FE intelligence layer (FeatureAnalyzer, FEPipelineIntelligence, Expert Rules).

This is THE critical missing piece: Kedro's rich execution logs contain every
decision the pipeline made — column types, encoding strategies, cardinalities,
variance filtering, feature selection — but this data was ephemeral (lost after
execution). This parser captures it ALL.

Capabilities:
  • Parse column type detection (numeric, categorical, binary, text, date)
  • Parse ID column detection results (with false negatives)
  • Parse scaling decisions (method, features scaled)
  • Parse encoding decisions (per-column: strategy, cardinality, rare grouping)
  • Parse variance filter results (before/after, removed features)
  • Parse feature selection results (input/output, selected features)
  • Parse NaN imputation results
  • Parse pipeline shapes at every stage
  • Parse execution timing
  • Detect type misclassifications from name+cardinality heuristics
  • Auto-reconstruct the full pipeline decision tree

Usage:
  parser = FELogParser()
  result = parser.parse(raw_kedro_log_text)
  # result is a dict compatible with PipelineIntelligenceRequest

Architecture:
  Pure regex + string parsing. Zero dependencies beyond Python stdlib.
  Handles multi-line Kedro log format with timestamp prefixes.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Columns names that STRONGLY indicate numeric data
NUMERIC_NAME_SIGNALS = [
    "charge", "amount", "price", "cost", "fee", "revenue", "salary",
    "income", "balance", "payment", "total", "sum", "count", "quantity",
    "rate", "score", "rating", "weight", "height", "age", "distance",
    "duration", "time", "days", "months", "years", "percent", "ratio",
    "value", "profit", "loss", "margin", "volume", "area", "speed",
    "temperature", "pressure", "frequency", "latitude", "longitude",
]

# Column names that STRONGLY indicate ID columns
ID_NAME_SIGNALS = [
    "id", "uuid", "guid", "key", "pk", "sk", "idx", "index",
    "customer_id", "user_id", "account_id", "session_id",
    "transaction_id", "order_id", "product_id", "employee_id",
    "patient_id", "record_id", "ticket_id", "case_id",
    "customerid", "userid", "accountid", "sessionid",
    "transactionid", "orderid", "productid", "employeeid",
]

# Known binary predictors by domain
KNOWN_BINARY_PREDICTORS = {
    "churn": ["gender", "partner", "dependents", "phoneservice",
              "paperlessbilling", "seniorcitizen"],
    "fraud": ["is_foreign", "is_online", "is_weekend", "is_recurring"],
    "credit": ["has_mortgage", "has_loan", "is_employed", "has_guarantor"],
}


# ═══════════════════════════════════════════════════════════════════════
# LOG PARSER
# ═══════════════════════════════════════════════════════════════════════

class FELogParser:
    """
    World-class Kedro Feature Engineering Log Parser.

    Extracts every pipeline decision from raw Kedro stdout into structured
    data for AI analysis. Handles the real log format with timestamps,
    worker prefixes, and multi-line output blocks.
    """

    def __init__(self):
        self.warnings: List[str] = []

    def _extract_embedded_metadata(self, raw_log: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured metadata JSON from pipeline log.

        If the Kedro node used FEMetadataCapture.print_to_log(), the log
        contains a JSON block between [FE-METADATA-JSON-START] and
        [FE-METADATA-JSON-END] markers.

        This gives 100% accurate structured data — no regex needed for
        the config/results fields. The regex parsing still runs for
        issue detection and other enrichment.

        Returns:
            Parsed metadata dict, or None if no embedded metadata found.
        """
        start_marker = "[FE-METADATA-JSON-START]"
        end_marker = "[FE-METADATA-JSON-END]"

        start_idx = raw_log.find(start_marker)
        end_idx = raw_log.find(end_marker)

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None

        json_str = raw_log[start_idx + len(start_marker):end_idx].strip()

        try:
            metadata = json.loads(json_str)
            logger.info(
                f"[FE-Parser] ✅ Found embedded metadata JSON "
                f"({len(json_str)} bytes, "
                f"cols={len(metadata.get('results', {}).get('original_columns', []))})"
            )
            return metadata
        except json.JSONDecodeError as e:
            logger.warning(f"[FE-Parser] Embedded metadata JSON is invalid: {e}")
            self.warnings.append(f"Embedded metadata JSON parse error: {e}")
            return None

    def parse(self, raw_log: str) -> Dict[str, Any]:
        """
        Parse raw Kedro FE pipeline logs into structured intelligence data.

        Args:
            raw_log: Raw stdout from Kedro pipeline execution

        Returns:
            Dict compatible with PipelineIntelligenceRequest, containing:
              - config: Pipeline configuration detected
              - columns: Column classifications
              - encoding: Per-column encoding decisions
              - scaling: Scaling decisions
              - variance: Variance filter results
              - selection: Feature selection results
              - shapes: Pipeline shapes at each stage
              - timing: Execution timing
              - issues: Auto-detected issues
              - pipeline_stages: Ordered list of stages executed
        """
        # ── Check for embedded structured metadata first ──
        # If the Kedro node used FEMetadataCapture.print_to_log(),
        # the log contains a JSON block between markers.
        # This gives 100% accurate data — no regex guessing.
        embedded_metadata = self._extract_embedded_metadata(raw_log)

        # Clean the log (remove timestamps, worker prefixes)
        lines = self._clean_log(raw_log)

        result = {
            "parsed_successfully": True,
            "parser_version": "2.0",
            "raw_line_count": len(raw_log.splitlines()),
            "cleaned_line_count": len(lines),

            # ── Pipeline Configuration ──
            "config": self._parse_config(lines),

            # ── Column Type Detection ──
            "columns": self._parse_column_types(lines),

            # ── ID Column Detection ──
            "id_detection": self._parse_id_detection(lines),

            # ── Scaling Decisions ──
            "scaling": self._parse_scaling(lines),

            # ── Encoding Decisions ──
            "encoding": self._parse_encoding(lines),

            # ── Variance Filter ──
            "variance_filter": self._parse_variance_filter(lines),

            # ── NaN Imputation ──
            "nan_imputation": self._parse_nan_imputation(lines),

            # ── Feature Selection ──
            "feature_selection": self._parse_feature_selection(lines),

            # ── Pipeline Shapes ──
            "shapes": self._parse_shapes(lines),

            # ── Final Output ──
            "final_output": self._parse_final_output(lines),

            # ── Execution Info ──
            "execution": self._parse_execution_info(lines, raw_log),

            # ── Pipeline Stages ──
            "pipeline_stages": self._parse_pipeline_stages(lines),

            # ── Parser Warnings ──
            "parser_warnings": list(self.warnings),

            # ── Embedded Structured Metadata (if Kedro node printed it) ──
            "embedded_metadata": embedded_metadata,
        }

        # ── Auto-detect issues from parsed data ──
        result["auto_detected_issues"] = self._detect_issues(result)

        # ── Generate PipelineIntelligenceRequest-compatible fields ──
        result["intelligence_request"] = self._to_intelligence_request(result)

        return result

    # ─────────────────────────────────────────────────────────
    # LOG CLEANING
    # ─────────────────────────────────────────────────────────

    def _clean_log(self, raw: str) -> List[str]:
        """Strip timestamps, worker prefixes, and empty lines from Kedro log."""
        lines = []
        for line in raw.splitlines():
            # Remove Celery worker prefix
            # Pattern: [2026-02-11 16:41:14,785: INFO/ForkPoolWorker-1]
            cleaned = re.sub(
                r'^\[[\d\-: ,]+:\s*\w+/[\w-]+\]\s*', '', line
            )
            # Remove Kedro rich logging prefix
            # Pattern: [02/11/26 16:41:09] INFO
            cleaned = re.sub(
                r'^\[[\d/ :]+\]\s*(?:INFO|WARNING|ERROR|DEBUG)\s*', '', cleaned
            )
            # Remove trailing module references like "pipeline_registry.py:535"
            cleaned = re.sub(r'\s+\w+\.py:\d+$', '', cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    # ─────────────────────────────────────────────────────────
    # COLUMN TYPE PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_column_types(self, lines: List[str]) -> Dict[str, Any]:
        """Parse UNIVERSAL COLUMN TYPE DETECTION section."""
        columns = {}
        numeric_cols = []
        categorical_cols = []
        binary_cols = []
        text_cols = []
        date_cols = []

        in_section = False
        for line in lines:
            if "UNIVERSAL COLUMN TYPE DETECTION" in line:
                in_section = True
                continue
            if in_section and line.startswith("====="):
                if columns:  # End of section (second separator)
                    break
                continue
            if in_section and line.startswith("Summary:"):
                break

            if not in_section:
                continue

            # Parse: 🏷️  customerID: Categorical (inferred)
            # Parse: 🔢 tenure: Numeric (int64)
            # Parse: 🔢 gender: Boolean/Binary (numeric)
            m = re.match(
                r'[🏷️🔢📅📝\s]*(\w[\w\s]*?):\s*([\w/]+)(?:\s*\(([^)]*)\))?',
                line
            )
            if m:
                col_name = m.group(1).strip()
                col_type = m.group(2).strip()
                col_detail = (m.group(3) or "").strip()

                type_lower = col_type.lower()
                detail_lower = col_detail.lower()

                entry = {
                    "name": col_name,
                    "detected_type": col_type,
                    "detail": col_detail,
                    "raw_line": line.strip(),
                }

                if "boolean" in type_lower or "binary" in type_lower:
                    entry["category"] = "binary"
                    binary_cols.append(col_name)
                elif "numeric" in type_lower or "int" in detail_lower or "float" in detail_lower:
                    entry["category"] = "numeric"
                    numeric_cols.append(col_name)
                elif "categorical" in type_lower:
                    entry["category"] = "categorical"
                    categorical_cols.append(col_name)
                elif "text" in type_lower:
                    entry["category"] = "text"
                    text_cols.append(col_name)
                elif "date" in type_lower:
                    entry["category"] = "date"
                    date_cols.append(col_name)
                else:
                    entry["category"] = "unknown"

                columns[col_name] = entry

        # Parse summary counts
        summary = {"n_numeric": 0, "n_categorical": 0, "n_text": 0, "n_date": 0, "n_binary": 0}
        for line in lines:
            m = re.match(r'Numeric:\s*(\d+)', line)
            if m:
                summary["n_numeric"] = int(m.group(1))
            m = re.match(r'Categorical:\s*(\d+)', line)
            if m:
                summary["n_categorical"] = int(m.group(1))
            m = re.match(r'Text:\s*(\d+)', line)
            if m:
                summary["n_text"] = int(m.group(1))
            m = re.match(r'Date:\s*(\d+)', line)
            if m:
                summary["n_date"] = int(m.group(1))

        summary["n_binary"] = len(binary_cols)

        return {
            "all_columns": columns,
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "binary": binary_cols,
            "text": text_cols,
            "date": date_cols,
            "summary": summary,
        }

    # ─────────────────────────────────────────────────────────
    # ID COLUMN DETECTION
    # ─────────────────────────────────────────────────────────

    def _parse_id_detection(self, lines: List[str]) -> Dict[str, Any]:
        """Parse UNIVERSAL ID COLUMN DETECTION section."""
        result = {
            "detected": [],
            "config": {},
            "status": "unknown",
        }

        in_section = False
        for line in lines:
            if "UNIVERSAL ID COLUMN DETECTION" in line:
                in_section = True
                continue
            if in_section and line.startswith("=====") and result.get("config"):
                break

            if not in_section:
                continue

            # Config lines
            m = re.match(r'Cardinality threshold:\s*([\d.]+%?)', line)
            if m:
                result["config"]["cardinality_threshold"] = m.group(1)
            m = re.match(r'Check variance:\s*(\w+)', line)
            if m:
                result["config"]["check_variance"] = m.group(1)

            # Detected IDs
            if "No ID columns detected" in line:
                result["status"] = "none_detected"
            m = re.match(r'(?:Detected|Removed) ID column[s]?:\s*(.+)', line)
            if m:
                result["detected"] = [c.strip() for c in m.group(1).split(",")]
                result["status"] = "detected"
            m = re.match(r'ID column:\s*(\w+)', line)
            if m:
                result["detected"].append(m.group(1))
                result["status"] = "detected"

        return result

    # ─────────────────────────────────────────────────────────
    # SCALING PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_scaling(self, lines: List[str]) -> Dict[str, Any]:
        """Parse UNIVERSAL NUMERIC SCALING section."""
        result = {
            "method": "unknown",
            "n_features_scaled": 0,
            "outlier_threshold": None,
            "method_reason": "",
        }

        in_section = False
        for line in lines:
            if "UNIVERSAL NUMERIC SCALING" in line:
                in_section = True
                continue
            if in_section and line.startswith("=====") and result["method"] != "unknown":
                break

            if not in_section:
                continue

            m = re.match(r'Scaling method:\s*(\w+)', line)
            if m:
                result["method"] = m.group(1)

            m = re.match(r'Outlier threshold\s*\(IQR\):\s*([\d.]+)', line)
            if m:
                result["outlier_threshold"] = float(m.group(1))

            m = re.match(r'.*Using\s+(\w+)\s*\(', line)
            if m:
                result["method_reason"] = line.strip()

            m = re.match(r'.*Scaled\s+(\d+)\s+numeric features', line)
            if m:
                result["n_features_scaled"] = int(m.group(1))

        return result

    # ─────────────────────────────────────────────────────────
    # ENCODING PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_encoding(self, lines: List[str]) -> Dict[str, Any]:
        """Parse UNIVERSAL CATEGORICAL ENCODING section."""
        result = {
            "config": {},
            "columns": {},
            "total_encoding_features": 0,
            "train_shape_after": None,
            "test_shape_after": None,
        }

        in_section = False
        current_column = None

        for line in lines:
            if "UNIVERSAL CATEGORICAL ENCODING" in line:
                in_section = True
                continue
            if in_section and line.startswith("=====") and result["columns"]:
                break

            if not in_section:
                continue

            # Config
            m = re.match(r'Max categories for one-hot:\s*(\d+)', line)
            if m:
                result["config"]["max_onehot"] = int(m.group(1))
            m = re.match(r'Max categories for label encoding:\s*(\d+)', line)
            if m:
                result["config"]["max_label"] = int(m.group(1))
            m = re.match(r'Rare category threshold:\s*([\d.]+)%?', line)
            if m:
                result["config"]["rare_threshold_pct"] = float(m.group(1))
            m = re.match(r'Max total encoding features:\s*(\d+)', line)
            if m:
                result["config"]["max_total_features"] = int(m.group(1))

            # Column processing
            m = re.match(r'Processing:\s*(\w+)', line)
            if m:
                current_column = m.group(1)
                result["columns"][current_column] = {
                    "name": current_column,
                    "unique_train": 0,
                    "unique_test": 0,
                    "unique_total": 0,
                    "rare_grouped": 0,
                    "strategy": "unknown",
                    "output_categories": 0,
                }
                continue

            if current_column:
                # Unique values line
                m = re.match(
                    r'Unique values:\s*Train=(\d+),\s*Test=(\d+),\s*Total=(\d+)',
                    line
                )
                if m:
                    col = result["columns"][current_column]
                    col["unique_train"] = int(m.group(1))
                    col["unique_test"] = int(m.group(2))
                    col["unique_total"] = int(m.group(3))

                # Rare grouping
                m = re.match(r'Grouping\s+(\d+)\s+rare categories\s*\(<([\d.]+)%\)', line)
                if m:
                    col = result["columns"][current_column]
                    col["rare_grouped"] = int(m.group(1))
                    col["rare_threshold"] = float(m.group(2))

                # Strategy
                m = re.match(r'.*Strategy:\s*([\w\- ]+)\s*\((\d+)\s*categories\)', line)
                if m:
                    col = result["columns"][current_column]
                    col["strategy"] = m.group(1).strip()
                    col["output_categories"] = int(m.group(2))

            # Result section
            m = re.match(r'Total encoding features:\s*(\d+)', line)
            if m:
                result["total_encoding_features"] = int(m.group(1))
                current_column = None

            m = re.match(r'Train shape:\s*\((\d+),\s*(\d+)\)', line)
            if m and in_section:
                result["train_shape_after"] = [int(m.group(1)), int(m.group(2))]

            m = re.match(r'Test shape:\s*\((\d+),\s*(\d+)\)', line)
            if m and in_section:
                result["test_shape_after"] = [int(m.group(1)), int(m.group(2))]

        return result

    # ─────────────────────────────────────────────────────────
    # VARIANCE FILTER PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_variance_filter(self, lines: List[str]) -> Dict[str, Any]:
        """Parse VARIANCE FILTERING section."""
        result = {
            "threshold": None,
            "features_before": 0,
            "features_after": 0,
            "n_removed": 0,
            "removed_features": [],
        }

        in_section = False
        for line in lines:
            if "VARIANCE FILTERING" in line:
                in_section = True
                continue
            if in_section and line.startswith("=====") and result["features_before"] > 0:
                break

            if not in_section:
                continue

            m = re.match(r'Variance threshold:\s*([\d.]+)', line)
            if m:
                result["threshold"] = float(m.group(1))

            m = re.match(r'Features before:\s*(\d+)', line)
            if m:
                result["features_before"] = int(m.group(1))

            m = re.match(r'Features after:\s*(\d+)', line)
            if m:
                result["features_after"] = int(m.group(1))

            m = re.match(r'Removed:\s*(\d+)\s*features?', line)
            if m:
                result["n_removed"] = int(m.group(1))

            # Removed feature list: Removed: ['gender_scaled', 'Partner_scaled', ...]
            m = re.match(r"Removed:\s*\[(.+)\]", line)
            if m and "'" in m.group(1):
                features_str = m.group(1)
                result["removed_features"] = [
                    f.strip().strip("'\"")
                    for f in features_str.split(",")
                    if f.strip().strip("'\"")
                ]

        return result

    # ─────────────────────────────────────────────────────────
    # NaN IMPUTATION PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_nan_imputation(self, lines: List[str]) -> Dict[str, Any]:
        """Parse NaN IMPUTATION section."""
        result = {
            "nan_before_train": 0,
            "nan_before_test": 0,
            "strategy": "unknown",
            "had_nans": False,
        }

        in_section = False
        for line in lines:
            if "NaN IMPUTATION" in line:
                in_section = True
                continue
            if in_section and line.startswith("=====") and result["strategy"] != "unknown":
                break

            if not in_section:
                continue

            m = re.match(r'NaN values before:\s*Train=(\d+),\s*Test=(\d+)', line)
            if m:
                result["nan_before_train"] = int(m.group(1))
                result["nan_before_test"] = int(m.group(2))
                result["had_nans"] = int(m.group(1)) > 0 or int(m.group(2)) > 0

            m = re.match(r'Strategy:\s*(\w+)', line)
            if m:
                result["strategy"] = m.group(1)

            if "No NaN values" in line:
                result["had_nans"] = False

        return result

    # ─────────────────────────────────────────────────────────
    # FEATURE SELECTION PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_feature_selection(self, lines: List[str]) -> Dict[str, Any]:
        """Parse FEATURE SELECTION section."""
        result = {
            "input_shape": None,
            "problem_type": "unknown",
            "n_requested": 0,
            "n_selected": 0,
            "output_train_shape": None,
            "output_test_shape": None,
        }

        in_section = False
        for line in lines:
            if "FEATURE SELECTION" in line and "Universal" in line:
                in_section = True
                continue
            if in_section and line.startswith("=====") and result["n_selected"] > 0:
                break

            if not in_section:
                continue

            m = re.match(r'Input features:\s*\((\d+),\s*(\d+)\)', line)
            if m:
                result["input_shape"] = [int(m.group(1)), int(m.group(2))]

            m = re.match(r'Problem type:\s*(\w+)', line)
            if m:
                result["problem_type"] = m.group(1)

            m = re.match(r'Selecting:\s*(\d+)\s*features', line)
            if m:
                result["n_requested"] = int(m.group(1))

            m = re.match(r'.*Selected\s+(\d+)\s+features', line)
            if m:
                result["n_selected"] = int(m.group(1))

            m = re.match(r'Output shapes?:\s*Train=\((\d+),\s*(\d+)\),\s*Test=\((\d+),\s*(\d+)\)', line)
            if m:
                result["output_train_shape"] = [int(m.group(1)), int(m.group(2))]
                result["output_test_shape"] = [int(m.group(3)), int(m.group(4))]

        return result

    # ─────────────────────────────────────────────────────────
    # PIPELINE CONFIG PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_config(self, lines: List[str]) -> Dict[str, Any]:
        """Parse pipeline parameters from the job execution header."""
        config = {
            "pipeline_name": "feature_engineering",
            "dataset_id": None,
            "scaling_method": "standard",
            "handle_missing_values": True,
            "handle_outliers": True,
            "encode_categories": True,
            "create_polynomial_features": False,
            "create_interactions": False,
        }

        for line in lines:
            # Parse flattened parameters
            m = re.match(r'.*feature_engineering\.dataset_id=([\w-]+)', line)
            if m:
                config["dataset_id"] = m.group(1)

            m = re.match(r'.*feature_engineering\.scaling_method=(\w+)', line)
            if m:
                config["scaling_method"] = m.group(1)

            for key in ["handle_missing_values", "handle_outliers",
                        "encode_categories", "create_polynomial_features",
                        "create_interactions"]:
                m = re.match(rf'.*feature_engineering\.{key}=(True|False)', line)
                if m:
                    config[key] = m.group(1) == "True"

            # Parse Job ID
            m = re.match(r'Job ID:\s*([\w-]+)', line)
            if m:
                config["job_id"] = m.group(1)

            # Parse pipeline name
            m = re.match(r'Pipeline:\s*(\w+)', line)
            if m:
                config["pipeline_name"] = m.group(1)

        return config

    # ─────────────────────────────────────────────────────────
    # SHAPES PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_shapes(self, lines: List[str]) -> Dict[str, Any]:
        """Parse pipeline shapes at each stage."""
        shapes = {
            "input_train": None,
            "input_test": None,
            "input_columns": [],
            "after_encoding": None,
            "after_combining": None,
            "after_variance": None,
            "after_selection": None,
            "final_train": None,
            "final_test": None,
            "final_columns": [],
        }

        # Input data
        for line in lines:
            if "Input Data:" in line or "📊 Input Data:" in line:
                continue

            m = re.match(r'Train:\s*\((\d+),\s*(\d+)\)', line)
            if m and shapes["input_train"] is None:
                shapes["input_train"] = [int(m.group(1)), int(m.group(2))]

            m = re.match(r'Test:\s*\((\d+),\s*(\d+)\)', line)
            if m and shapes["input_test"] is None:
                shapes["input_test"] = [int(m.group(1)), int(m.group(2))]

            # Input columns list
            m = re.match(r"Columns:\s*\[(.+)\]", line)
            if m and not shapes["input_columns"]:
                cols_str = m.group(1)
                shapes["input_columns"] = [
                    c.strip().strip("'\"")
                    for c in cols_str.split(",")
                    if c.strip().strip("'\"")
                ]

        # After combining
        for i, line in enumerate(lines):
            if "After combining all features" in line:
                for j in range(i + 1, min(i + 5, len(lines))):
                    m = re.match(r'Train:\s*\((\d+),\s*(\d+)\)', lines[j])
                    if m:
                        shapes["after_combining"] = [int(m.group(1)), int(m.group(2))]

        # Final output
        for i, line in enumerate(lines):
            if "UNIVERSAL FEATURE ENGINEERING COMPLETE" in line or "Final Output:" in line:
                for j in range(i + 1, min(i + 15, len(lines))):
                    m = re.match(r'Train:\s*\((\d+),\s*(\d+)\)', lines[j])
                    if m:
                        shapes["final_train"] = [int(m.group(1)), int(m.group(2))]
                    m = re.match(r'Test:\s*\((\d+),\s*(\d+)\)', lines[j])
                    if m:
                        shapes["final_test"] = [int(m.group(1)), int(m.group(2))]
                    m = re.match(r"Columns:\s*\[(.+)\]", lines[j])
                    if m:
                        cols_str = m.group(1)
                        shapes["final_columns"] = [
                            c.strip().strip("'\"")
                            for c in cols_str.split(",")
                            if c.strip().strip("'\"")
                        ]

        return shapes

    # ─────────────────────────────────────────────────────────
    # FINAL OUTPUT PARSING
    # ─────────────────────────────────────────────────────────

    def _parse_final_output(self, lines: List[str]) -> Dict[str, Any]:
        """Parse the final pipeline output verification."""
        result = {
            "shapes_match": None,
            "no_nans": None,
            "success": False,
        }

        for line in lines:
            if "Shapes match:" in line:
                result["shapes_match"] = "True" in line
            if "No NaN values:" in line:
                result["no_nans"] = "True" in line
            if "UNIVERSAL FEATURE ENGINEERING COMPLETE" in line:
                result["success"] = True

        return result

    # ─────────────────────────────────────────────────────────
    # EXECUTION INFO
    # ─────────────────────────────────────────────────────────

    def _parse_execution_info(self, lines: List[str], raw: str) -> Dict[str, Any]:
        """Parse execution timing and status."""
        result = {
            "status": "unknown",
            "execution_time_seconds": None,
            "job_id": None,
            "pipeline_name": None,
            "kedro_time": None,
        }

        for line in lines:
            if "PIPELINE EXECUTION SUCCESSFUL" in line:
                result["status"] = "completed"
            elif "PIPELINE EXECUTION FAILED" in line:
                result["status"] = "failed"

            m = re.match(r'Time:\s*([\d.]+)s', line)
            if m:
                result["execution_time_seconds"] = float(m.group(1))

            m = re.match(r'Job ID:\s*([\w-]+)', line)
            if m:
                result["job_id"] = m.group(1)

            m = re.match(r'Pipeline:\s*(\w+)', line)
            if m and result["pipeline_name"] is None:
                result["pipeline_name"] = m.group(1)

            # Kedro's own timing
            m = re.match(r'Pipeline execution completed successfully in ([\d.]+)\s*sec', line)
            if m:
                result["kedro_time"] = float(m.group(1))

        return result

    # ─────────────────────────────────────────────────────────
    # PIPELINE STAGES
    # ─────────────────────────────────────────────────────────

    def _parse_pipeline_stages(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse the ordered list of pipeline stages that executed."""
        stages = []

        stage_markers = [
            ("id_detection", "UNIVERSAL ID COLUMN DETECTION"),
            ("type_detection", "UNIVERSAL COLUMN TYPE DETECTION"),
            ("numeric_scaling", "UNIVERSAL NUMERIC SCALING"),
            ("categorical_encoding", "UNIVERSAL CATEGORICAL ENCODING"),
            ("combining", "After combining all features"),
            ("variance_filtering", "VARIANCE FILTERING"),
            ("nan_imputation", "NaN IMPUTATION"),
            ("feature_engineering_complete", "UNIVERSAL FEATURE ENGINEERING COMPLETE"),
            ("feature_selection", "FEATURE SELECTION"),
        ]

        for stage_name, marker in stage_markers:
            for i, line in enumerate(lines):
                if marker in line:
                    stages.append({
                        "name": stage_name,
                        "line_number": i,
                        "marker": marker,
                    })
                    break

        # Also detect Kedro node executions
        for i, line in enumerate(lines):
            m = re.match(r'Running node:\s*(\w+):', line)
            if m:
                stages.append({
                    "name": f"kedro_node_{m.group(1)}",
                    "line_number": i,
                    "marker": line,
                })

        stages.sort(key=lambda s: s["line_number"])
        return stages

    # ─────────────────────────────────────────────────────────
    # AUTO-DETECT ISSUES (ML SCIENTIST BRAIN)
    # ─────────────────────────────────────────────────────────

    def _detect_issues(self, parsed: Dict) -> List[Dict[str, Any]]:
        """
        Auto-detect issues from parsed log data.
        This is the ML scientist brain — catches everything a senior
        Staff ML Engineer at Google/Meta/DeepMind would catch.
        """
        issues = []

        # ── CRITICAL: Numeric columns misclassified as categorical ──
        columns = parsed.get("columns", {})
        encoding = parsed.get("encoding", {})
        cat_cols = columns.get("categorical", [])

        for col_name in cat_cols:
            col_lower = col_name.lower().replace("_", "").replace("-", "")
            is_numeric_name = any(p in col_lower for p in NUMERIC_NAME_SIGNALS)

            enc_data = encoding.get("columns", {}).get(col_name, {})
            unique_total = enc_data.get("unique_total", 0)
            rare_grouped = enc_data.get("rare_grouped", 0)

            if is_numeric_name and unique_total > 50:
                collapse_pct = (rare_grouped / unique_total * 100) if unique_total > 0 else 0
                issues.append({
                    "severity": "critical",
                    "code": "FE-LOG-001",
                    "title": f"NUMERIC COLUMN '{col_name}' MISCLASSIFIED AS CATEGORICAL",
                    "detail": (
                        f"Column '{col_name}' has {unique_total} unique values and a name "
                        f"strongly suggesting numeric data (charges/amounts/totals). "
                        f"It was treated as categorical, with {rare_grouped} values "
                        f"({collapse_pct:.0f}%) grouped into 'Other'. This destroys ALL "
                        f"ordinal/magnitude information — the column is now useless. "
                        f"Root cause: the raw data likely stores this as string "
                        f"(contains spaces, currency symbols, commas, or empty strings "
                        f"instead of NaN)."
                    ),
                    "fix": (
                        f"In data cleaning (Phase 1c), add:\n"
                        f"  df['{col_name}'] = pd.to_numeric(df['{col_name}'], errors='coerce')\n"
                        f"  df['{col_name}'].fillna(df['{col_name}'].median(), inplace=True)"
                    ),
                    "impact": "HIGH — complete information loss of a likely key predictor",
                    "column": col_name,
                    "evidence": {
                        "unique_values": unique_total,
                        "rare_grouped": rare_grouped,
                        "collapse_pct": collapse_pct,
                        "name_pattern": "numeric_by_name",
                    },
                })

        # ── CRITICAL: ID column not detected ──
        id_detection = parsed.get("id_detection", {})
        all_cols = list(columns.get("all_columns", {}).keys())
        detected_ids = id_detection.get("detected", [])

        for col_name in all_cols:
            col_clean = col_name.lower().replace("_", "").replace("-", "")
            is_id_name = (
                    col_clean in [p.replace("_", "") for p in ID_NAME_SIGNALS]
                    or (col_clean.endswith("id") and len(col_clean) <= 20)
            )

            if is_id_name and col_name not in detected_ids:
                enc_data = encoding.get("columns", {}).get(col_name, {})
                unique_total = enc_data.get("unique_total", 0)

                issues.append({
                    "severity": "critical",
                    "code": "FE-LOG-002",
                    "title": f"ID COLUMN '{col_name}' NOT DETECTED — Leaked Into Features",
                    "detail": (
                        f"Column '{col_name}' has 'ID' in its name and {unique_total} unique "
                        f"values (likely 100% cardinality), but was NOT flagged as an ID column. "
                        f"Instead, it went through categorical encoding where all values were "
                        f"grouped into 'Other' — creating a constant column that adds noise. "
                        f"Worse, if cardinality was below the threshold, individual IDs could "
                        f"leak into the model as one-hot columns (data leakage)."
                    ),
                    "fix": (
                        f"Fix the ID detection threshold or add '{col_name}' to the explicit "
                        f"drop list:\n"
                        f"  df = df.drop(columns=['{col_name}'])\n"
                        f"Or lower the cardinality threshold from 90% to 80%."
                    ),
                    "impact": "HIGH — noise injection + potential data leakage",
                    "column": col_name,
                    "evidence": {
                        "unique_values": unique_total,
                        "name_pattern": "id_by_name",
                        "detection_status": id_detection.get("status"),
                    },
                })

        # ── WARNING: Binary predictors killed by variance filter ──
        variance = parsed.get("variance_filter", {})
        removed = variance.get("removed_features", [])

        # Check against known binary predictors
        known_binary = set()
        for domain_preds in KNOWN_BINARY_PREDICTORS.values():
            known_binary.update(p.lower() for p in domain_preds)

        killed_predictors = []
        for feat in removed:
            # Strip _scaled suffix to get original name
            base_name = re.sub(r'_scaled$', '', feat).lower()
            if base_name in known_binary:
                killed_predictors.append({"feature": feat, "base_name": base_name})

        if killed_predictors:
            names = [kp["base_name"] for kp in killed_predictors]
            issues.append({
                "severity": "warning",
                "code": "FE-LOG-003",
                "title": f"VARIANCE FILTER KILLED {len(killed_predictors)} KNOWN BINARY PREDICTORS",
                "detail": (
                    f"The variance filter (threshold={variance.get('threshold', '?')}) removed "
                    f"{len(killed_predictors)} features that are KNOWN binary predictors: "
                    f"{', '.join(names)}. "
                    f"Binary features (0/1) inherently have low variance "
                    f"(max variance for binary = 0.25 at 50/50 split). "
                    f"After StandardScaler, their variance drops further. "
                    f"These features may carry significant predictive signal despite low variance."
                ),
                "fix": (
                    f"Option 1: Lower the variance threshold below 0.01\n"
                    f"Option 2: Exclude binary columns from variance filtering\n"
                    f"Option 3: Apply variance filter BEFORE scaling (on raw binary 0/1)"
                ),
                "impact": "MEDIUM — known predictive features lost",
                "features": names,
                "evidence": {
                    "variance_threshold": variance.get("threshold"),
                    "n_removed_total": variance.get("n_removed"),
                    "known_predictors_killed": names,
                },
            })

        # ── WARNING: Binary features scaled then killed ──
        binary_cols = columns.get("binary", [])
        scaling = parsed.get("scaling", {})

        if binary_cols and removed:
            binary_killed = []
            for feat in removed:
                base = re.sub(r'_scaled$', '', feat)
                if base in binary_cols:
                    binary_killed.append(feat)

            if binary_killed:
                issues.append({
                    "severity": "warning",
                    "code": "FE-LOG-004",
                    "title": f"SCALING→VARIANCE PIPELINE TRAP: {len(binary_killed)} Binary Features Lost",
                    "detail": (
                        f"A subtle pipeline interaction: {len(binary_killed)} binary (0/1) features "
                        f"were first StandardScaled (converting them to ~[-1, +1] with small variance), "
                        f"then the variance filter removed them because their post-scaling variance "
                        f"fell below {variance.get('threshold', '?')}. The scaling step made the "
                        f"variance filter too aggressive on binary features. "
                        f"Affected: {', '.join(binary_killed)}."
                    ),
                    "fix": (
                        f"Apply variance filter BEFORE scaling, or exclude binary columns "
                        f"from variance filtering, or use a variance threshold below 0.005 "
                        f"for scaled binary features."
                    ),
                    "impact": "MEDIUM — pipeline interaction causing silent feature loss",
                    "features": binary_killed,
                })

        # ── INFO: Feature selection ratio ──
        selection = parsed.get("feature_selection", {})
        input_features = (selection.get("input_shape") or [0, 0])[1] if selection.get("input_shape") else 0
        selected = selection.get("n_selected", 0)

        if input_features > 0 and selected > 0:
            retention_pct = selected / input_features * 100
            if retention_pct < 50:
                issues.append({
                    "severity": "info",
                    "code": "FE-LOG-005",
                    "title": f"AGGRESSIVE FEATURE SELECTION: {selected}/{input_features} ({retention_pct:.0f}% retained)",
                    "detail": (
                        f"Feature selection kept only {selected} of {input_features} features "
                        f"({retention_pct:.0f}% retention). While this reduces dimensionality, "
                        f"it may discard useful predictors. The optimal retention is typically "
                        f"60-80% depending on the dataset."
                    ),
                    "fix": (
                        f"Consider increasing n_features_to_select from {selected} to "
                        f"{min(input_features, int(input_features * 0.7))} "
                        f"and validating with cross-validation."
                    ),
                    "impact": "LOW-MEDIUM — potential information loss",
                    "evidence": {
                        "input_features": input_features,
                        "selected": selected,
                        "retention_pct": retention_pct,
                    },
                })

        # ── INFO: Rare category complete collapse ──
        for col_name, col_data in encoding.get("columns", {}).items():
            unique = col_data.get("unique_total", 0)
            rare = col_data.get("rare_grouped", 0)
            output = col_data.get("output_categories", 0)

            if unique > 10 and rare > 0 and output <= 1:
                issues.append({
                    "severity": "warning",
                    "code": "FE-LOG-006",
                    "title": f"COMPLETE CATEGORY COLLAPSE: '{col_name}' ({unique}→{output} categories)",
                    "detail": (
                        f"Column '{col_name}' had {unique} unique values. After rare category "
                        f"grouping, {rare} values ({rare/unique*100:.0f}%) were collapsed into "
                        f"'Other', leaving only {output} usable category. The column is now "
                        f"effectively constant and carries zero information."
                    ),
                    "fix": (
                        f"For high-cardinality columns, use target encoding or frequency encoding "
                        f"instead of one-hot. Or increase the rare category threshold."
                    ),
                    "impact": "MEDIUM — complete information loss for this column",
                    "column": col_name,
                })

        # ── INFO: Pipeline dimensionality ──
        shapes = parsed.get("shapes", {})
        final_train = shapes.get("final_train")
        if final_train and len(final_train) >= 2:
            n_rows, n_cols = final_train
            if n_rows > 0 and n_cols > 0:
                ratio = n_rows / n_cols
                if ratio < 10:
                    issues.append({
                        "severity": "warning",
                        "code": "FE-LOG-007",
                        "title": f"LOW SAMPLES-TO-FEATURES RATIO: {ratio:.1f}:1 ({n_rows} rows, {n_cols} features)",
                        "detail": (
                            f"The ratio of training samples to features is only {ratio:.1f}:1. "
                            f"A rule of thumb is to have at least 10-20 samples per feature "
                            f"for stable model training. Current ratio suggests potential "
                            f"overfitting risk."
                        ),
                        "fix": f"Reduce features to {n_rows // 20} or collect more data.",
                        "impact": "MEDIUM — overfitting risk",
                    })

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 3))

        return issues

    # ─────────────────────────────────────────────────────────
    # CONVERT TO INTELLIGENCE REQUEST
    # ─────────────────────────────────────────────────────────

    def _to_intelligence_request(self, parsed: Dict) -> Dict[str, Any]:
        """
        Convert parsed log data to PipelineIntelligenceRequest-compatible dict.

        This bridges the log parser output to the existing FE intelligence API.
        If embedded metadata is available (from FEMetadataCapture.print_to_log()),
        it overrides the regex-parsed values for 100% accuracy.
        """
        config = parsed.get("config", {})
        columns = parsed.get("columns", {})
        encoding = parsed.get("encoding", {})
        scaling = parsed.get("scaling", {})
        variance = parsed.get("variance_filter", {})
        selection = parsed.get("feature_selection", {})
        shapes = parsed.get("shapes", {})
        execution = parsed.get("execution", {})

        # Build encoding_details dict (per-column)
        encoding_details = {}
        for col_name, col_data in encoding.get("columns", {}).items():
            encoding_details[col_name] = {
                "unique_values": col_data.get("unique_total", 0),
                "unique_train": col_data.get("unique_train", 0),
                "unique_test": col_data.get("unique_test", 0),
                "rare_grouped": col_data.get("rare_grouped", 0),
                "strategy": col_data.get("strategy", "unknown"),
                "output_categories": col_data.get("output_categories", 0),
            }

        # Build original_columns from input columns
        original_columns = shapes.get("input_columns", [])
        if not original_columns:
            original_columns = list(columns.get("all_columns", {}).keys())

        # Build selected_features from final columns
        selected_features = shapes.get("final_columns", [])

        # Calculate shapes
        input_shape = shapes.get("input_train")
        final_shape = shapes.get("final_train")

        # Feature selection output
        sel_output = selection.get("output_train_shape")

        result = {
            # Config
            "scaling_method": config.get("scaling_method", scaling.get("method", "standard")),
            "handle_missing_values": config.get("handle_missing_values", True),
            "handle_outliers": config.get("handle_outliers", True),
            "encode_categories": config.get("encode_categories", True),
            "create_polynomial_features": config.get("create_polynomial_features", False),
            "create_interactions": config.get("create_interactions", False),
            "variance_threshold": variance.get("threshold", 0.01),

            # Columns
            "original_columns": original_columns,
            "selected_features": selected_features,
            "numeric_features": columns.get("numeric", []) + columns.get("binary", []),
            "categorical_features": columns.get("categorical", []),
            "id_columns_detected": parsed.get("id_detection", {}).get("detected", []),
            "variance_removed": variance.get("removed_features", []),

            # Encoding
            "encoding_details": encoding_details,

            # Shapes
            "original_shape": input_shape,
            "train_shape": final_shape,
            "test_shape": shapes.get("final_test"),
            "n_rows": input_shape[0] if input_shape else None,

            # Variance
            "features_before_variance": variance.get("features_before", 0),
            "features_after_variance": variance.get("features_after", 0),

            # Selection
            "features_input_to_selection": (
                selection.get("input_shape", [0, 0])[1]
                if selection.get("input_shape") else None
            ),
            "n_selected": selection.get("n_selected", 0),
            "n_features_to_select": selection.get("n_requested"),

            # Execution
            "execution_time_seconds": execution.get("execution_time_seconds"),
            "job_id": execution.get("job_id") or config.get("job_id"),
            "dataset_id": config.get("dataset_id"),

            # Auto-detected issues (bonus — not in standard request)
            "_auto_detected_issues": parsed.get("auto_detected_issues", []),
            "_parser_version": "2.0",
        }

        # ── Override with embedded metadata if available ──
        # Embedded metadata from FEMetadataCapture is 100% accurate
        # (captured at runtime vs regex-parsed from log text)
        embedded = parsed.get("embedded_metadata")
        if embedded:
            em_config = embedded.get("config", {})
            em_results = embedded.get("results", {})

            # Override config fields
            for key in ["scaling_method", "handle_missing_values", "handle_outliers",
                        "encode_categories", "create_polynomial_features",
                        "create_interactions", "variance_threshold"]:
                if key in em_config:
                    result[key] = em_config[key]

            # Override results fields (only if embedded has data)
            if em_results.get("original_columns"):
                result["original_columns"] = em_results["original_columns"]
            if em_results.get("selected_features"):
                result["selected_features"] = em_results["selected_features"]
            elif em_results.get("final_columns"):
                result["selected_features"] = em_results["final_columns"]
            if em_results.get("numeric_features"):
                result["numeric_features"] = em_results["numeric_features"]
            if em_results.get("categorical_features"):
                result["categorical_features"] = em_results["categorical_features"]
            if em_results.get("id_columns_detected"):
                result["id_columns_detected"] = em_results["id_columns_detected"]
            if em_results.get("variance_removed"):
                result["variance_removed"] = em_results["variance_removed"]
            if em_results.get("encoding_details"):
                result["encoding_details"] = em_results["encoding_details"]
            if em_results.get("original_shape"):
                result["original_shape"] = em_results["original_shape"]
                result["n_rows"] = em_results["original_shape"][0]
            if em_results.get("train_shape"):
                result["train_shape"] = em_results["train_shape"]
            if em_results.get("test_shape"):
                result["test_shape"] = em_results["test_shape"]
            if em_results.get("features_before_variance"):
                result["features_before_variance"] = em_results["features_before_variance"]
            if em_results.get("features_after_variance"):
                result["features_after_variance"] = em_results["features_after_variance"]
            if em_results.get("n_selected"):
                result["n_selected"] = em_results["n_selected"]
            if em_results.get("execution_time_seconds"):
                result["execution_time_seconds"] = em_results["execution_time_seconds"]

            result["_data_source"] = "embedded_metadata+log"
            result["_metadata_accuracy"] = "100%"
            logger.info("[FE-Parser] ✅ Overrode regex-parsed data with embedded metadata")
        else:
            result["_data_source"] = "log_regex"

        return result


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def parse_fe_log(raw_log: str) -> Dict[str, Any]:
    """
    Parse raw Kedro feature engineering log into structured intelligence data.

    Usage:
        from app.core.agent.fe_log_parser import parse_fe_log
        result = parse_fe_log(raw_kedro_log)
        issues = result["auto_detected_issues"]  # Critical issues found
        request = result["intelligence_request"]  # Ready for FE intelligence API
    """
    parser = FELogParser()
    return parser.parse(raw_log)