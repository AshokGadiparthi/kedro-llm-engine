"""
Feature Engineering Unified Intelligence Endpoint
=====================================================
The ULTIMATE FE intelligence endpoint that accepts raw Kedro pipeline logs
and produces world-class analysis — combining:

  1. FE Log Parser      → Extracts structured data from raw Kedro stdout
  2. FeatureAnalyzer     → Deep analysis (transformations, selection, quality)
  3. FEPipelineIntel     → 16 expert analyzers (type detection, leakage, etc.)
  4. DB Enrichment       → EDA statistics, model importance, job history
  5. Expert Rules        → FE-010 to FE-025 domain-specific rules

Endpoints:
  POST /features/log-intelligence    — THE BIG ONE: Raw logs → Full analysis
  POST /features/pipeline-intelligence — Structured data → Full analysis (existing)
  POST /features/quick-health        — Fast health check

Data Flow:
  Raw Kedro Logs → FELogParser → Structured Data → All Analyzers → Rich JSON
                                      ↑
                               DB Enrichment (EDA, Models, Jobs)
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/features", tags=["Feature Intelligence"])

# ── Server-side cache for AI Chat integration ──
# Stores the last FE analysis per dataset_id so the /ask endpoint
# can reference it when users chat about their FE results.
_fe_analysis_cache: Dict[str, Dict[str, Any]] = {}


def get_cached_fe_analysis(dataset_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the last FE analysis for a dataset (used by AI Chat context compiler)."""
    if dataset_id and dataset_id in _fe_analysis_cache:
        return _fe_analysis_cache[dataset_id]
    # Return the most recent analysis if no dataset_id specified
    if _fe_analysis_cache:
        return next(iter(reversed(_fe_analysis_cache.values())), None)
    return None


# ═══════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════

class LogIntelligenceRequest(BaseModel):
    """
    Accept raw Kedro pipeline logs for world-class AI analysis.

    Send the full stdout from the Kedro FE pipeline execution.
    The parser extracts every decision automatically.
    """
    pipeline_log: str = Field(..., description="Raw Kedro pipeline stdout log")
    dataset_id: Optional[str] = Field(None, description="Dataset ID for EDA enrichment")
    project_id: Optional[str] = Field(None, description="Project ID for model enrichment")
    model_id: Optional[str] = Field(None, description="Model ID for feature importance")

    class Config:
        json_schema_extra = {
            "example": {
                "pipeline_log": "[2026-02-11 16:41:14,782: INFO/ForkPoolWorker-1] 📊 UNIVERSAL COLUMN TYPE DETECTION...",
                "dataset_id": "8750650f-0479-428b-a134-3b681dae7492",
                "project_id": "some-project-id",
            }
        }


class PipelineIntelligenceRequest(BaseModel):
    """
    Structured FE pipeline data for analysis.
    (The existing request model — enhanced with encoding_details)
    """
    # Config
    scaling_method: str = "standard"
    handle_missing_values: bool = True
    handle_outliers: bool = True
    encode_categories: bool = True
    create_polynomial_features: bool = False
    create_interactions: bool = False
    variance_threshold: float = 0.01
    n_features_to_select: Optional[int] = None

    # Results
    original_columns: List[str] = []
    selected_features: List[str] = []
    final_columns: List[str] = []
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    variance_removed: List[str] = []
    id_columns_detected: List[str] = []
    original_shape: Optional[List[int]] = None
    final_shape: Optional[List[int]] = None
    train_shape: Optional[List[int]] = None
    test_shape: Optional[List[int]] = None
    n_rows: Optional[int] = None
    execution_time_seconds: Optional[float] = None
    target_column: Optional[str] = None

    # Encoding details (per-column)
    encoding_details: Dict[str, Any] = {}

    # Variance filter details
    features_before_variance: Optional[int] = None
    features_after_variance: Optional[int] = None

    # Feature selection
    features_input_to_selection: Optional[int] = None
    n_selected: Optional[int] = None

    # DB references
    dataset_id: Optional[str] = None
    project_id: Optional[str] = None
    model_id: Optional[str] = None
    job_id: Optional[str] = None


class QuickHealthRequest(BaseModel):
    """Minimal request for fast health check."""
    pipeline_log: Optional[str] = None
    original_columns: Optional[List[str]] = None
    selected_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    variance_removed: Optional[List[str]] = None
    encoding_details: Optional[Dict[str, Any]] = None
    original_shape: Optional[List[int]] = None
    final_shape: Optional[List[int]] = None


class SmartIntelligenceRequest(BaseModel):
    """
    🧠 SMART-INTELLIGENCE: Works with OR without pipeline logs.

    The system auto-selects the best data source:
      1. If pipeline_log is provided → parse it (most granular)
      2. If metadata file exists → use it (100% accurate)
      3. If dataset_id has EDA → infer from DB (always available)

    The frontend should ALWAYS use this request model.
    Just send whatever data is available — the system figures out the rest.
    """
    pipeline_log: Optional[str] = Field(None, description="Raw Kedro pipeline stdout log (optional)")
    dataset_id: Optional[str] = Field(None, description="Dataset ID for EDA/DB analysis")
    project_id: Optional[str] = Field(None, description="Project ID for model enrichment")
    model_id: Optional[str] = Field(None, description="Model ID for feature importance")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "8750650f-0479-428b-a134-3b681dae7492",
                "project_id": "some-project-id",
            }
        }


# ═══════════════════════════════════════════════════════════════
# DB ENRICHMENT HELPERS
# ═══════════════════════════════════════════════════════════════

async def _get_eda_data(db, dataset_id: Optional[str]) -> Dict[str, Any]:
    """Fetch EDA results from database for enrichment."""
    if not dataset_id or not db:
        return {}
    try:
        from app.models.models import EdaResult
        eda = db.query(EdaResult).filter(EdaResult.dataset_id == dataset_id).first()
        if not eda:
            return {}
        result = {}
        for field in ["summary", "statistics", "quality", "correlations"]:
            raw = getattr(eda, field, None)
            if raw:
                try:
                    result[field] = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    result[field] = {}
        logger.info(f"[FE-Intel] Loaded EDA: stats={len(result.get('statistics', {}))}, "
                    f"correlations={'yes' if result.get('correlations') else 'no'}")
        return result
    except Exception as e:
        logger.warning(f"[FE-Intel] EDA load failed: {e}")
        return {}


async def _get_model_versions(db, project_id: Optional[str],
                              model_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch model versions for feature importance data."""
    if not db or (not project_id and not model_id):
        return {"versions": []}
    try:
        from app.models.models import RegisteredModel, ModelVersion
        query = db.query(ModelVersion)
        if model_id:
            query = query.filter(ModelVersion.model_id == model_id)
        elif project_id:
            model_ids = db.query(RegisteredModel.id).filter(
                RegisteredModel.project_id == project_id
            ).all()
            if model_ids:
                query = query.filter(ModelVersion.model_id.in_([m[0] for m in model_ids]))
            else:
                return {"versions": []}

        versions = query.order_by(ModelVersion.created_at.desc()).limit(20).all()
        version_list = []
        for v in versions:
            vd = {
                "algorithm": getattr(v, "algorithm", None),
                "accuracy": getattr(v, "accuracy", None),
                "test_score": getattr(v, "test_score", None),
                "train_score": getattr(v, "train_score", None),
                "feature_names": getattr(v, "feature_names", None),
                "feature_importances": getattr(v, "feature_importances", None),
            }
            for field in ["feature_names", "feature_importances"]:
                raw = vd.get(field)
                if isinstance(raw, str):
                    try:
                        vd[field] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass
            version_list.append(vd)
        return {"versions": version_list}
    except Exception as e:
        logger.warning(f"[FE-Intel] Model versions load failed: {e}")
        return {"versions": []}


async def _get_job_history(db, project_id: Optional[str],
                           job_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch pipeline job history."""
    if not db:
        return {"jobs": [], "last_fe_job": None}
    try:
        from app.models.models import Job
        query = db.query(Job)

        if job_id:
            job = query.filter(Job.id == job_id).first()
            if job:
                job_data = {
                    "status": getattr(job, "status", None),
                    "duration": getattr(job, "duration_seconds", None),
                    "metrics": None,
                    "config": None,
                }
                for field in ["metrics", "config"]:
                    raw = getattr(job, field, None)
                    if isinstance(raw, str):
                        try:
                            job_data[field] = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            pass
                return {"jobs": [job_data], "last_fe_job": job_data}

        if project_id:
            query = query.filter(Job.project_id == project_id)

        jobs = query.order_by(Job.created_at.desc()).limit(20).all()
        fe_jobs = []
        for job in jobs:
            pl = (getattr(job, "pipeline_name", None) or "").lower()
            if "feature" in pl or "phase2" in pl:
                fe_jobs.append({
                    "status": getattr(job, "status", None),
                    "duration": getattr(job, "duration_seconds", None),
                    "created_at": str(getattr(job, "created_at", "")),
                })

        return {
            "jobs": fe_jobs,
            "last_fe_job": fe_jobs[0] if fe_jobs else None,
        }
    except Exception as e:
        logger.warning(f"[FE-Intel] Job history load failed: {e}")
        return {"jobs": [], "last_fe_job": None}


# ═══════════════════════════════════════════════════════════════
# CORE ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

def _run_full_analysis(
        config: Dict[str, Any],
        results: Dict[str, Any],
        eda_data: Dict[str, Any],
        model_versions: Dict[str, Any],
        auto_issues: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Run the full FE intelligence analysis pipeline.

    Combines FeatureAnalyzer + FEPipelineIntelligence + auto-detected issues
    into a single comprehensive analysis output.
    """
    analysis = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_version": "3.0",
    }

    # ── 1. FeatureAnalyzer (deep analysis) ──
    # Each method runs independently so one crash doesn't kill the rest.
    _analyzer_errors = []
    try:
        from app.core.agent.feature_analyzer import FeatureAnalyzer
        analyzer = FeatureAnalyzer()
    except Exception as e:
        logger.error(f"[FE-Intel] FeatureAnalyzer import failed: {e}")
        analyzer = None
        _analyzer_errors.append(f"import: {e}")

    if analyzer:
        # 1a. Transformation audit
        try:
            analysis["transformation_audit"] = analyzer.analyze_transformations(
                feature_config=config, feature_results=results,
                eda_data=eda_data, model_versions=model_versions,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] analyze_transformations failed: {e}")
            _analyzer_errors.append(f"transformation_audit: {e}")

        # 1b. Selection explanation
        try:
            analysis["selection_explanation"] = analyzer.explain_feature_selection(
                feature_results=results, eda_data=eda_data, model_versions=model_versions,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] explain_feature_selection failed: {e}")
            _analyzer_errors.append(f"selection_explanation: {e}")

        # 1c. Error patterns
        try:
            analysis["error_patterns"] = analyzer.detect_error_patterns(
                feature_config=config, feature_results=results,
                eda_data=eda_data, model_versions=model_versions,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] detect_error_patterns failed: {e}")
            _analyzer_errors.append(f"error_patterns: {e}")

        # 1d. Quality scorecard
        try:
            analysis["quality_scorecard"] = analyzer.assess_quality(
                feature_config=config, feature_results=results,
                eda_data=eda_data, model_versions=model_versions,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] assess_quality failed: {e}")
            _analyzer_errors.append(f"quality_scorecard: {e}")

        # 1e. Smart config recommendations
        try:
            analysis["smart_config"] = analyzer.generate_smart_config(
                feature_results=results, eda_data=eda_data, model_versions=model_versions,
                current_config=config,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] generate_smart_config failed: {e}")
            _analyzer_errors.append(f"smart_config: {e}")

        # 1f. Next steps
        try:
            analysis["next_steps"] = analyzer.generate_next_steps(
                feature_config=config, feature_results=results,
                eda_data=eda_data, model_versions=model_versions,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] generate_next_steps failed: {e}")
            _analyzer_errors.append(f"next_steps: {e}")

        # 1g. Feature interactions
        try:
            analysis["feature_interactions"] = analyzer.analyze_feature_interactions(
                feature_results=results, eda_data=eda_data,
            )
        except Exception as e:
            logger.warning(f"[FE-Intel] analyze_feature_interactions failed: {e}")
            _analyzer_errors.append(f"feature_interactions: {e}")

        n_ok = 7 - len(_analyzer_errors)
        if _analyzer_errors:
            logger.warning(f"[FE-Intel] FeatureAnalyzer: {n_ok}/7 succeeded, {len(_analyzer_errors)} failed")
            analysis["_analyzer_errors"] = _analyzer_errors
        else:
            logger.info("[FE-Intel] FeatureAnalyzer completed all 7 analyses ✅")

    # ── 2. FEPipelineIntelligence (16 expert analyzers) ──
    try:
        from app.core.agent.fe_pipeline_intelligence import FEPipelineIntelligence
        pi = FEPipelineIntelligence()

        # Full pipeline analysis (the main method — runs all 16 analyzers)
        pipeline_analysis = pi.analyze_pipeline(
            config=config, results=results,
            eda_data=eda_data, model_versions=model_versions,
        )
        analysis["pipeline_intelligence"] = pipeline_analysis

        # Extract findings for downstream methods
        findings = pipeline_analysis.get("findings", [])

        # Pipeline quality score (standalone — may differ from internal score)
        try:
            quality = pi.score_pipeline_quality(
                findings=findings, config=config, results=results,
                eda=eda_data,
            )
            analysis["pipeline_quality_score"] = quality
        except Exception as e:
            logger.warning(f"[FE-Intel] score_pipeline_quality failed: {e}")

        # Optimal config suggestion
        try:
            optimal = pi.generate_optimal_config(
                config=config, results=results,
                eda=eda_data, findings=findings,
            )
            analysis["optimal_config"] = optimal
        except Exception as e:
            logger.warning(f"[FE-Intel] generate_optimal_config failed: {e}")

        logger.info("[FE-Intel] FEPipelineIntelligence completed")

    except Exception as e:
        logger.error(f"[FE-Intel] FEPipelineIntelligence error: {e}")
        analysis["_pipeline_intel_error"] = str(e)

    # ── 3. Auto-detected issues from log parser ──
    if auto_issues:
        analysis["log_detected_issues"] = auto_issues
        analysis["critical_count"] = sum(
            1 for i in auto_issues if i.get("severity") == "critical"
        )
        analysis["warning_count"] = sum(
            1 for i in auto_issues if i.get("severity") == "warning"
        )
        analysis["info_count"] = sum(
            1 for i in auto_issues if i.get("severity") == "info"
        )

    # ── 4. Executive Summary ──
    analysis["executive_summary"] = _generate_executive_summary(analysis, config, results)

    return analysis


def _generate_executive_summary(
        analysis: Dict, config: Dict, results: Dict
) -> Dict[str, Any]:
    """Generate the executive summary combining all analysis results."""

    # Collect all issues across all analyzers
    all_issues = []

    # From log parser
    log_issues = analysis.get("log_detected_issues", [])
    for issue in log_issues:
        all_issues.append({
            "source": "log_parser",
            "severity": issue.get("severity", "info"),
            "code": issue.get("code", ""),
            "title": issue.get("title", ""),
            "fix": issue.get("fix", ""),
        })

    # From transformation audit
    audit = analysis.get("transformation_audit", {})
    for section_name in ["scaling", "encoding", "variance", "missing_values",
                         "outlier_handling", "id_detection", "feature_selection"]:
        section = audit.get(section_name, {})
        if isinstance(section, dict):
            for finding in section.get("findings", []):
                if isinstance(finding, dict):
                    sev = finding.get("severity", "info")
                    if sev in ("critical", "warning"):
                        all_issues.append({
                            "source": "transformation_audit",
                            "severity": sev,
                            "title": finding.get("message", finding.get("title", "")),
                        })

    # From error patterns
    errors = analysis.get("error_patterns", {})
    for pattern in errors.get("patterns", []):
        if isinstance(pattern, dict):
            all_issues.append({
                "source": "error_patterns",
                "severity": pattern.get("severity", "warning"),
                "title": pattern.get("name", pattern.get("title", "")),
            })

    # From pipeline intelligence
    pi = analysis.get("pipeline_intelligence", {})
    for analyzer_name, analyzer_result in pi.items():
        if isinstance(analyzer_result, dict):
            for finding in analyzer_result.get("findings", []):
                if isinstance(finding, dict):
                    sev = finding.get("severity", "info")
                    if sev in ("critical", "warning"):
                        all_issues.append({
                            "source": f"pipeline_intel.{analyzer_name}",
                            "severity": sev,
                            "title": finding.get("title", finding.get("message", "")),
                        })

    # Count by severity
    n_critical = sum(1 for i in all_issues if i["severity"] == "critical")
    n_warning = sum(1 for i in all_issues if i["severity"] == "warning")
    n_info = sum(1 for i in all_issues if i["severity"] == "info")

    # Determine overall health
    if n_critical >= 2:
        health = "critical"
        health_icon = "🔴"
        verdict = "Pipeline has critical issues that will significantly impact model quality"
    elif n_critical == 1:
        health = "needs_attention"
        health_icon = "🟠"
        verdict = "Pipeline has one critical issue — fix before training"
    elif n_warning >= 3:
        health = "fair"
        health_icon = "🟡"
        verdict = "Pipeline works but has several warnings worth addressing"
    elif n_warning >= 1:
        health = "good"
        health_icon = "🟢"
        verdict = "Pipeline is good with minor improvements possible"
    else:
        health = "excellent"
        health_icon = "✅"
        verdict = "World-class feature engineering pipeline — no issues detected"

    # Quality score — prefer FeatureAnalyzer scorecard, fallback to PipelineIntelligence
    quality = analysis.get("quality_scorecard", {})
    quality_pct = quality.get("percentage", 0)
    quality_grade = quality.get("grade", "")

    # Fallback: if FeatureAnalyzer didn't produce a quality score, use PipelineIntelligence
    pi_data = analysis.get("pipeline_intelligence", {})
    if not quality_pct and pi_data:
        # Use the quality_score from analyze_pipeline (which internally runs score_pipeline_quality)
        quality_pct = pi_data.get("quality_score", 0)
        quality_grade = quality_grade or pi_data.get("quality_grade", "?")
        # Also try the breakdown to get a more granular score
        breakdown = pi_data.get("quality_breakdown", {})
        if breakdown and not quality_pct:
            total = sum(v.get("score", 0) for v in breakdown.values() if isinstance(v, dict))
            max_total = sum(v.get("max_score", 0) for v in breakdown.values() if isinstance(v, dict))
            quality_pct = round(total / max_total * 100) if max_total > 0 else 0

    if not quality_grade or quality_grade == "?":
        if quality_pct >= 90: quality_grade = "A"
        elif quality_pct >= 80: quality_grade = "B+"
        elif quality_pct >= 70: quality_grade = "B"
        elif quality_pct >= 60: quality_grade = "C"
        else: quality_grade = "D"

    # Pipeline score (standalone)
    pi_quality = analysis.get("pipeline_quality_score", {})
    pi_pct = pi_quality.get("percentage", pi_quality.get("pct", quality_pct))

    # Shape tracking
    original_shape = results.get("original_shape", config.get("original_shape"))
    final_shape = results.get("train_shape", results.get("final_shape"))

    n_original = original_shape[1] if original_shape and len(original_shape) >= 2 else "?"
    n_final = final_shape[1] if final_shape and len(final_shape) >= 2 else "?"

    return {
        "health": health,
        "health_icon": health_icon,
        "verdict": verdict,
        "quality_score_pct": quality_pct,
        "quality_grade": quality_grade,
        "pipeline_score_pct": pi_pct,
        "total_issues": len(all_issues),
        "critical_issues": n_critical,
        "warnings": n_warning,
        "info": n_info,
        "feature_journey": f"{n_original} columns → {n_final} features",
        "top_issues": all_issues[:5],  # Top 5 most important issues
        "all_issues": all_issues,
    }


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@router.post("/log-intelligence")
async def feature_log_intelligence(
        request: LogIntelligenceRequest,
        db=Depends(get_db),
):
    """
    🧠 THE BIG ONE: Raw Kedro logs → World-class FE intelligence.

    Send your raw Kedro pipeline stdout log and get back:
      - Every pipeline decision dissected
      - Every issue a senior ML scientist would catch
      - Quality scorecard with grade
      - Smart config recommendations
      - Prioritized next steps
      - Feature interaction analysis
      - Executive summary

    This endpoint replaces the need for the frontend to manually extract
    pipeline results — the log parser does it ALL automatically.
    """
    t0 = time.time()

    # ── 1. Parse the raw log ──
    try:
        from app.core.agent.fe_log_parser import FELogParser
        parser = FELogParser()
        parsed = parser.parse(request.pipeline_log)
        intel_request = parsed.get("intelligence_request", {})
        auto_issues = parsed.get("auto_detected_issues", [])

        logger.info(
            f"[FE-Intel] Parsed log: {parsed['cleaned_line_count']} lines, "
            f"{len(auto_issues)} auto-detected issues "
            f"({sum(1 for i in auto_issues if i['severity'] == 'critical')} critical)"
        )
    except Exception as e:
        logger.error(f"[FE-Intel] Log parsing failed: {e}")
        return {
            "error": "log_parsing_failed",
            "message": str(e),
            "hint": "Ensure the pipeline_log contains the full Kedro stdout output",
        }

    # ── 2. Build config and results from parsed data ──
    config = {
        "scaling_method": intel_request.get("scaling_method", "standard"),
        "handle_missing_values": intel_request.get("handle_missing_values", True),
        "handle_outliers": intel_request.get("handle_outliers", True),
        "encode_categories": intel_request.get("encode_categories", True),
        "create_polynomial_features": intel_request.get("create_polynomial_features", False),
        "create_interactions": intel_request.get("create_interactions", False),
        "variance_threshold": intel_request.get("variance_threshold", 0.01),
    }

    results = {
        "original_columns": intel_request.get("original_columns", []),
        "selected_features": intel_request.get("selected_features", []),
        "numeric_features": intel_request.get("numeric_features", []),
        "categorical_features": intel_request.get("categorical_features", []),
        "id_columns_detected": intel_request.get("id_columns_detected", []),
        "variance_removed": intel_request.get("variance_removed", []),
        "encoding_details": intel_request.get("encoding_details", {}),
        "original_shape": intel_request.get("original_shape"),
        "train_shape": intel_request.get("train_shape"),
        "test_shape": intel_request.get("test_shape"),
        "n_rows": intel_request.get("n_rows"),
        "features_before_variance": intel_request.get("features_before_variance"),
        "features_after_variance": intel_request.get("features_after_variance"),
        "features_input_to_selection": intel_request.get("features_input_to_selection"),
        "n_selected": intel_request.get("n_selected"),
        "execution_time_seconds": intel_request.get("execution_time_seconds"),
    }

    # ── 3. DB Enrichment ──
    dataset_id = request.dataset_id or intel_request.get("dataset_id")
    project_id = request.project_id
    model_id = request.model_id

    eda_data = await _get_eda_data(db, dataset_id)
    model_versions = await _get_model_versions(db, project_id, model_id)
    job_history = await _get_job_history(
        db, project_id, intel_request.get("job_id")
    )

    # Enrich results from EDA if available
    if eda_data:
        results = _enrich_from_eda(results, eda_data)

    # ── 4. Run full analysis ──
    analysis = _run_full_analysis(
        config=config,
        results=results,
        eda_data=eda_data,
        model_versions=model_versions,
        auto_issues=auto_issues,
    )

    # ── 5. Cache for AI Chat ──
    cache_key = dataset_id or "latest"
    _fe_analysis_cache[cache_key] = {
        "config": config,
        "results": results,
        "analysis": analysis,
        "auto_issues": auto_issues,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Keep cache small — max 10 entries
    if len(_fe_analysis_cache) > 10:
        oldest = next(iter(_fe_analysis_cache))
        del _fe_analysis_cache[oldest]

    # ── 6. Build response ──
    elapsed = time.time() - t0

    return {
        "status": "success",
        "analysis_time_seconds": round(elapsed, 3),
        "data_sources": {
            "log_parsed": True,
            "eda_enriched": bool(eda_data),
            "model_versions_available": bool(model_versions.get("versions")),
            "job_history_available": bool(job_history.get("jobs")),
        },

        # The parsed log data (for frontend reference)
        "parsed_log": {
            "columns_detected": len(parsed.get("columns", {}).get("all_columns", {})),
            "encoding_columns": len(parsed.get("encoding", {}).get("columns", {})),
            "pipeline_stages": len(parsed.get("pipeline_stages", [])),
            "execution_time": parsed.get("execution", {}).get("execution_time_seconds"),
            "job_id": parsed.get("execution", {}).get("job_id"),
        },

        # The full analysis
        **analysis,
    }


@router.post("/pipeline-intelligence")
async def feature_pipeline_intelligence(
        raw_request: Request,
        db=Depends(get_db),
):
    """
    Full FE pipeline intelligence from structured data.

    Accepts BOTH formats:
      1. Flat:   {"scaling_method": "standard", "original_columns": [...]}
      2. Nested: {"config": {"scaling_method": ...}, "results": {"original_columns": ...}}

    The nested format is what fe_metadata_capture.py saves to JSON.
    The flat format is what post_to_api() sends.
    """
    t0 = time.time()

    # Parse raw JSON body
    body = await raw_request.json()

    # ── Detect & flatten nested metadata format ──
    if "config" in body and "results" in body:
        # Nested format from fe_metadata_capture.py / fe_metadata_latest.json
        nested_config = body.get("config", {})
        nested_results = body.get("results", {})
        flat = {**nested_config, **nested_results}
        # Carry over any top-level fields that aren't config/results
        for k, v in body.items():
            if k not in ("config", "results", "metadata_version", "captured_at"):
                flat[k] = v
        logger.info(
            f"[FE-Intel] Received nested metadata format — flattened "
            f"{len(nested_config)} config + {len(nested_results)} result fields"
        )
    else:
        flat = body

    # Validate through Pydantic model
    try:
        request = PipelineIntelligenceRequest(**flat)
    except Exception as e:
        logger.warning(f"[FE-Intel] Pydantic validation warning: {e}")
        request = PipelineIntelligenceRequest()
        # Manually set what we can
        for field in PipelineIntelligenceRequest.model_fields:
            if field in flat:
                try:
                    setattr(request, field, flat[field])
                except Exception:
                    pass

    # Build config and results from request
    config = {
        "scaling_method": request.scaling_method,
        "handle_missing_values": request.handle_missing_values,
        "handle_outliers": request.handle_outliers,
        "encode_categories": request.encode_categories,
        "create_polynomial_features": request.create_polynomial_features,
        "create_interactions": request.create_interactions,
        "variance_threshold": request.variance_threshold,
    }

    results = {
        "original_columns": request.original_columns,
        "selected_features": request.selected_features or request.final_columns,
        "numeric_features": request.numeric_features,
        "categorical_features": request.categorical_features,
        "id_columns_detected": request.id_columns_detected,
        "variance_removed": request.variance_removed,
        "encoding_details": request.encoding_details,
        "original_shape": request.original_shape,
        "train_shape": request.train_shape or request.final_shape,
        "test_shape": request.test_shape,
        "n_rows": request.n_rows,
        "features_before_variance": request.features_before_variance,
        "features_after_variance": request.features_after_variance,
        "features_input_to_selection": request.features_input_to_selection,
        "n_selected": request.n_selected,
        "execution_time_seconds": request.execution_time_seconds,
    }

    # DB Enrichment
    eda_data = await _get_eda_data(db, request.dataset_id)
    model_versions = await _get_model_versions(db, request.project_id, request.model_id)

    if eda_data:
        results = _enrich_from_eda(results, eda_data)

    # Run full analysis
    analysis = _run_full_analysis(
        config=config, results=results,
        eda_data=eda_data, model_versions=model_versions,
    )

    elapsed = time.time() - t0
    return {
        "status": "success",
        "analysis_time_seconds": round(elapsed, 3),
        "data_sources": {
            "log_parsed": False,
            "eda_enriched": bool(eda_data),
            "model_versions_available": bool(model_versions.get("versions")),
        },
        **analysis,
    }


@router.post("/quick-health")
async def feature_quick_health(request: QuickHealthRequest):
    """
    Fast health check — returns just the critical issues.

    Accepts either raw logs or minimal structured data.
    No DB enrichment (for speed). Returns in <100ms.
    """
    t0 = time.time()

    issues = []

    # If raw log provided, parse it
    if request.pipeline_log:
        try:
            from app.core.agent.fe_log_parser import FELogParser
            parser = FELogParser()
            parsed = parser.parse(request.pipeline_log)
            issues = parsed.get("auto_detected_issues", [])
        except Exception as e:
            return {"error": str(e)}
    else:
        # Quick checks from structured data
        if request.categorical_features and request.encoding_details:
            from app.core.agent.fe_log_parser import NUMERIC_NAME_SIGNALS
            for col in request.categorical_features:
                col_lower = col.lower()
                if any(p in col_lower for p in NUMERIC_NAME_SIGNALS):
                    enc = request.encoding_details.get(col, {})
                    unique = enc.get("unique_values", enc.get("unique_total", 0))
                    if unique > 50:
                        issues.append({
                            "severity": "critical",
                            "code": "FE-LOG-001",
                            "title": f"'{col}' is likely NUMERIC but classified as Categorical",
                        })

        if request.variance_removed:
            known_binary = {"gender", "partner", "dependents", "phoneservice",
                            "paperlessbilling", "seniorcitizen"}
            killed = [
                f for f in request.variance_removed
                if f.lower().replace("_scaled", "") in known_binary
            ]
            if killed:
                issues.append({
                    "severity": "warning",
                    "code": "FE-LOG-003",
                    "title": f"Variance filter removed {len(killed)} known predictors",
                })

    elapsed = time.time() - t0

    n_critical = sum(1 for i in issues if i.get("severity") == "critical")
    n_warning = sum(1 for i in issues if i.get("severity") == "warning")

    health = "critical" if n_critical > 0 else "warning" if n_warning > 0 else "healthy"

    return {
        "health": health,
        "health_icon": "🔴" if health == "critical" else "🟡" if health == "warning" else "✅",
        "issues_count": len(issues),
        "critical": n_critical,
        "warnings": n_warning,
        "issues": issues,
        "analysis_time_ms": round(elapsed * 1000, 1),
    }


# ═══════════════════════════════════════════════════════════════
# HELPER: EDA ENRICHMENT
# ═══════════════════════════════════════════════════════════════

def _enrich_from_eda(results: Dict, eda_data: Dict) -> Dict:
    """Enrich results with EDA data (statistics, correlations, etc.)."""
    stats = eda_data.get("statistics") or {}
    summary = eda_data.get("summary") or {}

    # Infer numeric/categorical from EDA if not in results
    if not results.get("numeric_features") and stats:
        numeric = []
        categorical = []
        for col, cs in stats.items():
            if isinstance(cs, dict):
                dtype = cs.get("dtype", "").lower()
                if any(t in dtype for t in ["int", "float", "numeric"]):
                    numeric.append(col)
                elif any(t in dtype for t in ["object", "category", "string"]):
                    categorical.append(col)
        if numeric:
            results["numeric_features"] = numeric
        if categorical and not results.get("categorical_features"):
            results["categorical_features"] = categorical

    # Infer original_columns from EDA
    if not results.get("original_columns") and stats:
        results["original_columns"] = list(stats.keys())

    # Infer n_rows
    if not results.get("n_rows") and summary:
        n_rows = summary.get("total_rows") or summary.get("n_rows") or summary.get("row_count")
        if n_rows:
            results["n_rows"] = int(n_rows)

    # Add correlations for interaction analysis
    correlations = eda_data.get("correlations")
    if correlations:
        results["_correlations"] = correlations

    # Add column statistics for scaling recommendations
    if stats:
        results["_column_stats"] = stats

    return results


# ═══════════════════════════════════════════════════════════════
# NEW: AUTO-INTELLIGENCE (DB-only — no logs required)
# ═══════════════════════════════════════════════════════════════

class AutoIntelligenceRequest(BaseModel):
    """
    DB-only intelligence — no pipeline logs needed.

    Just provide the dataset_id and optionally project/model/job IDs.
    The system assembles everything from:
      - EDA results (column stats, correlations, quality)
      - Job history (pipeline config used)
      - Model versions (selected features, importances)

    Missing fields are intelligently inferred.
    """
    dataset_id: str = Field(..., description="Dataset ID (required — links to EDA)")
    project_id: Optional[str] = Field(None, description="Project ID for model/job lookup")
    model_id: Optional[str] = Field(None, description="Model ID for feature importance")
    job_id: Optional[str] = Field(None, description="Specific job ID to analyze")

    # Optional overrides (if the frontend knows these)
    scaling_method: Optional[str] = Field(None, description="Override: scaling method used")
    selected_features: Optional[List[str]] = Field(None, description="Override: final feature list")
    variance_removed: Optional[List[str]] = Field(None, description="Override: features removed by variance filter")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "8750650f-0479-428b-a134-3b681dae7492",
                "project_id": "my-project-id",
            }
        }


@router.post("/auto-intelligence")
async def feature_auto_intelligence(
        request: AutoIntelligenceRequest,
        db=Depends(get_db),
):
    """
    🧠 AUTO-INTELLIGENCE: Full FE analysis from DB data alone — no logs needed.

    This endpoint is the complement to /log-intelligence:
      - /log-intelligence: Parse raw Kedro logs → analysis (most granular)
      - /auto-intelligence: DB data → inference → analysis (always available)

    Works by assembling config/results from 3 DB sources:
      1. EDA Results   → column names, types, statistics, correlations
      2. Job History   → pipeline parameters (scaling_method, etc.)
      3. Model Versions → selected features, importances

    Missing data is intelligently inferred:
      - Encoding details from EDA cardinality + standard pipeline thresholds
      - Variance-removed from comparing original vs selected features
      - ID columns from name patterns + cardinality > 80%
      - Training shapes from EDA shape + standard 80/20 split

    The output format is IDENTICAL to /log-intelligence, so the frontend
    can use the same rendering code for both paths.
    """
    t0 = time.time()

    # ── 1. Assemble from DB ──
    try:
        from app.core.agent.fe_data_assembler import FEDataAssembler
        assembler = FEDataAssembler(db)
        config, results, eda_data = await assembler.assemble(
            dataset_id=request.dataset_id,
            project_id=request.project_id,
            job_id=request.job_id,
            model_id=request.model_id,
        )
    except Exception as e:
        logger.error(f"[FE-Auto] DB assembly failed: {e}")
        return {
            "error": "db_assembly_failed",
            "message": str(e),
            "hint": "Ensure dataset_id exists and has EDA results",
        }

    # ── 2. Apply frontend overrides ──
    if request.scaling_method:
        config["scaling_method"] = request.scaling_method
    if request.selected_features:
        results["selected_features"] = request.selected_features
        results["n_selected"] = len(request.selected_features)
    if request.variance_removed:
        results["variance_removed"] = request.variance_removed

    # ── 3. Load model versions for analyzers ──
    model_versions = await _get_model_versions(
        db, request.project_id, request.model_id
    )

    # ── 4. Run full analysis (same analyzers as log path!) ──
    analysis = _run_full_analysis(
        config=config,
        results=results,
        eda_data=eda_data,
        model_versions=model_versions,
        auto_issues=[],  # No log-parsed issues in DB-only mode
    )

    # ── 5. Cache for AI Chat ──
    cache_key = request.dataset_id or "latest"
    _fe_analysis_cache[cache_key] = {
        "config": config,
        "results": results,
        "analysis": analysis,
        "auto_issues": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if len(_fe_analysis_cache) > 10:
        oldest = next(iter(_fe_analysis_cache))
        del _fe_analysis_cache[oldest]

    # ── 6. Build response (same format as /log-intelligence) ──
    elapsed = time.time() - t0

    return {
        "status": "success",
        "analysis_time_seconds": round(elapsed, 3),
        "data_sources": {
            "log_parsed": False,  # Key difference: no logs used
            "db_assembled": True,
            "eda_enriched": bool(eda_data.get("statistics")),
            "model_versions_available": bool(model_versions.get("versions")),
            "fields_inferred": [
                k for k in ["encoding_details", "variance_removed",
                            "id_columns_detected", "train_shape"]
                if results.get(k)
            ],
        },

        # No parsed_log section (since there's no log)
        "assembly_info": {
            "original_columns": len(results.get("original_columns", [])),
            "numeric_features": len(results.get("numeric_features", [])),
            "categorical_features": len(results.get("categorical_features", [])),
            "selected_features": len(results.get("selected_features", [])),
            "id_columns_inferred": results.get("id_columns_detected", []),
            "variance_removed_inferred": results.get("variance_removed", []),
        },

        # The full analysis (same structure as /log-intelligence)
        **analysis,
    }


@router.post("/smart-intelligence")
async def feature_smart_intelligence(
        request: SmartIntelligenceRequest,
        db=Depends(get_db),
):
    """
    🧠 SMART-INTELLIGENCE: Auto-selects the best data path available.

    Priority order:
      1. Raw pipeline logs (most granular — exact decisions from stdout)
      2. Structured metadata file (saved by Kedro node — 100% accurate)
      3. DB inference (EDA + Job + Model — always available after EDA runs)

    The frontend can ALWAYS call this endpoint the same way.
    It will pick the richest data source automatically.

    This is the RECOMMENDED endpoint for the frontend to call.
    """
    dataset_id = request.dataset_id
    project_id = request.project_id
    model_id = request.model_id

    # ── Path 1: Raw logs provided → use log parser ──
    has_meaningful_log = (
            request.pipeline_log
            and len(request.pipeline_log.strip()) > 200
            and "PIPELINE" in request.pipeline_log.upper()
    )
    if has_meaningful_log:
        logger.info("[FE-Smart] Using PATH 1: Log parsing")
        # Forward to log-intelligence with a LogIntelligenceRequest
        log_request = LogIntelligenceRequest(
            pipeline_log=request.pipeline_log,
            dataset_id=dataset_id,
            project_id=project_id,
            model_id=model_id,
        )
        return await feature_log_intelligence(log_request, db)

    # ── Path 2: Check for structured metadata file ──
    if dataset_id:
        try:
            from app.core.agent.fe_metadata_capture import (
                load_fe_metadata_for_dataset,
                load_fe_metadata,
            )
            metadata = load_fe_metadata_for_dataset(dataset_id)
            if not metadata:
                metadata = load_fe_metadata()  # Try 'latest'

            if metadata and metadata.get("config") and metadata.get("results"):
                logger.info("[FE-Smart] Using PATH 2: Structured metadata file")
                return await _run_from_metadata(metadata, dataset_id, project_id, model_id, db)
        except Exception as e:
            logger.warning(f"[FE-Smart] Metadata load failed: {e}")

    # ── Path 3: DB inference (always available if dataset_id exists) ──
    if dataset_id:
        logger.info("[FE-Smart] Using PATH 3: DB inference")
        auto_request = AutoIntelligenceRequest(
            dataset_id=dataset_id,
            project_id=project_id,
            model_id=model_id,
        )
        return await feature_auto_intelligence(auto_request, db)

    return {
        "error": "no_data_source",
        "message": "No data source available for analysis",
        "hint": "Provide one of: pipeline_log (raw Kedro output), dataset_id (for DB analysis), or ensure fe_metadata_latest.json exists",
        "available_paths": {
            "path_1_logs": "POST pipeline_log with raw Kedro stdout",
            "path_2_metadata": "Run FE pipeline with metadata capture enabled",
            "path_3_db": "POST dataset_id to use EDA + Job + Model data",
        },
    }


async def _run_from_metadata(
        metadata: Dict, dataset_id: str, project_id: str, model_id: str, db
) -> Dict[str, Any]:
    """Run analysis from structured metadata file (Path 2)."""
    t0 = time.time()

    config = metadata["config"]
    results = metadata["results"]

    # DB enrichment on top of metadata
    eda_data = await _get_eda_data(db, dataset_id)
    model_versions = await _get_model_versions(db, project_id, model_id)

    if eda_data:
        results = _enrich_from_eda(results, eda_data)

    # Run full analysis
    analysis = _run_full_analysis(
        config=config,
        results=results,
        eda_data=eda_data,
        model_versions=model_versions,
        auto_issues=[],
    )

    # Cache for AI Chat
    cache_key = dataset_id or "latest"
    _fe_analysis_cache[cache_key] = {
        "config": config,
        "results": results,
        "analysis": analysis,
        "auto_issues": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if len(_fe_analysis_cache) > 10:
        oldest = next(iter(_fe_analysis_cache))
        del _fe_analysis_cache[oldest]

    elapsed = time.time() - t0

    return {
        "status": "success",
        "analysis_time_seconds": round(elapsed, 3),
        "data_sources": {
            "log_parsed": False,
            "metadata_file": True,
            "db_assembled": False,
            "eda_enriched": bool(eda_data.get("statistics") if eda_data else False),
            "model_versions_available": bool(model_versions.get("versions")),
            "metadata_captured_at": metadata.get("captured_at"),
        },
        "assembly_info": {
            "original_columns": len(results.get("original_columns", [])),
            "numeric_features": len(results.get("numeric_features", [])),
            "categorical_features": len(results.get("categorical_features", [])),
            "selected_features": len(results.get("selected_features", [])),
            "source": "metadata_file",
        },
        **analysis,
    }