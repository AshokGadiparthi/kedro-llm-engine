# ML Expert Agent — Engine Integration Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          API Layer                                    │
│  POST /api/v1/agent/insights | /ask | /validate | /recommend         │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────────┐
│                    AgentOrchestrator (orchestrator.py)                │
│  Single entry point — wires ALL components per request type          │
│                                                                       │
│  INSIGHTS: Context → Stats → Patterns → Rules → Recs → Impact → LLM │
│  ASK:      Context → Stats → Rules → QA Engine → LLM → Memory       │
│  VALIDATE: Context → Stats → Validation Engine → Rules → Response    │
│  RECOMMEND: Context → Stats → Patterns → Recs → Impact              │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────────┐
│                    Component Layer (16 modules)                       │
│                                                                       │
│  context_compiler.py (682L)  — Reads ALL platform DB models          │
│  rule_engine.py (1556L)      — 120+ base rules (15 categories)      │
│  rules_extended.py (1092L)   — 90+ advanced rules (mixin)           │
│  statistical_analyzer.py (890L) — Distribution/cardinality/readiness │
│  pattern_detector.py (796L)  — 10 anti-pattern detectors            │
│  recommendation_engine.py (1042L) — 10 recommendation generators    │
│  business_impact.py (503L)   — Metrics → dollar value               │
│  validation_engine.py (483L) — Pre-action gates (40+ checks)        │
│  qa_engine.py (1271L)        — Deterministic Q&A (18 intents)       │
│  drift_analyzer.py (799L)    — 12 drift detection methods           │
│  threshold_optimizer.py (503L) — Business-optimal thresholds        │
│  knowledge_base.py (481L)    — Curated ML decision frameworks       │
│  llm_reasoner.py (524L)      — Multi-provider LLM (optional)        │
│  memory_store.py (827L)      — Persistent learning                  │
│  __init__.py (86L)           — Public API exports                   │
│  orchestrator.py (827L)      — Central brain                        │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────────┐
│                    Data Layer (READ-ONLY metadata)                    │
│                                                                       │
│  EdaResult    → summary, statistics, quality, correlations (JSON)    │
│  Dataset      → name, file_info, schema, row_count, quality_score   │
│  DataProfile  → full_report (if available)                           │
│  Job          → pipeline_name, status, parameters, execution_time   │
│  RegisteredModel → name, status, best_accuracy, is_deployed         │
│  ModelVersion → accuracy, precision, recall, f1, roc_auc,           │
│                 hyperparameters, feature_importances, confusion_matrix│
│  AgentMemory  → user feedback, preferences, interaction history     │
│  DatasetCollection, CollectionTable → multi-table metadata          │
└──────────────────────────────────────────────────────────────────────┘
```

## Step 1: Copy Files

```bash
# Engine core (16 files)
cp -r app/core/agent/ YOUR_PROJECT/app/core/agent/

# API router (1 file)
cp app/api/agent.py YOUR_PROJECT/app/api/agent.py

# Database model (1 file)
cp app/models/agent_memory.py YOUR_PROJECT/app/models/agent_memory.py
```

## Step 2: Register the Router (main.py)

Add 2 lines to your FastAPI `main.py`:

```python
# Import
from app.api.agent import router as agent_router

# Register (after other routers)
app.include_router(agent_router, prefix="/api/v1/agent", tags=["ML Expert Agent"])
```

## Step 3: Create the Database Table

Add to your `init_db()` or startup:

```python
from app.models.agent_memory import AgentMemory
from app.core.database import engine, Base

# This creates the agent_memory table if it doesn't exist
Base.metadata.create_all(bind=engine)
```

Or run a migration:
```sql
CREATE TABLE agent_memory (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    key VARCHAR(500) NOT NULL,
    value TEXT,
    confidence FLOAT DEFAULT 1.0,
    decay_factor FLOAT DEFAULT 1.0,
    reinforcement_count INTEGER DEFAULT 1,
    access_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    project_id VARCHAR(36),
    dataset_id VARCHAR(36),
    screen VARCHAR(50),
    expires_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME,
    last_accessed_at DATETIME
);
CREATE INDEX ix_agent_memory_user_type ON agent_memory(user_id, memory_type);
CREATE INDEX ix_agent_memory_user_key ON agent_memory(user_id, key);
CREATE INDEX ix_agent_memory_active ON agent_memory(user_id, is_active);
```

## Step 4: (Optional) Configure LLM Provider

The engine works perfectly in **rules-only mode** (no LLM needed).
To enable LLM synthesis, set environment variables:

```bash
# Option A: OpenAI
export AGENT_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Option B: Anthropic
export AGENT_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Option C: Local Ollama
export AGENT_LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434

# Option D: Rules only (default — no LLM)
# Just don't set any of the above
```

## Step 5: Verify

```bash
# Health check
curl http://localhost:8000/api/v1/agent/health

# Test insights
curl -X POST http://localhost:8000/api/v1/agent/insights \
  -H "Content-Type: application/json" \
  -d '{"screen": "eda", "dataset_id": "YOUR_DATASET_ID"}'

# Test question
curl -X POST http://localhost:8000/api/v1/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"screen": "eda", "question": "How is my data quality?", "dataset_id": "YOUR_DATASET_ID"}'

# Test validation
curl -X POST http://localhost:8000/api/v1/agent/validate \
  -H "Content-Type: application/json" \
  -d '{"action": "start_training", "screen": "training", "dataset_id": "YOUR_DATASET_ID"}'
```

## API Reference

### POST /api/v1/agent/insights
Auto-generates screen-aware insights. Full pipeline.

**Request:**
```json
{
  "screen": "eda",
  "project_id": "proj-123",
  "dataset_id": "ds-456",
  "model_id": null,
  "user_id": "user-1",
  "extra": {
    "target_column": "Churn",
    "active_tab": "correlations"
  }
}
```

**Response:**
```json
{
  "insights": [
    {
      "severity": "critical",
      "category": "Target Variable",
      "title": "Class Imbalance Detected",
      "message": "Positive class is only 26.6% — standard accuracy will be misleading.",
      "action": "Use SMOTE or class_weight='balanced' and evaluate with F1 Score, not accuracy.",
      "evidence": "26.6% positive, 73.4% negative",
      "confidence": 0.98,
      "rule_id": "TV-001"
    }
  ],
  "patterns": [...],
  "recommendations": {
    "feature_selection": {...},
    "imputation": {...}
  },
  "business_impact": {...},
  "advice": "...",
  "source": "rules_only",
  "counts": {"critical": 2, "warning": 4, "total": 12}
}
```

### POST /api/v1/agent/ask
Context-aware question answering. 18 intent categories.

### POST /api/v1/agent/validate
Pre-action readiness gate. Returns blockers/warnings/pass.

### POST /api/v1/agent/recommend
Standalone recommendations (algorithms, features, encoding, etc).

### POST /api/v1/agent/feedback
Record user feedback for learning.

### GET /api/v1/agent/readiness?dataset_id=...
Data readiness report.

### GET /api/v1/agent/health
Component health check.

## Key Design Decisions

1. **NEVER accesses raw data.** Context Compiler reads only from:
   - EdaResult (pre-computed statistics)
   - Dataset metadata (row count, column count, schema)
   - Job records (pipeline status)
   - Model registry (metrics, hyperparameters)

2. **Rules-first, LLM-second.** The 200+ deterministic rules provide the foundation.
   LLM is optional enhancement for natural language synthesis. If LLM fails,
   the response is still useful.

3. **Graceful degradation everywhere.** Every component is wrapped in try/except.
   If statistical analysis fails, rules still run. If patterns fail, insights still
   generate. If memory fails, personalization is skipped.

4. **Caching.** Expensive computations (statistical analysis, pattern detection)
   are cached per (dataset_id, screen) with 5-minute TTL.

5. **Memory learning.** User feedback (thumbs up/down) adjusts future insight
   priority. Reinforced memories gain confidence; unhelpful insights get
   deprioritized.

## File Sizes (Total: 12,786 lines)

| File | Lines | Purpose |
|------|-------|---------|
| rule_engine.py | 1,556 | 120+ base rules (15 categories) |
| qa_engine.py | 1,271 | Deterministic Q&A (18 intents) |
| rules_extended.py | 1,092 | 90+ advanced rules (mixin) |
| recommendation_engine.py | 1,042 | 10 recommendation generators |
| statistical_analyzer.py | 890 | Deep metadata-driven statistics |
| memory_store.py | 827 | Persistent learning & preferences |
| orchestrator.py | 827 | Central brain |
| drift_analyzer.py | 799 | 12 drift detection methods |
| pattern_detector.py | 796 | 10 anti-pattern detectors |
| context_compiler.py | 682 | Reads ALL platform DB models |
| llm_reasoner.py | 524 | Multi-provider LLM (optional) |
| business_impact.py | 503 | Metrics → dollar value |
| threshold_optimizer.py | 503 | Business-optimal thresholds |
| validation_engine.py | 483 | Pre-action gates (40+ checks) |
| knowledge_base.py | 481 | Curated ML decision frameworks |
| api/agent.py | 356 | FastAPI endpoints |
| __init__.py | 86 | Public exports |
| agent_memory.py | 68 | SQLAlchemy model |
