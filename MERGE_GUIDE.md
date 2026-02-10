# Merge Guide — Integrate into Your Existing FastAPI App (Port 8000)

## Two Options

| Option | Effort | Best For |
|--------|--------|----------|
| **A: Run Standalone on 8001** | 2 min | Quick start, microservice |
| **B: Merge into port 8000** | 15 min | Single app, shared DB |

---

## Option A: Standalone (Port 8001)

```bash
pip install -r requirements.txt
python main.py
# → http://localhost:8001/docs (23 endpoints)
```

Call from port 8000:
```python
import httpx
async def get_insights(screen, dataset_id):
    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8001/api/v1/agent/insights",
            json={"screen": screen, "dataset_id": dataset_id})
        return r.json()
```

To share the same database: set the same `DATABASE_URL` in both apps.

---

## Option B: Merge into Port 8000

### Step 1: Copy Engine (one command)

```bash
cp -r ml-agent-engine/app/core/agent/ your-app/app/core/agent/
```

This copies all 22 engine files (16 original + 6 enhancements).

### Step 2: Copy API Routes

```bash
cp ml-agent-engine/app/api/v1/agent.py your-app/app/api/v1/agent.py
cp ml-agent-engine/app/api/v1/enhanced.py your-app/app/api/v1/enhanced.py
```

### Step 3: Copy Agent Memory Model

```bash
cp ml-agent-engine/app/models/agent_memory.py your-app/app/models/agent_memory.py
```

### Step 4: Skip These (you already have them)

- `app/core/database.py` — keep yours
- `app/models/platform.py` — keep your real models
- `app/models/models.py` — keep yours
- `app/config.py` — keep yours
- `main.py` — keep yours

### Step 5: Register Routes (add 2 lines to your main.py)

```python
from app.api.v1.agent import router as agent_router
from app.api.v1.enhanced import router as enhanced_router
app.include_router(agent_router, prefix="/api/v1/agent", tags=["ML Expert Agent"])
app.include_router(enhanced_router, prefix="/api/v1/agent", tags=["ML Agent Enhanced"])
```

### Step 6: Register Model (add 1 line to your models/__init__.py)

```python
from app.models.agent_memory import AgentMemory  # noqa
```

### Step 7: Verify

```bash
curl http://localhost:8000/api/v1/agent/health/enhanced
```

---

## File Checklist

| File | Action |
|------|--------|
| `app/core/agent/` (22 .py files) | **COPY** entire directory |
| `app/core/agent/tests/` | **COPY** |
| `app/api/v1/agent.py` | **COPY** (or keep if you already have it) |
| `app/api/v1/enhanced.py` | **COPY** (new) |
| `app/models/agent_memory.py` | **COPY** (if missing) |
| `app/models/platform.py` | **SKIP** (you have real models) |
| `app/models/models.py` | **SKIP** |
| `app/models/data_management.py` | **SKIP** |
| `app/core/database.py` | **SKIP** |
| `app/config.py` | **SKIP** |
| `main.py` | **SKIP** (add router includes only) |

---

## Dependencies

Your existing app likely already has these. Just verify:

```
fastapi>=0.104.0
uvicorn>=0.24.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
httpx>=0.25.0      # only if using LLM reasoner
```

The 7 enhancement modules are **pure Python** — zero new dependencies.
