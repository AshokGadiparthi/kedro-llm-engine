"""
Debug Context Compiler — simulates exactly what happens during a /ask request.
Run: python debug_context.py
"""
import os, sys, json, asyncio

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Must match what FastAPI does
from app.core.database import SessionLocal, engine
from app.models.platform import EdaResult, Dataset

TEST_ID = "8f2b4f94-41e8-43c9-9278-ca86959bcfd4"

print(f"\n{'='*60}")
print(f"  Debug Context Compiler — dataset_id: {TEST_ID}")
print(f"{'='*60}\n")

# ── Test 1: Raw SQLAlchemy query (like context_compiler does) ──
print("--- Test 1: SQLAlchemy ORM query ---\n")
db = SessionLocal()
try:
    # This is exactly what context_compiler._get_dataset_profile does
    print(f"  Querying EdaResult.dataset_id == '{TEST_ID}'...")
    eda = db.query(EdaResult).filter(EdaResult.dataset_id == TEST_ID).first()
    if eda:
        print(f"  ✅ Found EDA result: id={eda.id}")
        print(f"  ✅ summary type: {type(eda.summary).__name__}, length: {len(eda.summary) if eda.summary else 0}")
        print(f"  ✅ quality type: {type(eda.quality).__name__}, length: {len(eda.quality) if eda.quality else 0}")
        print(f"  ✅ correlations: {len(eda.correlations) if eda.correlations else 0} chars")
        print(f"  ✅ statistics: {len(eda.statistics) if eda.statistics else 0} chars")

        if eda.summary:
            s = json.loads(eda.summary)
            print(f"\n  Summary parsed: shape={s.get('shape')}, columns={s.get('columns')}")
    else:
        print(f"  ❌ No EdaResult found for dataset_id={TEST_ID}")

    print(f"\n  Querying Dataset.id == '{TEST_ID}'...")
    ds = db.query(Dataset).filter(Dataset.id == TEST_ID).first()
    if ds:
        print(f"  ✅ Found dataset: name={ds.name}, file_name={ds.file_name}")
    else:
        print(f"  ⚠️  No Dataset row for id={TEST_ID} (EDA may reference datasets by different ID)")
        # Try listing all datasets
        all_ds = db.query(Dataset).limit(5).all()
        print(f"  Available datasets: {[(d.id[:12]+'...', d.name) for d in all_ds]}")

except Exception as e:
    print(f"  ❌ SQLAlchemy query failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()

# ── Test 2: Full Context Compiler (async) ──
print(f"\n--- Test 2: Full ContextCompiler.compile() ---\n")

async def test_context_compiler():
    db = SessionLocal()
    try:
        from app.core.agent.context_compiler import ContextCompiler
        compiler = ContextCompiler(db)

        ctx = await compiler.compile(
            screen="eda",
            dataset_id=TEST_ID,
        )

        print(f"  Context keys: {list(ctx.keys())}")

        # Check dataset_profile
        profile = ctx.get("dataset_profile", {})
        print(f"\n  dataset_profile:")
        print(f"    rows: {profile.get('rows')}")
        print(f"    columns: {profile.get('columns')}")
        print(f"    file_name: {profile.get('file_name')}")
        print(f"    numeric_columns: {profile.get('numeric_columns')}")
        print(f"    categorical_columns: {profile.get('categorical_columns')}")

        # Check data_quality
        quality = ctx.get("data_quality", {})
        print(f"\n  data_quality:")
        print(f"    completeness: {quality.get('completeness')}")
        print(f"    overall_quality_score: {quality.get('overall_quality_score')}")
        print(f"    missing_cells: {quality.get('missing_cells')}")

        # Check correlations
        corr = ctx.get("correlations", {})
        print(f"\n  correlations:")
        print(f"    high_pairs: {corr.get('high_pairs')}")
        print(f"    pair_count: {corr.get('pair_count')}")

        if profile.get("rows", 0) > 0:
            print(f"\n  🎉 SUCCESS — Context has real data! ({profile['rows']} rows, {profile['columns']} cols)")
            print(f"  Claude will see: {profile.get('file_name')}, {profile['rows']}×{profile['columns']}, "
                  f"{profile.get('numeric_count',0)} numeric, {profile.get('categorical_count',0)} categorical")
        else:
            print(f"\n  ❌ PROBLEM — Context is empty (rows=0). Data didn't flow through.")

    except Exception as e:
        print(f"  ❌ ContextCompiler failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

asyncio.run(test_context_compiler())

# ── Test 3: Full Orchestrator ask() ──
print(f"\n--- Test 3: Full Orchestrator.ask() ---\n")

async def test_orchestrator():
    db = SessionLocal()
    try:
        from app.core.agent.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator(db)

        bundle = await orch.ask(
            screen="eda",
            dataset_id=TEST_ID,
            question="What algorithm should I use?",
        )

        d = bundle.to_dict()
        print(f"  source: {d.get('source')}")
        print(f"  confidence: {d.get('confidence')}")
        print(f"  timing: {d.get('timing')}")
        print(f"  answer preview: {d.get('answer', '')[:150]}...")
        print(f"  insights count: {len(d.get('supporting_insights', []))}")

        if d.get('timing', {}).get('context_compile', 0) > 0:
            print(f"\n  🎉 Context compile ran! ({d['timing']['context_compile']}s)")
        else:
            print(f"\n  ❌ context_compile=0 — still not loading data")

    except Exception as e:
        print(f"  ❌ Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

asyncio.run(test_orchestrator())