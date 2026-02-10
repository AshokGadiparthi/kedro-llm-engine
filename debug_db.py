"""
Quick DB diagnostic — Run this to check if agent engine can see your data.

Usage: python debug_db.py
"""
import os, sys

# Load .env exactly as main.py does
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded")
except ImportError:
    print("⚠️  dotenv not installed")

# Check what DATABASE_URL resolved to
raw_env = os.getenv("DATABASE_URL", "NOT SET")
print(f"\n📌 DATABASE_URL from env: '{raw_env}'")

# Check config.py resolution
from app.config import settings
print(f"📌 settings.DATABASE_URL:  '{settings.DATABASE_URL}'")

# Check if file exists (for SQLite)
if "sqlite" in settings.DATABASE_URL:
    # Extract path from sqlite:///path or sqlite:////path
    path = settings.DATABASE_URL.replace("sqlite:///", "")
    # Remove leading slash duplication
    if path.startswith("/") and not path.startswith("//"):
        abs_path = path
    else:
        abs_path = os.path.abspath(path)
    print(f"📌 SQLite file path: '{abs_path}'")
    print(f"📌 File exists: {os.path.exists(abs_path)}")
    if os.path.exists(abs_path):
        print(f"📌 File size: {os.path.getsize(abs_path):,} bytes")

# Try connecting
print("\n--- Database Connection Test ---\n")
from sqlalchemy import create_engine, text, inspect

try:
    engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
    with engine.connect() as conn:
        # List tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"✅ Connected! Found {len(tables)} tables:")
        for t in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM [{t}]")).scalar()
            print(f"   • {t}: {count} rows")

        # Check specific dataset
        print("\n--- EDA Results Check ---\n")
        result = conn.execute(text("SELECT id, dataset_id, LENGTH(summary), LENGTH(quality), LENGTH(correlations), LENGTH(statistics) FROM eda_results LIMIT 5"))
        rows = result.fetchall()
        if rows:
            print(f"✅ Found {len(rows)} EDA results:")
            for r in rows:
                print(f"   dataset_id={r[1]}")
                print(f"   summary={r[2]} chars, quality={r[3]} chars, correlations={r[4]} chars, statistics={r[5]} chars")
        else:
            print("❌ No EDA results found!")

        # Test the exact query the context_compiler runs
        test_id = "8f2b4f94-41e8-43c9-9278-ca86959bcfd4"
        print(f"\n--- Context Compiler Simulation (dataset_id={test_id}) ---\n")

        eda = conn.execute(text(f"SELECT summary, quality, correlations, statistics FROM eda_results WHERE dataset_id = :did"), {"did": test_id}).first()
        if eda:
            import json
            summary = json.loads(eda[0]) if eda[0] else {}
            quality = json.loads(eda[1]) if eda[1] else {}
            correlations = json.loads(eda[2]) if eda[2] else {}

            print(f"✅ Dataset found!")
            print(f"   Shape: {summary.get('shape')}")
            print(f"   Columns: {summary.get('columns')}")
            print(f"   Quality score: {quality.get('overall_quality_score')}")
            print(f"   Completeness: {quality.get('completeness')}")
            print(f"   Correlations: {correlations.get('correlations')}")
            print(f"\n🎯 THIS data should flow into Claude when you POST:")
            print(f'   {{"screen":"eda", "dataset_id":"{test_id}"}}')
        else:
            print(f"❌ dataset_id '{test_id}' NOT found in eda_results")
            print("   Check: is this the right database?")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()