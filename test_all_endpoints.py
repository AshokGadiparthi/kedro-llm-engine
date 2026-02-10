"""
ML Expert Agent Engine — Complete API Test Suite v2
=====================================================
Tests all 23 endpoints. Prints response body on failure for debugging.

HOW TO RUN:
  Step 1:  python main.py           (Terminal 1 — keep running)
  Step 2:  python test_all_endpoints.py   (Terminal 2)
"""

import json
import sys
import time

# Use requests (stdlib-friendly) or httpx
try:
    import httpx
    def get(url, **kw): return httpx.get(url, timeout=30, **kw)
    def post(url, **kw): return httpx.post(url, timeout=30, **kw)
    ConnError = httpx.ConnectError
except ImportError:
    import requests
    def get(url, **kw): return requests.get(url, timeout=30, **kw)
    def post(url, **kw): return requests.post(url, timeout=30, **kw)
    ConnError = requests.ConnectionError

BASE = "http://localhost:8001/api/v1/agent"
PASS = 0
FAIL = 0
TOTAL = 0


def test(name, method, url, expected_status=200, body=None, params=None,
         check_field=None, check_value=None, check_no_error=True):
    """Run one test. On failure, prints actual response for debugging."""
    global PASS, FAIL, TOTAL
    TOTAL += 1
    full_url = f"{BASE}{url}"
    try:
        if method == "GET":
            r = get(full_url, params=params)
        else:
            r = post(full_url, json=body)

        status_ok = r.status_code == expected_status

        try:
            data = r.json()
        except Exception:
            data = r.text

        # Check for error response
        has_error = isinstance(data, dict) and "error" in data and check_no_error
        field_ok = True
        value_ok = True

        if check_field and isinstance(data, dict):
            if check_field not in data:
                field_ok = False
        if check_value and isinstance(data, dict) and check_field:
            if data.get(check_field) != check_value:
                value_ok = False

        success = status_ok and field_ok and value_ok and not has_error

        if success:
            PASS += 1
            print(f"  ✅ TEST {TOTAL:2d} PASS │ {name}")
        else:
            FAIL += 1
            reason = ""
            if not status_ok:
                reason += f" status={r.status_code}"
            if has_error:
                reason += f' error="{data.get("error", "")[:80]}"'
            if not field_ok:
                reason += f" missing '{check_field}'"
            if not value_ok:
                reason += f" {check_field}={data.get(check_field)!r}"
            print(f"  ❌ TEST {TOTAL:2d} FAIL │ {name} │{reason}")
            # Print full response for debugging
            if isinstance(data, dict):
                truncated = json.dumps(data, default=str)[:200]
                print(f"         ↳ Response: {truncated}")

        return data

    except ConnError:
        FAIL += 1
        print(f"  ❌ TEST {TOTAL:2d} FAIL │ {name} │ Cannot connect — is server running on :8001?")
        return None
    except Exception as e:
        FAIL += 1
        print(f"  ❌ TEST {TOTAL:2d} FAIL │ {name} │ {type(e).__name__}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 72)
print("  ML Expert Agent Engine — Complete API Test Suite v2")
print("  Testing all 23 endpoints on http://localhost:8001")
print("=" * 72)
print()

try:
    r = get("http://localhost:8001/")
    data = r.json()
    print(f"  🟢 Server online: {data.get('service', '?')} v{data.get('version', '?')}")
except ConnError:
    print("  🔴 Server NOT running! Start with: python main.py")
    sys.exit(1)
except Exception as e:
    print(f"  🔴 Server error: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# GROUP 1: CORE ENDPOINTS (7 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 1: Core Agent Endpoints (7 tests)")
print("─" * 72)
print()

# Test 1: Health
test("GET /health", "GET", "/health",
     check_field="status", check_value="healthy")

# Test 2: Insights (EDA)
test("POST /insights — EDA screen", "POST", "/insights",
     body={"screen": "eda"}, check_field="source")

# Test 3: Insights (Training)
test("POST /insights — Training screen", "POST", "/insights",
     body={"screen": "training", "user_id": "test_user"}, check_field="insights")

# Test 4: Ask
test("POST /ask — Algorithm question", "POST", "/ask",
     body={"screen": "eda", "question": "What algorithm should I use for classification?"},
     check_field="answer")

# Test 5: Validate
test("POST /validate — Training readiness", "POST", "/validate",
     body={"action": "start_training", "screen": "training"},
     check_field="can_proceed")

# Test 6: Recommend
test("POST /recommend", "POST", "/recommend", body={"screen": "eda"})

# Test 7: Feedback
test("POST /feedback — Thumbs up", "POST", "/feedback",
     body={"insight_rule_id": "DQ-001", "helpful": True, "user_id": "test_user"},
     check_field="status", check_value="ok")


# ═══════════════════════════════════════════════════════════════
# GROUP 2: DOMAIN PROFILES (3 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 2: Domain Profiles (3 tests)")
print("─" * 72)
print()

# Test 8: List domains
data = test("GET /enhanced/domains — List all profiles", "GET", "/enhanced/domains",
            check_field="count")
if data and isinstance(data, dict) and data.get("count"):
    print(f"         → {data['count']} domains found")

# Test 9: Detect healthcare
data = test("POST /enhanced/detect-domain — Healthcare", "POST", "/enhanced/detect-domain",
            body={"column_names": ["patient_id", "blood_pressure", "diagnosis_code", "heart_rate", "bmi", "age"]},
            check_field="domain")
if data and isinstance(data, dict) and data.get("domain"):
    print(f"         → Detected: {data['domain']} (confidence={data.get('confidence', 0):.2f})")

# Test 10: Detect finance
data = test("POST /enhanced/detect-domain — Finance", "POST", "/enhanced/detect-domain",
            body={"column_names": ["account_id", "transaction_amount", "credit_score", "loan_amount", "interest_rate"]},
            check_field="domain")
if data and isinstance(data, dict) and data.get("domain"):
    print(f"         → Detected: {data['domain']} (confidence={data.get('confidence', 0):.2f})")


# ═══════════════════════════════════════════════════════════════
# GROUP 3: SEMANTIC INTENT (3 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 3: Semantic Intent Classification (3 tests)")
print("─" * 72)
print()

# Test 11: Algorithm selection
data = test("POST /classify-intent — Algorithm selection", "POST", "/classify-intent",
            body={"question": "my random forest sucks, what else can I try?", "top_k": 5},
            check_field="intent")
if data and isinstance(data, dict) and data.get("intent"):
    print(f"         → {data['intent']} (conf={data.get('confidence', 0):.2f}, method={data.get('method')})")

# Test 12: Overfitting
data = test("POST /classify-intent — Overfitting", "POST", "/classify-intent",
            body={"question": "training accuracy is 99% but validation is only 60%", "screen": "evaluation"},
            check_field="intent")
if data and isinstance(data, dict) and data.get("intent"):
    print(f"         → {data['intent']} (conf={data.get('confidence', 0):.2f})")

# Test 13: Missing data
data = test("POST /classify-intent — Missing data", "POST", "/classify-intent",
            body={"question": "lots of nulls in my dataset, how to handle them?"},
            check_field="intent")
if data and isinstance(data, dict) and data.get("intent"):
    print(f"         → {data['intent']} (conf={data.get('confidence', 0):.2f})")


# ═══════════════════════════════════════════════════════════════
# GROUP 4: STATISTICAL TESTS (6 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 4: Statistical Tests (6 tests)")
print("─" * 72)
print()

# Test 14: PSI
data = test("POST /statistical-tests — PSI", "POST", "/enhanced/statistical-tests",
            body={"test_type": "psi",
                  "reference_histogram": [100, 200, 300, 250, 150],
                  "current_histogram": [300, 100, 50, 200, 350]},
            check_field="test")
if data and isinstance(data, dict) and "statistic" in data:
    print(f"         → PSI={data['statistic']:.4f}, significant={data.get('is_significant')}")

# Test 15: Cohen's d
data = test("POST /statistical-tests — Cohen's d", "POST", "/enhanced/statistical-tests",
            body={"test_type": "cohens_d",
                  "ref_mean": 50.0, "ref_std": 10.0, "cur_mean": 55.0, "cur_std": 10.0},
            check_field="test")
if data and isinstance(data, dict) and "statistic" in data:
    print(f"         → d={data['statistic']:.4f}")

# Test 16: Chi-Square
data = test("POST /statistical-tests — Chi-Square", "POST", "/enhanced/statistical-tests",
            body={"test_type": "chi_square",
                  "reference_distribution": {"cat_A": 0.4, "cat_B": 0.35, "cat_C": 0.25},
                  "current_distribution": {"cat_A": 0.2, "cat_B": 0.5, "cat_C": 0.3}},
            check_field="test")

# Test 17: KS
data = test("POST /statistical-tests — KS", "POST", "/enhanced/statistical-tests",
            body={"test_type": "ks",
                  "reference_percentiles": {"1%": 1, "25%": 25, "50%": 50, "75%": 75, "99%": 99},
                  "current_percentiles": {"1%": 10, "25%": 35, "50%": 55, "75%": 80, "99%": 99}},
            check_field="test")

# Test 18: Normality
data = test("POST /statistical-tests — Normality", "POST", "/enhanced/statistical-tests",
            body={"test_type": "normality",
                  "ref_mean": 0.0, "ref_std": 1.0, "skewness": 0.1, "kurtosis": 3.05, "n_samples": 1000},
            check_field="is_normal")

# Test 19: Drift Suite
data = test("POST /drift-suite — Full drift analysis", "POST", "/enhanced/drift-suite",
            body={"feature": "age",
                  "ref_stats": {"mean": 35.0, "std": 10.0},
                  "cur_stats": {"mean": 45.0, "std": 12.0},
                  "ref_histogram": [100, 200, 300, 200, 100],
                  "cur_histogram": [50, 100, 200, 300, 250],
                  "ref_percentiles": {"1%": 15, "25%": 28, "50%": 35, "75%": 42, "99%": 55},
                  "cur_percentiles": {"1%": 20, "25%": 36, "50%": 45, "75%": 52, "99%": 65}},
            check_field="feature")
if data and isinstance(data, dict) and data.get("feature"):
    print(f"         → Feature: {data['feature']}, consensus={data.get('consensus_score', 0):.2f}")


# ═══════════════════════════════════════════════════════════════
# GROUP 5: FEEDBACK LOOP (5 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 5: Closed-Loop Feedback Tracking (5 tests)")
print("─" * 72)
print()

# Test 20: Record action
test("POST /feedback/record-action — Followed", "POST", "/feedback/record-action",
     body={"user_id": "user_123", "recommendation_id": "rec_001", "action": "followed",
           "details": {"algorithm": "xgboost"}},
     check_field="success")

# Test 21: Record outcome
data = test("POST /feedback/record-outcome — Improved", "POST", "/feedback/record-outcome",
            body={"user_id": "user_123", "recommendation_id": "rec_001",
                  "metrics_after": {"accuracy": 0.92, "f1": 0.88},
                  "metrics_before": {"accuracy": 0.85, "f1": 0.80}})

# Test 22: Auto-detect
test("POST /feedback/auto-detect", "POST", "/feedback/auto-detect",
     body={"user_id": "user_123", "project_id": "project_001", "current_metrics": {"accuracy": 0.93}})

# Test 23: Effectiveness
test("GET /feedback/effectiveness", "GET", "/feedback/effectiveness",
     params={"user_id": "user_123"})

# Test 24: Benchmarking
test("GET /feedback/benchmarking", "GET", "/feedback/benchmarking",
     params={"user_id": "user_123"})


# ═══════════════════════════════════════════════════════════════
# GROUP 6: COST MATRICES (2 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 6: Domain Cost Matrices & Business Impact (2 tests)")
print("─" * 72)
print()

# Test 25: List matrices
data = test("GET /cost-matrices — List available", "GET", "/cost-matrices",
            check_field="matrices")
if data and isinstance(data, dict) and isinstance(data.get("matrices"), list):
    print(f"         → {len(data['matrices'])} domain cost matrices")

# Test 26: Healthcare impact
data = test("POST /cost-matrices/impact — Healthcare", "POST", "/cost-matrices/impact",
            body={"precision": 0.85, "recall": 0.92, "domain": "healthcare",
                  "use_case": "diagnosis", "n_predictions": 10000},
            check_field="costs")
if data and isinstance(data, dict) and "costs" in data:
    print(f"         → Net savings: ${data['costs'].get('net_savings', 0):,.0f}")


# ═══════════════════════════════════════════════════════════════
# GROUP 7: ENHANCED ENDPOINTS (2 tests)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 7: Enhanced Domain-Aware Endpoints (2 tests)")
print("─" * 72)
print()

# Test 27: Enhanced insights
data = test("POST /enhanced/insights — Healthcare domain", "POST", "/enhanced/insights",
            body={"screen": "eda", "user_domain": "healthcare", "user_id": "doctor_123"})
if data and isinstance(data, dict):
    src = data.get("source", data.get("insights", ""))
    if not isinstance(src, str):
        src = str(type(src).__name__)
    print(f"         → source={src}")

# Test 28: Enhanced ask
data = test("POST /enhanced/ask — Semantic Q&A", "POST", "/enhanced/ask",
            body={"question": "my model is overfitting badly, what should I do?", "screen": "evaluation"})
if data and isinstance(data, dict):
    ans = str(data.get("answer", ""))[:80]
    print(f"         → answer={ans}")


# ═══════════════════════════════════════════════════════════════
# GROUP 8: ENHANCED HEALTH (1 test)
# ═══════════════════════════════════════════════════════════════

print()
print("─" * 72)
print("  GROUP 8: Enhanced Health Check (1 test)")
print("─" * 72)
print()

# Test 29: Enhanced health
data = test("GET /health/enhanced — Full status", "GET", "/health/enhanced",
            check_field="components")
if data and isinstance(data, dict):
    comps = data.get("components", {})
    print(f"         → Status: {data.get('status')}, Version: {data.get('version')}")
    for comp, info in comps.items():
        if isinstance(info, dict):
            st = info.get("status", "?")
            det = info.get("detail", "")
            extra = ""
            if "rules" in info: extra = f" ({info['rules']} rules)"
            if "domains" in info: extra = f" ({info['domains']} domains)"
            if "intents" in info: extra = f" ({info['intents']} intents)"
            if "matrices" in info: extra = f" ({info['matrices']} matrices)"
            if det: extra += f" [{det[:50]}]"
            print(f"           • {comp}: {st}{extra}")


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 72)
if FAIL == 0:
    print(f"  🎉 ALL {TOTAL} TESTS PASSED — 100% SUCCESS")
else:
    pct = PASS / TOTAL * 100 if TOTAL else 0
    print(f"  📊 RESULTS: {PASS}/{TOTAL} passed, {FAIL} failed ({pct:.0f}%)")
    if FAIL > 0:
        print()
        print("  💡 TIP: Check the 'Response:' lines above for error details.")
        print("     Share the output with Claude to get targeted fixes.")

print()
print("  Groups:  1=Core(7)  2=Domains(3)  3=Intent(3)  4=Stats(6)")
print("           5=Feedback(5)  6=CostMatrix(2)  7=Enhanced(2)  8=Health(1)")
print(f"  Total:   {TOTAL} tests covering all 23 endpoints")
print()
print("=" * 72)
print()
