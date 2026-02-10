# Step-by-Step Testing Guide — 100% Coverage

## You already did ✅
```
pip install -r requirements.txt
```

---

## STEP 1: Start the Server (Terminal 1)

Open your first terminal/command prompt and run:

```bash
cd C:\work\git\ML\ui-fast-kedro\apache2\ML_LLM_engine-v3
python main.py
```

You should see output like:
```
2026-02-10 08:30:00 | ml_agent | INFO | Creating database tables...
2026-02-10 08:30:00 | ml_agent | INFO | Database tables ready
2026-02-10 08:30:00 | ml_agent | INFO | Engine warmed up: 200+ rules, semantic classifier ready, 13 domain profiles
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to stop)
```

**⚠️ Keep this terminal running! Don't close it.**

---

## STEP 2: Quick Verify (Terminal 2)

Open a **second** terminal/command prompt:

```bash
curl http://localhost:8001/
```

Or open browser: **http://localhost:8001/docs** (Swagger UI — you can test every endpoint visually here!)

---

## STEP 3: Run ALL 29 Tests Automatically (Terminal 2)

```bash
cd C:\work\git\ML\ui-fast-kedro\apache2\ML_LLM_engine-v3
python test_all_endpoints.py
```

This will test all 23 endpoints and print results like:
```
  ✅ TEST  1 PASS │ GET /health — Component health check
  ✅ TEST  2 PASS │ POST /insights — EDA screen auto-insights
  ...
  🎉 ALL 29 TESTS PASSED — 100% SUCCESS
```

---

## STEP 4: Test ONE BY ONE Manually (curl commands)

If you want to test each endpoint individually, copy-paste these commands in Terminal 2:

### TEST 1: Health Check
```bash
curl http://localhost:8001/api/v1/agent/health
```
**Expected:** `{"status":"healthy","components":{...},"version":"3.0.0",...}`

---

### TEST 2: Enhanced Health (all components)
```bash
curl http://localhost:8001/api/v1/agent/health/enhanced
```
**Expected:** `{"status":"healthy","version":"4.0.0","components":{"rule_engine":{"status":"active",...},...}}`

---

### TEST 3: Auto-Insights (EDA screen)
```bash
curl -X POST http://localhost:8001/api/v1/agent/insights -H "Content-Type: application/json" -d "{\"screen\":\"eda\"}"
```
**Expected:** `{"insights":[...],"source":"rules_only",...}`

---

### TEST 4: Auto-Insights (Training screen)
```bash
curl -X POST http://localhost:8001/api/v1/agent/insights -H "Content-Type: application/json" -d "{\"screen\":\"training\",\"user_id\":\"test_user\"}"
```
**Expected:** `{"insights":[...],...}`

---

### TEST 5: Ask ML Expert a Question
```bash
curl -X POST http://localhost:8001/api/v1/agent/ask -H "Content-Type: application/json" -d "{\"screen\":\"eda\",\"question\":\"What algorithm should I use for classification?\"}"
```
**Expected:** `{"answer":"...","source":"qa_engine","confidence":...}`

---

### TEST 6: Validate Training Readiness
```bash
curl -X POST http://localhost:8001/api/v1/agent/validate -H "Content-Type: application/json" -d "{\"action\":\"start_training\",\"screen\":\"training\"}"
```
**Expected:** `{"can_proceed":true/false,"verdict":"...","blockers":[...],...}`

---

### TEST 7: Get Recommendations
```bash
curl -X POST http://localhost:8001/api/v1/agent/recommend -H "Content-Type: application/json" -d "{\"screen\":\"eda\"}"
```
**Expected:** `{...recommendations...}`

---

### TEST 8: Submit Feedback
```bash
curl -X POST http://localhost:8001/api/v1/agent/feedback -H "Content-Type: application/json" -d "{\"insight_rule_id\":\"DQ-001\",\"helpful\":true,\"user_id\":\"test_user\"}"
```
**Expected:** `{"status":"ok","message":"Feedback recorded: helpful","rule_id":"DQ-001"}`

---

### TEST 9: List All Domain Profiles
```bash
curl http://localhost:8001/api/v1/agent/enhanced/domains
```
**Expected:** `{"domains":{"healthcare":{...},"finance":{...},...},"count":13}`

---

### TEST 10: Auto-Detect Healthcare Domain
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/detect-domain -H "Content-Type: application/json" -d "{\"column_names\":[\"patient_id\",\"blood_pressure\",\"diagnosis_code\",\"heart_rate\",\"bmi\"]}"
```
**Expected:** `{"domain":"healthcare","confidence":0.5+,"description":"...","thresholds":{...}}`

---

### TEST 11: Auto-Detect Finance Domain
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/detect-domain -H "Content-Type: application/json" -d "{\"column_names\":[\"account_id\",\"transaction_amount\",\"credit_score\",\"loan_amount\",\"interest_rate\"]}"
```
**Expected:** `{"domain":"finance",...}`

---

### TEST 12: Classify Intent — Algorithm Selection
```bash
curl -X POST http://localhost:8001/api/v1/agent/classify-intent -H "Content-Type: application/json" -d "{\"question\":\"my random forest sucks, what else can I try?\",\"top_k\":5}"
```
**Expected:** `{"intent":"algorithm_selection","confidence":0.5+,"method":"...","top_intents":[...]}`

---

### TEST 13: Classify Intent — Overfitting
```bash
curl -X POST http://localhost:8001/api/v1/agent/classify-intent -H "Content-Type: application/json" -d "{\"question\":\"training accuracy is 99% but validation is only 60%\",\"screen\":\"evaluation\"}"
```
**Expected:** `{"intent":"overfitting",...}`

---

### TEST 14: Classify Intent — Missing Data
```bash
curl -X POST http://localhost:8001/api/v1/agent/classify-intent -H "Content-Type: application/json" -d "{\"question\":\"lots of nulls in my dataset, how to handle them?\"}"
```
**Expected:** `{"intent":"missing_data",...}`

---

### TEST 15: Statistical Test — PSI (Drift Detection)
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/statistical-tests -H "Content-Type: application/json" -d "{\"test_type\":\"psi\",\"reference_histogram\":[100,200,300,250,150],\"current_histogram\":[300,100,50,200,350]}"
```
**Expected:** `{"test":"PSI","statistic":0.7+,"is_significant":true,"severity":"critical",...}`

---

### TEST 16: Statistical Test — Cohen's d (Effect Size)
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/statistical-tests -H "Content-Type: application/json" -d "{\"test_type\":\"cohens_d\",\"ref_mean\":50.0,\"ref_std\":10.0,\"cur_mean\":55.0,\"cur_std\":10.0}"
```
**Expected:** `{"test":"Cohen's d","statistic":0.5,"interpretation":"medium effect size",...}`

---

### TEST 17: Statistical Test — Chi-Square (Categorical Drift)
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/statistical-tests -H "Content-Type: application/json" -d "{\"test_type\":\"chi_square\",\"reference_distribution\":{\"cat_A\":0.4,\"cat_B\":0.35,\"cat_C\":0.25},\"current_distribution\":{\"cat_A\":0.2,\"cat_B\":0.5,\"cat_C\":0.3}}"
```
**Expected:** `{"test":"Chi-Square","statistic":...,"is_significant":...}`

---

### TEST 18: Statistical Test — KS (Distribution Shift)
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/statistical-tests -H "Content-Type: application/json" -d "{\"test_type\":\"ks\",\"reference_percentiles\":{\"1%\":1,\"25%\":25,\"50%\":50,\"75%\":75,\"99%\":99},\"current_percentiles\":{\"1%\":10,\"25%\":35,\"50%\":55,\"75%\":80,\"99%\":99}}"
```
**Expected:** `{"test":"KS","statistic":0.1,...}`

---

### TEST 19: Statistical Test — Normality
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/statistical-tests -H "Content-Type: application/json" -d "{\"test_type\":\"normality\",\"ref_mean\":0.0,\"ref_std\":1.0,\"skewness\":0.1,\"kurtosis\":3.05,\"n_samples\":1000}"
```
**Expected:** `{"is_normal":true,"tests":[...],"suggested_transform":"none"}`

---

### TEST 20: Full Drift Suite (Multi-Test)
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/drift-suite -H "Content-Type: application/json" -d "{\"feature\":\"age\",\"ref_stats\":{\"mean\":35,\"std\":10},\"cur_stats\":{\"mean\":45,\"std\":12},\"ref_histogram\":[100,200,300,200,100],\"cur_histogram\":[50,100,200,300,250],\"ref_percentiles\":{\"1%%\":15,\"25%%\":28,\"50%%\":35,\"75%%\":42,\"99%%\":55},\"cur_percentiles\":{\"1%%\":20,\"25%%\":36,\"50%%\":45,\"75%%\":52,\"99%%\":65}}"
```
**Expected:** `{"feature":"age","consensus_score":0.7+,"tests_run":...}`

---

### TEST 21: Record User Action on Recommendation
```bash
curl -X POST http://localhost:8001/api/v1/agent/feedback/record-action -H "Content-Type: application/json" -d "{\"user_id\":\"user_123\",\"recommendation_id\":\"rec_001\",\"action\":\"followed\",\"details\":{\"algorithm\":\"xgboost\"}}"
```
**Expected:** `{"success":true,"recommendation_id":"rec_001"}`

---

### TEST 22: Record Outcome
```bash
curl -X POST http://localhost:8001/api/v1/agent/feedback/record-outcome -H "Content-Type: application/json" -d "{\"user_id\":\"user_123\",\"recommendation_id\":\"rec_001\",\"metrics_after\":{\"accuracy\":0.92,\"f1\":0.88},\"metrics_before\":{\"accuracy\":0.85,\"f1\":0.80}}"
```
**Expected:** `{"verdict":"improved",...}`

---

### TEST 23: Auto-Detect Outcomes
```bash
curl -X POST http://localhost:8001/api/v1/agent/feedback/auto-detect -H "Content-Type: application/json" -d "{\"user_id\":\"user_123\",\"project_id\":\"project_001\",\"current_metrics\":{\"accuracy\":0.93}}"
```
**Expected:** `{"resolved":[...],"count":...}`

---

### TEST 24: Effectiveness Report
```bash
curl "http://localhost:8001/api/v1/agent/feedback/effectiveness?user_id=user_123"
```
**Expected:** `{...effectiveness metrics...}`

---

### TEST 25: A/B Benchmarking (Followed vs Ignored)
```bash
curl "http://localhost:8001/api/v1/agent/feedback/benchmarking?user_id=user_123"
```
**Expected:** `{...benchmarking report...}`

---

### TEST 26: List Cost Matrices
```bash
curl http://localhost:8001/api/v1/agent/cost-matrices
```
**Expected:** `{"matrices":[{"domain":"healthcare","use_cases":[...]},...]}`

---

### TEST 27: Healthcare Business Impact
```bash
curl -X POST http://localhost:8001/api/v1/agent/cost-matrices/impact -H "Content-Type: application/json" -d "{\"precision\":0.85,\"recall\":0.92,\"domain\":\"healthcare\",\"use_case\":\"diagnosis\",\"n_predictions\":10000}"
```
**Expected:** `{"costs":{"net_savings":16000000+,...},...}`

---

### TEST 28: Enhanced Domain-Aware Insights
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/insights -H "Content-Type: application/json" -d "{\"screen\":\"eda\",\"user_domain\":\"healthcare\",\"user_id\":\"doctor_123\"}"
```
**Expected:** `{"insights":[...],...}`

---

### TEST 29: Enhanced Semantic Q&A
```bash
curl -X POST http://localhost:8001/api/v1/agent/enhanced/ask -H "Content-Type: application/json" -d "{\"question\":\"my model is overfitting badly, what should I do?\",\"screen\":\"evaluation\"}"
```
**Expected:** `{"answer":"...","source":"...",...}`

---

## SUMMARY

| # | Endpoint | Method | What It Tests |
|---|----------|--------|---------------|
| 1 | `/health` | GET | Core component status |
| 2 | `/health/enhanced` | GET | ALL component status (v4) |
| 3-4 | `/insights` | POST | Auto-insights (EDA, Training screens) |
| 5 | `/ask` | POST | Q&A (algorithm advice) |
| 6 | `/validate` | POST | Training readiness gate |
| 7 | `/recommend` | POST | Recommendations |
| 8 | `/feedback` | POST | Thumbs up/down |
| 9 | `/enhanced/domains` | GET | 13 domain profiles |
| 10-11 | `/enhanced/detect-domain` | POST | Healthcare & Finance detection |
| 12-14 | `/classify-intent` | POST | 3 intent types |
| 15-19 | `/enhanced/statistical-tests` | POST | PSI, Cohen's d, Chi², KS, Normality |
| 20 | `/enhanced/drift-suite` | POST | Multi-test drift |
| 21 | `/feedback/record-action` | POST | Log followed/ignored |
| 22 | `/feedback/record-outcome` | POST | Log outcome metrics |
| 23 | `/feedback/auto-detect` | POST | Auto-resolve outcomes |
| 24 | `/feedback/effectiveness` | GET | Success rates |
| 25 | `/feedback/benchmarking` | GET | A/B comparison |
| 26 | `/cost-matrices` | GET | List cost matrices |
| 27 | `/cost-matrices/impact` | POST | Business $ impact |
| 28 | `/enhanced/insights` | POST | Domain-aware insights |
| 29 | `/enhanced/ask` | POST | Semantic Q&A |

**All 23 unique endpoints × 29 test cases = 100% API coverage ✅**
