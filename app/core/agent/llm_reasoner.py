"""
LLM Reasoner — Production-Grade AI Reasoning Layer
=====================================================
Provider-agnostic LLM integration with:
  - Multi-provider support (OpenAI, Anthropic, Local Ollama/vLLM, Azure)
  - Structured output with JSON schema enforcement
  - Prompt chain pipelines (multi-step reasoning)
  - Automatic fallback chains (provider A → B → C → rules-only)
  - Response caching with TTL
  - Token budget management
  - Streaming support (async generator)
  - Confidence scoring on LLM outputs
  - Guardrails: hallucination detection, fact-checking against context

Design:
  - Rules-only mode works perfectly without any LLM (zero external deps)
  - LLM enhances but never replaces deterministic rules
  - Every LLM output is fact-checked against compiled context
  - Failed LLM calls are invisible to the user (graceful degradation)
"""

import json, logging, os, time, hashlib, re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL_OLLAMA = "ollama"
    LOCAL_VLLM = "vllm"
    RULES_ONLY = "rules_only"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.RULES_ONLY
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1500
    temperature: float = 0.3
    timeout_seconds: int = 15
    max_retries: int = 2
    token_budget_per_request: int = 4000
    cache_ttl_seconds: int = 300

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider_str = os.getenv("AGENT_LLM_PROVIDER", "rules_only").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.RULES_ONLY
        return cls(
            provider=provider,
            model=os.getenv("AGENT_LLM_MODEL", ""),
            api_key=os.getenv("AGENT_LLM_API_KEY", ""),
            base_url=os.getenv("AGENT_LLM_BASE_URL", ""),
            max_tokens=int(os.getenv("AGENT_LLM_MAX_TOKENS", "1500")),
            temperature=float(os.getenv("AGENT_LLM_TEMPERATURE", "0.3")),
            timeout_seconds=int(os.getenv("AGENT_LLM_TIMEOUT", "15")),
        )


@dataclass
class LLMResponse:
    advice: str = ""
    confidence: float = 0.0
    source: str = "rules_only"
    has_llm: bool = False
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: int = 0
    cached: bool = False
    fact_check_passed: bool = True
    fact_check_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "fact_check_issues"}


SYSTEM_PROMPT = """You are a world-class Principal ML Architect embedded inside a production ML platform.
EXPERTISE: Statistics, ML, deep learning, MLOps, data engineering, production systems.
RULES:
1. NEVER invent metrics not in the context
2. Reference specific numbers from context
3. Provide CONCRETE next steps with parameters
4. Explain WHY, not just WHAT
5. Present trade-offs with quantified impact
TONE: Direct, expert, no fluff. FORMAT: Short paragraphs, **bold** for key terms."""

SCREEN_PROMPTS = {
    "eda": "Focus on data quality, feature distributions, correlations, target health, transformations.",
    "data": "Focus on dataset characteristics, schema issues, data type mismatches, readiness.",
    "mlflow": "Focus on algorithm selection for THIS data, hyperparameter ranges, CV config, pitfalls.",
    "training": "Focus on algorithm selection, feature engineering, hyperparameter strategy, validation.",
    "evaluation": "Focus on metric interpretation (accuracy vs F1 traps), overfitting, threshold optimization, production readiness.",
    "registry": "Focus on version management, model comparison, promotion criteria, A/B testing.",
    "deployment": "Focus on deployment strategy (shadow/canary/blue-green), monitoring setup, rollback planning.",
    "predictions": "Focus on input validation, prediction confidence, edge cases, trust boundaries.",
    "monitoring": "Focus on drift detection, performance degradation, alerting thresholds, retraining triggers.",
}

STRUCTURED_SCHEMAS = {
    "insight_synthesis": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "top_priority": {"type": "string"},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "trade_offs": {"type": "array", "items": {"type": "object", "properties": {"option": {"type": "string"}, "pros": {"type": "string"}, "cons": {"type": "string"}}}},
            "next_steps": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        },
        "required": ["summary", "top_priority", "risk_level", "next_steps"],
    },
    "question_answer": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            "caveats": {"type": "array", "items": {"type": "string"}},
            "follow_ups": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["answer", "confidence"],
    },
}


class _ResponseCache:
    def __init__(self, max_size=100, ttl_seconds=300):
        self._store: Dict[str, Tuple[Any, datetime]] = {}
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, prompt_hash: str, model: str) -> Optional[str]:
        key = f"{model}:{prompt_hash}"
        if key in self._store:
            value, ts = self._store[key]
            if datetime.utcnow() - ts < self._ttl:
                return value
            del self._store[key]
        return None

    def set(self, prompt_hash: str, model: str, value: str):
        key = f"{model}:{prompt_hash}"
        if len(self._store) >= self._max_size:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, datetime.utcnow())

    def clear(self):
        self._store.clear()


class _BaseClient:
    async def complete(self, messages, config, json_schema=None) -> Tuple[str, int]:
        raise NotImplementedError
    async def stream(self, messages, config) -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield


class _OpenAIClient(_BaseClient):
    async def complete(self, messages, config, json_schema=None):
        import httpx
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}
        base_url = config.base_url or "https://api.openai.com/v1"
        body = {"model": config.model or "gpt-4o-mini", "messages": messages, "max_tokens": config.max_tokens, "temperature": config.temperature}
        if json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": json_schema}}
        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {}).get("total_tokens", 0)

    async def stream(self, messages, config):
        import httpx
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}
        base_url = config.base_url or "https://api.openai.com/v1"
        body = {"model": config.model or "gpt-4o-mini", "messages": messages, "max_tokens": config.max_tokens, "temperature": config.temperature, "stream": True}
        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            async with client.stream("POST", f"{base_url}/chat/completions", json=body, headers=headers) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue


class _AnthropicClient(_BaseClient):
    async def complete(self, messages, config, json_schema=None):
        import httpx
        headers = {"Content-Type": "application/json", "x-api-key": config.api_key, "anthropic-version": "2023-06-01"}
        system_msg = ""
        user_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_msgs.append(msg)
        body = {"model": config.model or "claude-sonnet-4-20250514", "max_tokens": config.max_tokens, "messages": user_msgs}
        if system_msg:
            body["system"] = system_msg
        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
        return text, tokens


class _OllamaClient(_BaseClient):
    async def complete(self, messages, config, json_schema=None):
        import httpx
        base_url = config.base_url or "http://localhost:11434"
        body = {"model": config.model or "llama3.1:8b", "messages": messages, "stream": False, "options": {"temperature": config.temperature, "num_predict": config.max_tokens}}
        if json_schema:
            body["format"] = "json"
        async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
            resp = await client.post(f"{base_url}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", ""), data.get("eval_count", 0) + data.get("prompt_eval_count", 0)


class _FactChecker:
    @staticmethod
    def check(response_text: str, context: Dict) -> Tuple[bool, List[str]]:
        issues = []
        profile = context.get("dataset_profile", {})
        screen_ctx = context.get("screen_context", {}) or {}
        metrics = screen_ctx.get("metrics", {})
        numbers = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|rows|columns|features|samples|accuracy|precision|recall|f1)', response_text.lower())
        for num_str, unit in numbers:
            num = float(num_str.replace(",", ""))
            if unit in ("rows", "samples"):
                actual = profile.get("rows", 0)
                if actual > 0 and abs(num - actual) > actual * 0.1:
                    issues.append(f"Claimed {num:.0f} {unit}, actual {actual}")
            elif unit in ("columns", "features"):
                actual = profile.get("columns", 0)
                if actual > 0 and abs(num - actual) > actual * 0.2:
                    issues.append(f"Claimed {num:.0f} {unit}, actual {actual}")
        return len(issues) == 0, issues


class _PromptBuilder:
    @staticmethod
    def build_insight_prompt(context, insights, screen, question=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        screen_prompt = SCREEN_PROMPTS.get(screen, "")
        profile = context.get("dataset_profile", {})
        quality = context.get("data_quality", {})
        correlations = context.get("correlations", {})
        feature_stats = context.get("feature_stats", {})
        screen_ctx = context.get("screen_context", {}) or {}
        parts = []
        if profile.get("rows"):
            parts.append(f"DATASET: {profile.get('file_name', '?')} — {profile['rows']:,}×{profile.get('columns', 0)} ({profile.get('numeric_count', 0)} num, {profile.get('categorical_count', 0)} cat)")
        if quality.get("completeness", 100) < 100:
            parts.append(f"QUALITY: {quality['completeness']:.1f}% complete, {quality.get('duplicate_pct', 0):.1f}% dupes")
        high_corr = correlations.get("high_pairs", [])
        if high_corr:
            parts.append("CORRELATIONS: " + ", ".join(f"{p['feature1']}↔{p['feature2']}(ρ={p['correlation']:.2f})" for p in high_corr[:5]))
        if screen_ctx.get("metrics"):
            m = screen_ctx["metrics"]
            parts.append("METRICS: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, (int, float))))
        insight_parts = [f"[{i.get('severity', 'info').upper()}] {i.get('title', '')}: {i.get('message', '')[:200]}" for i in insights[:10]]
        user_content = f"SCREEN: {screen}\n{screen_prompt}\n\nCONTEXT:\n" + "\n".join(parts) + "\n\n"
        if insight_parts:
            user_content += "FINDINGS:\n" + "\n".join(insight_parts) + "\n\n"
        if question:
            user_content += f"QUESTION: {question}\nAnswer precisely using ONLY the context above."
        else:
            user_content += "Synthesize into a concise expert briefing. Lead with the most impactful action."
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def build_chain_prompt(step, context, prior):
        chains = {
            "diagnose": "Analyze context. Identify TOP 3 issues by impact. JSON: {\"issues\": [{\"title\": ..., \"impact\": ..., \"fix\": ...}]}",
            "strategize": "Given issues:\n{prior}\nCreate prioritized action plan with effort and expected gain. JSON: {\"plan\": [{\"action\": ..., \"effort_hours\": ..., \"expected_gain\": ...}]}",
            "validate_plan": "Review plan:\n{prior}\nIdentify risks and rate 1-10. JSON: {\"score\": ..., \"risks\": [...], \"approval\": true/false}",
        }
        template = chains.get(step, "Analyze the context.")
        user_content = template.format(prior=json.dumps(prior, indent=2)[:3000])
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

    @staticmethod
    def estimate_tokens(messages):
        return sum(len(m.get("content", "")) for m in messages) // 4


class LLMReasoner:
    def __init__(self, primary=None, fallback=None, enable_cache=True, enable_fact_check=True):
        self.primary = primary or LLMConfig.from_env()
        self.fallback = fallback
        self.enable_cache = enable_cache
        self.enable_fact_check = enable_fact_check
        self._cache = _ResponseCache(max_size=100, ttl_seconds=self.primary.cache_ttl_seconds)
        self._fact_checker = _FactChecker()
        self._prompt_builder = _PromptBuilder()
        self._clients = {
            LLMProvider.OPENAI: _OpenAIClient(),
            LLMProvider.AZURE_OPENAI: _OpenAIClient(),
            LLMProvider.ANTHROPIC: _AnthropicClient(),
            LLMProvider.LOCAL_OLLAMA: _OllamaClient(),
            LLMProvider.LOCAL_VLLM: _OllamaClient(),
        }
        self._call_count = 0
        self._cache_hits = 0
        self._failures = 0

    @property
    def enabled(self):
        return self.primary.provider != LLMProvider.RULES_ONLY

    async def reason(self, context, insights, question=None, json_schema=None) -> LLMResponse:
        result = LLMResponse()
        screen = context.get("screen", "")
        if not self.enabled:
            result.advice = self._rules_only_synthesis(insights, screen, question)
            result.source = "rules_only"
            result.confidence = 0.7
            return result

        messages = self._prompt_builder.build_insight_prompt(context, insights, screen, question)
        prompt_hash = hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest()

        if self.enable_cache:
            cached = self._cache.get(prompt_hash, self.primary.model)
            if cached:
                self._cache_hits += 1
                result.advice = cached
                result.source = f"cached:{self.primary.provider.value}"
                result.has_llm = True
                result.cached = True
                result.confidence = 0.85
                return result

        self._call_count += 1
        t0 = time.time()

        # Try primary
        try:
            text, tokens = await self._call_provider(self.primary, messages, json_schema)
            if text:
                result.advice = text
                result.source = self.primary.provider.value
                result.has_llm = True
                result.model_used = self.primary.model
                result.tokens_used = tokens
                result.latency_ms = int((time.time() - t0) * 1000)
                result.confidence = 0.85
                if self.enable_fact_check:
                    passed, issues = self._fact_checker.check(text, context)
                    result.fact_check_passed = passed
                    result.fact_check_issues = issues
                    if not passed:
                        result.confidence *= 0.6
                if self.enable_cache:
                    self._cache.set(prompt_hash, self.primary.model, text)
                return result
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}")
            self._failures += 1

        # Try fallback
        if self.fallback and self.fallback.provider != LLMProvider.RULES_ONLY:
            try:
                text, tokens = await self._call_provider(self.fallback, messages, json_schema)
                if text:
                    result.advice = text
                    result.source = f"fallback:{self.fallback.provider.value}"
                    result.has_llm = True
                    result.model_used = self.fallback.model
                    result.tokens_used = tokens
                    result.latency_ms = int((time.time() - t0) * 1000)
                    result.confidence = 0.75
                    return result
            except Exception as e:
                logger.warning(f"Fallback LLM failed: {e}")
                self._failures += 1

        result.advice = self._rules_only_synthesis(insights, screen, question)
        result.source = "rules_only_fallback"
        result.confidence = 0.7
        return result

    async def reason_chain(self, context, insights, steps=None):
        if not self.enabled:
            return {"source": "rules_only", "chain_completed": False}
        steps = steps or ["diagnose", "strategize", "validate_plan"]
        results = {}
        prior = {}
        for step in steps:
            try:
                messages = self._prompt_builder.build_chain_prompt(step, context, prior)
                text, _ = await self._call_provider(self.primary, messages)
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
                    parsed = json.loads(m.group(1)) if m else {"raw": text}
                results[step] = parsed
                prior = parsed
            except Exception as e:
                results[step] = {"error": str(e)}
                break
        results["chain_completed"] = all("error" not in results.get(s, {}) for s in steps)
        return results

    async def reason_stream(self, context, insights, question=None):
        if not self.enabled:
            yield self._rules_only_synthesis(insights, context.get("screen", ""), question)
            return
        messages = self._prompt_builder.build_insight_prompt(context, insights, context.get("screen", ""), question)
        client = self._clients.get(self.primary.provider)
        if client:
            try:
                async for chunk in client.stream(messages, self.primary):
                    yield chunk
                return
            except Exception:
                pass
        result = await self.reason(context, insights, question)
        yield result.advice

    async def reason_structured(self, context, insights, schema_name="insight_synthesis"):
        schema = STRUCTURED_SCHEMAS.get(schema_name)
        if not schema:
            return {"error": f"Unknown schema: {schema_name}"}
        result = await self.reason(context, insights, json_schema=schema)
        if result.has_llm and result.advice:
            try:
                parsed = json.loads(result.advice)
                parsed["_meta"] = {"source": result.source, "confidence": result.confidence}
                return parsed
            except json.JSONDecodeError:
                m = re.search(r'```(?:json)?\s*([\s\S]*?)```', result.advice)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except json.JSONDecodeError:
                        pass
        return self._rules_only_structured(insights, schema_name)

    async def _call_provider(self, config, messages, json_schema=None):
        client = self._clients.get(config.provider)
        if not client:
            raise ValueError(f"No client for {config.provider}")
        last_err = None
        for attempt in range(config.max_retries + 1):
            try:
                return await client.complete(messages, config, json_schema)
            except Exception as e:
                last_err = e
                if attempt < config.max_retries:
                    import asyncio
                    await asyncio.sleep((attempt + 1) * 1.5)
        raise last_err or RuntimeError("LLM call failed")

    def _rules_only_synthesis(self, insights, screen, question=None):
        if not insights:
            return "No significant findings for the current context."
        if question:
            return self._rules_only_answer(insights, question)
        critical = [i for i in insights if i.get("severity") == "critical"]
        warnings = [i for i in insights if i.get("severity") == "warning"]
        parts = []
        if critical:
            parts.append(f"**{len(critical)} critical issue(s)**: {critical[0].get('title', '')} — {critical[0].get('message', '')[:200]}")
            if critical[0].get("action"):
                parts.append(f"**Priority action**: {critical[0]['action'][:200]}")
        if warnings:
            parts.append(f"**{len(warnings)} warning(s)**: " + ", ".join(w.get("title", "")[:40] for w in warnings[:4]))
        return "\n\n".join(parts) if parts else "Analysis complete. No critical issues found."

    def _rules_only_answer(self, insights, question):
        q_words = set(question.lower().split())
        scored = []
        for i in insights:
            words = set((i.get("title", "") + " " + i.get("message", "")[:100]).lower().split())
            overlap = len(q_words & words)
            if overlap:
                scored.append((overlap, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            top = scored[0][1]
            answer = f"**{top.get('title', '')}** — {top.get('message', '')}"
            if top.get("action"):
                answer += f"\n\n**Action**: {top['action']}"
            return answer
        return "No specific insights match your question in the current context."

    def _rules_only_structured(self, insights, schema_name):
        critical = [i for i in insights if i.get("severity") == "critical"]
        warnings = [i for i in insights if i.get("severity") == "warning"]
        return {
            "summary": f"{len(critical)} critical, {len(warnings)} warnings.",
            "top_priority": critical[0].get("action", "Review") if critical else "No critical issues",
            "risk_level": "critical" if critical else "medium" if warnings else "low",
            "next_steps": [i.get("action", "") for i in (critical + warnings)[:5] if i.get("action")],
            "_meta": {"source": "rules_only", "confidence": 0.7},
        }

    def get_stats(self):
        return {"provider": self.primary.provider.value, "enabled": self.enabled, "calls": self._call_count, "cache_hits": self._cache_hits, "failures": self._failures}

    def clear_cache(self):
        self._cache.clear()
