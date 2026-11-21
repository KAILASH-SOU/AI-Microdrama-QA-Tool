"""Swappable LLM Adapter, Deterministic SHA-256 Caching & Token Cost Tracker.

Supports:
- Google Gemini (via google-genai SDK, default gemini-2.5-flash / gemini-1.5-flash)
- OpenAI (via openai SDK, default gpt-4o-mini)
- Anthropic (via anthropic SDK, default claude-3-5-haiku-20241022)
- Groq / OpenAI-compatible endpoints (configurable base_url + api_key + model)
- Deterministic Mock mode for offline execution and cost-free CI/CD testing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

# Load dotenv if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

T = TypeVar("T", bound=BaseModel)


class LLMProvider(str, Enum):
    """Supported LLM Provider identifiers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GENERIC_OPENAI = "generic_openai"
    MOCK = "mock"


# Pricing in USD per 1 Million tokens (prompt_cost_per_m, completion_cost_per_m)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Gemini
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Anthropic
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    # Groq
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # Mock
    "mock-model": (0.0, 0.0),
}


class TokenUsage(BaseModel):
    """Tracks token consumption, latency, and estimated cost for a single LLM invocation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_sec: float = 0.0
    cached: bool = False
    provider: str = ""
    model: str = ""
    stage: str = "general"


class UsageTracker:
    """Thread-safe aggregator for token counts, latencies, and financial costs across pipeline stages."""

    def __init__(self):
        self.records: List[TokenUsage] = []

    def record(self, usage: TokenUsage) -> None:
        self.records.append(usage)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def cached_calls(self) -> int:
        return sum(1 for r in self.records if r.cached)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.estimated_cost_usd for r in self.records), 6)

    @property
    def total_latency_sec(self) -> float:
        return round(sum(r.latency_sec for r in self.records), 3)

    def summary(self) -> Dict[str, Any]:
        """Returns structured usage metrics summary."""
        stages: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            st = r.stage or "general"
            if st not in stages:
                stages[st] = {
                    "calls": 0,
                    "cached": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": 0.0,
                }
            stages[st]["calls"] += 1
            if r.cached:
                stages[st]["cached"] += 1
            stages[st]["prompt_tokens"] += r.prompt_tokens
            stages[st]["completion_tokens"] += r.completion_tokens
            stages[st]["cost_usd"] = round(stages[st]["cost_usd"] + r.estimated_cost_usd, 6)
            stages[st]["latency_sec"] = round(stages[st]["latency_sec"] + r.latency_sec, 3)

        return {
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "cache_hit_rate_pct": round((self.cached_calls / self.total_calls * 100.0) if self.total_calls > 0 else 0.0, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_sec": self.total_latency_sec,
            "by_stage": stages,
        }

    def reset(self) -> None:
        self.records.clear()


def clean_and_extract_json(raw_text: str) -> Dict[str, Any]:
    """Robustly cleans LLM output, extracting JSON from markdown fences or text boundaries.

    Handles:
    - ```json ... ``` blocks
    - Stray prose before/after JSON
    - Trailing commas before closing braces/brackets
    """
    cleaned = raw_text.strip()

    # Step 1: Strip markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    # Step 2: Locate outermost JSON delimiters
    start_bracket = cleaned.find("[")
    start_brace = cleaned.find("{")

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        # Object starts first
        end_brace = cleaned.rfind("}")
        if end_brace != -1:
            cleaned = cleaned[start_brace : end_brace + 1]
    elif start_bracket != -1:
        # Array starts first
        end_bracket = cleaned.rfind("]")
        if end_bracket != -1:
            cleaned = cleaned[start_bracket : end_bracket + 1]

    # Step 3: Remove trailing commas before closing braces
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse valid JSON from text: {raw_text[:200]}... Error: {exc}") from exc


class LLMClient:
    """Production swappable LLM client with disk caching, token tracking, and exponential backoff."""

    def __init__(
        self,
        provider: Optional[Union[LLMProvider, str]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Union[str, Path] = ".cache",
        use_cache: bool = True,
        mock: bool = False,
        max_retries: int = 3,
        initial_retry_delay_sec: float = 1.0,
    ):
        self.mock = mock
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.initial_retry_delay_sec = initial_retry_delay_sec
        self.tracker = UsageTracker()

        # Determine Provider
        if isinstance(provider, LLMProvider):
            self.provider = provider
        elif isinstance(provider, str):
            try:
                self.provider = LLMProvider(provider.strip().lower())
            except ValueError:
                self.provider = LLMProvider.GEMINI
        else:
            env_provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
            try:
                self.provider = LLMProvider(env_provider)
            except ValueError:
                self.provider = LLMProvider.GEMINI

        # Resolve Model
        self.model = model or self._default_model_for_provider(self.provider)

        # Resolve API Key
        self.api_key = api_key or self._resolve_api_key(self.provider)
        self.base_url = base_url or os.environ.get("GENERIC_OPENAI_BASE_URL")

        # Auto-fallback to mock mode if no key is provided
        if not self.api_key and self.provider != LLMProvider.MOCK:
            self.mock = True

        # Initialize underlying SDK clients lazily
        self._gemini_client = None
        self._openai_client = None
        self._anthropic_client = None
        self._groq_client = None

    def _default_model_for_provider(self, provider: LLMProvider) -> str:
        if provider == LLMProvider.GEMINI:
            return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        elif provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        elif provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        elif provider == LLMProvider.GROQ:
            return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif provider == LLMProvider.GENERIC_OPENAI:
            return os.environ.get("GENERIC_OPENAI_MODEL", "default")
        return "mock-model"

    def _resolve_api_key(self, provider: LLMProvider) -> Optional[str]:
        if provider == LLMProvider.GEMINI:
            return os.environ.get("GEMINI_API_KEY")
        elif provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        elif provider == LLMProvider.GROQ:
            return os.environ.get("GROQ_API_KEY")
        elif provider == LLMProvider.GENERIC_OPENAI:
            return os.environ.get("GENERIC_OPENAI_API_KEY")
        return None

    def _compute_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        prompt_version: str,
    ) -> str:
        """Generates deterministic SHA-256 hash key for disk caching."""
        hasher = hashlib.sha256()
        payload = f"{self.provider.value}:{self.model}:{prompt_version}:{system_prompt or ''}:{prompt}"
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        prov_dir = self.cache_dir / self.provider.value
        prov_dir.mkdir(parents=True, exist_ok=True)
        return prov_dir / f"{cache_key}.json"

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates USD cost based on published model token pricing."""
        rates = MODEL_PRICING.get(self.model, (0.10, 0.40))
        cost = (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000.0
        return round(cost, 6)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        stage: str = "general",
        prompt_version: str = "v1",
        temperature: float = 0.0,
    ) -> Tuple[Union[Dict[str, Any], T], TokenUsage]:
        """Executes LLM request with disk caching, exponential backoff, and JSON/Pydantic validation.

        Returns:
            Tuple of (parsed_json_or_pydantic_instance, TokenUsage)
        """
        cache_key = self._compute_cache_key(prompt, system_prompt, prompt_version)
        cache_path = self._get_cache_path(cache_key)

        # Check Disk Cache
        if self.use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as fp:
                    cached_data = json.load(fp)

                raw_json = cached_data["response_json"]
                usage = TokenUsage(
                    prompt_tokens=cached_data.get("prompt_tokens", 0),
                    completion_tokens=cached_data.get("completion_tokens", 0),
                    total_tokens=cached_data.get("total_tokens", 0),
                    estimated_cost_usd=0.0,  # Cached runs cost $0
                    latency_sec=0.0,
                    cached=True,
                    provider=self.provider.value,
                    model=self.model,
                    stage=stage,
                )
                self.tracker.record(usage)

                if response_model is not None:
                    parsed_model = response_model.model_validate(raw_json)
                    return parsed_model, usage
                return raw_json, usage
            except Exception:
                # If cache read or validation fails, proceed to live call
                pass

        # If Mock Mode or No API Key, generate deterministic mock
        if self.mock or not self.api_key:
            return self._generate_mock(prompt, system_prompt, response_model, stage, cache_path)

        # Live Execution with Exponential Backoff and Pydantic Self-Correction
        current_prompt = prompt
        attempts = 0
        last_error: Optional[Exception] = None

        while attempts < self.max_retries:
            attempts += 1
            t0 = time.time()

            try:
                raw_response_text, p_tokens, c_tokens = self._dispatch_call(
                    current_prompt, system_prompt, temperature
                )
                latency = round(time.time() - t0, 3)

                # Extract and parse JSON
                parsed_json = clean_and_extract_json(raw_response_text)

                # Validate against Pydantic model if requested
                result_obj: Any = parsed_json
                if response_model is not None:
                    result_obj = response_model.model_validate(parsed_json)

                # Record token metrics
                cost = self._estimate_cost(p_tokens, c_tokens)
                usage = TokenUsage(
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens,
                    estimated_cost_usd=cost,
                    latency_sec=latency,
                    cached=False,
                    provider=self.provider.value,
                    model=self.model,
                    stage=stage,
                )
                self.tracker.record(usage)

                # Save to Disk Cache
                if self.use_cache:
                    with open(cache_path, "w", encoding="utf-8") as fp:
                        json.dump(
                            {
                                "response_json": parsed_json,
                                "prompt_tokens": p_tokens,
                                "completion_tokens": c_tokens,
                                "total_tokens": p_tokens + c_tokens,
                                "timestamp": time.time(),
                            },
                            fp,
                            indent=2,
                        )

                return result_obj, usage

            except (ValidationError, ValueError) as val_err:
                last_error = val_err
                # Self-correction prompt for next attempt
                current_prompt = (
                    f"{prompt}\n\n"
                    f"[ERROR ON PREVIOUS ATTEMPT]: Output failed validation: {str(val_err)}\n"
                    f"Please correct the output format strictly according to the required JSON schema."
                )
            except Exception as api_err:
                last_error = api_err
                # Rate limit (429) backoff
                err_str = str(api_err).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str or "exhausted" in err_str
                delay = self.initial_retry_delay_sec * (2 ** (attempts - 1))
                if is_rate_limit:
                    delay += 1.0
                time.sleep(delay)

        # Fallback to mock if live attempts failed
        print(f"Warning: Live call failed after {self.max_retries} attempts ({last_error}). Falling back to mock generator.")
        return self._generate_mock(prompt, system_prompt, response_model, stage, cache_path)

    def _dispatch_call(
        self, prompt: str, system_prompt: Optional[str], temperature: float
    ) -> Tuple[str, int, int]:
        """Routes call to the configured SDK provider."""
        if self.provider == LLMProvider.GEMINI:
            return self._call_gemini(prompt, system_prompt, temperature)
        elif self.provider in [LLMProvider.OPENAI, LLMProvider.GENERIC_OPENAI]:
            return self._call_openai(prompt, system_prompt, temperature)
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(prompt, system_prompt, temperature)
        elif self.provider == LLMProvider.GROQ:
            return self._call_groq(prompt, system_prompt, temperature)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_gemini(
        self, prompt: str, system_prompt: Optional[str], temperature: float
    ) -> Tuple[str, int, int]:
        from google import genai
        from google.genai import types

        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.api_key)

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            system_instruction=system_prompt if system_prompt else None,
        )

        response = self._gemini_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        p_tokens = getattr(response.usage_metadata, "prompt_token_count", len(prompt) // 4)
        c_tokens = getattr(response.usage_metadata, "candidates_token_count", len(response.text or "") // 4)
        return response.text or "{}", p_tokens, c_tokens

    def _call_openai(
        self, prompt: str, system_prompt: Optional[str], temperature: float
    ) -> Tuple[str, int, int]:
        from openai import OpenAI

        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or "{}"
        p_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
        c_tokens = response.usage.completion_tokens if response.usage else len(text) // 4
        return text, p_tokens, c_tokens

    def _call_anthropic(
        self, prompt: str, system_prompt: Optional[str], temperature: float
    ) -> Tuple[str, int, int]:
        from anthropic import Anthropic

        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=self.api_key)

        response = self._anthropic_client.messages.create(
            model=self.model,
            system=system_prompt or "You are a precise JSON response generator.",
            messages=[{"role": "user", "content": f"{prompt}\n\nRespond with valid JSON only."}],
            temperature=temperature,
            max_tokens=2048,
        )

        text = response.content[0].text if response.content else "{}"
        p_tokens = response.usage.input_tokens if hasattr(response, "usage") else len(prompt) // 4
        c_tokens = response.usage.output_tokens if hasattr(response, "usage") else len(text) // 4
        return text, p_tokens, c_tokens

    def _call_groq(
        self, prompt: str, system_prompt: Optional[str], temperature: float
    ) -> Tuple[str, int, int]:
        from groq import Groq

        if self._groq_client is None:
            self._groq_client = Groq(api_key=self.api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._groq_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or "{}"
        p_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
        c_tokens = response.usage.completion_tokens if response.usage else len(text) // 4
        return text, p_tokens, c_tokens

    def _generate_mock(
        self,
        prompt: str,
        system_prompt: Optional[str],
        response_model: Optional[Type[T]],
        stage: str,
        cache_path: Path,
    ) -> Tuple[Union[Dict[str, Any], T], TokenUsage]:
        """Generates deterministic mock responses when no API key is available."""
        p_tokens = len(prompt) // 4
        c_tokens = 60

        mock_data: Dict[str, Any] = {}

        # Heuristic to detect Module 1 EndingAudit / EpisodeEndingLLMResponse schema
        if response_model is not None and response_model.__name__ in ["EndingAudit", "EpisodeEndingLLMResponse"]:
            # Extract episode number from prompt if possible
            m_ep = re.search(r"episode\s*([0-9]+)", prompt, re.IGNORECASE)
            ep_num = int(m_ep.group(1)) if m_ep else 1

            # Deterministic classification based on episode pattern
            # For demonstration: ep 5-8 cliffhanger streak, ep 9 payoff
            if ep_num in [5, 6, 7, 8, 10, 11, 12]:
                classification = "HARD_CLIFFHANGER"
                tension = 9
                payoff = 2
                verdict = f"Episode {ep_num} ends on an unresolved acute danger or confrontation without payoff."
                pivot = "Add a micro-payoff confirming one revelation or emotional resolution before cut to black."
            elif ep_num in [9, 14, 25]:
                classification = "MICRO_PAYOFF_WITH_HOOK"
                tension = 7
                payoff = 6
                verdict = f"Episode {ep_num} resolves immediate danger while setting up next conflict hook."
                pivot = None
            elif ep_num == 50:
                classification = "SATISFYING_CLOSE"
                tension = 4
                payoff = 9
                verdict = "Season finale provides strong emotional payoff and arc resolution."
                pivot = None
            else:
                classification = "HARD_CLIFFHANGER" if ep_num % 2 == 1 else "MICRO_PAYOFF_WITH_HOOK"
                tension = 8 if ep_num % 2 == 1 else 6
                payoff = 2 if ep_num % 2 == 1 else 6
                verdict = f"Episode {ep_num} pacing audit."
                pivot = None

            # Extract beat_ids mentioned in prompt
            cited_beats = re.findall(r"E\d+_B\d+", prompt)
            evidence_id = cited_beats[-1] if cited_beats else f"E{ep_num:02d}_B020"

            if response_model.__name__ == "EpisodeEndingLLMResponse":
                mock_data = {
                    "ending_type": classification,
                    "tension_score": tension,
                    "payoff_score": payoff,
                    "evidence_beat_id": evidence_id,
                    "one_line_verdict": verdict,
                    "suggested_pivot": pivot,
                }
            else:
                mock_data = {
                    "episode_number": ep_num,
                    "ending_type": classification,
                    "tension_score": tension,
                    "payoff_score": payoff,
                    "evidence_beat_id": evidence_id,
                    "evidence_beat_valid": True,
                    "one_line_verdict": verdict,
                    "is_paywall_danger_zone": 5 <= ep_num <= 15,
                    "streak_count": 1,
                    "fatigue_flag": False,
                    "suggested_pivot": pivot,
                }
        elif response_model is not None and response_model.__name__ == "VoicePrediction":
            m_beat = re.search(r"E\d+_B\d+", prompt)
            beat_id = m_beat.group(0) if m_beat else "E01_B001"
            # Extract candidates list from prompt if available
            cand_match = re.findall(r"[A-Z]{3,}", prompt)
            candidates = [c for c in cand_match if c not in ["JSON", "PROMPT", "SPEAKER", "THE"]]
            predicted = candidates[0] if candidates else "JULIANA"

            mock_data = {
                "beat_id": beat_id,
                "masked_text": "Sample dialogue line",
                "true_speaker": predicted,
                "predicted_speaker": predicted,
                "is_correct": True,
                "confidence": 0.88,
                "reasoning": "Mock voice test attribution based on vocabulary tone.",
                "interchangeable_flag": False,
                "stylometric_scores": {},
            }
        else:
            mock_data = {
                "status": "success",
                "message": "Deterministic mock response for smoke testing",
                "detected_stage": stage,
            }

        usage = TokenUsage(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
            estimated_cost_usd=0.0,
            latency_sec=0.005,
            cached=False,
            provider="mock",
            model="mock-model",
            stage=stage,
        )
        self.tracker.record(usage)

        # Cache mock data
        if self.use_cache:
            with open(cache_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "response_json": mock_data,
                        "prompt_tokens": p_tokens,
                        "completion_tokens": c_tokens,
                        "total_tokens": p_tokens + c_tokens,
                        "timestamp": time.time(),
                    },
                    fp,
                    indent=2,
                )

        if response_model is not None:
            return response_model.model_validate(mock_data), usage
        return mock_data, usage
