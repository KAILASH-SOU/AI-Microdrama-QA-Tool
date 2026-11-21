"""Unit and smoke tests for Swappable LLM Adapter, Disk Caching, and Cost Tracker."""

import shutil
from pathlib import Path
import pytest

from src.llm_client import (
    LLMClient,
    LLMProvider,
    TokenUsage,
    UsageTracker,
    clean_and_extract_json,
)
from src.schemas import EndingAudit, EndingClassification


def test_clean_and_extract_json():
    """Tests JSON cleaning with markdown code blocks, prose, and trailing commas."""
    # Plain JSON
    assert clean_and_extract_json('{"key": "value"}') == {"key": "value"}

    # Markdown block
    md_text = '```json\n{"status": "ok", "count": 42}\n```'
    assert clean_and_extract_json(md_text) == {"status": "ok", "count": 42}

    # Surrounding prose
    prose_text = 'Here is the analysis:\n{"result": true}\nHope that helps!'
    assert clean_and_extract_json(prose_text) == {"result": True}

    # Trailing commas
    trailing_comma = '{"items": [1, 2, ], "active": true, }'
    assert clean_and_extract_json(trailing_comma) == {"items": [1, 2], "active": True}


def test_token_usage_and_tracker():
    """Tests token accumulation and stage-level cost summaries."""
    tracker = UsageTracker()

    u1 = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        estimated_cost_usd=0.00015,
        latency_sec=0.5,
        cached=False,
        stage="fatigue_auditor",
    )
    u2 = TokenUsage(
        prompt_tokens=500,
        completion_tokens=100,
        total_tokens=600,
        estimated_cost_usd=0.0,
        latency_sec=0.0,
        cached=True,
        stage="fatigue_auditor",
    )
    u3 = TokenUsage(
        prompt_tokens=2000,
        completion_tokens=400,
        total_tokens=2400,
        estimated_cost_usd=0.0005,
        latency_sec=0.8,
        cached=False,
        stage="voice_auditor",
    )

    tracker.record(u1)
    tracker.record(u2)
    tracker.record(u3)

    summary = tracker.summary()
    assert summary["total_calls"] == 3
    assert summary["cached_calls"] == 1
    assert summary["cache_hit_rate_pct"] == 33.3
    assert summary["total_tokens"] == 4200
    assert summary["total_cost_usd"] == 0.00065
    assert "fatigue_auditor" in summary["by_stage"]
    assert summary["by_stage"]["fatigue_auditor"]["cached"] == 1


def test_gate2_smoke_test_and_caching_hit(tmp_path):
    """GATE 2 CHECK: 1-call smoke test verifying JSON parsing, disk caching hit, and cost tracking."""
    cache_dir = tmp_path / "test_cache"
    client = LLMClient(provider=LLMProvider.MOCK, cache_dir=cache_dir, use_cache=True)

    test_prompt = "Audit episode 5 ending beats: E05_B020, E05_B021 cut to black on a cliffhanger."

    # Call 1: Cold call (Must not be cached)
    result_1, usage_1 = client.generate(
        prompt=test_prompt,
        response_model=EndingAudit,
        stage="fatigue_auditor",
        prompt_version="v1",
    )

    assert isinstance(result_1, EndingAudit)
    assert result_1.episode_number == 5
    assert result_1.classification in [
        EndingClassification.HARD_CLIFFHANGER,
        EndingClassification.MICRO_PAYOFF_WITH_HOOK,
    ]
    assert usage_1.cached is False
    assert usage_1.total_tokens > 0

    # Verify cache file was created on disk
    prov_cache = cache_dir / "llm" / "mock"
    cache_files = list(prov_cache.glob("*.json"))
    assert len(cache_files) == 1, "Cache file should exist on disk"

    # Call 2: Repeat call with exact same parameters (Must hit cache, $0 cost, 0 latency)
    result_2, usage_2 = client.generate(
        prompt=test_prompt,
        response_model=EndingAudit,
        stage="fatigue_auditor",
        prompt_version="v1",
    )

    assert isinstance(result_2, EndingAudit)
    assert result_2.episode_number == result_1.episode_number
    assert result_2.classification == result_1.classification
    assert usage_2.cached is True
    assert usage_2.estimated_cost_usd == 0.0
    assert usage_2.latency_sec == 0.0

    # Summary assertion
    summary = client.tracker.summary()
    assert summary["total_calls"] == 2
    assert summary["cached_calls"] == 1
    assert summary["cache_hit_rate_pct"] == 50.0
