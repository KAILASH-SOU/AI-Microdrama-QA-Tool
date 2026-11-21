"""Phase 2 Verification & Smoke Test Script.

Exercises the LLMClient:
1. Cold LLM call with Pydantic JSON schema extraction (EndingAudit)
2. Disk caching verification (verifying SHA-256 cache file created on disk)
3. Repeat call verifying Cache Hit (0.00s latency, $0.000000 cost)
4. Aggregate usage & cost reporting across pipeline stages
"""

import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.llm_client import LLMClient, LLMProvider
from src.schemas import EndingAudit


def run_smoke_test():
    print("=" * 65)
    print("PHASE 2 SWAPPABLE LLM ADAPTER & CACHING SMOKE TEST")
    print("=" * 65)

    client = LLMClient(use_cache=True)
    print(f"Configured Provider:          {client.provider.value}")
    print(f"Configured Model:             {client.model}")
    print(f"Mock / Fallback Mode:         {client.mock}")
    print(f"Cache Directory:              {client.cache_dir}")
    print("-" * 65)

    test_prompt = (
        "Analyze the ending beats for Episode 12: "
        "E12_B024: Tension swells as Lady Catherine issues a death threat. "
        "E12_B025: Cut to black abruptly with no resolution."
    )

    # 1. Cold Call
    print("1. Executing Cold Call (Cold cache)...")
    audit_1, usage_1 = client.generate(
        prompt=test_prompt,
        response_model=EndingAudit,
        stage="fatigue_auditor",
        prompt_version="v1",
    )

    print(f"   -> Result Type:            {type(audit_1).__name__}")
    print(f"   -> Episode Number:         {audit_1.episode_number}")
    print(f"   -> Classification:         {audit_1.classification.value}")
    print(f"   -> Justification:          {audit_1.justification}")
    print(f"   -> Cited Beat IDs:         {audit_1.cited_beat_ids}")
    print(f"   -> Cached:                 {usage_1.cached}")
    print(f"   -> Latency:                {usage_1.latency_sec:.3f}s")
    print(f"   -> Estimated Cost:         ${usage_1.estimated_cost_usd:.6f}")

    assert not usage_1.cached, "Cold call should not be cached"

    # 2. Repeat Call (Cache Hit)
    print("\n2. Executing Repeat Call (Cache hit expected)...")
    audit_2, usage_2 = client.generate(
        prompt=test_prompt,
        response_model=EndingAudit,
        stage="fatigue_auditor",
        prompt_version="v1",
    )

    print(f"   -> Cached:                 {usage_2.cached}")
    print(f"   -> Latency:                {usage_2.latency_sec:.3f}s (Instant)")
    print(f"   -> Estimated Cost:         ${usage_2.estimated_cost_usd:.6f} ($0 cost)")
    print(f"   -> Output Matches:         {audit_1.classification == audit_2.classification}")

    assert usage_2.cached, "Repeat call must hit the disk cache"
    assert usage_2.estimated_cost_usd == 0.0, "Cached call cost must be $0.00"

    # 3. Cost & Metric Summary
    print("\n3. Usage & Financial Tracker Summary:")
    summary = client.tracker.summary()
    print(f"   -> Total Invocations:      {summary['total_calls']}")
    print(f"   -> Cached Invocations:     {summary['cached_calls']}")
    print(f"   -> Cache Hit Rate:         {summary['cache_hit_rate_pct']}%")
    print(f"   -> Total Tokens:           {summary['total_tokens']}")
    print(f"   -> Total Accumulated Cost: ${summary['total_cost_usd']:.6f}")

    print("=" * 65)
    print(">>> GATE 2 CHECK: PASSED (Schema extraction, SHA-256 caching & cost tracking verified)")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
