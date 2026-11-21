#!/usr/bin/env python3
"""CLI Runner for Pre-Render Script Health Gatekeeper.

Usage:
    python run_audit.py --input sample_data --output outputs/sample_run
    python run_audit.py --input data/raw_scripts --output outputs/run_full_50 --provider gemini --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.llm_client import LLMClient, LLMProvider
from src.pipeline import AuditPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-Render Script Health Gatekeeper (AI Microdrama QA Tool)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="sample_data",
        help="Path to screenplays directory (e.g. sample_data or data/raw_scripts) or a .zip archive.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/run_audit",
        help="Directory to save the 6 standardized Raw Outputs.",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="LLM provider: 'gemini', 'openai', 'anthropic', 'groq', 'generic_openai', or 'mock'.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name (e.g., 'gemini-2.5-flash', 'gpt-4o-mini', 'claude-3-5-haiku-20241022').",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="API key for the chosen provider (defaults to environment variables).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top characters to audit in Module 2 Voice Tester (default: 4).",
    )
    parser.add_argument(
        "--samples-per-char",
        type=int,
        default=15,
        help="Number of dialogue lines sampled per character in Module 2 (default: 15).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory for SHA-256 disk caching (default: .cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk caching.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force deterministic mock mode (cost $0.00, no API key needed).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Microdrama Season 1",
        help="Season title name for the summary report.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    # Initialize LLM Client
    client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        mock=args.mock,
    )

    # Initialize Pipeline Orchestrator
    pipeline = AuditPipeline(
        llm_client=client,
        output_dir=args.output,
        top_k_characters=args.top_k,
        samples_per_character=args.samples_per_char,
    )

    # Execute
    report, exported_files = pipeline.run(
        input_source=input_path,
        season_title=args.title,
    )

    # Print Terminal Summary
    print("=" * 72)
    print("PRE-RENDER SCRIPT HEALTH GATEKEEPER — AUDIT SUMMARY")
    print("=" * 72)
    print(f"Overall Verdict:           {report.overall_verdict}")
    print(f"Episodes Analyzed:         {report.total_episodes} episodes ({report.total_duration_sec / 60:.1f} min)")
    print(f"Total Beats:               {report.total_beats} beats ({report.total_dialogue_lines} dialogue lines)")
    print("-" * 72)
    print(f"MODULE 1 (Fatigue Auditor):")
    print(f"  Season Pacing Score:     {report.fatigue_audit.season_pacing_score} / 100 ({report.fatigue_audit.status})")
    print(f"  Max Cliffhanger Streak:  {report.fatigue_audit.max_consecutive_cliffhangers} episodes")
    print(f"  Danger Zone Alerts:      {len(report.fatigue_audit.danger_zone_fatigue_episodes)} (Episodes {report.fatigue_audit.danger_zone_fatigue_episodes})")
    print(f"  Cliffhanger / Payoff:    {report.fatigue_audit.hard_cliffhanger_count} cliffhangers / {report.fatigue_audit.micro_payoff_count + report.fatigue_audit.satisfying_close_count} payoffs (Ratio: {report.fatigue_audit.cliffhanger_to_payoff_ratio})")
    print("-" * 72)
    print(f"MODULE 2 (Voice Tester):")
    print(f"  Blind Voice Accuracy:    {report.voice_audit.accuracy * 100:.1f}% (Random Baseline: {report.voice_audit.random_baseline * 100:.1f}%)")
    print(f"  Macro-F1 Score:          {report.voice_audit.f1_macro:.4f} ({report.voice_audit.status})")
    print(f"  Flagged Interchangeable: {report.voice_audit.interchangeable_dialogue_count} lines")
    if report.voice_audit.most_confused_pairs:
        top_confused = report.voice_audit.most_confused_pairs[0]
        print(f"  Most Confused Pair:      {top_confused['pair'][0]} <-> {top_confused['pair'][1]} ({top_confused['confusion_count']} confusions)")
    print("-" * 72)
    print(f"PRODUCTION & COST REPORT:")
    cost_info = report.llm_cost_and_usage
    print(f"  Total API Calls:         {cost_info.get('total_calls', 0)} ({cost_info.get('cached_calls', 0)} cached, {cost_info.get('cache_hit_rate_pct', 0)}% hit rate)")
    print(f"  Tokens In/Out:           {cost_info.get('prompt_tokens', 0)} in / {cost_info.get('completion_tokens', 0)} out")
    print(f"  Run Cost:                ${cost_info.get('total_cost_usd', 0.0):.6f}")
    print("-" * 72)
    print("RAW OUTPUT ARTIFACTS GENERATED:")
    for key, path in exported_files.items():
        print(f"  - [{key}]: {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
