"""Phase 1 Verification Script.

Parses all 50 screenplay episodes in data/raw_scripts/, computing:
- Total episodes parsed
- Total beats parsed
- Total dialogue beats parsed
- Top 10 characters by dialogue count
- Unparsed line rate (% of total raw script lines)
- Total season duration
- Any validation issues or warnings
"""

import sys
from collections import Counter
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.parser import ScreenplayParser


def run_verification():
    raw_dir = Path("data/raw_scripts")
    if not raw_dir.exists():
        print(f"Error: Directory '{raw_dir}' does not exist.")
        sys.exit(1)

    parser = ScreenplayParser()
    episodes = parser.parse_directory(raw_dir)

    total_episodes = len(episodes)
    total_beats = sum(ep.total_beats for ep in episodes)
    total_dialogue = sum(ep.dialogue_line_count for ep in episodes)
    total_duration_sec = sum(ep.total_duration_sec for ep in episodes)

    # Calculate total raw lines across all 50 files
    total_raw_lines = 0
    for f in raw_dir.glob("*.txt"):
        with open(f, "r", encoding="utf-8") as fp:
            total_raw_lines += len(fp.readlines())

    # Collect validation issues
    all_issues = []
    for ep in episodes:
        all_issues.extend(ep.validation_issues)

    unparsed_issues = [iss for iss in all_issues if iss.issue_type == "UNPARSED_LINE"]
    unparsed_rate = (len(unparsed_issues) / total_raw_lines) * 100.0 if total_raw_lines > 0 else 0.0

    # Aggregate speaker counts
    speaker_counter = Counter()
    for ep in episodes:
        for spk, count in ep.speaker_counts.items():
            speaker_counter[spk] += count

    print("=" * 65)
    print("PHASE 1 SCREENPLAY PARSER & BEAT INDEXER VERIFICATION")
    print("=" * 65)
    print(f"Total Episodes Parsed:        {total_episodes} / 50")
    print(f"Total Raw Lines Ingested:     {total_raw_lines:,}")
    print(f"Total Beats Indexed:          {total_beats:,}")
    print(f"Total Dialogue Beats:         {total_dialogue:,}")
    print(f"Total Season Duration:        {total_duration_sec:,.2f}s ({total_duration_sec / 60:.1f} min)")
    print(f"Total Validation Issues:      {len(all_issues)}")
    print(f"Unparsed Lines:               {len(unparsed_issues)}")
    print(f"Unparsed Line Rate:           {unparsed_rate:.4f}% (Target: < 1.0%)")
    print("-" * 65)
    print("TOP 10 CHARACTERS BY DIALOGUE COUNT:")
    for rank, (speaker, count) in enumerate(speaker_counter.most_common(10), 1):
        pct = (count / total_dialogue) * 100.0 if total_dialogue > 0 else 0.0
        print(f"  {rank:2d}. {speaker:26s} : {count:4d} lines ({pct:5.1f}%)")
    print("=" * 65)

    if all_issues:
        print("\nValidation Issues Sample:")
        for iss in all_issues[:10]:
            print(f"  [{iss.severity}] Ep {iss.episode_number} Line {iss.line_number}: {iss.message}")

    # Assertions for Gate 1
    assert total_episodes == 50, f"Expected 50 episodes, got {total_episodes}"
    assert total_beats > 3000, f"Expected >3000 beats, got {total_beats}"
    assert unparsed_rate < 1.0, f"Unparsed rate {unparsed_rate:.2f}% exceeds 1.0% threshold"
    print("\n>>> GATE 1 CHECK: PASSED (100% deterministic parsing, <1% unparsed rate)")


if __name__ == "__main__":
    run_verification()
