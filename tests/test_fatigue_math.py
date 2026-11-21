"""Unit tests for Module 1 — Cliffhanger-to-Payoff Fatigue Auditor.

Verifies:
- Deterministic streak counting and payoff resets
- Paywall Danger Zone (Episodes 5-15) alerting and penalty multipliers
- Invalid / hallucinated beat_id rejection
- Season pacing score and ratio calculations
"""

import pytest

from src.fatigue_auditor import (
    PAYWALL_END_EPISODE,
    PAYWALL_START_EPISODE,
    audit_single_episode,
    compute_fatigue_engine,
    extract_ending_window,
    render_fatigue_curve_image,
)
from src.llm_client import LLMClient, LLMProvider
from src.parser import ScreenplayParser
from src.schemas import (
    Beat,
    BeatType,
    EndingAudit,
    EndingClassification,
    EpisodeData,
)


def make_dummy_audit(
    ep_num: int,
    ending_type: EndingClassification,
    payoff_score: int = 2,
    tension_score: int = 8,
    beat_id: str = "E01_B020",
) -> EndingAudit:
    """Helper to generate an EndingAudit for deterministic engine testing."""
    return EndingAudit(
        episode_number=ep_num,
        ending_type=ending_type,
        tension_score=tension_score,
        payoff_score=payoff_score,
        evidence_beat_id=beat_id,
        evidence_beat_valid=True,
        evidence_beat_text="Dummy ending action",
        evidence_beat_ts="01:20.00 - 01:25.00",
        one_line_verdict=f"Test verdict for Ep {ep_num}",
        is_paywall_danger_zone=(PAYWALL_START_EPISODE <= ep_num <= PAYWALL_END_EPISODE),
    )


def test_streak_detection_and_payoff_reset():
    """Verifies that consecutive cliffhangers increment the streak and payoffs reset it to 0."""
    # Sequence: 3 cliffhangers, 1 micro-payoff, 2 cliffhangers, 1 satisfying close
    audits = [
        make_dummy_audit(1, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(2, EndingClassification.HARD_CLIFFHANGER, payoff_score=1),
        make_dummy_audit(3, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(4, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=7),
        make_dummy_audit(5, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(6, EndingClassification.HARD_CLIFFHANGER, payoff_score=3),
        make_dummy_audit(7, EndingClassification.SATISFYING_CLOSE, payoff_score=9),
    ]

    result = compute_fatigue_engine(audits)

    streaks = [a.streak_count for a in result.audits]
    assert streaks == [1, 2, 3, 0, 1, 2, 0]

    # Episode 3 had streak=3, so it should be flagged for general fatigue
    assert 3 in result.fatigue_flagged_episodes
    assert result.max_consecutive_cliffhangers == 3
    assert result.hard_cliffhanger_count == 5
    assert result.micro_payoff_count == 1
    assert result.satisfying_close_count == 1


def test_paywall_danger_zone_multiplier():
    """Verifies stricter alerting and higher penalties in the Paywall Danger Zone (Episodes 5-15)."""
    # Create a severe 4-cliffhanger streak inside the danger zone (Episodes 7, 8, 9, 10)
    audits = [
        make_dummy_audit(1, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=7),
        make_dummy_audit(2, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=6),
        make_dummy_audit(7, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(8, EndingClassification.HARD_CLIFFHANGER, payoff_score=1),
        make_dummy_audit(9, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(10, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
    ]

    result = compute_fatigue_engine(audits)

    # Danger zone threshold is streak >= 2 in Ep 5-15
    assert result.danger_zone_cliffhangers == 4
    assert 8 in result.danger_zone_fatigue_episodes
    assert 9 in result.danger_zone_fatigue_episodes
    assert 10 in result.danger_zone_fatigue_episodes

    # Pacing score should take heavy penalties
    assert result.season_pacing_score < 75.0
    assert result.status in ["MODERATE_FATIGUE_RISK", "CRITICAL_PAYWALL_FATIGUE"]


def test_invalid_beat_id_rejection():
    """Verifies that hallucinated beat IDs are detected, flagged as invalid, and given a fallback."""
    parser = ScreenplayParser()
    ep = parser.parse_file("sample_data/episode-001-Episode 1000001.txt")

    # Mock client that cites a completely non-existent beat_id
    mock_client = LLMClient(provider=LLMProvider.MOCK)

    # We run the audit
    audit = audit_single_episode(ep, mock_client)

    # Test with genuine valid beats from ep 1
    valid_ids = {b.beat_id for b in ep.beats}
    assert audit.evidence_beat_id in valid_ids or not audit.evidence_beat_valid

    # Now simulate an explicitly invalid citation
    fake_client = LLMClient(provider=LLMProvider.MOCK)
    # Force client to return hallucinated ID
    original_generate = fake_client.generate

    def hallucinating_generate(*args, **kwargs):
        res, usage = original_generate(*args, **kwargs)
        # Modify cited beat to a fake one
        fake_res = res.model_copy(update={"evidence_beat_id": "E01_B999_HALLUCINATED"})
        return fake_res, usage

    fake_client.generate = hallucinating_generate

    audit_with_hallucination = audit_single_episode(ep, fake_client)

    # Verification: must flag evidence_beat_valid as False
    assert audit_with_hallucination.evidence_beat_id == "E01_B999_HALLUCINATED"
    assert audit_with_hallucination.evidence_beat_valid is False
    assert "UNVERIFIED CITATION" in audit_with_hallucination.evidence_beat_text


def test_season_pacing_score_healthy_vs_fatigued():
    """Tests that a well-paced show receives a high score and a fatigued show receives a low score."""
    # Healthy show: Alternating cliffhangers and payoffs
    healthy_audits = [
        make_dummy_audit(1, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(2, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=7),
        make_dummy_audit(3, EndingClassification.HARD_CLIFFHANGER, payoff_score=3),
        make_dummy_audit(4, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=8),
        make_dummy_audit(5, EndingClassification.HARD_CLIFFHANGER, payoff_score=2),
        make_dummy_audit(6, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=7),
    ]

    healthy_result = compute_fatigue_engine(healthy_audits)
    assert healthy_result.season_pacing_score >= 85.0
    assert healthy_result.status == "HEALTHY_PACING"
    assert healthy_result.cliffhanger_to_payoff_ratio <= 1.0

    # Highly fatigued show: 8 consecutive hard cliffhangers without payoff
    fatigued_audits = [
        make_dummy_audit(i, EndingClassification.HARD_CLIFFHANGER, payoff_score=1)
        for i in range(1, 9)
    ]
    fatigued_result = compute_fatigue_engine(fatigued_audits)
    assert fatigued_result.season_pacing_score < 50.0
    assert fatigued_result.status == "CRITICAL_PAYWALL_FATIGUE"


def test_100_episode_season_scaling_invariance():
    """Verifies that a 100-episode season is not unfairly penalized for length.

    If 90% of episodes are well-balanced and paywall danger zone (Ep 5-15) is healthy,
    the season pacing score must remain >= 85 (HEALTHY_PACING), rather than collapsing
    to 50 due to raw count accumulation.
    """
    audits = []
    # Build 100 episodes:
    # Danger zone (Ep 5-15): every 2nd episode has a micro-payoff (no fatigue)
    # Outside danger zone: alternating cliffhangers and payoffs, with only 8 fatigue spikes
    for ep in range(1, 101):
        if 5 <= ep <= 15:
            # Alternating in danger zone: no danger fatigue
            if ep % 2 == 0:
                audits.append(make_dummy_audit(ep, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=7))
            else:
                audits.append(make_dummy_audit(ep, EndingClassification.HARD_CLIFFHANGER, payoff_score=2))
        elif 30 <= ep <= 33 or 70 <= ep <= 73:
            # Two small 4-episode cliffhanger streaks outside danger zone (total ~6 fatigue episodes)
            audits.append(make_dummy_audit(ep, EndingClassification.HARD_CLIFFHANGER, payoff_score=2))
        else:
            # Healthy alternating cliffhangers and payoffs
            if ep % 2 == 0:
                audits.append(make_dummy_audit(ep, EndingClassification.MICRO_PAYOFF_WITH_HOOK, payoff_score=7))
            else:
                audits.append(make_dummy_audit(ep, EndingClassification.HARD_CLIFFHANGER, payoff_score=2))

    result = compute_fatigue_engine(audits)

    # 100-episode season must score >= 85 and retain HEALTHY_PACING because 90%+ is healthy
    assert result.total_episodes_audited == 100
    assert len(result.danger_zone_fatigue_episodes) == 0
    assert result.season_pacing_score >= 85.0
    assert result.status == "HEALTHY_PACING"


def test_extract_ending_window():
    """Verifies that ending window captures the final beats within the duration window."""
    parser = ScreenplayParser()
    ep = parser.parse_file("sample_data/episode-001-Episode 1000001.txt")

    ending_beats = extract_ending_window(ep, window_sec=20.0, min_beats=4)
    assert len(ending_beats) >= 4
    # The last beat of ending window must be the last beat of the episode
    assert ending_beats[-1].beat_id == ep.beats[-1].beat_id


def test_render_fatigue_curve_image(tmp_path):
    """Verifies that render_fatigue_curve_image creates a valid high-resolution PNG image file."""
    audits = [
        make_dummy_audit(1, EndingClassification.HARD_CLIFFHANGER, tension_score=8, payoff_score=2),
        make_dummy_audit(2, EndingClassification.MICRO_PAYOFF_WITH_HOOK, tension_score=4, payoff_score=7),
        make_dummy_audit(5, EndingClassification.HARD_CLIFFHANGER, tension_score=9, payoff_score=1),
    ]
    result = compute_fatigue_engine(audits)

    out_file = tmp_path / "02_episode_fatigue_curve.png"
    ret = render_fatigue_curve_image(result, out_file)

    assert ret.exists()
    assert ret.suffix == ".png"
    assert ret.stat().st_size > 1000  # Valid non-empty PNG binary
