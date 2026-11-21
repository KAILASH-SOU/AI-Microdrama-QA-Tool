"""Module 1: Cliffhanger-to-Payoff Fatigue Auditor.

Performs:
1. Python Pre-Processing: Extracting ending windows and condensed episode context.
2. Swappable LLM Evaluation: Classifying ending type, tension, payoff, citing exact beat_id.
3. Deterministic Python Fatigue Engine:
   - Calculating consecutive unresolved cliffhanger streaks (Episodes 1..N).
   - Alerting on Fatigue Risk Zones (streak >= 3).
   - Applying Paywall Danger Zone Multipliers (Episodes 5-15).
   - Validating evidence beat_ids against raw parsed episode beats (rejecting hallucinations).
   - Computing Season Pacing Score (0-100) and Cliffhanger-to-Payoff Ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
from src.llm_client import LLMClient
from src.schemas import (
    Beat,
    EndingAudit,
    EndingClassification,
    EpisodeData,
    EpisodeEndingLLMResponse,
    FatigueAuditResult,
)

# Constants for Microdrama Pacing Economics
PAYWALL_START_EPISODE = 5
PAYWALL_END_EPISODE = 15
FATIGUE_STREAK_THRESHOLD = 3
PAYWALL_DANGER_STREAK_THRESHOLD = 2  # Stricter threshold in monetization zone
PAYWALL_PENALTY_MULTIPLIER = 2.0


def extract_ending_window(
    episode: EpisodeData,
    window_sec: float = 20.0,
    min_beats: int = 5,
) -> List[Beat]:
    """Extracts the final beats of an episode within the specified window (or last N beats).

    Guarantees at least min_beats (e.g. 5-6 beats) or all beats if episode is short.
    """
    if not episode.beats:
        return []

    if len(episode.beats) <= min_beats:
        return list(episode.beats)

    total_duration = episode.total_duration_sec
    threshold_time = max(0.0, total_duration - window_sec)

    # Beats that start or end within the last `window_sec` seconds
    window_beats = [b for b in episode.beats if b.end_sec >= threshold_time]

    # Ensure at least `min_beats` are returned
    if len(window_beats) < min_beats:
        return episode.beats[-min_beats:]

    return window_beats


def extract_condensed_beats(episode: EpisodeData, max_beats: int = 12) -> str:
    """Creates a high-signal condensed timeline of the episode's primary setup and conflict."""
    if not episode.beats:
        return "No beats recorded."

    # Sample beats evenly across the episode
    dialogue_and_action = [b for b in episode.beats if b.is_dialogue or b.beat_type.value == "ACTION"]
    if not dialogue_and_action:
        dialogue_and_action = episode.beats

    step = max(1, len(dialogue_and_action) // max_beats)
    sampled = dialogue_and_action[::step][:max_beats]

    lines = []
    for b in sampled:
        speaker_prefix = f"[{b.speaker}]: " if b.speaker else ""
        lines.append(f"- ({b.start_ts}) {b.beat_id}: {speaker_prefix}{b.text[:90]}")

    return "\n".join(lines)


def build_ending_prompt(episode: EpisodeData, ending_window: List[Beat]) -> Tuple[str, str]:
    """Constructs system and user prompts for LLM narrative ending audit."""
    system_prompt = (
        "You are an expert Senior Creative Producer and Microdrama Showrunner specializing in short-form "
        "vertical video pacing (ReelShort, DramaBox, ShortMax format).\n"
        "Your task is to analyze the final moments of a 1-to-3 minute microdrama episode to audit viewer fatigue.\n"
        "Microdramas thrive on high tension, but consecutive unresolved cliffhangers without micro-payoffs "
        "cause audience drop-off and paywall churn.\n\n"
        "You must return STRICT JSON adhering to this schema:\n"
        "- ending_type: 'HARD_CLIFFHANGER' | 'MICRO_PAYOFF_WITH_HOOK' | 'SATISFYING_CLOSE' | 'FLAT_ENDING'\n"
        "- tension_score: Integer 1-10 (intensity of the final beat)\n"
        "- payoff_score: Integer 1-10 (1 = zero questions answered/pure cliffhanger, 10 = complete resolution)\n"
        "- evidence_beat_id: EXACT beat_id from the provided ending window that justifies your classification\n"
        "- one_line_verdict: Concise editorial rationale (max 20 words)\n"
        "- suggested_pivot: Actionable recommendation to add a micro-payoff if the ending causes cliffhanger fatigue"
    )

    ending_lines = []
    for b in ending_window:
        spk = f"[{b.speaker}] " if b.speaker else f"[{b.beat_type.value}] "
        tone = f"({b.delivery_tone}) " if b.delivery_tone else ""
        ending_lines.append(f"  {b.beat_id} ({b.start_ts} - {b.end_ts}): {spk}{tone}\"{b.text}\"")

    ending_text = "\n".join(ending_lines)
    condensed_context = extract_condensed_beats(episode)

    user_prompt = (
        f"EPISODE AUDIT REQUEST:\n"
        f"Episode Number: {episode.episode_number}\n"
        f"Logline: {episode.logline}\n"
        f"Continuity Notes: {episode.continuity_notes}\n"
        f"Duration: {episode.formatted_duration} ({episode.total_duration_sec:.1f}s)\n\n"
        f"EPISODE TIMELINE CONTEXT:\n"
        f"{condensed_context}\n\n"
        f"DEDICATED ENDING WINDOW (Final Beats):\n"
        f"{ending_text}\n\n"
        f"Evaluate the ending window strictly. You must cite an EXACT beat_id from the ending window above.\n"
        f"Respond with valid JSON only."
    )

    return system_prompt, user_prompt


def audit_single_episode(
    episode: EpisodeData,
    llm_client: Optional[LLMClient] = None,
) -> EndingAudit:
    """Evaluates an episode's ending using the LLM and validates cited beat_id against Python parsed beats."""
    client = llm_client or LLMClient()
    ending_window = extract_ending_window(episode, window_sec=20.0, min_beats=5)
    valid_beat_ids = {b.beat_id: b for b in episode.beats}

    system_prompt, user_prompt = build_ending_prompt(episode, ending_window)

    llm_response, _ = client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        response_model=EpisodeEndingLLMResponse,
        stage="fatigue_auditor",
        prompt_version="v1.0",
        temperature=0.0,
    )

    # Deterministic Python Verification of cited beat_id
    cited_id = llm_response.evidence_beat_id.strip()
    is_valid = cited_id in valid_beat_ids
    beat_text = valid_beat_ids[cited_id].text if is_valid else None
    beat_ts = f"{valid_beat_ids[cited_id].start_ts} - {valid_beat_ids[cited_id].end_ts}" if is_valid else None

    # Fallback if LLM cited an invalid beat_id
    if not is_valid and ending_window:
        # Use final beat of ending window as fallback
        fallback_beat = ending_window[-1]
        beat_text = f"[UNVERIFIED CITATION '{cited_id}'] Fallback: {fallback_beat.text}"
        beat_ts = f"{fallback_beat.start_ts} - {fallback_beat.end_ts}"

    is_danger_zone = PAYWALL_START_EPISODE <= episode.episode_number <= PAYWALL_END_EPISODE

    return EndingAudit(
        episode_number=episode.episode_number,
        ending_type=llm_response.ending_type,
        tension_score=llm_response.tension_score,
        payoff_score=llm_response.payoff_score,
        evidence_beat_id=cited_id,
        evidence_beat_valid=is_valid,
        evidence_beat_text=beat_text,
        evidence_beat_ts=beat_ts,
        one_line_verdict=llm_response.one_line_verdict,
        is_paywall_danger_zone=is_danger_zone,
        streak_count=0,  # Computed subsequently by the sequence engine
        fatigue_flag=False,
        danger_zone_fatigue=False,
        suggested_pivot=llm_response.suggested_pivot,
    )


def compute_fatigue_engine(audits: List[EndingAudit]) -> FatigueAuditResult:
    """100% Deterministic Python Fatigue Engine.

    Calculates:
    - consecutive_cliffhanger_streak (increments on HARD_CLIFFHANGER with payoff <= 3, resets on payoff)
    - Fatigue Risk Alerts (streak >= 3)
    - Paywall Danger Zone Multipliers (Episodes 5-15)
    - Season Pacing Score (0-100)
    - Cliffhanger-to-Payoff Ratio
    """
    sorted_audits = sorted(audits, key=lambda a: a.episode_number)

    current_streak = 0
    max_streak = 0
    hard_cliffhangers = 0
    micro_payoffs = 0
    satisfying_closes = 0
    flat_endings = 0
    danger_zone_cliffhangers = 0
    danger_zone_fatigue_eps = []
    fatigue_flagged_eps = []

    enriched_audits: List[EndingAudit] = []

    for audit in sorted_audits:
        ep_num = audit.episode_number
        is_danger_zone = PAYWALL_START_EPISODE <= ep_num <= PAYWALL_END_EPISODE

        # Count distribution
        if audit.ending_type == EndingClassification.HARD_CLIFFHANGER:
            hard_cliffhangers += 1
            if is_danger_zone:
                danger_zone_cliffhangers += 1
        elif audit.ending_type == EndingClassification.MICRO_PAYOFF_WITH_HOOK:
            micro_payoffs += 1
        elif audit.ending_type == EndingClassification.SATISFYING_CLOSE:
            satisfying_closes += 1
        elif audit.ending_type == EndingClassification.FLAT_ENDING:
            flat_endings += 1

        # Streak Logic: Increments on HARD_CLIFFHANGER with low payoff (payoff <= 3)
        # Resets on payoffs (MICRO_PAYOFF_WITH_HOOK, SATISFYING_CLOSE, or payoff >= 6)
        is_unresolved_cliffhanger = (
            audit.ending_type == EndingClassification.HARD_CLIFFHANGER
            and audit.payoff_score <= 3
        )
        is_payoff = (
            audit.ending_type in [
                EndingClassification.MICRO_PAYOFF_WITH_HOOK,
                EndingClassification.SATISFYING_CLOSE,
            ]
            or audit.payoff_score >= 6
        )

        if is_unresolved_cliffhanger:
            current_streak += 1
        elif is_payoff:
            current_streak = 0
        else:
            # Flat ending: does not resolve payoff, but tension dropped
            # Keep streak or increment with lower severity
            current_streak += 1

        if current_streak > max_streak:
            max_streak = current_streak

        # Fatigue Flagging
        is_fatigued = current_streak >= FATIGUE_STREAK_THRESHOLD
        is_danger_zone_fatigued = is_danger_zone and current_streak >= PAYWALL_DANGER_STREAK_THRESHOLD

        if is_fatigued:
            fatigue_flagged_eps.append(ep_num)
        if is_danger_zone_fatigued:
            danger_zone_fatigue_eps.append(ep_num)

        # Create updated audit record with deterministic streak attributes
        enriched_audit = audit.model_copy(
            update={
                "streak_count": current_streak,
                "fatigue_flag": is_fatigued,
                "danger_zone_fatigue": is_danger_zone_fatigued,
            }
        )
        enriched_audits.append(enriched_audit)

    # Cliffhanger to Payoff Ratio
    total_payoffs = micro_payoffs + satisfying_closes
    ratio = round(hard_cliffhangers / max(1, total_payoffs), 2)

    # Season Pacing Score Math (0 - 100)
    # Starts at 100
    # Scale-Invariant Normalization:
    # In long series (e.g. 100 episodes), absolute count penalties unfairly punish shows
    # with low fatigue density (e.g. 10 fatigue episodes out of 100 is a 90% healthy season).
    # We normalize general fatigue and flat endings by total episode count while preserving
    # strict fixed-window penalties for the Paywall Danger Zone (Ep 5-15).
    base_score = 100.0
    total_eps = max(1, len(sorted_audits))

    general_fatigue_count = len([ep for ep in fatigue_flagged_eps if ep not in danger_zone_fatigue_eps])
    danger_fatigue_count = len(danger_zone_fatigue_eps)

    # 1. Monetization Danger Zone penalties (fixed 11-episode churn window Ep 5-15)
    danger_penalty = min(40.0, danger_fatigue_count * 8.0)

    # 2. General Fatigue normalized by season length (scale-invariant rate)
    # E.g. 10 fatigue episodes in a 100-ep season = 10% rate -> 6.0 pts penalty (not 50 pts!)
    general_fatigue_rate = general_fatigue_count / total_eps
    general_fatigue_penalty = min(30.0, general_fatigue_rate * 60.0)

    # 3. Flat Endings normalized by season length
    flat_ending_rate = flat_endings / total_eps
    flat_ending_penalty = min(15.0, flat_ending_rate * 30.0)

    # 4. Imbalance Ratio & Peak Fatigue Streak
    ratio_penalty = 12.0 if ratio > 2.5 else 0.0
    streak_penalty = 8.0 if max_streak >= 4 else 0.0

    deductions = (
        danger_penalty
        + general_fatigue_penalty
        + flat_ending_penalty
        + ratio_penalty
        + streak_penalty
    )

    score = max(0.0, min(100.0, base_score - deductions))

    # Determine status
    if score >= 80.0 and len(danger_zone_fatigue_eps) == 0:
        status = "HEALTHY_PACING"
    elif score >= 60.0 and len(danger_zone_fatigue_eps) <= 2:
        status = "MODERATE_FATIGUE_RISK"
    else:
        status = "CRITICAL_PAYWALL_FATIGUE"

    return FatigueAuditResult(
        total_episodes_audited=len(sorted_audits),
        hard_cliffhanger_count=hard_cliffhangers,
        micro_payoff_count=micro_payoffs,
        satisfying_close_count=satisfying_closes,
        flat_ending_count=flat_endings,
        cliffhanger_to_payoff_ratio=ratio,
        max_consecutive_cliffhangers=max_streak,
        danger_zone_cliffhangers=danger_zone_cliffhangers,
        danger_zone_fatigue_episodes=sorted(list(set(danger_zone_fatigue_eps))),
        fatigue_flagged_episodes=sorted(list(set(fatigue_flagged_eps))),
        season_pacing_score=round(score, 1),
        status=status,
        audits=enriched_audits,
    )


def audit_season_fatigue(
    episodes: List[EpisodeData],
    llm_client: Optional[LLMClient] = None,
) -> FatigueAuditResult:
    """End-to-end Module 1 pipeline: Audits each episode, then runs the deterministic fatigue engine."""
    client = llm_client or LLMClient()
    raw_audits: List[EndingAudit] = []

    for ep in episodes:
        audit = audit_single_episode(ep, client)
        raw_audits.append(audit)

    return compute_fatigue_engine(raw_audits)


def render_fatigue_curve_image(
    fatigue: FatigueAuditResult,
    output_path: Path | str,
) -> Path:
    """Renders a high-resolution PNG image of the Cliffhanger-to-Payoff Fatigue Curve."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 650
    img = Image.new("RGB", (width, height), color="#0E131F")
    draw = ImageDraw.Draw(img)

    pad_left, pad_right, pad_top, pad_bottom = 80, 50, 95, 85
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Plot Background
    draw.rectangle(
        [pad_left, pad_top, pad_left + plot_w, pad_top + plot_h],
        fill="#141A29",
        outline="#1E293B",
        width=1,
    )

    # Grid lines and Y labels
    for y_val in range(0, 11, 2):
        y_pos = pad_top + plot_h - (y_val / 10.0) * plot_h
        draw.line([pad_left, y_pos, pad_left + plot_w, y_pos], fill="#1E293B", width=1)
        draw.text((pad_left - 30, y_pos - 6), str(y_val), fill="#94A3B8")

    draw.text((15, pad_top + plot_h // 2 - 20), "Intensity\n(1-10)", fill="#64748B")

    audits = fatigue.audits or []
    N = max(1, len(audits))

    def get_x(idx: int) -> float:
        if N <= 1:
            return pad_left + plot_w / 2
        return pad_left + (idx / (N - 1)) * (plot_w - 40) + 20

    def get_y(val: float) -> float:
        return pad_top + plot_h - (max(0.0, min(10.0, float(val))) / 10.0) * plot_h

    # Paywall Zone (Ep 5-15) based on audits ordinal positions
    zone_indices = [
        i
        for i, a in enumerate(audits)
        if PAYWALL_START_EPISODE <= a.episode_number <= PAYWALL_END_EPISODE
    ]
    if zone_indices:
        min_idx = min(zone_indices)
        max_idx = max(zone_indices)
        x_start = max(pad_left, get_x(min_idx) - 15)
        x_end = min(pad_left + plot_w, get_x(max_idx) + 15)
        if x_end <= x_start:
            x_end = min(pad_left + plot_w, x_start + 30)
            x_start = max(pad_left, x_end - 30)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        o_draw = ImageDraw.Draw(overlay)
        o_draw.rectangle([x_start, pad_top, x_end, pad_top + plot_h], fill=(239, 68, 68, 30))
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x_start, pad_top, x_end, pad_top + plot_h], outline="#7F1D1D", width=1)
        draw.text((x_start + 10, pad_top + 10), "Monetization Paywall Danger Zone (Ep 5-15)", fill="#F87171")

    # X axis ticks & labels
    step = 1 if N <= 20 else (2 if N <= 40 else (5 if N <= 80 else 10))
    for i, a in enumerate(audits):
        num = a.episode_number
        x = get_x(i)
        draw.line([x, pad_top + plot_h, x, pad_top + plot_h + 5], fill="#64748B", width=1)
        if i == 0 or i == N - 1 or (i + 1) % step == 0:
            draw.text((x - 10, pad_top + plot_h + 8), f"Ep {num}", fill="#94A3B8")

    # Lines & markers
    tension_pts = [(get_x(i), get_y(a.tension_score)) for i, a in enumerate(audits)]
    payoff_pts = [(get_x(i), get_y(a.payoff_score)) for i, a in enumerate(audits)]

    for i in range(len(tension_pts) - 1):
        draw.line([tension_pts[i], tension_pts[i + 1]], fill="#EF4444", width=3)
    for pt in tension_pts:
        draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill="#EF4444", outline="#FFFFFF", width=1)

    for i in range(len(payoff_pts) - 1):
        draw.line([payoff_pts[i], payoff_pts[i + 1]], fill="#10B981", width=3)
    for pt in payoff_pts:
        draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill="#10B981", outline="#FFFFFF", width=1)

    # Fatigue alerts
    for i, a in enumerate(audits):
        if a.fatigue_flag or a.danger_zone_fatigue:
            x = get_x(i)
            y = get_y(a.tension_score)
            tri = [(x, y - 16), (x - 7, y - 4), (x + 7, y - 4)]
            draw.polygon(tri, fill="#B91C1C", outline="#FFFFFF")

    # Header Title
    draw.text((pad_left, 22), "Season Tension vs. Payoff Pacing Trajectory", fill="#F8FAFC")
    draw.text((pad_left, 48), "Pre-Render Script Health Gatekeeper — Cliffhanger & Paywall Fatigue Curve", fill="#94A3B8")

    # Legend
    leg_x = width - 420
    draw.line([leg_x, 30, leg_x + 20, 30], fill="#EF4444", width=3)
    draw.ellipse([leg_x + 7, 27, leg_x + 13, 33], fill="#EF4444")
    draw.text((leg_x + 25, 24), "Tension", fill="#E2E8F0")

    draw.line([leg_x + 90, 30, leg_x + 110, 30], fill="#10B981", width=3)
    draw.ellipse([leg_x + 97, 27, leg_x + 103, 33], fill="#10B981")
    draw.text((leg_x + 115, 24), "Payoff", fill="#E2E8F0")

    draw.polygon([(leg_x + 175, 22), (leg_x + 170, 34), (leg_x + 180, 34)], fill="#B91C1C", outline="#FFFFFF")
    draw.text((leg_x + 185, 24), "Fatigue Alert", fill="#E2E8F0")

    # Footer scorecard
    footer = (
        f"Season Pacing Score: {fatigue.season_pacing_score}/100  |  "
        f"Status: {fatigue.status}  |  "
        f"Max Streak: {fatigue.max_consecutive_cliffhangers} eps  |  "
        f"Paywall Alerts: {len(fatigue.danger_zone_fatigue_episodes)}"
    )
    draw.rectangle([pad_left, height - 45, pad_left + plot_w, height - 15], fill="#141A29", outline="#1E293B", width=1)
    draw.text((pad_left + 15, height - 35), footer, fill="#38BDF8")

    img.save(out_p, "PNG")
    return out_p
