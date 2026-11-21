"""Pipeline Orchestrator & Raw Outputs Generator.

Orchestrates the end-to-end Pre-Render Script Health Gatekeeper workflow:
1. Ingestion & Deterministic Parsing (src.parser)
2. Module 1: Cliffhanger-to-Payoff Fatigue Audit (src.fatigue_auditor)
3. Module 2: Zero-Leakage Blind Character-Voice Testing (src.voice_auditor)
4. Comprehensive Validation, Failure & Hallucination Logging
5. Raw Outputs Export (6 standardized JSON & CSV audit artifacts)
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.fatigue_auditor import audit_season_fatigue, render_fatigue_curve_image
from src.llm_client import LLMClient, LLMProvider
from src.parser import ScreenplayParser, parse_season
from src.schemas import (
    EpisodeData,
    FatigueAuditResult,
    SeasonHealthReport,
    VoiceAuditResult,
)
from src.voice_auditor import audit_season_voice


class AuditPipeline:
    """End-to-End Orchestrator for Microdrama Pre-Render Script Health QA."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        output_dir: Union[str, Path] = "outputs/run_audit",
        top_k_characters: int = 4,
        samples_per_character: int = 15,
    ):
        self.client = llm_client or LLMClient()
        self.output_dir = Path(output_dir)
        self.top_k_characters = top_k_characters
        self.samples_per_character = samples_per_character
        self.parser = ScreenplayParser()

    def run(
        self,
        input_source: Union[str, Path, List[EpisodeData]],
        season_title: str = "Microdrama Season 1",
    ) -> Tuple[SeasonHealthReport, Dict[str, Path]]:
        """Executes the full audit pipeline and exports all 6 raw artifacts."""
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # STEP 1: Screenplay Parsing & Deterministic Beat Indexing
        # -------------------------------------------------------------
        if isinstance(input_source, list):
            episodes: List[EpisodeData] = input_source
            parse_duration = 0.0
            source_display = f"in_memory_episodes ({len(episodes)} episodes)"
        else:
            t0 = time.time()
            episodes = parse_season(input_source)
            parse_duration = round(time.time() - t0, 3)
            source_display = str(input_source)

        if not episodes:
            raise ValueError(f"No valid screenplay episodes could be found or parsed from: {source_display}")

        print(f"\n[PIPELINE] Starting Pre-Render Script Health Gatekeeper Audit...")
        print(f"[PIPELINE] Input Source:      {source_display}")
        print(f"[PIPELINE] Output Directory:  {self.output_dir}")
        print(f"[PIPELINE] LLM Provider:      {self.client.provider.value} ({self.client.model})")
        print(f"[PIPELINE] Mock Mode:         {self.client.mock}")

        total_episodes = len(episodes)
        total_beats = sum(ep.total_beats for ep in episodes)
        total_dialogue = sum(ep.dialogue_line_count for ep in episodes)
        total_duration_sec = sum(ep.total_duration_sec for ep in episodes)

        print(f"[PIPELINE] Step 1 Complete: Ingested {total_episodes} episodes, {total_beats} beats, {total_dialogue} dialogue lines in {parse_duration}s.")

        # -------------------------------------------------------------
        # STEP 2: Module 1 — Cliffhanger-to-Payoff Fatigue Audit
        # -------------------------------------------------------------
        t0 = time.time()
        print(f"[PIPELINE] Step 2 Running: Auditing cliffhanger streaks & paywall danger zone...")
        fatigue_result: FatigueAuditResult = audit_season_fatigue(episodes, self.client)
        fatigue_duration = round(time.time() - t0, 3)
        print(f"[PIPELINE] Step 2 Complete: Pacing Score = {fatigue_result.season_pacing_score}/100, Max Streak = {fatigue_result.max_consecutive_cliffhangers} ({fatigue_duration}s).")

        # -------------------------------------------------------------
        # STEP 3: Module 2 — Zero-Leakage Blind Character-Voice Testing
        # -------------------------------------------------------------
        t0 = time.time()
        print(f"[PIPELINE] Step 3 Running: Stylometrics profiling & blind voice prediction test...")
        voice_result: VoiceAuditResult = audit_season_voice(
            episodes,
            top_k=self.top_k_characters,
            samples_per_char=self.samples_per_character,
            client=self.client,
        )
        voice_duration = round(time.time() - t0, 3)
        print(f"[PIPELINE] Step 3 Complete: Blind Accuracy = {voice_result.accuracy * 100:.1f}%, F1-macro = {voice_result.f1_macro:.4f} ({voice_duration}s).")

        # -------------------------------------------------------------
        # STEP 4: Synthesize Master Health Report & Overall Verdict
        # -------------------------------------------------------------
        # Overall Verdict Logic
        pacing_ok = fatigue_result.season_pacing_score >= 75.0 and len(fatigue_result.danger_zone_fatigue_episodes) <= 1
        voice_ok = voice_result.accuracy >= 0.50 and voice_result.f1_macro >= 0.45

        if pacing_ok and voice_ok:
            overall_verdict = "READY_FOR_RENDER"
        elif fatigue_result.season_pacing_score < 60.0 or len(fatigue_result.danger_zone_fatigue_episodes) >= 3:
            overall_verdict = "REVISION_REQUIRED"
        else:
            overall_verdict = "CAUTION"

        # Executive Summary
        exec_summary = (
            f"Pre-Render Health Gatekeeper evaluated {total_episodes} episodes ({total_duration_sec / 60:.1f} min). "
            f"Overall Verdict: {overall_verdict}. "
            f"Narrative Pacing scored {fatigue_result.season_pacing_score}/100 with a max cliffhanger streak of {fatigue_result.max_consecutive_cliffhangers} "
            f"and {len(fatigue_result.danger_zone_fatigue_episodes)} fatigue alerts in the monetization paywall zone (Ep 5-15). "
            f"Character Voice testing achieved {voice_result.accuracy * 100:.1f}% accuracy across {len(voice_result.top_speakers)} lead characters "
            f"(random baseline: {voice_result.random_baseline * 100:.1f}%), flagging {len(voice_result.flagged_predictions)} interchangeable dialogue lines."
        )

        # Aggregate speaker distribution
        speaker_distribution: Dict[str, int] = {}
        for ep in episodes:
            for spk, cnt in ep.speaker_counts.items():
                speaker_distribution[spk] = speaker_distribution.get(spk, 0) + cnt

        usage_summary = self.client.tracker.summary()

        master_report = SeasonHealthReport(
            season_name=season_title,
            total_episodes=total_episodes,
            total_duration_sec=round(total_duration_sec, 2),
            total_beats=total_beats,
            total_dialogue_lines=total_dialogue,
            speaker_distribution=speaker_distribution,
            fatigue_audit=fatigue_result,
            voice_audit=voice_result,
            validation_summary={
                "total_parsing_issues": sum(len(ep.validation_issues) for ep in episodes),
                "leakage_filtered_lines": voice_result.leakage_filtered_count,
                "unverified_citations_count": sum(1 for a in fatigue_result.audits if not a.evidence_beat_valid),
            },
            llm_cost_and_usage=usage_summary,
            overall_verdict=overall_verdict,
            executive_summary=exec_summary,
        )

        # -------------------------------------------------------------
        # STEP 5: Export The 6 Required Raw Outputs
        # -------------------------------------------------------------
        exported_files = self._export_raw_outputs(episodes, fatigue_result, voice_result, master_report)

        total_runtime = round(time.time() - start_time, 2)
        print(f"\n[PIPELINE] Audit Completed in {total_runtime}s!")
        print(f"[PIPELINE] Artifacts written to: {self.output_dir.resolve()}\n")

        return master_report, exported_files

    def _export_raw_outputs(
        self,
        episodes: List[EpisodeData],
        fatigue: FatigueAuditResult,
        voice: VoiceAuditResult,
        report: SeasonHealthReport,
    ) -> Dict[str, Path]:
        """Writes the 6 required Raw Outputs into the output directory."""
        files: Dict[str, Path] = {}

        # ---------------------------------------------------------
        # File 1: 01_season_health_summary.json
        # ---------------------------------------------------------
        p1 = self.output_dir / "01_season_health_summary.json"
        summary_payload = {
            "season_name": report.season_name,
            "overall_verdict": report.overall_verdict,
            "executive_summary": report.executive_summary,
            "headline_kpis": {
                "total_episodes": report.total_episodes,
                "total_runtime_minutes": round(report.total_duration_sec / 60, 2),
                "total_beats": report.total_beats,
                "total_dialogue_lines": report.total_dialogue_lines,
                "season_pacing_score": fatigue.season_pacing_score,
                "max_cliffhanger_streak": fatigue.max_consecutive_cliffhangers,
                "cliffhanger_to_payoff_ratio": fatigue.cliffhanger_to_payoff_ratio,
                "danger_zone_fatigue_episodes_count": len(fatigue.danger_zone_fatigue_episodes),
                "blind_voice_accuracy": voice.accuracy,
                "voice_macro_f1": voice.f1_macro,
                "random_baseline": voice.random_baseline,
                "interchangeable_lines_flagged": voice.interchangeable_dialogue_count,
            },
            "top_red_flags": {
                "fatigue_risk_episodes": fatigue.fatigue_flagged_episodes,
                "monetization_danger_zone_alerts": fatigue.danger_zone_fatigue_episodes,
                "most_confused_character_pairs": voice.most_confused_pairs[:3],
                "unverified_llm_citations": [
                    {"episode": a.episode_number, "cited_beat_id": a.evidence_beat_id}
                    for a in fatigue.audits
                    if not a.evidence_beat_valid
                ],
            },
            "assumptions_and_parameters": {
                "paywall_danger_zone_range": "Episodes 5-15",
                "fatigue_streak_threshold": 3,
                "paywall_streak_threshold": 2,
                "monetization_penalty_multiplier": 2.0,
                "voice_top_k_characters": self.top_k_characters,
                "voice_sample_lines_per_character": self.samples_per_character,
            },
            "model_configuration": {
                "provider": self.client.provider.value,
                "model_name": self.client.model,
                "mock_mode": self.client.mock,
                "use_disk_cache": self.client.use_cache,
            },
        }
        with open(p1, "w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2)
        files["01_season_health_summary"] = p1

        # ---------------------------------------------------------
        # File 2: 02_episode_fatigue_curve.png (Image Format) & .csv
        # ---------------------------------------------------------
        p2_png = self.output_dir / "02_episode_fatigue_curve.png"
        render_fatigue_curve_image(fatigue, p2_png)
        files["02_episode_fatigue_curve"] = p2_png
        files["02_episode_fatigue_curve_png"] = p2_png

        # Also preserve CSV as tabular data companion
        p2_csv = self.output_dir / "02_episode_fatigue_curve.csv"
        ep_duration_map = {ep.episode_number: ep.total_duration_sec for ep in episodes}
        ep_formatted_map = {ep.episode_number: ep.formatted_duration for ep in episodes}

        with open(p2_csv, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "episode_number",
                "duration_seconds",
                "formatted_duration",
                "ending_type",
                "tension_score",
                "payoff_score",
                "streak_count",
                "is_paywall_danger_zone",
                "fatigue_flag",
                "danger_zone_fatigue",
                "evidence_beat_id",
                "evidence_beat_valid",
                "evidence_timestamp",
                "evidence_beat_text",
                "verdict",
                "suggested_pivot",
            ])
            for a in fatigue.audits:
                writer.writerow([
                    a.episode_number,
                    ep_duration_map.get(a.episode_number, 0.0),
                    ep_formatted_map.get(a.episode_number, "00:00.00"),
                    a.ending_type.value,
                    a.tension_score,
                    a.payoff_score,
                    a.streak_count,
                    a.is_paywall_danger_zone,
                    a.fatigue_flag,
                    a.danger_zone_fatigue,
                    a.evidence_beat_id,
                    a.evidence_beat_valid,
                    a.evidence_beat_ts or "",
                    (a.evidence_beat_text or "").replace("\n", " "),
                    a.one_line_verdict,
                    a.suggested_pivot or "",
                ])
        files["02_episode_fatigue_curve_csv"] = p2_csv

        # ---------------------------------------------------------
        # File 3: 03_blind_voice_metrics_and_confusion_matrix.json + .csv
        # ---------------------------------------------------------
        p3_json = self.output_dir / "03_blind_voice_metrics_and_confusion_matrix.json"
        voice_payload = {
            "accuracy": voice.accuracy,
            "f1_macro": voice.f1_macro,
            "random_baseline": voice.random_baseline,
            "status": voice.status,
            "top_speakers": voice.top_speakers,
            "tested_lines_count": voice.tested_lines_count,
            "interchangeable_dialogue_count": voice.interchangeable_dialogue_count,
            "leakage_filtered_count": voice.leakage_filtered_count,
            "character_archetypes": voice.character_archetypes,
            "per_character_metrics": voice.per_character_metrics,
            "confusion_matrix": voice.confusion_matrix,
            "most_confused_pairs": voice.most_confused_pairs,
            "stylometric_summary": voice.stylometric_summary,
            "pair_similarity_matrix": voice.pair_similarity_matrix,
        }
        with open(p3_json, "w", encoding="utf-8") as fp:
            json.dump(voice_payload, fp, indent=2)
        files["03_voice_metrics_json"] = p3_json

        p3_csv = self.output_dir / "03_blind_voice_confusion_matrix.csv"
        with open(p3_csv, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            # Header
            writer.writerow(["True_Speaker \\ Predicted_Speaker"] + voice.top_speakers + ["Total_Actual"])
            for true_spk in voice.top_speakers:
                row_counts = [voice.confusion_matrix[true_spk].get(pred_spk, 0) for pred_spk in voice.top_speakers]
                total_actual = sum(row_counts)
                writer.writerow([true_spk] + row_counts + [total_actual])
            # Column totals
            col_totals = [
                sum(voice.confusion_matrix[tr].get(pr, 0) for tr in voice.top_speakers)
                for pr in voice.top_speakers
            ]
            writer.writerow(["Total_Predicted"] + col_totals + [sum(col_totals)])
        files["03_voice_metrics_csv"] = p3_csv

        # ---------------------------------------------------------
        # File 4: 04_flagged_interchangeable_lines.csv
        # ---------------------------------------------------------
        p4 = self.output_dir / "04_flagged_interchangeable_lines.csv"
        with open(p4, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "beat_id",
                "episode_number",
                "timestamp",
                "true_speaker",
                "predicted_speaker",
                "confidence",
                "masked_text",
                "voice_cue_reason",
            ])
            for pred in voice.top_flagged_lines:
                writer.writerow([
                    pred.beat_id,
                    pred.episode_number or "",
                    pred.timestamp or "",
                    pred.true_speaker,
                    pred.predicted_speaker,
                    pred.confidence,
                    pred.masked_text.replace("\n", " "),
                    pred.voice_cue_reason or pred.reasoning,
                ])
        files["04_flagged_interchangeable_lines"] = p4

        # ---------------------------------------------------------
        # File 5: 05_validation_and_failure_log.json
        # ---------------------------------------------------------
        p5 = self.output_dir / "05_validation_and_failure_log.json"
        all_parsing_issues = []
        for ep in episodes:
            for iss in ep.validation_issues:
                all_parsing_issues.append({
                    "episode_number": ep.episode_number,
                    "line_number": iss.line_number,
                    "issue_type": iss.issue_type,
                    "message": iss.message,
                    "raw_content": iss.raw_content,
                    "severity": iss.severity,
                })

        hallucination_citations = [
            {
                "episode_number": a.episode_number,
                "cited_beat_id": a.evidence_beat_id,
                "status": "REJECTED_AS_HALLUCINATION",
                "resolution": "Fallback to ending window boundary",
            }
            for a in fatigue.audits
            if not a.evidence_beat_valid
        ]

        failure_payload = {
            "summary": {
                "total_parsing_issues": len(all_parsing_issues),
                "unparsed_line_rate_pct": round(len(all_parsing_issues) / max(1, sum(len(ep.beats) for ep in episodes)) * 100, 4),
                "leakage_filtered_lines_count": voice.leakage_filtered_count,
                "invalid_llm_citations_count": len(hallucination_citations),
                "interchangeable_dialogue_misclassifications": len(voice.flagged_predictions),
            },
            "parser_warnings": all_parsing_issues,
            "citation_hallucination_rejections": hallucination_citations,
            "leakage_filter_details": {
                "description": "Lines excluded from blind voice test because speaker named themselves or stated their title",
                "filtered_count": voice.leakage_filtered_count,
            },
            "low_confidence_voice_predictions": [
                {
                    "beat_id": p.beat_id,
                    "true_speaker": p.true_speaker,
                    "predicted_speaker": p.predicted_speaker,
                    "confidence": p.confidence,
                    "reason": p.voice_cue_reason,
                }
                for p in voice.flagged_predictions
                if p.confidence < 0.65
            ],
        }
        with open(p5, "w", encoding="utf-8") as fp:
            json.dump(failure_payload, fp, indent=2)
        files["05_validation_and_failure_log"] = p5

        # ---------------------------------------------------------
        # File 6: 06_cost_and_latency_report.json
        # ---------------------------------------------------------
        p6 = self.output_dir / "06_cost_and_latency_report.json"
        usage = self.client.tracker.summary()
        show_cost = usage["total_cost_usd"]
        annual_1000_shows_cost = round(show_cost * 1000.0, 2)

        cost_payload = {
            "provider": self.client.provider.value,
            "model": self.client.model,
            "mock_mode": self.client.mock,
            "current_run_metrics": {
                "total_llm_calls": usage["total_calls"],
                "cached_llm_calls": usage["cached_calls"],
                "cache_hit_rate_pct": usage["cache_hit_rate_pct"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "total_latency_seconds": usage["total_latency_sec"],
                "usd_cost_this_run": show_cost,
            },
            "stage_breakdown": usage["by_stage"],
            "production_roi_projections": {
                "unit_cost_per_50_ep_season_usd": show_cost,
                "projected_annual_cost_1000_shows_usd": annual_1000_shows_cost,
                "estimated_ai_video_render_cost_saved_per_prevented_reshoot_usd": 1500.0,
                "projected_net_savings_per_season_usd": round(1500.0 - show_cost, 2),
                "roi_multiple": round(1500.0 / max(0.01, show_cost), 1),
            },
        }
        with open(p6, "w", encoding="utf-8") as fp:
            json.dump(cost_payload, fp, indent=2)
        files["06_cost_and_latency_report"] = p6

        return files
