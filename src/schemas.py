"""Pydantic data schemas for Script Health Gatekeeper.

Defines strict types for Screenplay Beats, Episodes, Cliffhanger Audits,
Blind Voice Predictions, Validation Issues, and Aggregate Health Reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class BeatType(str, Enum):
    """Categorical classification of screenplay block types."""
    DIALOGUE = "DIALOGUE"
    VOICEOVER = "VOICEOVER"
    ACTION = "ACTION"
    CAMERA = "CAMERA"
    SOUND_FX = "SOUND_FX"
    SCENE_HEADER = "SCENE_HEADER"
    OTHER = "OTHER"


class EndingClassification(str, Enum):
    """Classification of microdrama episode final beats."""
    HARD_CLIFFHANGER = "HARD_CLIFFHANGER"
    MICRO_PAYOFF_WITH_HOOK = "MICRO_PAYOFF_WITH_HOOK"
    SATISFYING_CLOSE = "SATISFYING_CLOSE"
    FLAT_ENDING = "FLAT_ENDING"


class ValidationIssue(BaseModel):
    """Tracks unparsed lines, syntax mismatches, or malformed data."""
    file_path: Optional[str] = None
    episode_number: Optional[int] = None
    line_number: Optional[int] = None
    issue_type: str = Field(..., description="E.g., UNPARSED_LINE, MALFORMED_TIMESTAMP, MISSING_TAG")
    message: str = Field(..., description="Human-readable issue description")
    raw_content: Optional[str] = None
    severity: str = Field(default="WARNING", description="INFO, WARNING, or ERROR")


class Beat(BaseModel):
    """Atomic unit of a screenplay (Camera shot, Action beat, Sound FX, or Dialogue line)."""
    beat_id: str = Field(..., description="Deterministic ID in format E{ep:02d}_B{idx:03d}")
    episode_number: int = Field(..., ge=1, description="Episode sequence number (1-based)")
    scene_number: int = Field(..., ge=1, description="Scene index within episode")
    scene_heading: str = Field(..., description="INT./EXT. slugline")
    beat_type: BeatType = Field(..., description="Block classification")
    start_ts: str = Field(..., description="Start timestamp string MM:SS.ss")
    end_ts: str = Field(..., description="End timestamp string MM:SS.ss")
    start_sec: float = Field(..., ge=0.0, description="Start time in seconds")
    end_sec: float = Field(..., ge=0.0, description="End time in seconds")
    duration_sec: float = Field(..., ge=0.0, description="Duration in seconds")
    speaker: Optional[str] = Field(default=None, description="Normalized canonical uppercase speaker name")
    raw_speaker: Optional[str] = Field(default=None, description="Exact author attribution string")
    delivery_tone: Optional[str] = Field(default=None, description="Parenthetical performance direction")
    modifiers: List[str] = Field(default_factory=list, description="Sub-tags e.g. O.S., V.O., Off-screen")
    text: str = Field(..., description="Dialogue text or technical/action description")
    is_dialogue: bool = Field(default=False, description="True if block contains character dialogue")

    @field_validator("duration_sec")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Duration cannot be negative: {v}")
        return round(v, 3)


class SceneData(BaseModel):
    """Structured container for a single scene within an episode."""
    scene_number: int = Field(..., ge=1)
    heading: str
    start_ts: str
    end_ts: str
    start_sec: float
    end_sec: float
    duration_sec: float
    beat_ids: List[str] = Field(default_factory=list)


class EpisodeData(BaseModel):
    """Structured container for an entire parsed episode screenplay."""
    episode_number: int = Field(..., ge=1)
    title: Optional[str] = None
    logline: str = ""
    continuity_notes: str = ""
    total_duration_sec: float = 0.0
    total_beats: int = 0
    dialogue_line_count: int = 0
    speaker_counts: Dict[str, int] = Field(default_factory=dict)
    scenes: List[SceneData] = Field(default_factory=list)
    beats: List[Beat] = Field(default_factory=list)
    validation_issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def formatted_duration(self) -> str:
        """Returns MM:SS formatted duration string."""
        minutes = int(self.total_duration_sec // 60)
        seconds = self.total_duration_sec % 60
        return f"{minutes:02d}:{seconds:05.2f}"

    def get_closing_beats(self, count: int = 5) -> List[Beat]:
        """Returns the final N beats of the episode for ending classification."""
        return self.beats[-count:] if len(self.beats) >= count else self.beats


class EndingAudit(BaseModel):
    """Enriched audit of an episode ending combining LLM narrative judgment with deterministic Python checks."""
    episode_number: int
    ending_type: EndingClassification
    tension_score: int = Field(..., ge=1, le=10, description="Tension rating 1-10")
    payoff_score: int = Field(..., ge=1, le=10, description="Payoff rating 1-10 (1=unresolved, 10=major payoff)")
    evidence_beat_id: str = Field(..., description="Deterministic beat_id cited by LLM")
    evidence_beat_valid: bool = Field(default=True, description="True if cited beat_id exists in the episode")
    evidence_beat_text: Optional[str] = Field(default=None, description="Actual script text from verified beat_id")
    evidence_beat_ts: Optional[str] = Field(default=None, description="Exact timestamp of verified beat_id")
    one_line_verdict: str = Field(..., description="Concise editorial reason")
    is_paywall_danger_zone: bool = Field(default=False, description="True for Episodes 5-15")
    streak_count: int = Field(default=0, description="Consecutive unresolved cliffhangers streak length")
    fatigue_flag: bool = Field(default=False, description="True if streak >= 3")
    danger_zone_fatigue: bool = Field(default=False, description="True if fatigue occurs in paywall zone (Ep 5-15)")
    suggested_pivot: Optional[str] = Field(default=None, description="Actionable micro-payoff recommendation")
    raw_llm_response: Optional[str] = None

    # Convenience properties for backward compatibility
    @property
    def classification(self) -> EndingClassification:
        return self.ending_type

    @property
    def justification(self) -> str:
        return self.one_line_verdict

    @property
    def cited_beat_ids(self) -> List[str]:
        return [self.evidence_beat_id]

    @property
    def cliffhanger_fatigue_flag(self) -> bool:
        return self.fatigue_flag


class EpisodeEndingLLMResponse(BaseModel):
    """Raw JSON schema requested from the LLM for episode ending classification."""
    ending_type: EndingClassification = Field(..., description="HARD_CLIFFHANGER | MICRO_PAYOFF_WITH_HOOK | SATISFYING_CLOSE | FLAT_ENDING")
    tension_score: int = Field(..., ge=1, le=10, description="1-10 intensity score")
    payoff_score: int = Field(..., ge=1, le=10, description="1-10 narrative question payoff score")
    evidence_beat_id: str = Field(..., description="Exact beat_id from the episode justifying judgment")
    one_line_verdict: str = Field(..., description="Concise editorial reason")
    suggested_pivot: Optional[str] = Field(default=None, description="Optional micro-payoff recommendation if streak is high")


class FatigueAuditResult(BaseModel):
    """Aggregate report across all episodes for cliffhanger and payoff pacing."""
    total_episodes_audited: int
    hard_cliffhanger_count: int = 0
    micro_payoff_count: int = 0
    satisfying_close_count: int = 0
    flat_ending_count: int = 0
    cliffhanger_to_payoff_ratio: float = Field(default=0.0, description="Hard cliffhangers / total payoffs")
    max_consecutive_cliffhangers: int = 0
    danger_zone_cliffhangers: int = 0
    danger_zone_fatigue_episodes: List[int] = Field(default_factory=list)
    fatigue_flagged_episodes: List[int] = Field(default_factory=list)
    season_pacing_score: float = Field(default=100.0, ge=0.0, le=100.0, description="0-100 narrative health metric")
    status: str = Field(default="HEALTHY_PACING", description="HEALTHY_PACING, MODERATE_FATIGUE_RISK, or CRITICAL_PAYWALL_FATIGUE")
    audits: List[EndingAudit] = Field(default_factory=list)

    @property
    def fatigue_index_score(self) -> float:
        return round(100.0 - self.season_pacing_score, 1)


class VoicePrediction(BaseModel):
    """Single blind speaker prediction outcome from zero-leakage voice testing."""
    beat_id: str
    episode_number: Optional[int] = None
    timestamp: Optional[str] = None
    masked_text: str
    true_speaker: str
    predicted_speaker: str
    is_correct: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""
    voice_cue_reason: Optional[str] = None
    interchangeable_flag: bool = Field(default=False, description="Flagged if model misclassifies with high confidence")
    stylometric_scores: Dict[str, float] = Field(default_factory=dict)
    rewritten_text: Optional[str] = None
    retest_predicted_speaker: Optional[str] = None
    retest_is_correct: Optional[bool] = None


class VoiceAuditResult(BaseModel):
    """Aggregate stylometric and blind prediction results for character distinction."""
    accuracy: float = Field(..., ge=0.0, le=1.0)
    f1_macro: float = Field(..., ge=0.0, le=1.0)
    random_baseline: float = Field(default=0.25, ge=0.0, le=1.0)
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    top_speakers: List[str] = Field(default_factory=list)
    tested_lines_count: int = 0
    interchangeable_dialogue_count: int = 0
    leakage_filtered_count: int = 0
    character_archetypes: Dict[str, str] = Field(default_factory=dict)
    per_character_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    most_confused_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    flagged_predictions: List[VoicePrediction] = Field(default_factory=list)
    top_flagged_lines: List[VoicePrediction] = Field(default_factory=list)
    stylometric_summary: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    pair_similarity_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    status: str = Field(default="PASS", description="PASS or LOW_VOICE_DISTINCTION")


class SeasonHealthReport(BaseModel):
    """Master pre-render health assessment combining Module 1, Module 2, and Parser health."""
    season_name: str = "Microdrama Season 1"
    total_episodes: int
    total_duration_sec: float
    total_beats: int
    total_dialogue_lines: int
    speaker_distribution: Dict[str, int]
    fatigue_audit: FatigueAuditResult
    voice_audit: VoiceAuditResult
    validation_summary: Dict[str, Any] = Field(default_factory=dict)
    llm_cost_and_usage: Dict[str, Any] = Field(default_factory=dict)
    overall_verdict: str = Field(..., description="READY_FOR_RENDER, CAUTION, or REVISION_REQUIRED")
    executive_summary: str = ""
