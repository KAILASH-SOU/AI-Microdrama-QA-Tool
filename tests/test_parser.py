"""Unit tests for Screenplay Parser and Beat Indexer."""

import io
import zipfile
from pathlib import Path
import pytest

from src.parser import (
    ScreenplayParser,
    canonicalize_speaker,
    parse_speaker_tag,
    seconds_to_timestamp,
    timestamp_to_seconds,
)
from src.schemas import BeatType, EpisodeData


def test_timestamp_conversions():
    """Tests bidirectional timestamp conversion math."""
    assert timestamp_to_seconds("00:00.00") == 0.0
    assert timestamp_to_seconds("00:15.50") == 15.5
    assert timestamp_to_seconds("01:23.45") == 83.45
    assert timestamp_to_seconds("02:18.00") == 138.0

    assert seconds_to_timestamp(0.0) == "00:00.00"
    assert seconds_to_timestamp(15.5) == "00:15.50"
    assert seconds_to_timestamp(83.45) == "01:23.45"
    assert seconds_to_timestamp(138.0) == "02:18.00"


def test_parse_speaker_tag():
    """Tests extraction of base speaker, modifiers, and delivery tone."""
    base, mods, tone = parse_speaker_tag("MATRIARCH (Cold, Commanding):")
    assert base == "MATRIARCH"
    assert mods == []
    assert tone == "Cold, Commanding"

    base, mods, tone = parse_speaker_tag("JULIANA (V.O.) (Reflective, Vulnerable):")
    assert base == "JULIANA"
    assert mods == ["V.O."]
    assert tone == "Reflective, Vulnerable"

    base, mods, tone = parse_speaker_tag("BLOND NOBLEMAN (Rupert) (Cold, Menacing):")
    assert base == "BLOND NOBLEMAN"
    assert mods == ["Rupert"]
    assert tone == "Cold, Menacing"

    base, mods, tone = parse_speaker_tag("SOLDIER:")
    assert base == "SOLDIER"
    assert mods == []
    assert tone is None


def test_canonicalize_speaker():
    """Tests normalization of character name variants."""
    assert canonicalize_speaker("LADY JULIANA", []) == "JULIANA"
    assert canonicalize_speaker("JULIANA", []) == "JULIANA"
    assert canonicalize_speaker("BLOND NOBLEMAN", ["Rupert"]) == "RUPERT"
    assert canonicalize_speaker("NOBLE MATRIARCH", []) == "MATRIARCH"
    assert canonicalize_speaker("CAPTAIN MAYA VANCE", []) == "CAPTAIN MAYA VANCE"


def test_parse_sample_episode_1():
    """Tests parsing synthetic Episode 1 and deterministic beat indexing."""
    parser = ScreenplayParser()
    ep = parser.parse_file("sample_data/episode-001-Episode 1000001.txt")

    assert isinstance(ep, EpisodeData)
    assert ep.episode_number == 1
    assert "Orion Singularity" in ep.logline
    assert ep.total_beats > 0
    assert ep.dialogue_line_count > 0

    # Test deterministic Beat ID indexing
    assert ep.beats[0].beat_id == "E01_B001"
    assert ep.beats[1].beat_id == "E01_B002"

    # Test Beat types and dialogue extraction
    dialogue_beats = [b for b in ep.beats if b.is_dialogue]
    assert len(dialogue_beats) >= 4

    # Check first dialogue beat
    vane_beat = next(b for b in ep.beats if b.speaker == "PROSECUTOR VANE")
    assert vane_beat.beat_type == BeatType.DIALOGUE
    assert vane_beat.delivery_tone == "Arrogant, Resonant"
    assert not vane_beat.text.startswith('"')
    assert not vane_beat.text.endswith('"')
    assert "highest treason" in vane_beat.text

    # Check Voiceover beat
    vo_beat = next(b for b in ep.beats if b.beat_type == BeatType.VOICEOVER)
    assert vo_beat.speaker == "CAPTAIN MAYA VANCE"
    assert "V.O." in vo_beat.modifiers
    assert "shadows leave prints" in vo_beat.text.lower()


def test_parse_multi_scene_episode():
    """Tests multi-scene parsing on synthetic Episode 3."""
    parser = ScreenplayParser()
    ep = parser.parse_file("sample_data/episode-003-Episode 1000003.txt")

    assert ep.episode_number == 3
    assert len(ep.scenes) == 2

    scene1 = ep.scenes[0]
    assert scene1.scene_number == 1
    assert "HANGAR BAY" in scene1.heading.upper()
    assert scene1.start_ts == "00:00.00"
    assert scene1.end_ts == "00:45.00"

    scene2 = ep.scenes[1]
    assert scene2.scene_number == 2
    assert "BRIDGE" in scene2.heading.upper()
    assert scene2.start_ts == "00:45.00"
    assert scene2.end_ts == "01:35.00"


def test_parse_zip_in_memory():
    """Tests parsing episodes directly from an in-memory zip archive."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        p1 = Path("sample_data/episode-001-Episode 1000001.txt")
        p2 = Path("sample_data/episode-002-Episode 1000002.txt")
        zf.writestr(p1.name, p1.read_text(encoding="utf-8"))
        zf.writestr(p2.name, p2.read_text(encoding="utf-8"))

    zip_bytes = zip_buffer.getvalue()
    parser = ScreenplayParser()
    episodes = parser.parse_zip(zip_bytes)

    assert len(episodes) == 2
    assert episodes[0].episode_number == 1
    assert episodes[1].episode_number == 2


def test_unparsed_line_logging():
    """Tests that malformed or unrecognized syntax is captured in ValidationIssue."""
    corrupted_script = """EPISODE NUMBER: Episode 99
EPISODE LOGLINE: Test corrupted script.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. ROOM – DAY
Scene start: 00:00.00
Scene end: 00:30.00
Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Normal camera framing.

THIS_IS_A_CORRUPTED_LINE_WITHOUT_PROPER_SYNTAX

**ACTION:**
Start: (00:05.00) | End: (00:10.00)
Normal action line.

--- SCRIPT END ---
"""
    parser = ScreenplayParser()
    ep = parser.parse_text(corrupted_script, file_path="test_corrupt.txt")

    assert ep.episode_number == 99
    assert len(ep.beats) == 2
    assert len(ep.validation_issues) == 1
    assert ep.validation_issues[0].issue_type == "UNPARSED_LINE"
    assert "THIS_IS_A_CORRUPTED_LINE" in ep.validation_issues[0].raw_content
