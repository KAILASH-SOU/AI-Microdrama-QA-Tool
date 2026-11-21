"""Deterministic Screenplay Parser & Beat Indexer.

Parses single .txt files, directories of .txt episodes, or .zip screenplay archives
into structured, strictly validated EpisodeData objects with deterministic beat IDs,
exact timestamp math, and zero hallucinated citations.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.schemas import (
    Beat,
    BeatType,
    EpisodeData,
    SceneData,
    ValidationIssue,
)

# Regex Patterns for Parsing
RE_METADATA_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s+)?([0-9]+)\s*$", re.IGNORECASE)
RE_METADATA_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.+)$", re.IGNORECASE)
RE_METADATA_CONTINUITY = re.compile(r"^CONTINUITY NOTES:\s*(.+)$", re.IGNORECASE)
RE_SCRIPT_START = re.compile(r"^---\s*SCRIPT START\s*---$", re.IGNORECASE)
RE_SCRIPT_END = re.compile(r"^---\s*SCRIPT END\s*---$", re.IGNORECASE)

RE_SCENE_NUM = re.compile(r"^SCENE\s+([0-9]+)\s*$", re.IGNORECASE)
RE_SCENE_HEADING = re.compile(r"^Scene heading:\s*(.+)$", re.IGNORECASE)
RE_SCENE_START = re.compile(r"^Scene start:\s*([0-9]{2}:[0-9]{2}\.[0-9]{2})\s*$", re.IGNORECASE)
RE_SCENE_END = re.compile(r"^Scene end:\s*([0-9]{2}:[0-9]{2}\.[0-9]{2})\s*$", re.IGNORECASE)
RE_ACTIONS_DIVIDER = re.compile(r"^Actions and dialogue:\s*$", re.IGNORECASE)

RE_BLOCK_TAG = re.compile(r"^\*\*([^*]+)\*\*\s*$")
RE_TIMESTAMP = re.compile(
    r"^Start:\s*\(([0-9]{2}):([0-9]{2}\.[0-9]{2})\)\s*\|\s*End:\s*\(([0-9]{2}):([0-9]{2}\.[0-9]{2})\)$"
)

# Canonical speaker alias dictionary for cross-episode voice testing
DEFAULT_SPEAKER_CANONICAL_MAP = {
    "LADY JULIANA": "JULIANA",
    "LADY CATHERINE": "CATHERINE",
    "LADY CLARISSA": "CLARISSA",
    "NOBLE MATRIARCH": "MATRIARCH",
    "THE FORMER GROOM": "GERALD",
    "BLOND NOBLEMAN": "RUPERT",
    "MYSTERIOUS BEARDED MAN": "THE BEARDED STRANGER",
    "THE PRIEST": "FATHER THOMAS",
    "PRIEST": "FATHER THOMAS",
    "ROYAL GUARD": "GUARD",
    "POLICE CONSTABLE": "CONSTABLE",
}


def timestamp_to_seconds(ts_str: str) -> float:
    """Converts MM:SS.ss timestamp string into precise float seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp format: '{ts_str}'. Expected 'MM:SS.ss'")
    minutes = int(parts[0])
    seconds = float(parts[1])
    return round(minutes * 60.0 + seconds, 3)


def seconds_to_timestamp(total_seconds: float) -> str:
    """Converts float seconds back into MM:SS.ss timestamp string."""
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def parse_speaker_tag(tag_str: str) -> Tuple[str, List[str], Optional[str]]:
    """Extracts raw base speaker name, modifier tags, and delivery tone from header.

    Example:
        'MATRIARCH (Cold, Commanding):' -> ('MATRIARCH', [], 'Cold, Commanding')
        'JULIANA (V.O.) (Reflective, Vulnerable):' -> ('JULIANA', ['V.O.'], 'Reflective, Vulnerable')
        'BLOND NOBLEMAN (Rupert) (Cold):' -> ('BLOND NOBLEMAN', ['Rupert'], 'Cold')
    """
    clean_tag = tag_str.strip().rstrip(":")
    parens = re.findall(r"\(([^)]+)\)", clean_tag)
    base_name = re.sub(r"\(.*", "", clean_tag).strip().upper()

    if not parens:
        return base_name, [], None

    delivery_tone = parens[-1].strip()
    modifiers = [p.strip() for p in parens[:-1]]
    return base_name, modifiers, delivery_tone


def canonicalize_speaker(
    raw_base: str,
    modifiers: List[str],
    alias_map: Optional[Dict[str, str]] = None,
) -> str:
    """Normalizes minor name variations to a canonical speaker label."""
    norm = raw_base.strip().upper()
    active_map = alias_map if alias_map is not None else DEFAULT_SPEAKER_CANONICAL_MAP

    # If an alias/character name is explicitly embedded in modifiers (e.g. Blond Nobleman (Rupert))
    for mod in modifiers:
        mod_upper = mod.upper()
        if mod_upper in active_map:
            return active_map[mod_upper]
        if mod_upper in ["RUPERT", "CLARA", "ALEXANDER", "JULIANA", "CATHERINE", "MAYA"]:
            return mod_upper

    return active_map.get(norm, norm)


class ScreenplayParser:
    """Deterministic parser converting screenplay files into structured EpisodeData."""

    def __init__(self, speaker_alias_map: Optional[Dict[str, str]] = None):
        self.speaker_alias_map = speaker_alias_map if speaker_alias_map is not None else DEFAULT_SPEAKER_CANONICAL_MAP

    def parse_text(
        self,
        text_content: str,
        file_path: Optional[str] = None,
        fallback_episode_num: int = 1,
    ) -> EpisodeData:
        """Parses a raw screenplay text string into EpisodeData."""
        lines = text_content.splitlines()
        validation_issues: List[ValidationIssue] = []

        episode_num = fallback_episode_num
        logline = ""
        continuity_notes = ""

        # Phase A: Parse Metadata Preamble
        script_start_idx = -1
        script_end_idx = len(lines)

        for i, line in enumerate(lines):
            line_str = line.strip()
            if RE_SCRIPT_START.match(line_str):
                script_start_idx = i
                break

            m_ep = RE_METADATA_EP_NUM.match(line_str)
            if m_ep:
                episode_num = int(m_ep.group(1))
                continue

            m_log = RE_METADATA_LOGLINE.match(line_str)
            if m_log:
                logline = m_log.group(1).strip()
                continue

            m_con = RE_METADATA_CONTINUITY.match(line_str)
            if m_con:
                continuity_notes = m_con.group(1).strip()
                continue

        if script_start_idx == -1:
            validation_issues.append(
                ValidationIssue(
                    file_path=file_path,
                    episode_number=episode_num,
                    line_number=1,
                    issue_type="MISSING_SCRIPT_START",
                    message="Delimiter '--- SCRIPT START ---' not found. Parsing entire file as body.",
                    severity="WARNING",
                )
            )
            body_start = 0
        else:
            body_start = script_start_idx + 1

        # Locate Script End
        for i in range(body_start, len(lines)):
            if RE_SCRIPT_END.match(lines[i].strip()):
                script_end_idx = i
                break

        body_lines = lines[body_start:script_end_idx]

        # Phase B: Parse Body (Scenes, Beats, Timestamps)
        scenes: List[SceneData] = []
        beats: List[Beat] = []

        current_scene_num = 1
        current_scene_heading = "UNTITLED SCENE"
        current_scene_start_ts = "00:00.00"
        current_scene_end_ts = "00:00.00"
        current_scene_beats: List[str] = []

        line_idx = 0
        total_body_lines = len(body_lines)
        beat_counter = 1

        while line_idx < total_body_lines:
            raw_line = body_lines[line_idx]
            line_str = raw_line.strip()
            current_file_line = body_start + line_idx + 1

            if not line_str:
                line_idx += 1
                continue

            # Scene Number
            m_scene = RE_SCENE_NUM.match(line_str)
            if m_scene:
                # Close out previous scene if it had beats
                if current_scene_beats:
                    s_start_sec = timestamp_to_seconds(current_scene_start_ts)
                    s_end_sec = timestamp_to_seconds(current_scene_end_ts)
                    scenes.append(
                        SceneData(
                            scene_number=current_scene_num,
                            heading=current_scene_heading,
                            start_ts=current_scene_start_ts,
                            end_ts=current_scene_end_ts,
                            start_sec=s_start_sec,
                            end_sec=s_end_sec,
                            duration_sec=round(s_end_sec - s_start_sec, 3),
                            beat_ids=list(current_scene_beats),
                        )
                    )
                    current_scene_beats = []

                current_scene_num = int(m_scene.group(1))
                line_idx += 1
                continue

            # Scene Heading
            m_heading = RE_SCENE_HEADING.match(line_str)
            if m_heading:
                current_scene_heading = m_heading.group(1).strip()
                line_idx += 1
                continue

            # Scene Start Time
            m_sstart = RE_SCENE_START.match(line_str)
            if m_sstart:
                current_scene_start_ts = m_sstart.group(1).strip()
                line_idx += 1
                continue

            # Scene End Time
            m_send = RE_SCENE_END.match(line_str)
            if m_send:
                current_scene_end_ts = m_send.group(1).strip()
                line_idx += 1
                continue

            # Actions and dialogue divider
            if RE_ACTIONS_DIVIDER.match(line_str):
                line_idx += 1
                continue

            # Beat Block Header Check (**TAG:**)
            m_tag = RE_BLOCK_TAG.match(line_str)
            if m_tag:
                tag_content = m_tag.group(1).strip()

                # Next line must be timestamp
                if line_idx + 1 >= total_body_lines:
                    validation_issues.append(
                        ValidationIssue(
                            file_path=file_path,
                            episode_number=episode_num,
                            line_number=current_file_line,
                            issue_type="UNEXPECTED_EOF_AFTER_TAG",
                            message=f"Missing timestamp line after block tag: '{line_str}'",
                            raw_content=line_str,
                            severity="ERROR",
                        )
                    )
                    break

                ts_line = body_lines[line_idx + 1].strip()
                m_ts = RE_TIMESTAMP.match(ts_line)

                if not m_ts:
                    validation_issues.append(
                        ValidationIssue(
                            file_path=file_path,
                            episode_number=episode_num,
                            line_number=current_file_line + 1,
                            issue_type="MALFORMED_TIMESTAMP",
                            message=f"Expected Start/End timestamp, got: '{ts_line}'",
                            raw_content=ts_line,
                            severity="ERROR",
                        )
                    )
                    line_idx += 2
                    continue

                start_ts = f"{m_ts.group(1)}:{m_ts.group(2)}"
                end_ts = f"{m_ts.group(3)}:{m_ts.group(4)}"
                start_sec = timestamp_to_seconds(start_ts)
                end_sec = timestamp_to_seconds(end_ts)
                duration_sec = round(end_sec - start_sec, 3)

                # Content line (dialogue or description)
                content_lines: List[str] = []
                j = line_idx + 2
                while (
                    j < total_body_lines
                    and body_lines[j].strip() != ""
                    and not RE_BLOCK_TAG.match(body_lines[j].strip())
                    and not RE_SCENE_NUM.match(body_lines[j].strip())
                    and not RE_SCRIPT_END.match(body_lines[j].strip())
                ):
                    content_lines.append(body_lines[j].strip())
                    j += 1

                content_text = " ".join(content_lines).strip()
                consumed_lines = max(1, j - (line_idx + 2))

                # Determine Beat Type and Speaker Attribution
                tag_upper = tag_content.upper().rstrip(":")
                is_dialogue = False
                beat_type = BeatType.OTHER
                speaker: Optional[str] = None
                raw_speaker: Optional[str] = None
                delivery_tone: Optional[str] = None
                modifiers: List[str] = []

                if tag_upper == "CAMERA":
                    beat_type = BeatType.CAMERA
                elif tag_upper == "ACTION":
                    beat_type = BeatType.ACTION
                elif tag_upper in ["SOUND FX", "SOUND", "SFX", "MUSIC"]:
                    beat_type = BeatType.SOUND_FX
                else:
                    # Character Dialogue or Voiceover
                    is_dialogue = True
                    raw_base, mods, tone = parse_speaker_tag(tag_content)
                    raw_speaker = raw_base
                    delivery_tone = tone
                    modifiers = mods

                    # Check for voiceover tags
                    is_vo = any(
                        m.upper() in ["V.O.", "VO", "VOICEOVER", "V.O"]
                        for m in modifiers
                    ) or "V.O." in tag_content.upper()

                    beat_type = BeatType.VOICEOVER if is_vo else BeatType.DIALOGUE
                    speaker = canonicalize_speaker(
                        raw_base, modifiers, self.speaker_alias_map
                    )

                    # Dialogue quotation extraction
                    if content_text.startswith('"') and content_text.endswith('"'):
                        content_text = content_text[1:-1].strip()

                beat_id = f"E{episode_num:02d}_B{beat_counter:03d}"

                beat = Beat(
                    beat_id=beat_id,
                    episode_number=episode_num,
                    scene_number=current_scene_num,
                    scene_heading=current_scene_heading,
                    beat_type=beat_type,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                    speaker=speaker,
                    raw_speaker=raw_speaker,
                    delivery_tone=delivery_tone,
                    modifiers=modifiers,
                    text=content_text,
                    is_dialogue=is_dialogue,
                )

                beats.append(beat)
                current_scene_beats.append(beat_id)
                beat_counter += 1

                line_idx += 2 + consumed_lines
                continue

            # If line reached here, it's an unrecognized/unparsed line
            validation_issues.append(
                ValidationIssue(
                    file_path=file_path,
                    episode_number=episode_num,
                    line_number=current_file_line,
                    issue_type="UNPARSED_LINE",
                    message=f"Unrecognized syntax: '{line_str}'",
                    raw_content=line_str,
                    severity="WARNING",
                )
            )
            line_idx += 1

        # Close final scene
        if current_scene_beats:
            s_start_sec = timestamp_to_seconds(current_scene_start_ts)
            s_end_sec = timestamp_to_seconds(current_scene_end_ts)
            scenes.append(
                SceneData(
                    scene_number=current_scene_num,
                    heading=current_scene_heading,
                    start_ts=current_scene_start_ts,
                    end_ts=current_scene_end_ts,
                    start_sec=s_start_sec,
                    end_sec=s_end_sec,
                    duration_sec=round(s_end_sec - s_start_sec, 3),
                    beat_ids=list(current_scene_beats),
                )
            )

        # Compute aggregate episode statistics
        total_beats = len(beats)
        dialogue_beats = [b for b in beats if b.is_dialogue and b.speaker]
        dialogue_count = len(dialogue_beats)

        speaker_counts: Dict[str, int] = dict(
            Counter(b.speaker for b in dialogue_beats if b.speaker)
        )

        total_duration = max((b.end_sec for b in beats), default=0.0)
        if scenes:
            scene_max = max(s.end_sec for s in scenes)
            total_duration = max(total_duration, scene_max)

        return EpisodeData(
            episode_number=episode_num,
            title=f"Episode {episode_num}",
            logline=logline,
            continuity_notes=continuity_notes,
            total_duration_sec=round(total_duration, 3),
            total_beats=total_beats,
            dialogue_line_count=dialogue_count,
            speaker_counts=speaker_counts,
            scenes=scenes,
            beats=beats,
            validation_issues=validation_issues,
        )

    def parse_file(self, file_path: Union[str, Path]) -> EpisodeData:
        """Parses a single .txt screenplay file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as fp:
            text = fp.read()

        # Extract episode number from filename as fallback
        fallback_ep = 1
        m = re.search(r"episode-?([0-9]+)", path.name, re.IGNORECASE)
        if m:
            fallback_ep = int(m.group(1))

        return self.parse_text(text, file_path=str(path), fallback_episode_num=fallback_ep)

    def parse_directory(
        self, directory_path: Union[str, Path], pattern: str = "*.txt"
    ) -> List[EpisodeData]:
        """Parses all screenplay files in a directory, returning them sorted by episode number."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Screenplay directory not found: {directory_path}")

        files = sorted(dir_path.glob(pattern))
        episodes: List[EpisodeData] = []

        for f in files:
            ep = self.parse_file(f)
            episodes.append(ep)

        # Sort strictly by episode number
        episodes.sort(key=lambda e: e.episode_number)
        return episodes

    def parse_zip(self, zip_source: Union[str, Path, io.BytesIO, bytes]) -> List[EpisodeData]:
        """Parses screenplay text files directly from a .zip archive without disk unpacking."""
        episodes: List[EpisodeData] = []

        if isinstance(zip_source, bytes):
            zip_file = zipfile.ZipFile(io.BytesIO(zip_source))
        elif isinstance(zip_source, io.BytesIO):
            zip_file = zipfile.ZipFile(zip_source)
        else:
            zip_file = zipfile.ZipFile(Path(zip_source))

        with zip_file:
            txt_names = [
                n
                for n in zip_file.namelist()
                if n.endswith(".txt") and not n.startswith("__MACOSX") and not Path(n).name.startswith(".")
            ]

            for name in txt_names:
                with zip_file.open(name) as fp:
                    raw_bytes = fp.read()
                    text = raw_bytes.decode("utf-8", errors="replace")

                fallback_ep = 1
                m = re.search(r"episode-?([0-9]+)", Path(name).name, re.IGNORECASE)
                if m:
                    fallback_ep = int(m.group(1))

                ep = self.parse_text(text, file_path=name, fallback_episode_num=fallback_ep)
                episodes.append(ep)

        episodes.sort(key=lambda e: e.episode_number)
        return episodes


# Module-level convenience functions
def parse_episode(file_path: Union[str, Path]) -> EpisodeData:
    """Helper to parse a single episode file."""
    return ScreenplayParser().parse_file(file_path)


def parse_season(directory_or_zip: Union[str, Path]) -> List[EpisodeData]:
    """Helper to parse an entire season from directory or zip file."""
    path = Path(directory_or_zip)
    parser = ScreenplayParser()
    if path.is_file() and path.suffix.lower() == ".zip":
        return parser.parse_zip(path)
    if path.is_dir():
        return parser.parse_directory(path)
    raise ValueError(f"Path must be a directory or .zip file: {directory_or_zip}")
