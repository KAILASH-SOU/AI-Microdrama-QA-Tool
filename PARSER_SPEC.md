# PARSER SPECIFICATION: Microdrama Screenplay Structure & Extraction Grammar

> **Gate 0 Verification Complete:** Generated from empirical reconnaissance of 50 proprietary microdrama screenplays (`data/raw_scripts/`) across 3,727 total script blocks and 1,487 dialogue lines.

---

## 1. Executive Summary & Reconnaissance Metrics

| Metric | Empirical Count | Notes |
| :--- | :--- | :--- |
| **Total Episode Files** | 50 files | `episode-001-*` to `episode-050-*` |
| **Total Script Blocks (Beats)** | 3,727 blocks | Camera, Action, Sound FX, Dialogue |
| **Total Dialogue Blocks** | 1,487 blocks | 100% quoted with double quotes |
| **Distinct Character Names** | 64 speakers | 15 primary characters represent >85% of lines |
| **Timestamp Match Rate** | 100.0% (3,727 / 3,727) | Strict `Start: (MM:SS.ss) \| End: (MM:SS.ss)` |
| **Negative Duration Blocks** | 0 blocks | All timestamps are monotonic and valid |
| **Multi-Scene Episodes** | 25 episodes (50%) | Up to 8 scenes per episode (e.g., Episode 006) |
| **Episode Duration Range** | 70.5s – 200.75s | Mean duration: 123.25s (~2.05 min/episode) |

---

## 2. File Organization & Ingestion

### File Naming Convention
Files follow the exact naming convention:
```text
episode-(\d{3})-Episode \d+\.txt
```
*Example:* `episode-001-Episode 1671703.txt`, `episode-025-Episode 1678578.txt`, `episode-050-Episode 1697563.txt`.

### Episode Sorting
File sorting must be numeric based on the 3-digit episode index (`int(group(1))`), not raw alphabetical sorting, to guarantee sequential order across 1 to 50.

---

## 3. Screenplay Document Structure

Every episode file consists of three sequential zones:
1. **Preamble Zone:** Metadata preceding script start.
2. **Body Zone:** Scene headers and discrete 3-line Beat blocks.
3. **Post-Script Zone:** Delimited by `--- SCRIPT END ---` (0 trailing lines observed).

```text
┌─────────────────────────────────────────────────────────────┐
│ PREAMBLE ZONE                                               │
│ EPISODE NUMBER: Episode <N>                                 │
│ EPISODE LOGLINE: <Logline text>                             │
│ CONTINUITY NOTES: <Continuity notes text>                   │
├─────────────────────────────────────────────────────────────┤
│ --- SCRIPT START ---                                        │
├─────────────────────────────────────────────────────────────┤
│ BODY ZONE                                                   │
│   SCENE <NN>                                                │
│   Scene heading: <INT./EXT. SLUGLINE - TIME>                │
│   Scene start: MM:SS.ss                                     │
│   Scene end: MM:SS.ss                                       │
│   Actions and dialogue:                                     │
│                                                             │
│   **TAG:**                                                  │
│   Start: (MM:SS.ss) | End: (MM:SS.ss)                       │
│   <Content / "Dialogue">                                    │
│   ...                                                       │
├─────────────────────────────────────────────────────────────┤
│ --- SCRIPT END ---                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Metadata Grammar (Preamble)

All 50 episodes contain exactly three metadata keys:
- `^EPISODE NUMBER:\s*(?:Episode\s+)?([0-9]+)\s*$` -> Extracts integer `episode_number`.
- `^EPISODE LOGLINE:\s*(.+)$` -> Extracts string `logline`.
- `^CONTINUITY NOTES:\s*(.+)$` -> Extracts string `continuity_notes`.
- Delimiters:
  - `^---\s*SCRIPT START\s*---$`
  - `^---\s*SCRIPT END\s*---$`

---

## 5. Scene Header Grammar

Episodes contain between 1 and 8 scenes. Scene timestamps run **continuously** throughout the episode without resetting to zero.

### Case Variance Anomaly (Identified in Gate 0)
Most episodes use title case (`Scene heading:`), but Episode 006 uses uppercase (`SCENE HEADING:`).
Therefore, all scene header regexes must use case-insensitive matching (`re.IGNORECASE`).

### Scene Matching Grammar
```regex
# Scene Index:
^SCENE\s+([0-9]+)\s*$

# Slugline:
^Scene heading:\s*(.+)$   (with re.IGNORECASE)

# Scene Start Timestamp:
^Scene start:\s*([0-9]{2}:[0-9]{2}\.[0-9]{2})\s*$   (with re.IGNORECASE)

# Scene End Timestamp:
^Scene end:\s*([0-9]{2}:[0-9]{2}\.[0-9]{2})\s*$   (with re.IGNORECASE)

# Section Divider:
^Actions and dialogue:\s*$   (with re.IGNORECASE)
```

---

## 6. Beat Block Structure (The 3-Line Unit)

Across all 3,727 blocks in the dataset, each beat is strictly represented as an atomic 3-line block followed by an optional blank separator:

1. **Line 1 (Block Header):** `**<TAG>:**`
2. **Line 2 (Timestamp):** `Start: (MM:SS.ss) | End: (MM:SS.ss)`
3. **Line 3 (Content):** Single-line description text or double-quoted dialogue.

### Block Types
- **Technical Beats:**
  - `**CAMERA:**` (1,632 blocks across season)
  - `**ACTION:**` (558 blocks across season)
  - `**SOUND FX:**` (50 blocks across season)
- **Dialogue Beats:**
  - `**SPEAKER (Delivery Direction):**` (1,487 blocks across season)

---

## 7. Speaker Tag & Delivery Grammar (Crucial Edge Cases)

Empirical analysis discovered that speaker headers fall into two specific patterns:

### Pattern A: Standard Speaker Tag (1,431 / 1,487 = 96.2%)
Contains a character name and a single parenthetical specifying tone/delivery:
```text
**MATRIARCH (Cold, Commanding):**
**LADY JULIANA (Confused, Breathless):**
**ALEXANDER (Authoritative, Protective):**
```
- **Base Speaker:** `MATRIARCH`, `LADY JULIANA`, `ALEXANDER`
- **Delivery Direction:** `Cold, Commanding`, `Confused, Breathless`, `Authoritative, Protective`

### Pattern B: Modified Speaker Tag with Sub-Tags (56 / 1,487 = 3.8%)
Contains multiple parentheticals. Gate 0 discovered four modifier categories:
1. Off-screen: `(O.S.)` or `(Off-screen)` (e.g. `**BLOND NOBLEMAN (O.S.) (Contemptuous):**`)
2. Voice-over: `(V.O.)` (e.g. `**JULIANA (V.O.) (Reflective, Vulnerable):**`)
3. Character Alias/Identity: `(Rupert)` (e.g. `**BLOND NOBLEMAN (Rupert) (Authoritative):**`)

### Parsing Strategy for Speaker Tags
```python
# 1. Match enclosing asterisks and trailing colon:
# Line matches: ^\*\*(.+?):\*\*\s*$
# 2. Extract all parentheticals:
parens = re.findall(r"\(([^)]+)\)", inner_content)
# 3. Base Name: Text before the first '('
base_name = re.sub(r"\(.*", "", inner_content).strip()
# 4. Modifiers vs Delivery Tone:
if parens:
    delivery_tone = parens[-1].strip()  # Last paren is always delivery direction
    modifiers = [p.strip() for p in parens[:-1]]  # Preceding parens are modifiers (O.S., V.O., Alias)
```

### Character Name Canonicalization (For Module 2 Voice Auditor)
Certain characters appear under formal and informal titles across episodes:
- `LADY JULIANA` <-> `JULIANA`
- `LADY CATHERINE` <-> `CATHERINE`
- `LADY CLARISSA` <-> `CLARISSA`
- `THE BEARDED STRANGER` <-> `MYSTERIOUS BEARDED MAN`
- `THE PRIEST` <-> `FATHER THOMAS` / `PRIEST`

The parser retains the `raw_speaker` name exactly as authored, and adds a `canonical_speaker` field for aggregate stylometric and confusion matrix calculations.

---

## 8. Timestamp Math & Conversion Grammar

### Timestamp Regex
```regex
^Start:\s*\(([0-9]{2}):([0-9]{2}\.[0-9]{2})\)\s*\|\s*End:\s*\(([0-9]{2}):([0-9]{2}\.[0-9]{2})\)$
```

### Conversion Formula
Every timestamp is deterministically converted into float seconds:
```python
def timestamp_to_seconds(ts_str: str) -> float:
    # ts_str format: "MM:SS.ss"
    minutes_str, seconds_str = ts_str.split(":")
    return round(int(minutes_str) * 60 + float(seconds_str), 3)

def seconds_to_timestamp(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"
```

---

## 9. Dialogue Line Extraction Grammar

- **Quotes:** All 1,487 dialogue lines begin with `"` and end with `"`.
- **Character Encoding:** 100% standard ASCII double quotes (`"`, `0x22`); zero smart quotes (`“`, `”`).
- **Extraction Rule:**
  ```python
  raw_line = content_line.strip()
  if raw_line.startswith('"') and raw_line.endswith('"'):
      dialogue_text = raw_line[1:-1].strip()
  ```

---

## 10. Deterministic Beat ID Schema

To prevent LLM hallucination of citations, line numbers, or timestamps, every single beat receives a globally unique, deterministic ID:

```text
E{episode_number:02d}_B{beat_index:03d}
```
*Examples:*
- `E01_B001` (First beat of Episode 1)
- `E01_B014` (Dialogue beat 14 of Episode 1)
- `E25_B042` (Beat 42 of Episode 25)
- `E50_B085` (Closing beat of Episode 50)

### Bidirectional Mapping
1. **Forward Mapping:** Script text -> Parsed Beat Object with `beat_id`.
2. **Reverse Mapping:** LLM output (`beat_id="E01_B014"`) -> Python retrieves exact timestamp `(00:13.50 - 00:14.30)`, raw speaker, and original text.

---

## 11. Edge Cases & Safeguards Checklist (Gate 0 Sign-Off)

| Potential Failure Point | Observed in Dataset? | Resolution Strategy |
| :--- | :--- | :--- |
| **All-caps scene headers** | Yes (`SCENE HEADING:` in Ep 006) | Use `re.IGNORECASE` for all scene headers and dividers. |
| **Multiple parentheticals in speaker tags** | Yes (56 beats with `(O.S.)`, `(V.O.)`, `(Rupert)`) | Use multi-paren extraction: last paren = tone, prior parens = modifiers. |
| **Multiline dialogue lines** | No (0 occurrences across 1,487 lines) | Read single line following timestamp; assert no stray dialogue lines. |
| **Overlapping timestamps** | Yes (Camera & Action blocks run in parallel) | Normal behavior in screenplay timeline; do not treat as error. |
| **Proprietary text leakage** | High risk | Strictly enforce `.gitignore`; never commit raw script text to repository. |
