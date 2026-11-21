# 🎬 Pre-Render Script Health Gatekeeper
### *AI Microdrama Production QA Tool — High-ROI Pre-Render Health Gatekeeper*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2.7+-green.svg)](https://docs.pydantic.dev/)
[![Streamlit Ready](https://img.shields.io/badge/streamlit-cloud%20ready-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-20%20passed%20%7C%20100%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Production Objective

In the high-velocity vertical microdrama industry (ReelShort, DramaBox, ShortMax), producing a 50-to-80 episode season through AI video synthesis tools (Runway, Kling, Sora, Hailuo) costs thousands of dollars per season in GPU compute and rendering fees. 

The **Pre-Render Script Health Gatekeeper** is an autonomous quality-assurance system tailored for an **AI Microdrama Writer / Creative Producer**. It automates the critical pre-render gatekeeping step: **before** sending screenplays to expensive video generation and voice-cloning pipelines, it deterministically parses screenplays and executes two targeted QA audits:

1. **Module 1 — Cliffhanger-to-Payoff Fatigue Auditor:**
   - Detects consecutive unresolved cliffhanger streaks across the season.
   - Monitors the critical **Paywall Danger Zone (Episodes 5–15)** where viewers are prompted to spend coins. Unearned cliffhangers without micro-payoffs trigger viewer fatigue and churn.
2. **Module 2 — Zero-Leakage Blind Character-Voice Tester:**
   - Tests whether lead characters possess distinct voices or sound interchangeable.
   - Strips speaker tags, applies an anti-leakage filter, computes deterministic stylometrics (Type-Token Ratio, syntax, TF-IDF cosine similarity), and runs a blind attribution test via a swappable LLM to produce an exact Confusion Matrix, Macro-F1 score, and flagged dialogue lines.

---

## 2. Architecture: Strict Code vs. AI Split

| Component | Technology | Role & Guarantees |
| :--- | :--- | :--- |
| **Deterministic Core** | Pure Python 3 + Pydantic v2 | File parsing, timestamp math, Beat-ID indexing (`E01_B014`), streak counting, anti-leakage speaker masking, stylometrics, confusion matrix math, JSON schema validation, and cost tracking. **Zero LLM hallucinations.** |
| **Narrative Judgment** | Swappable LLM Adapter | Evaluates ending structure (`HARD_CLIFFHANGER`, `MICRO_PAYOFF_WITH_HOOK`, `SATISFYING_CLOSE`, `FLAT_ENDING`), performs blind speaker prediction from 2-sentence archetype cards, and suggests dialogue rewrites. |
| **Free-Tier Resilience** | SHA-256 Disk Caching | All LLM calls are hashed by `(provider, model, prompt_version, payload)`. Repeat queries execute in `0.000s` at **\$0.00 marginal cost**. |

---

## 3. Repository Structure

```text
script-health-gatekeeper/
├── .gitignore                      # Enforces proprietary script confidentiality
├── .env.example                    # Swappable model & API credentials template
├── requirements.txt                # Pydantic, Streamlit, Plotly, Scikit-Learn, GenAI SDKs
├── PARSER_SPEC.md                  # Comprehensive empirical grammar & token specification
├── README.md                       # Production documentation & user guide
├── data/
│   └── raw_scripts/                # [GITIGNORED] 50 proprietary .txt episodes
├── sample_data/                    # [COMMITTED] 3-episode 100% synthetic dummy sci-fi show
├── src/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic models (Beat, EpisodeData, EndingAudit, VoicePrediction)
│   ├── parser.py                   # Deterministic Beat-ID & Screenplay parser
│   ├── llm_client.py               # Swappable LLM adapter + retry + SHA-256 disk cache + cost tracker
│   ├── fatigue_auditor.py          # Module 1: Cliffhanger-to-Payoff Fatigue Auditor
│   ├── voice_auditor.py            # Module 2: Zero-Leakage Blind Voice Tester + Stylometrics
│   └── pipeline.py                 # End-to-end orchestrator & raw outputs exporter
├── tests/
│   ├── conftest.py
│   ├── test_parser.py              # Parsing, timestamp math, multi-scene, zip tests
│   ├── test_llm_client.py          # JSON repair, disk caching, cost tracking tests
│   ├── test_fatigue_math.py        # Streak math, paywall zone alert, citation validation tests
│   └── test_voice_masking.py       # Anti-leakage, stylometrics, confusion matrix tests
├── scripts/
│   ├── verify_phase1.py            # Phase 1 empirical dataset verification (50 episodes)
│   ├── verify_phase2.py            # Phase 2 LLM adapter smoke test
│   ├── verify_phase3.py            # Phase 3 Module 1 smoke test
│   └── verify_phase4.py            # Phase 4 Module 2 smoke test
├── run_audit.py                    # CLI runner generating the 6 Raw Output files
└── app.py                          # Streamlit Community Cloud interactive UI
```

---

## 4. Quickstart & Local Setup

### 1. Clone & Create Virtual Environment
```bash
git clone <repo-url>
cd script-health-gatekeeper

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your preferred API key (Gemini, OpenAI, Anthropic, or Groq). If left blank, the system automatically runs in **Deterministic Mock Mode**, allowing full offline verification at \$0 cost.

### 3. Run the Test Suite
```bash
pytest tests/ -v
# Verified: 20 passed in <1.0s
```

---

## 5. CLI Execution & Raw Outputs Generation

Run the CLI against either the public synthetic demo (`sample_data/`) or a complete season (`data/raw_scripts/` or a `.zip` archive):

```bash
# Run on Public Synthetic Sample
python run_audit.py --input sample_data --output outputs/sample_run

# Run on Full 50-Episode Season with Google Gemini
python run_audit.py --input data/raw_scripts --output outputs/run_full_50 --provider gemini --model gemini-2.5-flash

# Run on Full Season with OpenAI
python run_audit.py --input data/raw_scripts --output outputs/run_full_50 --provider openai --model gpt-4o-mini
```

### The 6 Standardized Raw Output Files
The CLI generates the exact unedited audit files required for production handoff:
1. `01_season_health_summary.json`: Headline KPIs (verdict, pacing score, voice accuracy, max streak, model settings).
2. `02_episode_fatigue_curve.png`: High-resolution pacing curve image visualizing tension vs. payoff trajectories, consecutive cliffhanger streaks, and the Paywall Danger Zone (accompanied by `.csv` data companion).
3. `03_blind_voice_metrics_and_confusion_matrix.json` + `03_blind_voice_confusion_matrix.csv`: Pure Python stylometrics, pairwise TF-IDF cosine similarity, and $K \times K$ confusion matrix.
4. `04_flagged_interchangeable_lines.csv`: High-confidence misclassifications with exact timestamps, true vs. predicted speaker, and voice cues.
5. `05_validation_and_failure_log.json`: Transparent failure mode log (parser coverage, leakage-filtered lines, citation hallucination rejections).
6. `06_cost_and_latency_report.json`: Exact token usage, cache hits, runtime, show cost, and annual ROI projections.

---

## 6. Interactive Streamlit Web Application

Launch the recruiter-ready UI locally or deploy to **Streamlit Community Cloud**:

```bash
streamlit run app.py
```

### Dashboard Tabs
- **Tab 1: Executive Gatekeeper Scorecard**: Overall verdict banner (`READY_FOR_RENDER`, `CAUTION`, `REVISION_REQUIRED`), 4 headline KPI metric cards, red flags, and character dialogue breakdown.
- **Tab 2: Cliffhanger Fatigue Curve**: Interactive Plotly tension vs. payoff chart with shaded Paywall Danger Zone (Episodes 5–15) and searchable episode ending table.
- **Tab 3: Blind Character-Voice Audit**: Interactive Plotly confusion matrix heatmap, stylometric comparison table, TF-IDF similarity matrix, and an on-demand **"Rewrite & Re-Test Line"** assistant.
- **Tab 4: Failure Log & Raw Exports**: Transparent audit log display with 1-click download buttons for all 6 JSON & CSV outputs.

---

## 7. Swappable Model Configuration

Switching models requires zero code changes. You can set the provider in `.env`, via CLI arguments (`--provider`), or directly in the Streamlit sidebar:

| Provider | Supported Models | Configuration Key |
| :--- | :--- | :--- |
| **Google Gemini** *(Default Free Tier)* | `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash` | `GEMINI_API_KEY` |
| **OpenAI** | `gpt-4o-mini`, `gpt-4o` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-5-haiku-20241022`, `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **Groq** | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | `GROQ_API_KEY` |
| **OpenAI-Compatible** | Any local / vLLM / Ollama endpoint | `GENERIC_OPENAI_BASE_URL` |

---

## 8. Confidentiality Safeguards

- **Proprietary Scripts**: All proprietary scripts in `data/raw_scripts/` are strictly ignored by `.gitignore`.
- **Public Synthetic Demo**: A 3-episode synthetic sci-fi courtroom drama (*"The Quantum Tribunal"*) is committed in `sample_data/` so recruiters and public evaluators can run the tool end-to-end without needing proprietary files.
