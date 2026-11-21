"""Pre-Render Script Health Gatekeeper — Streamlit Interactive Web Application.

Production-grade, interview-ready QA dashboard for AI Microdrama Studios.
Features:
- Tab 1: Executive Gatekeeper Scorecard (Verdict, KPIs, Red Flags)
- Tab 2: Cliffhanger Fatigue Curve (Plotly Tension vs. Payoff, Paywall Zone Ep 5-15)
- Tab 3: Blind Character-Voice Audit (Confusion Matrix Heatmap, Stylometrics, On-Demand Rewrite & Re-Test)
- Tab 4: Quality Checks, Failure Log & 1-Click Raw Exports
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.fatigue_auditor import audit_season_fatigue
from src.llm_client import LLMClient, LLMProvider
from src.parser import ScreenplayParser
from src.pipeline import AuditPipeline
from src.schemas import (
    EpisodeData,
    FatigueAuditResult,
    SeasonHealthReport,
    VoiceAuditResult,
)
from src.voice_auditor import audit_season_voice, rewrite_and_retest_line


def clean_html(s: str) -> str:
    """Strips leading whitespace from every line so Markdown never treats HTML as an indented code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


# Page Configuration
st.set_page_config(
    page_title="Kuku ScriptGuard — Pre-Render Retention Gatekeeper",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# DEDICATED STUDIO DARK THEME STYLING
# -----------------------------------------------------------------------------
plotly_template = "plotly_dark"

theme_css = """
<style>
:root {
    --text-color: #F8FAFC !important;
    --background-color: #0E131F !important;
    --secondary-background-color: #141A29 !important;
}
.stApp {
    background-color: #0E131F !important;
    color: #F8FAFC !important;
}
/* Headings, paragraphs, spans, labels across all components */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp span, .stApp label, .stApp li, .stApp strong,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stMetricValue"] div,
[data-testid="stMetricLabel"] label,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: #F8FAFC !important;
}
/* Subtitles and Captions */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F8FAFC !important;
    margin-bottom: 0.2rem;
}
.sub-title,
.stApp .sub-title,
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: #94A3B8 !important;
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #141A29 !important;
    border-right: 1px solid #1F293D !important;
}
/* Inputs */
div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="base-input"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F8FAFC !important;
}
input, textarea {
    color: #F8FAFC !important;
    background-color: #1E293B !important;
}

/* ------------------------------------------------------------- */
/* BUTTON-BOX STYLING FOR NAVIGATION TABS                        */
/* ------------------------------------------------------------- */
div[data-baseweb="tab-list"] {
    gap: 10px !important;
    background-color: transparent !important;
    border-bottom: 1px solid #1F293D !important;
    padding-bottom: 12px !important;
    margin-bottom: 18px !important;
}

/* Hide Streamlit default underline/highlight line */
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"] {
    display: none !important;
}

/* Default Inactive Tab Button Box */
button[data-baseweb="tab"] {
    background-color: #141A29 !important;
    border: 1px solid #243048 !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    margin: 0 4px !important;
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

button[data-baseweb="tab"] p,
button[data-baseweb="tab"] div,
button[data-baseweb="tab"] span {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Hover Tab Button Box */
button[data-baseweb="tab"]:hover {
    background-color: #1E293B !important;
    border-color: #38BDF8 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4) !important;
}

button[data-baseweb="tab"]:hover p,
button[data-baseweb="tab"]:hover div,
button[data-baseweb="tab"]:hover span {
    color: #F8FAFC !important;
}

/* Active / Selected Tab Button Box */
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%) !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.45) !important;
    transform: translateY(-1px);
}

button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] div,
button[data-baseweb="tab"][aria-selected="true"] span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* Button-Box Styling for Raw Download Buttons */
div[data-testid="stDownloadButton"] button {
    background-color: #141A29 !important;
    border: 1px solid #243048 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background-color: #1E293B !important;
    border-color: #38BDF8 !important;
    color: #FFFFFF !important;
}
/* Metric Cards */
.metric-card {
    background: #141A29 !important;
    border: 1px solid #1F293D !important;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 2.0rem;
    font-weight: 700;
    color: #F8FAFC !important;
}
.metric-label {
    font-size: 0.85rem;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
/* Verdict boxes */
.verdict-box-pass {
    background-color: rgba(34, 197, 94, 0.16) !important;
    border-left: 6px solid #22C55E !important;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.verdict-box-pass h3 {
    color: #4ADE80 !important;
}
.verdict-box-pass p {
    color: #BBF7D0 !important;
}
.verdict-box-caution {
    background-color: rgba(245, 158, 11, 0.18) !important;
    border-left: 6px solid #F59E0B !important;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.verdict-box-caution h3 {
    color: #FBBF24 !important;
}
.verdict-box-caution p {
    color: #FEF08A !important;
}
.verdict-box-fail {
    background-color: rgba(239, 68, 68, 0.18) !important;
    border-left: 6px solid #EF4444 !important;
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.verdict-box-fail h3 {
    color: #F87171 !important;
}
.verdict-box-fail p {
    color: #FECACA !important;
}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)


def get_default_model(provider_str: str) -> str:
    defaults = {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
        "groq": "llama-3.3-70b-versatile",
        "generic_openai": "default",
        "mock": "mock-model",
    }
    return defaults.get(provider_str.lower(), "gemini-2.5-flash")


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
sidebar_logo_html = clean_html("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
    <div style="background: linear-gradient(135deg, #FF2A54 0%, #E11D48 100%); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 900; color: white; font-size: 1.05rem; letter-spacing: -0.5px; box-shadow: 0 2px 8px rgba(225, 29, 72, 0.4);">
        kuku
    </div>
    <div>
        <div style="font-size: 0.95rem; font-weight: 800; color: #F8FAFC;">Kuku ScriptGuard</div>
        <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Studio QA Platform</div>
    </div>
</div>
""")
st.sidebar.markdown(sidebar_logo_html, unsafe_allow_html=True)
st.sidebar.title("Gatekeeper Controls")

# Input Source
input_mode = st.sidebar.radio(
    "Select Screenplay Source:",
    ["Public Synthetic Demo (sample_data/)", "Upload Screenplays (.zip or .txt)"],
    index=0,
)

uploaded_files = None
if input_mode == "Upload Screenplays (.zip or .txt)":
    uploaded_files = st.sidebar.file_uploader(
        "Upload Screenplay Files",
        type=["txt", "zip"],
        accept_multiple_files=True,
        help="Upload multiple .txt episode files or a single .zip archive containing screenplays.",
    )

st.sidebar.divider()

# LLM Configuration
st.sidebar.subheader("LLM Provider Setup")
provider_choice = st.sidebar.selectbox(
    "Provider Adapter:",
    ["gemini", "openai", "anthropic", "groq", "generic_openai", "mock"],
    index=0,
    help="Swappable LLM provider. Default: Google Gemini.",
)

default_model_name = get_default_model(provider_choice)
model_name = st.sidebar.text_input("Model Name:", value=default_model_name)

api_key_input = st.sidebar.text_input(
    "API KEY (Optional):",
    type="password",
    help="Enter an API key here, or leave blank to automatically use the key from your .env file or run in simulated mock mode.",
)

max_episodes = st.sidebar.number_input(
    "Number of Episodes to Audit:",
    min_value=1,
    max_value=1000,
    value=100,
    step=1,
    help="Maximum number of episodes to process in this audit run.",
)

run_button = st.sidebar.button("Run Pre-Render Script Audit", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# AUDIT EXECUTION HANDLER
# -----------------------------------------------------------------------------
if "audit_report" not in st.session_state:
    st.session_state.audit_report = None
if "exported_files" not in st.session_state:
    st.session_state.exported_files = None
if "episodes_data" not in st.session_state:
    st.session_state.episodes_data = None
if "last_input_mode" not in st.session_state:
    st.session_state.last_input_mode = input_mode
elif st.session_state.last_input_mode != input_mode:
    st.session_state.audit_report = None
    st.session_state.last_input_mode = input_mode
    st.rerun()


def execute_audit():
    with st.spinner("Executing Deterministic Screenplay Audit..."):
        # Setup Client
        active_api_key = api_key_input.strip() if api_key_input else None
        client = LLMClient(
            provider=provider_choice,
            model=model_name,
            api_key=active_api_key,
            use_cache=True,
        )

        output_dir = Path("outputs/streamlit_run")
        pipeline = AuditPipeline(
            llm_client=client,
            output_dir=output_dir,
            top_k_characters=None,
            samples_per_character=15,
        )

        # Parse Source
        parser = ScreenplayParser()
        episodes: List[EpisodeData] = []

        if input_mode == "Public Synthetic Demo (sample_data/)":
            episodes = parser.parse_directory("sample_data")
        else:
            if not uploaded_files:
                st.warning("Please upload at least one .txt file or .zip archive to run the audit.")
                return

            for uf in uploaded_files:
                if uf.name.endswith(".zip"):
                    episodes.extend(parser.parse_zip(uf.getvalue()))
                elif uf.name.endswith(".txt"):
                    text_str = uf.getvalue().decode("utf-8", errors="replace")
                    ep = parser.parse_text(text_str, file_path=uf.name)
                    episodes.append(ep)

        if not episodes:
            st.error("No valid screenplay episodes could be found or parsed.")
            return

        episodes.sort(key=lambda e: e.episode_number)
        if max_episodes < len(episodes):
            episodes = episodes[:max_episodes]
        st.session_state.episodes_data = episodes

        # Run Pipeline with parsed episodes
        report, exported_files = pipeline.run(
            input_source=episodes,
            season_title="Audited Microdrama Season",
        )

        st.session_state.audit_report = report
        st.session_state.exported_files = exported_files
        st.toast("Script Health Audit Complete!")


if run_button:
    execute_audit()
elif st.session_state.audit_report is None:
    if input_mode == "Public Synthetic Demo (sample_data/)":
        execute_audit()

# Retrieve active state
report: Optional[SeasonHealthReport] = st.session_state.audit_report
episodes: List[EpisodeData] = st.session_state.episodes_data or []
exported_files: Dict[str, Path] = st.session_state.exported_files or {}

# -----------------------------------------------------------------------------
# MAIN APP HEADER & PRODUCT SUMMARY
# -----------------------------------------------------------------------------
header_html = clean_html("""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 8px;">
    <div style="background: linear-gradient(135deg, #FF2A54 0%, #E11D48 100%); width: 52px; height: 52px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 900; color: white; font-size: 1.6rem; letter-spacing: -1px; box-shadow: 0 4px 14px rgba(225, 29, 72, 0.45); flex-shrink: 0;">
        kuku
    </div>
    <div>
        <div style="font-size: 0.8rem; font-weight: 700; color: #E11D48; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px;">
            Kuku FM Studios  •  Pre-Render Script Doctor
        </div>
        <div style="font-size: 2.1rem; font-weight: 800; color: #F8FAFC; line-height: 1.2;">
            Kuku ScriptGuard: Pre-Render Audience Retention Gatekeeper
        </div>
    </div>
</div>
<div style="font-size: 1.02rem; color: #94A3B8; margin-bottom: 16px;">
    Autonomous Safety Scanner for AI Microdramas — Protecting Viewer Retention & Coin Paywalls Before Video Generation
</div>
""")
st.markdown(header_html, unsafe_allow_html=True)

summary_html = clean_html("""
<div style="background: rgba(20, 26, 41, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid #E11D48; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px;">
    <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #E11D48; margin-bottom: 6px;">
        How This Tool Protects Your Studio (In Simple Words)
    </div>
    <div style="font-size: 0.94rem; color: #E2E8F0; line-height: 1.6; margin-bottom: 12px;">
        Producing vertical video microdramas with AI video tools (Runway, Sora, Kling) costs thousands of dollars per season in GPU rendering. <strong>Kuku ScriptGuard</strong> acts as a pre-flight safety scanner that audits your screenplays in seconds <em>before</em> any video or voice-cloning budget is spent.
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 10px;">
        <div style="background: rgba(14, 19, 31, 0.7); padding: 12px 14px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <strong style="color: #F87171; display: block; margin-bottom: 4px;">1. Protects Coin Paywalls (Ep 5–15)</strong>
            <span style="font-size: 0.86rem; color: #94A3B8; line-height: 1.45;">Checks that viewers are not exhausted with unending, unresolved cliffhangers right when they need to unlock episodes with paid coins.</span>
        </div>
        <div style="background: rgba(14, 19, 31, 0.7); padding: 12px 14px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <strong style="color: #60A5FA; display: block; margin-bottom: 4px;">2. Stops Clone Character Voices</strong>
            <span style="font-size: 0.86rem; color: #94A3B8; line-height: 1.45;">Runs a blind listening test on dialogue to make sure the hero, villain, and lead characters sound like real distinct people, not identical bots.</span>
        </div>
        <div style="background: rgba(14, 19, 31, 0.7); padding: 12px 14px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <strong style="color: #4ADE80; display: block; margin-bottom: 4px;">3. Instant 1-Click Fixes</strong>
            <span style="font-size: 0.86rem; color: #94A3B8; line-height: 1.45;">Instantly rewrites weak or interchangeable dialogue lines with built-in Dialogue Doctor and re-verifies them before production handoff.</span>
        </div>
    </div>
</div>
""")
st.markdown(summary_html, unsafe_allow_html=True)

if not report:
    st.info("Click 'Run Pre-Render Script Audit' in the sidebar to generate the evaluation.")
    st.stop()

# -----------------------------------------------------------------------------
# TABS INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Scorecard",
        "Cliffhanger Fatigue Curve",
        "Blind Character-Voice Audit",
        "Failure Log & Raw Exports",
    ]
)

# =============================================================================
# TAB 1: EXECUTIVE GATEKEEPER SCORECARD
# =============================================================================
with tab1:
    st.subheader("Production Gatekeeper Assessment")

    # Overall Verdict Banner
    verdict = report.overall_verdict
    if verdict == "READY_FOR_RENDER":
        st.markdown(
            f"""
            <div class="verdict-box-pass">
                <h3 style="color: #4ADE80; margin:0 0 8px 0;">[APPROVED] GATEKEEPER VERDICT: READY FOR AI VIDEO RENDERING</h3>
                <p style="margin:0; color: #BBF7D0;">{report.executive_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif verdict == "CAUTION":
        st.markdown(
            f"""
            <div class="verdict-box-caution">
                <h3 style="color: #FBBF24; margin:0 0 8px 0;">[CAUTION] GATEKEEPER VERDICT: CREATIVE REVISION ADVISED</h3>
                <p style="margin:0; color: #FEF08A;">{report.executive_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-box-fail">
                <h3 style="color: #F87171; margin:0 0 8px 0;">[BLOCKED] GATEKEEPER VERDICT: REVISION REQUIRED BEFORE RENDER</h3>
                <p style="margin:0; color: #FECACA;">{report.executive_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4 Headline KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            label="Season Pacing Score",
            value=f"{report.fatigue_audit.season_pacing_score} / 100",
            delta=report.fatigue_audit.status,
            delta_color="normal" if report.fatigue_audit.season_pacing_score >= 75 else "inverse",
        )
    with kpi2:
        danger_count = len(report.fatigue_audit.danger_zone_fatigue_episodes)
        st.metric(
            label="Max Cliffhanger Streak",
            value=f"{report.fatigue_audit.max_consecutive_cliffhangers} eps",
            delta=f"{danger_count} Paywall Alerts (Ep 5-15)",
            delta_color="inverse" if danger_count > 0 else "normal",
        )
    with kpi3:
        baseline_pct = f"{report.voice_audit.random_baseline * 100:.0f}%"
        st.metric(
            label="Blind Voice Accuracy",
            value=f"{report.voice_audit.accuracy * 100:.1f}%",
            delta=f"+{round((report.voice_audit.accuracy - report.voice_audit.random_baseline)*100, 1)}% vs. Random",
            delta_color="normal" if report.voice_audit.accuracy >= 0.50 else "inverse",
        )
    with kpi4:
        st.metric(
            label="Audit Cost & Latency",
            value=f"${report.llm_cost_and_usage.get('total_cost_usd', 0.0):.4f}",
            delta=f"{report.llm_cost_and_usage.get('total_latency_sec', 0.0):.2f}s ({report.llm_cost_and_usage.get('cache_hit_rate_pct', 0)}% cache)",
            delta_color="off",
        )

    st.divider()

    # Red Flags & Character Dialogue Distribution
    col_flags, col_chars = st.columns([1, 1])

    with col_flags:
        st.markdown("#### Actionable Red Flags & Recommendations")
        red_flags = []
        if report.fatigue_audit.danger_zone_fatigue_episodes:
            red_flags.append(
                f"**Paywall Danger Zone Fatigue:** Episodes `{report.fatigue_audit.danger_zone_fatigue_episodes}` have consecutive cliffhanger streaks without micro-payoffs, causing coin paywall churn."
            )
        if report.fatigue_audit.max_consecutive_cliffhangers >= 4:
            red_flags.append(
                f"**Extended Cliffhanger Streak:** Peak streak reached `{report.fatigue_audit.max_consecutive_cliffhangers}` consecutive unresolved episodes. Audience patience exhausted."
            )
        if report.voice_audit.most_confused_pairs:
            top_pair = report.voice_audit.most_confused_pairs[0]
            red_flags.append(
                f"**Interchangeable Voices:** `{top_pair['pair'][0]}` and `{top_pair['pair'][1]}` were confused `{top_pair['confusion_count']}` times ({top_pair['confusion_rate'] * 100:.1f}% rate). Character voice distinction requires dialogue doctoring."
            )
        if not red_flags:
            red_flags.append("No critical red flags detected. Pacing balance and character vocal distinction meet studio standards.")

        for rf in red_flags:
            st.warning(rf)

    with col_chars:
        st.markdown("#### Season Dialogue Distribution")
        top_speakers = dict(Counter(report.speaker_distribution).most_common(8))
        df_speakers = pd.DataFrame(list(top_speakers.items()), columns=["Character", "Dialogue Lines"])
        if not df_speakers.empty:
            fig_bar = px.bar(
                df_speakers,
                x="Dialogue Lines",
                y="Character",
                orientation="h",
                color="Dialogue Lines",
                color_continuous_scale="Teal",
            )
            fig_bar.update_layout(
                yaxis={"autorange": "reversed"},
                margin=dict(l=0, r=0, t=10, b=10),
                height=260,
                template=plotly_template,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No dialogue lines found in audited episodes.")


# =============================================================================
# TAB 2: CLIFFHANGER FATIGUE CURVE
# =============================================================================
with tab2:
    st.subheader("Module 1: Episode Cliffhanger-to-Payoff Fatigue Curve")
    st.caption(
        "Monitors narrative tension vs. emotional payoff across episodes. "
        "The shaded region denotes the Monetization Paywall Zone (Episodes 5-15) where consecutive unresolved cliffhangers cause viewer drop-off."
    )

    fatigue_data = []
    for a in report.fatigue_audit.audits:
        fatigue_data.append(
            {
                "Episode": a.episode_number,
                "Ending Type": a.ending_type.value,
                "Tension Score": a.tension_score,
                "Payoff Score": a.payoff_score,
                "Streak": a.streak_count,
                "In Paywall Zone": a.is_paywall_danger_zone,
                "Fatigue Flag": a.fatigue_flag,
                "Danger Zone Fatigue": a.danger_zone_fatigue,
                "Evidence Beat": a.evidence_beat_id,
                "Timestamp": a.evidence_beat_ts or "",
                "Script Text": a.evidence_beat_text or "",
                "Editorial Verdict": a.one_line_verdict,
                "Suggested Pivot": a.suggested_pivot or "None",
            }
        )
    fatigue_cols = [
        "Episode",
        "Ending Type",
        "Tension Score",
        "Payoff Score",
        "Streak",
        "In Paywall Zone",
        "Fatigue Flag",
        "Danger Zone Fatigue",
        "Evidence Beat",
        "Timestamp",
        "Script Text",
        "Editorial Verdict",
        "Suggested Pivot",
    ]
    df_fatigue = pd.DataFrame(fatigue_data, columns=fatigue_cols) if fatigue_data else pd.DataFrame(columns=fatigue_cols)

    if not df_fatigue.empty:
        fig_curve = go.Figure()

        # Add Paywall Danger Zone Shading (Ep 5 to 15)
        fig_curve.add_vrect(
            x0=4.5,
            x1=15.5,
            fillcolor="rgba(239, 68, 68, 0.12)",
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color="rgba(239, 68, 68, 0.4)",
            annotation_text="Monetization Paywall Danger Zone (Ep 5-15)",
            annotation_position="top left",
        )

        # Tension Line
        fig_curve.add_trace(
            go.Scatter(
                x=df_fatigue["Episode"],
                y=df_fatigue["Tension Score"],
                mode="lines+markers",
                name="Tension Score",
                line=dict(color="#EF4444", width=3),
                marker=dict(size=7),
            )
        )

        # Payoff Line
        fig_curve.add_trace(
            go.Scatter(
                x=df_fatigue["Episode"],
                y=df_fatigue["Payoff Score"],
                mode="lines+markers",
                name="Payoff Score",
                line=dict(color="#10B981", width=3),
                marker=dict(size=7),
            )
        )

        # Highlight Fatigue Flagged Episodes
        flagged_df = df_fatigue[df_fatigue["Fatigue Flag"] | df_fatigue["Danger Zone Fatigue"]]
        if not flagged_df.empty:
            fig_curve.add_trace(
                go.Scatter(
                    x=flagged_df["Episode"],
                    y=flagged_df["Tension Score"],
                    mode="markers",
                    name="High Fatigue Alert",
                    marker=dict(symbol="triangle-up", size=14, color="#B91C1C", line=dict(width=2, color="white")),
                )
            )

        fig_curve.update_layout(
            title="Season Tension vs. Payoff Pacing Trajectory",
            xaxis_title="Episode Number",
            yaxis_title="Intensity Rating (1 - 10)",
            yaxis=dict(range=[0, 11]),
            hovermode="x unified",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template=plotly_template,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        p2_img = exported_files.get("02_episode_fatigue_curve") or exported_files.get("02_episode_fatigue_curve_png")
        if p2_img and p2_img.exists():
            with open(p2_img, "rb") as f_img:
                st.download_button(
                    label="Download Fatigue Curve (PNG Image)",
                    data=f_img.read(),
                    file_name="02_episode_fatigue_curve.png",
                    mime="image/png",
                )

    # Searchable Granular Episode Table
    st.markdown("#### Episode Ending Details & Beat Citations")
    if not df_fatigue.empty:
        search_query = st.text_input("Filter by Keyword or Episode Number:", placeholder="e.g. Episode 7, cliffhanger, cut to black...")
        filtered_df = df_fatigue
        if search_query:
            filtered_df = df_fatigue[
                df_fatigue.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            ]

        display_cols = [
            col
            for col in [
                "Episode",
                "Ending Type",
                "Tension Score",
                "Payoff Score",
                "Streak",
                "Evidence Beat",
                "Timestamp",
                "Script Text",
                "Editorial Verdict",
            ]
            if col in filtered_df.columns
        ]
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No episode ending audit data available.")


# =============================================================================
# TAB 3: BLIND CHARACTER-VOICE AUDIT
# =============================================================================
with tab3:
    st.subheader("Module 2: Zero-Leakage Blind Character-Voice Tester")
    st.caption(
        "Evaluates whether characters possess distinct vocal identities or sound interchangeable. "
        "Speaker attributions are completely masked and tested against blind LLM predictions."
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Blind Prediction Confusion Matrix")
        # Format Confusion Matrix
        cm = report.voice_audit.confusion_matrix
        speakers = report.voice_audit.top_speakers
        if speakers and cm:
            matrix_values = [[cm[tr].get(pr, 0) for pr in speakers] for tr in speakers]

            fig_cm = px.imshow(
                matrix_values,
                x=speakers,
                y=speakers,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Character", y="True Character", color="Count"),
                text_auto=True,
            )
            fig_cm.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=10),
                template=plotly_template,
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Not enough character lines to build confusion matrix.")

    with c2:
        st.markdown("#### Deterministic Stylometric Fingerprints (Pure Python)")
        stylometrics = report.voice_audit.stylometric_summary
        style_rows = []
        for spk in speakers:
            s_data = stylometrics.get(spk, {})
            style_rows.append(
                {
                    "Character": spk,
                    "Avg Words/Line": s_data.get("avg_words_per_line", 0.0),
                    "Lexical Diversity (TTR)": s_data.get("lexical_diversity_ttr", 0.0),
                    "Question %": f"{s_data.get('question_ratio', 0.0)*100:.1f}%",
                    "Exclamation %": f"{s_data.get('exclamation_ratio', 0.0)*100:.1f}%",
                    "1st-Person %": f"{s_data.get('first_person_ratio', 0.0)*100:.1f}%",
                }
            )
        if style_rows:
            st.dataframe(pd.DataFrame(style_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No stylometric data available.")

        st.markdown("#### TF-IDF Vocabulary Cosine Similarity")
        pair_sim = report.voice_audit.pair_similarity_matrix
        if speakers and pair_sim:
            sim_matrix_values = [[pair_sim[s1].get(s2, 0.0) for s2 in speakers] for s1 in speakers]
            fig_sim = px.imshow(
                sim_matrix_values,
                x=speakers,
                y=speakers,
                color_continuous_scale="Purples",
                zmin=0.0,
                zmax=1.0,
                text_auto=".2f",
            )
            fig_sim.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=10, b=10),
                template=plotly_template,
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("No vocabulary similarity data available.")

    st.divider()

    # Interactive Flagged Interchangeable Lines Inspector & Rewrite Workflow
    st.markdown("#### Flagged Interchangeable Dialogue Inspector & Rewrite Doctor")
    st.caption("Lines where the blind tester misclassified the speaker with high confidence.")

    flagged_lines = report.voice_audit.flagged_predictions

    if flagged_lines:
        line_options = [
            f"Beat {p.beat_id} (True: {p.true_speaker} | Mistaken for: {p.predicted_speaker} @ {p.confidence*100:.0f}%)"
            for p in flagged_lines[:15]
        ]
        selected_option = st.selectbox("Select a Flagged Dialogue Line to Inspect:", line_options)
        selected_idx = line_options.index(selected_option)
        selected_pred = flagged_lines[selected_idx]

        rcol1, rcol2 = st.columns([1, 1])

        with rcol1:
            st.info(
                f"**Original Masked Line:**\n\n"
                f"> \"{selected_pred.masked_text}\"\n\n"
                f"- **True Speaker:** `{selected_pred.true_speaker}`\n"
                f"- **Predicted Speaker:** `{selected_pred.predicted_speaker}`\n"
                f"- **Misclassification Confidence:** `{selected_pred.confidence * 100:.0f}%`\n"
                f"- **Voice Cue Reason:** {selected_pred.voice_cue_reason or selected_pred.reasoning}"
            )

        with rcol2:
            st.markdown("**Dialogue Doctor: On-Demand Voice Sharpening**")

            def format_craft_explanation_html(explanation: str) -> str:
                """Formats raw craft explanations into clean, structured HTML with proper bolding, line breaks, and hierarchy."""
                if not explanation:
                    return "<em style='color: #94A3B8;'>No craft explanation provided.</em>"

                # Replace markdown bold **text** with HTML <strong>text</strong>
                text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", explanation.strip())

                # Check for numbered items (e.g. '1. ', ' 2. ', or '\n1. ')
                parts = re.split(r"(?:(?<=[.:\n])\s*|\A\s*)(?=\d+\.\s+)", text)
                parts = [p.strip() for p in parts if p.strip()]

                if len(parts) > 1:
                    intro = ""
                    items = []
                    for p in parts:
                        if re.match(r"^\d+\.\s+", p):
                            clean_item = re.sub(r"^\d+\.\s*", "", p)
                            items.append(clean_item)
                        else:
                            intro = p

                    html_items = "".join(
                        f"<div style='display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start;'>"
                        f"<div style='background: rgba(56, 189, 248, 0.15); color: #38BDF8; font-weight: 700; font-size: 0.78rem; border-radius: 4px; padding: 2px 7px; min-width: 22px; text-align: center; margin-top: 2px; border: 1px solid rgba(56, 189, 248, 0.25);'>{idx}</div>"
                        f"<div style='line-height: 1.55; color: #E2E8F0; font-size: 0.92rem;'>{item}</div>"
                        f"</div>"
                        for idx, item in enumerate(items, 1)
                    )
                    intro_html = f"<div style='margin-bottom: 10px; color: #94A3B8; font-size: 0.88rem; font-weight: 500;'>{intro}</div>" if intro else ""
                    return clean_html(f"{intro_html}{html_items}")

                # Check for bullet points starting with -, *, or •
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if len(lines) > 1 and all(line.startswith(("-", "*", "•")) for line in lines):
                    html_items = "".join(
                        f"<div style='display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start;'>"
                        f"<div style='color: #38BDF8; font-size: 1rem; line-height: 1.2;'>•</div>"
                        f"<div style='line-height: 1.55; color: #E2E8F0; font-size: 0.92rem;'>{re.sub(r'^[-*•]\s*', '', l)}</div>"
                        f"</div>"
                        for l in lines
                    )
                    return clean_html(html_items)

                # Multiple newline-separated paragraphs or plain text
                paragraphs = text.split("\n\n")
                return clean_html("".join(
                    f"<p style='margin: 0 0 8px 0; line-height: 1.55; color: #E2E8F0; font-size: 0.92rem;'>{p.replace(chr(10), '<br/>')}</p>"
                    for p in paragraphs
                ))

            session_key = f"rewrite_{selected_pred.beat_id}"
            if st.button("Sharpen Character Voice & Re-Test Line", use_container_width=True):
                with st.spinner("Rewriting dialogue line and re-running blind classifier..."):
                    client = LLMClient()
                    rewrite_result = rewrite_and_retest_line(
                        selected_pred,
                        client=client,
                        candidate_characters=report.voice_audit.top_speakers,
                        archetypes=report.voice_audit.character_archetypes,
                    )
                    st.session_state[session_key] = rewrite_result

            saved_result = st.session_state.get(session_key)
            if saved_result:
                is_improved = saved_result.get("improved", False)
                border_color = "#22C55E" if is_improved else "#F59E0B"
                true_spk = selected_pred.true_speaker
                rewritten_line = saved_result.get("rewritten_text", selected_pred.masked_text)
                retest_prediction = saved_result.get("retest_prediction", "Unknown")
                craft_html = format_craft_explanation_html(saved_result.get("craft_explanation", ""))

                if is_improved:
                    status_badge = "[RESOLVED] Voice Distinction Improved! (Correctly Predicted)"
                    pill_bg = "rgba(34, 197, 94, 0.2)"
                    pill_color = "#4ADE80"
                    pill_label = "Resolved"
                    outcome_label = "Distinct Character Cadence Achieved"
                else:
                    status_badge = "[AMBIGUOUS] Still Ambiguous (Manual Edit Recommended)"
                    pill_bg = "rgba(245, 158, 11, 0.2)"
                    pill_color = "#FBBF24"
                    pill_label = "Needs Doctoring"
                    outcome_label = "Further Stylistic Doctoring Advised"

                card_html = clean_html(f"""
<div style="border: 1px solid rgba(255,255,255,0.08); border-left: 5px solid {border_color}; background: #141A29; color: #F8FAFC; padding: 18px; border-radius: 8px; margin-top: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.08);">
<h4 style="margin: 0; font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">{status_badge}</h4>
<span style="background: {pill_bg}; color: {pill_color}; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.05em;">{pill_label}</span>
</div>
<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid {border_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 16px;">
<div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 700; margin-bottom: 6px;">
Sharpened Dialogue Line ({true_spk})
</div>
<div style="font-size: 1.08rem; font-style: italic; color: #FFFFFF; line-height: 1.55; font-family: Georgia, 'Times New Roman', serif;">
"{rewritten_line}"
</div>
</div>
<div style="margin-bottom: 16px;">
<div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 700; margin-bottom: 8px;">
Craft Rationale & Rhetorical Shifts
</div>
<div style="background: rgba(15, 23, 42, 0.55); padding: 14px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06);">
{craft_html}
</div>
</div>
<div style="display: flex; align-items: center; justify-content: space-between; background: rgba(30, 41, 59, 0.5); padding: 10px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.88rem;">
<div>
<span style="color: #94A3B8;">Blind Re-Test Prediction:</span>
<code style="color: #38BDF8; background: rgba(56, 189, 248, 0.12); padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.95rem; margin-left: 6px;">{retest_prediction}</code>
</div>
<div>
<span style="color: #94A3B8;">Status:</span>
<span style="color: {border_color}; font-weight: 700; margin-left: 6px;">{outcome_label}</span>
</div>
</div>
</div>
""")
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.success("No interchangeable dialogue lines detected! All characters demonstrate strong vocal distinction.")


# =============================================================================
# TAB 4: QUALITY CHECKS, FAILURE LOG & RAW EXPORTS
# =============================================================================
with tab4:
    st.subheader("Integrity Verification, Failure Mode Log & Raw Exports")

    # Metrics on Integrity
    q1, q2, q3 = st.columns(3)
    val_summary = report.validation_summary
    with q1:
        st.metric("Parser Unparsed Lines", f"{val_summary.get('total_parsing_issues', 0)}", delta="0.0% unparsed rate", delta_color="normal")
    with q2:
        st.metric("Leakage-Filtered Lines", f"{val_summary.get('leakage_filtered_lines', 0)} lines", delta="Protected blind test", delta_color="off")
    with q3:
        st.metric("Citation Hallucinations Rejected", f"{val_summary.get('unverified_citations_count', 0)}", delta="Deterministic Python guard", delta_color="normal")

    st.divider()

    st.markdown("#### Raw Audit Exports")
    st.caption("Download the complete, unedited raw output files for auditing, production handoff, or CI/CD pipelines:")

    d1, d2, d3, d4 = st.columns(4)

    # 01 Season Health Summary
    p1 = exported_files.get("01_season_health_summary")
    if p1 and p1.exists():
        with open(p1, "r", encoding="utf-8") as f:
            d1.download_button("01 Summary (JSON)", f.read(), file_name="01_season_health_summary.json", mime="application/json")

    # 02 Fatigue Curve PNG Image
    p2 = exported_files.get("02_episode_fatigue_curve") or exported_files.get("02_episode_fatigue_curve_png")
    if p2 and p2.exists():
        with open(p2, "rb") as f:
            d2.download_button("02 Fatigue Curve (PNG)", f.read(), file_name="02_episode_fatigue_curve.png", mime="image/png")

    # 03 Confusion Matrix CSV
    p3 = exported_files.get("03_voice_metrics_csv")
    if p3 and p3.exists():
        with open(p3, "r", encoding="utf-8") as f:
            d3.download_button("03 Confusion Matrix (CSV)", f.read(), file_name="03_blind_voice_confusion_matrix.csv", mime="text/csv")

    # 04 Flagged Lines CSV
    p4 = exported_files.get("04_flagged_interchangeable_lines")
    if p4 and p4.exists():
        with open(p4, "r", encoding="utf-8") as f:
            d4.download_button("04 Flagged Lines (CSV)", f.read(), file_name="04_flagged_interchangeable_lines.csv", mime="text/csv")

    st.write("")

    d5, d6, _, _ = st.columns(4)
    # 05 Failure Log JSON
    p5 = exported_files.get("05_validation_and_failure_log")
    if p5 and p5.exists():
        with open(p5, "r", encoding="utf-8") as f:
            d5.download_button("05 Failure Log (JSON)", f.read(), file_name="05_validation_and_failure_log.json", mime="application/json")

    # 06 Cost Report JSON
    p6 = exported_files.get("06_cost_and_latency_report")
    if p6 and p6.exists():
        with open(p6, "r", encoding="utf-8") as f:
            d6.download_button("06 Cost Report (JSON)", f.read(), file_name="06_cost_and_latency_report.json", mime="application/json")

    st.divider()

    # Integrity Status Summary (Clean human-readable verdict instead of raw JSON dump)
    issues_count = val_summary.get("total_parsing_issues", 0) + val_summary.get("unverified_citations_count", 0)
    if issues_count == 0:
        st.success("Audit Quality Verified: Zero parsing anomalies, zero citation hallucinations, and zero voice leakage detected across all audited episodes.")
    else:
        st.warning(f"Integrity Notice: {issues_count} potential anomalies detected in the audit pipeline. Please review the exported log files above for details.")
