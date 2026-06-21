"""
app.py — Streamlit frontend for TalentFilter Resume Screener.
Supports Anthropic Claude, OpenAI GPT, and Google Gemini (FREE).

Run with:  streamlit run app.py
"""

import json
import os
import sys
import time
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extractor import ResumeExtractor
from src.scorer import ScoringWeights, ScoreResult
from src.screener import ResumeScreener
from src.utils import (
    setup_logging, save_results_csv, save_results_json,
    format_duration, recommendation_color, estimate_tokens,
)

# ─────────────────────────── Page config ────────────────────────────
st.set_page_config(
    page_title="TalentFilter by Kaushik P",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-card: #12121a;
        --bg-card-hover: #1a1a28;
        --bg-surface: #16161f;
        --border: #1e1e2e;
        --border-hover: #2d2d44;
        --text-primary: #e8e8ed;
        --text-secondary: #8b8b9e;
        --text-muted: #5a5a72;
        --accent: #6c63ff;
        --accent-soft: rgba(108, 99, 255, 0.12);
        --green: #34d399;
        --green-soft: rgba(52, 211, 153, 0.12);
        --blue: #60a5fa;
        --blue-soft: rgba(96, 165, 250, 0.12);
        --amber: #fbbf24;
        --amber-soft: rgba(251, 191, 36, 0.12);
        --red: #f87171;
        --red-soft: rgba(248, 113, 113, 0.12);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2d2d44; border-radius: 3px; }

    /* ── Hide default Streamlit branding ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Main background ── */
    .stApp { background: var(--bg-primary); }
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }

    /* ── Hero Header ── */
    .hero {
        position: relative;
        padding: 2.5rem 2.8rem 2rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--green), var(--blue));
        border-radius: 16px 16px 0 0;
    }
    .hero-icon {
        width: 44px; height: 44px;
        background: var(--accent-soft);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0; font-size: 1.75rem; font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.4rem 0 0; font-size: 0.92rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    .hero-tag {
        display: inline-block;
        margin-top: 1rem;
        padding: 4px 14px;
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Cards ── */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    /* ── Candidate card ── */
    .candidate-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1rem;
    }
    .candidate-rank {
        width: 36px; height: 36px;
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .candidate-name {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .candidate-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green { background: var(--green-soft); color: var(--green); }
    .badge-blue  { background: var(--blue-soft);  color: var(--blue); }
    .badge-amber { background: var(--amber-soft); color: var(--amber); }
    .badge-red   { background: var(--red-soft);    color: var(--red); }

    /* ── Score pill ── */
    .score-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 100px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* ── Skill chips ── */
    .skill-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px 3px;
    }
    .skill-match { background: var(--green-soft); color: var(--green); border: 1px solid rgba(52,211,153,0.2); }
    .skill-miss  { background: var(--red-soft);   color: var(--red);   border: 1px solid rgba(248,113,113,0.2); }

    /* ── Score bar ── */
    .score-bar-container {
        background: var(--bg-surface);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin-top: 6px;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    /* ── Dimension label row ── */
    .dim-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
    }
    .dim-label {
        font-size: 0.82rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .dim-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        font-weight: 600;
        margin: 1.2rem 0 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Quote block ── */
    .quote-block {
        background: var(--bg-surface);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin: 0.5rem 0;
    }

    /* ── Sidebar badges ── */
    .provider-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        background: var(--green-soft);
        color: var(--green);
        margin-bottom: 10px;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
    .app-footer p {
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .app-footer a {
        color: var(--accent);
        text-decoration: none;
    }
    .app-footer a:hover { text-decoration: underline; }

    /* ── Streamlit overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    div[data-testid="stExpander"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    div[data-testid="stExpander"]:hover {
        border-color: var(--border-hover);
    }
    div[data-testid="stExpander"] summary span {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    .stTextArea textarea, .stTextInput input {
        background: var(--bg-surface) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    .stButton>button[kind="primary"] {
        background: var(--accent) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button[kind="primary"]:hover {
        background: #5a52e0 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.25) !important;
    }
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        border-color: var(--border) !important;
        color: var(--text-secondary) !important;
        background: var(--bg-card) !important;
    }

    .stDownloadButton>button {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton>button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="stDataFrame"] > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    .stProgress > div > div { background-color: var(--bg-surface) !important; border-radius: 8px; }
    .stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--green)) !important; border-radius: 8px; }

    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetricLabel"] label {
        color: var(--text-muted) !important;
    }

    hr { border-color: var(--border) !important; }

    .stSlider label { color: var(--text-secondary) !important; }
    
    .stFileUploader {
        background: var(--bg-surface);
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Session state ──────────────────────────
if "session_results" not in st.session_state:
    st.session_state.session_results = None
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────── Header ─────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🎯</div>
  <h1>TalentFilter</h1>
  <p>Screen resumes with AI precision — upload candidates, define the role, get ranked scores in seconds.</p>
  <span class="hero-tag">AI-Powered Hiring Intelligence</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════ SIDEBAR ════════════════════════════════
with st.sidebar:
    st.markdown("### Configuration")

    # -- Provider selection --
    provider = st.selectbox(
        "AI Provider",
        ["Google Gemini (FREE ✅)", "Anthropic (Claude)", "OpenAI (GPT)"],
        index=0,
    )

    if "Gemini" in provider:
        prov_key = "gemini"
        st.markdown('<div class="provider-badge">✦ Free Tier — No billing</div>', unsafe_allow_html=True)
        st.caption("Get your free key at [aistudio.google.com](https://aistudio.google.com) → Get API Key")
    elif "Anthropic" in provider:
        prov_key = "anthropic"
        st.caption("Get key at [console.anthropic.com](https://console.anthropic.com)")
    else:
        prov_key = "openai"
        st.caption("Get key at [platform.openai.com](https://platform.openai.com)")

    _secrets_map = {
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
    }
    _secret_name = _secrets_map.get(prov_key, "")
    try:
        api_key = st.secrets.get(_secret_name, "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.environ.get(_secret_name, "")

    if not api_key:
        st.warning("API key not found")
    else:
        st.success("API key loaded")

    model_override = st.text_input(
        "Model (optional override)",
        placeholder="e.g. gemini-2.0-flash",
        help="Leave blank to use the default model.",
    )

    st.divider()

    # -- Scoring weights --
    st.markdown("### Scoring Weights")
    st.caption("Tune each dimension's contribution to the overall score.")
    w_skills  = st.slider("Skills Match",  0, 100, 40)
    w_exp     = st.slider("Experience",    0, 100, 30)
    w_edu     = st.slider("Education",     0, 100, 15)
    w_culture = st.slider("Cultural Fit",  0, 100, 15)

    total_w = w_skills + w_exp + w_edu + w_culture
    if total_w == 0:
        st.error("At least one weight must be > 0.")
    else:
        st.caption(
            f"Skills {w_skills/total_w:.0%} · "
            f"Exp {w_exp/total_w:.0%} · "
            f"Edu {w_edu/total_w:.0%} · "
            f"Fit {w_culture/total_w:.0%}"
        )

    st.divider()
    st.markdown("### Quick Start")
    if st.button("Load sample data", use_container_width=True):
        st.session_state.load_samples = True


# ═══════════════════════════ MAIN TABS ══════════════════════════════
tab_screen, tab_results, tab_history = st.tabs(
    ["Screen", "Results", "History"]
)

# ──────────────────── TAB 1: Screen Resumes ─────────────────────────
with tab_screen:
    col_jd, col_up = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("##### Job Description")
        sample_jd = ""
        if st.session_state.get("load_samples"):
            sample_jd = """Senior Python Developer

We are looking for an experienced Python developer to join our platform team.

Requirements:
- 5+ years of professional Python development
- Strong knowledge of FastAPI or Django REST framework
- Experience with PostgreSQL and Redis
- Proficiency in Docker and Kubernetes
- Familiarity with AWS (EC2, RDS, S3, Lambda)
- Experience with CI/CD pipelines (GitHub Actions, Jenkins)
- Understanding of microservices architecture
- Strong problem-solving and communication skills

Nice to have:
- Open-source contributions
- Experience with ML/AI frameworks (PyTorch, TensorFlow)

Education: Bachelor's or Master's degree in Computer Science or equivalent."""

        job_description = st.text_area(
            "Paste the full job description",
            value=sample_jd,
            height=350,
            placeholder="Paste or type the full job description…",
            label_visibility="collapsed",
        )
        if job_description:
            st.caption(f"~{estimate_tokens(job_description)} tokens")

    with col_up:
        st.markdown("##### Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready")
            for f in uploaded_files:
                st.caption(f"› {f.name} ({len(f.getvalue())/1024:.1f} KB)")

        if st.session_state.get("load_samples"):
            st.info("Sample resumes loaded — click **Start Screening** to test.")
            st.session_state.load_samples = False

    st.divider()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_btn = st.button(
            "Start Screening →",
            use_container_width=True,
            type="primary",
            disabled=not (job_description),
        )
    with col_info:
        if not job_description:
            st.warning("Add a job description to continue.")

    # ── Run screening ────────────────────────────────────────────────
    if start_btn:
        if not job_description.strip():
            st.error("Job description is required.")
            st.stop()

        texts: dict = {}
        extractor = ResumeExtractor()

        if uploaded_files:
            for uf in uploaded_files:
                try:
                    tmp = Path(tempfile.gettempdir()) / uf.name
                    tmp.write_bytes(uf.getvalue())
                    texts[uf.name] = extractor.extract(tmp)
                    tmp.unlink(missing_ok=True)
                except Exception as e:
                    st.warning(f"Could not extract '{uf.name}': {e}")

        if not texts:
            texts = {
                "alice_chen_senior.txt": """Alice Chen | alice@email.com
Senior Software Engineer — 7 years experience

EXPERIENCE
Lead Python Developer @ TechCorp (2020-Present)
- Architected microservices platform using FastAPI + Kubernetes, 2M req/day
- Led team of 6, cut deployment time 60% via GitHub Actions CI/CD
- Tech: Python, FastAPI, PostgreSQL, Redis, Docker, K8s, AWS

Senior Python Developer @ DataFlow Inc (2018-2020)
- Real-time ETL pipelines with Apache Kafka + Python
- ML model serving with TensorFlow + FastAPI

SKILLS: Python, FastAPI, Django REST, PostgreSQL, Redis, Docker, Kubernetes,
AWS, GitHub Actions, Jenkins, TensorFlow, PyTorch

EDUCATION: M.Sc. Computer Science — Stanford University (2017)
OPEN SOURCE: Contributor to FastAPI, 3 published PyPI packages""",

                "bob_martinez_mid.txt": """Bob Martinez | bob.m@gmail.com
Python Developer — 3 years

EXPERIENCE
Python Developer @ Webagency (2022-Present)
- REST APIs with Django REST Framework
- PostgreSQL database design
- Basic Docker, AWS S3

SKILLS: Python, Django REST, PostgreSQL, Docker (basic), Git, AWS S3

EDUCATION: B.Sc. Information Technology — State University (2021)
Note: No Kubernetes or CI/CD pipeline experience.""",

                "carol_johnson_junior.txt": """Carol Johnson | carol@mail.com
Junior Developer — 1 year experience

EXPERIENCE
Junior Developer @ SmallBiz (2023-Present)
- Maintained legacy PHP website
- Basic Python scripts for data cleanup

SKILLS: Python (beginner), PHP, HTML, CSS, Git, MySQL

EDUCATION: B.A. Business Administration — Community College (2023)
Certifications: Python Basics (Coursera 2023)""",
            }

        weights = ScoringWeights(
            skills=w_skills / total_w,
            experience=w_exp / total_w,
            education=w_edu / total_w,
            cultural_fit=w_culture / total_w,
        )

        screener = ResumeScreener(
            api_key=api_key if api_key else "mock_key",
            provider=prov_key,
            model=model_override or None,
            weights=weights,
        )

        progress_bar = st.progress(0, text="Initialising…")
        status_text  = st.empty()
        start_time   = time.time()

        def update_progress(current, total, filename):
            pct = current / total if total else 0
            progress_bar.progress(pct, text=f"Processing {current}/{total}: {filename}")
            status_text.caption(f"⏱ Elapsed: {format_duration(time.time() - start_time)}")

        with st.spinner("Scoring resumes with AI…"):
            session = screener.screen_texts(
                texts=texts,
                job_description=job_description,
                progress_callback=update_progress,
            )

        progress_bar.progress(1.0, text="Screening complete")
        status_text.empty()

        st.session_state.session_results = session
        st.session_state.history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "provider": prov_key,
            "resumes": len(session.results),
            "elapsed_sec": round(session.elapsed_seconds, 1),
            "tokens_used": session.total_tokens,
        })

        st.success(
            f"Screened {len(session.results)} resume(s) in "
            f"{format_duration(session.elapsed_seconds)} · "
            f"{session.total_tokens:,} tokens used"
        )
        st.info("Switch to the **Results** tab to see rankings.")


# ──────────────────── TAB 2: Results ────────────────────────────────
with tab_results:
    session = st.session_state.session_results

    if session is None:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📭</div>
            <p style="color:var(--text-muted); font-size:0.95rem;">No results yet. Run a screening session to see rankings here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        ranked = session.ranked

        # Summary stat cards
        counts = {"Strong Yes": 0, "Yes": 0, "Maybe": 0, "No": 0}
        for r in ranked:
            counts[r.recommendation] = counts.get(r.recommendation, 0) + 1

        badge_map = {
            "Strong Yes": ("var(--green)", "badge-green"),
            "Yes": ("var(--blue)", "badge-blue"),
            "Maybe": ("var(--amber)", "badge-amber"),
            "No": ("var(--red)", "badge-red"),
        }

        mc = st.columns(4)
        for col, (label, count) in zip(mc, counts.items()):
            color, _ = badge_map[label]
            with col:
                st.markdown(f"""
                <div class="stat-card">
                  <div class="stat-value" style="color:{color}">{count}</div>
                  <div class="stat-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score bar chart
        if ranked:
            colour_map = {
                "Strong Yes": "#34d399", "Yes": "#60a5fa",
                "Maybe": "#fbbf24", "No": "#f87171",
            }
            fig = go.Figure(go.Bar(
                x=[r.overall_score for r in ranked],
                y=[r.filename[:28] for r in ranked],
                orientation="h",
                marker_color=[colour_map.get(r.recommendation, "#5a5a72") for r in ranked],
                marker_line_width=0,
                text=[f"{r.overall_score:.1f}" for r in ranked],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12, color="#e8e8ed"),
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 115],
                           title=dict(text="Score (0–100)", font=dict(color="#5a5a72", size=11)),
                           gridcolor="#1e1e2e", zerolinecolor="#1e1e2e",
                           tickfont=dict(color="#5a5a72")),
                yaxis=dict(autorange="reversed",
                           tickfont=dict(color="#8b8b9e", size=11)),
                height=max(220, len(ranked) * 55 + 60),
                margin=dict(l=0, r=60, t=20, b=40),
                plot_bgcolor="#0a0a0f",
                paper_bgcolor="#0a0a0f",
            )
            fig.add_vline(x=80, line_dash="dot", line_color="rgba(52,211,153,0.3)",
                         annotation_text="Strong Yes", annotation_font_color="#34d399",
                         annotation_font_size=10)
            fig.add_vline(x=65, line_dash="dot", line_color="rgba(96,165,250,0.3)",
                         annotation_text="Yes", annotation_font_color="#60a5fa",
                         annotation_font_size=10)
            fig.add_vline(x=45, line_dash="dot", line_color="rgba(251,191,36,0.3)",
                         annotation_text="Maybe", annotation_font_color="#fbbf24",
                         annotation_font_size=10)
            st.plotly_chart(fig, use_container_width=True)

        # Ranked candidates table
        st.markdown('<div class="section-title">Ranked Candidates</div>', unsafe_allow_html=True)
        rows = []
        for rank, r in enumerate(ranked, 1):
            rows.append({
                "Rank": rank,
                "Filename": r.filename,
                "Score": r.overall_score,
                "Recommendation": r.recommendation,
                "Experience (yrs)": r.years_of_experience,
                "Education": r.education_level,
                "Matched Skills": len(r.matched_skills),
                "Missing Skills": len(r.missing_skills),
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=100, format="%.1f"
                    ),
                },
            )

        # Candidate detail expanders
        st.markdown('<div class="section-title">Candidate Details</div>', unsafe_allow_html=True)

        for rank, r in enumerate(ranked, 1):
            rec_badge = badge_map.get(r.recommendation, ("var(--text-muted)", "badge-red"))
            badge_class = rec_badge[1]

            with st.expander(f"#{rank} · {r.filename}  —  {r.overall_score:.1f}  · {r.recommendation}"):
                if r.error:
                    st.error(f"Error: {r.error}")
                    continue

                # Header with rank badge
                st.markdown(f"""
                <div class="candidate-header">
                    <div class="candidate-rank">#{rank}</div>
                    <div>
                        <div class="candidate-name">{r.filename}</div>
                        <div class="candidate-meta">{r.years_of_experience:.0f} yrs · {r.education_level or 'N/A'} · <span class="badge {badge_class}">{r.recommendation}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                dcol1, dcol2 = st.columns(2)

                with dcol1:
                    # Dimension scores as custom bars
                    st.markdown('<div class="section-title">Dimension Scores</div>', unsafe_allow_html=True)
                    dims = [
                        ("Skills", r.skills_score, "var(--accent)"),
                        ("Experience", r.experience_score, "var(--green)"),
                        ("Education", r.education_score, "var(--blue)"),
                        ("Cultural Fit", r.cultural_fit_score, "var(--amber)"),
                    ]
                    for dim_name, dim_score, dim_color in dims:
                        st.markdown(f"""
                        <div class="dim-row">
                            <span class="dim-label">{dim_name}</span>
                            <span class="dim-value">{dim_score:.0f}</span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar-fill" style="width:{dim_score}%; background:{dim_color};"></div>
                        </div>
                        """, unsafe_allow_html=True)

                with dcol2:
                    # Skills chips
                    if r.matched_skills:
                        st.markdown('<div class="section-title">Matched Skills</div>', unsafe_allow_html=True)
                        chips = "".join(f'<span class="skill-chip skill-match">{s}</span>' for s in r.matched_skills[:8])
                        st.markdown(chips, unsafe_allow_html=True)

                    if r.missing_skills:
                        st.markdown('<div class="section-title">Missing Skills</div>', unsafe_allow_html=True)
                        chips = "".join(f'<span class="skill-chip skill-miss">{s}</span>' for s in r.missing_skills[:8])
                        st.markdown(chips, unsafe_allow_html=True)

                # Summary & Justification
                st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="quote-block">{r.summary}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">Justification</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="quote-block">{r.justification}</div>', unsafe_allow_html=True)

                # Radar chart
                cats = ["Skills", "Experience", "Education", "Cultural Fit"]
                vals = [r.skills_score, r.experience_score, r.education_score, r.cultural_fit_score]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor="rgba(108, 99, 255, 0.15)",
                    line_color="#6c63ff",
                    line_width=2,
                ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 100], showticklabels=False,
                            gridcolor="#1e1e2e", linecolor="#1e1e2e",
                        ),
                        angularaxis=dict(
                            gridcolor="#1e1e2e", linecolor="#1e1e2e",
                            tickfont=dict(color="#8b8b9e", size=11),
                        ),
                        bgcolor="#0a0a0f",
                    ),
                    height=260,
                    margin=dict(l=50, r=50, t=25, b=25),
                    paper_bgcolor="#12121a",
                )
                st.plotly_chart(fig_r, use_container_width=True)

        # Export buttons
        st.divider()
        if ranked:
            from io import StringIO
            import csv as csv_mod

            buf = StringIO()
            fieldnames = [
                "rank", "filename", "overall_score", "recommendation",
                "years_of_experience", "education_level",
                "skills_score", "experience_score", "education_score",
                "cultural_fit_score", "matched_skills", "missing_skills",
                "summary", "justification",
            ]
            writer = csv_mod.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rank, r in enumerate(ranked, 1):
                row = r.to_dict()
                row["rank"] = rank
                row["matched_skills"] = ", ".join(row.get("matched_skills") or [])
                row["missing_skills"]  = ", ".join(row.get("missing_skills")  or [])
                writer.writerow(row)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download CSV", data=buf.getvalue(),
                    file_name="screening_results.csv", mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download JSON",
                    data=json.dumps({"results": [r.to_dict() for r in ranked]}, indent=2),
                    file_name="screening_results.json", mime="application/json",
                    use_container_width=True,
                )


# ──────────────────── TAB 3: History ────────────────────────────────
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🕐</div>
            <p style="color:var(--text-muted); font-size:0.95rem;">No sessions yet — run a screening to see history here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">Screening History</div>', unsafe_allow_html=True)
        df_h = pd.DataFrame(st.session_state.history)
        df_h.index = range(1, len(df_h) + 1)
        st.dataframe(df_h, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ──────────────────── Footer ─────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>TalentFilter · Built by <a href="https://github.com/duo311" target="_blank">Kaushik P</a></p>
</div>
""", unsafe_allow_html=True)
