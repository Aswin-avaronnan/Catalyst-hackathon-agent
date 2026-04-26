import streamlit as st
import requests
import json

st.set_page_config(
    page_title="AI Talent Scout",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .candidate-card {
        background: #1a1a2e;
        border-left: 4px solid #6c63ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .rank-badge {
        background: #6c63ff;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .score-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
    .score-high { background: #1a4731; color: #4ade80; }
    .score-mid  { background: #3b2f00; color: #fbbf24; }
    .score-low  { background: #3b1515; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI Talent Scout")
    st.markdown("**Deccan AI Hackathon 2026**")
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
1. 📋 **Parse** — Extract skills, level & location from JD  
2. 🔍 **Discover** — Load 50 mock candidates  
3. 🧮 **Match** — Score by skills, experience, location & culture  
4. 💬 **Engage** — Simulate conversation with top 20  
5. 🏆 **Rank** — Combined score (60% match + 40% interest)  
    """)
    st.divider()
    api_url = st.text_input("API URL", value="http://localhost:8000")
    st.caption("Make sure FastAPI is running on port 8000")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 AI Talent Scouting Agent")
st.markdown("Paste a job description below and let the AI find, score, and engage the best candidates.")
st.divider()

# ── JD Input ─────────────────────────────────────────────────────────────────
DEFAULT_JD = """Senior Python Engineer

We're looking for a senior backend engineer with 5+ years Python experience to join our fast-paced startup.

Must have:
- Python, Django or FastAPI
- PostgreSQL, Redis
- Docker, AWS

Nice to have:
- Kubernetes
- React or GraphQL

Location: Remote (US timezone preferred)
Salary: $140k-$180k
Culture: Collaborative, startup, ownership-driven"""

col_input, col_info = st.columns([3, 1])

with col_input:
    jd_text = st.text_area(
        "Job Description",
        value=DEFAULT_JD,
        height=280,
        help="Paste any job description — structured or free-form",
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("#### 💡 Tips")
    st.markdown("""
- Include required skills  
- Mention experience level  
- Add location/remote info  
- Include salary if known  
    """)
    st.markdown("#### ⚡ Scoring weights")
    st.markdown("""
| Dimension | Max pts |
|---|---|
| Skills | 40 |
| Experience | 25 |
| Location | 15 |
| Culture | 20 |
    """)

# ── Scout Button ──────────────────────────────────────────────────────────────
scout_clicked = st.button("🚀 Scout Talent", type="primary", use_container_width=True)

if scout_clicked:
    if not jd_text.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("🔍 Parsing JD → Matching candidates → Simulating engagement …"):
            try:
                resp = requests.post(
                    f"{api_url}/scout",
                    json={"job_description": jd_text},
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()

                    # ── Summary metrics ───────────────────────────────────────
                    st.divider()
                    st.markdown("## 📊 Results")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("🏷️ Job Title", data["job_title"])
                    m2.metric("📂 Reviewed", data["total_candidates_reviewed"])
                    m3.metric("💬 Engaged", data["candidates_engaged"])
                    m4.metric("🏆 Shortlisted", len(data["shortlist"]))
                    m5.metric("🎯 Experience", data.get("experience_level", "—").title())

                    # ── Required skills pills ─────────────────────────────────
                    if data.get("required_skills"):
                        st.markdown("**Required skills detected:** " + " ".join(
                            [f"`{s}`" for s in data["required_skills"]]
                        ))

                    st.divider()
                    st.markdown("## 🏆 Top 10 Ranked Candidates")

                    for cand in data["shortlist"]:
                        rank = cand["rank"]
                        cs = cand["combined_score"]

                        # Score colour
                        if cs >= 75:
                            badge_cls = "score-high"
                        elif cs >= 50:
                            badge_cls = "score-mid"
                        else:
                            badge_cls = "score-low"

                        label = (
                            f"#{rank}  {cand['name']}  •  {cand['title']}  "
                            f"@{cand['current_company']}  •  Combined: {cs:.0f}/100"
                        )
                        with st.expander(label, expanded=(rank <= 3)):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Match Score", f"{cand['match_score']:.0f}/100")
                            c2.metric("Interest Score", f"{cand['interest_score']:.0f}/100")
                            c3.metric("Combined Score", f"{cs:.0f}/100",
                                      help="60% match + 40% interest")

                            st.markdown(f"📍 **Location:** {cand['location']}")
                            st.markdown(f"🧠 **Skills match:** {cand['skills_reason']}")
                            st.markdown(f"📅 **Experience:** {cand['experience_reason']}")

                            col_s, col_g = st.columns(2)
                            with col_s:
                                if cand["strengths"]:
                                    st.success("**✅ Strengths:** " + ", ".join(cand["strengths"]))
                            with col_g:
                                if cand["gaps"]:
                                    st.warning("**⚠️ Gaps:** " + ", ".join(cand["gaps"]))

                            st.info(
                                f"💬 **Engagement ({cand['conversation_turns']} turns):** "
                                f"{cand['conversation_summary']}"
                            )

                    # ── Download ──────────────────────────────────────────────
                    st.divider()
                    safe_title = data["job_title"].replace(" ", "_")
                    st.download_button(
                        label="📥 Download Full Results (JSON)",
                        data=json.dumps(data, indent=2),
                        file_name=f"talent_scout_{safe_title}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                elif resp.status_code == 422:
                    st.error(f"Validation error: {resp.json().get('detail', resp.text)}")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API.\n\n"
                    "Make sure FastAPI is running:\n```\npython api/server.py\n```"
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The LLM call may be slow — try again.")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🤖 AI Talent Scout • Deccan AI Hackathon 2026 • Powered by Gemini 2.5 Flash")
