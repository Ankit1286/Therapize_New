"""
Streamlit frontend for Therapize.

UI design:
1. Free text search box (primary input)
2. Optional questionnaire (collapsible sidebar)
3. Results displayed as cards with score breakdown
4. Feedback buttons on each result
5. Session continuity: query history in sidebar
"""
import json
import uuid
from datetime import datetime

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Therapize — Find Your Therapist",
    page_icon="🧠",
    layout="wide",
)

# ── Session state ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ── Sidebar — questionnaire ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Refine Your Search")
    st.caption("All fields are optional — the more you fill in, the better the match.")

    city = st.text_input("City", placeholder="e.g. San Francisco")
    insurance_options = [
        "", "blue_shield", "blue_cross", "aetna", "united_healthcare",
        "cigna", "kaiser", "magellan", "sliding_scale", "self_pay"
    ]
    insurance = st.selectbox("Insurance", insurance_options)
    max_budget = st.slider("Max budget per session ($)", 0, 500, 0, step=10)
    session_format = st.radio(
        "Session format",
        ["any", "in_person", "telehealth"],
        horizontal=True
    )

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

# ── Main area ────────────────────────────────────────────────────────────────
st.title("Find Your Therapist in California")
st.caption(
    "Describe what you're going through in your own words. "
    "Our AI will identify the best-fit therapists based on your needs."
)

query = st.text_area(
    "What brings you here?",
    placeholder=(
        "e.g. I've been having panic attacks at work and trouble sleeping. "
        "I've been going through a difficult divorce and need someone who "
        "understands anxiety and major life transitions."
    ),
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col2:
    if st.session_state.search_history:
        st.caption(f"Past searches: {len(st.session_state.search_history)}")

# ── Search execution ─────────────────────────────────────────────────────────
if search_clicked and (query or city):
    questionnaire = {}
    if city:
        questionnaire["city"] = city
    if insurance:
        questionnaire["insurance"] = insurance
    if max_budget > 0:
        questionnaire["max_budget_per_session"] = max_budget
    if session_format != "any":
        questionnaire["session_format"] = session_format

    payload = {
        "free_text": query,
        "questionnaire": questionnaire,
        "session_id": st.session_state.session_id,
    }

    with st.spinner("Finding your best matches..."):
        try:
            response = httpx.post(
                f"{API_BASE}/search",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Save to history
            st.session_state.search_history.append({
                "query": query[:50] + "..." if len(query) > 50 else query,
                "timestamp": datetime.now().strftime("%H:%M"),
                "num_results": len(data["results"]),
            })

            # ── Results ───────────────────────────────────────────────────
            st.divider()

            # Metadata bar
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Results", len(data["results"]))
            with col2:
                st.metric("Candidates searched", data["filtered_count"])
            with col3:
                st.metric("Latency", f"{data['latency_ms']:.0f}ms")
            with col4:
                cache_icon = "✓ Cached" if data["cache_hit"] else "Fresh"
                st.metric("Cache", cache_icon)

            # Extracted intent
            intent = data.get("extracted_intent", {})
            if intent.get("emotional_concerns"):
                st.info(
                    f"**Identified concerns:** {', '.join(intent['emotional_concerns'])} "
                    f"— *{intent.get('query_summary', '')}*"
                )

            # Therapist cards
            for i, therapist in enumerate(data["results"]):
                with st.container():
                    st.markdown(f"---")
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        rank_emoji = ["🥇", "🥈", "🥉"] + ["  "] * 10
                        st.subheader(
                            f"{rank_emoji[i]} {therapist['name']}, "
                            f"{', '.join(therapist['credentials'])}"
                        )
                        st.caption(
                            f"📍 {therapist['city']} | "
                            f"{'💻 Online' if 'telehealth' in str(therapist['session_formats']) else '🏢 In-person'} | "
                            f"{therapist.get('fee_range', 'Contact for pricing')}"
                        )
                        if therapist.get("matched_modalities"):
                            st.markdown(
                                "**Modalities:** " + " · ".join(
                                    m.replace("_", " ").title()
                                    for m in therapist["matched_modalities"][:4]
                                )
                            )
                        st.markdown(f"_{therapist['bio_excerpt']}_")
                        st.caption(f"*Why this match: {therapist['match_explanation']}*")

                    with col2:
                        # Score breakdown
                        score_pct = int(therapist["composite_score"] * 100)
                        st.metric("Match Score", f"{score_pct}%")

                        # Score breakdown (expandable)
                        with st.expander("Score breakdown"):
                            st.progress(therapist["modality_score"], text=f"Modality: {therapist['modality_score']:.2f}")
                            st.progress(therapist["semantic_score"], text=f"Semantic: {therapist['semantic_score']:.2f}")
                            st.progress(therapist["bm25_score"], text=f"Keyword: {therapist['bm25_score']:.2f}")

                        # Feedback buttons
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            if st.button("👍 Good match", key=f"good_{i}"):
                                _submit_feedback(data["query_id"], therapist["therapist_id"], 5)
                        with f_col2:
                            if st.button("👎 Poor match", key=f"bad_{i}"):
                                _submit_feedback(data["query_id"], therapist["therapist_id"], 1)

                        st.link_button("View Profile", therapist["source_url"])

        except httpx.TimeoutException:
            st.error("Search timed out. Please try again.")
        except httpx.HTTPStatusError as e:
            st.error(f"Search failed: {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif search_clicked:
    st.warning("Please enter a search query or fill in your location.")


def _submit_feedback(query_id: str, therapist_id: str, rating: int) -> None:
    """Submit feedback asynchronously (fire-and-forget)."""
    try:
        httpx.post(
            f"{API_BASE}/feedback",
            json={
                "query_id": query_id,
                "therapist_id": therapist_id,
                "rating": rating,
            },
            timeout=5,
        )
        st.toast("Feedback recorded — thank you!")
    except Exception:
        pass  # feedback failure should never interrupt the user


# ── Sidebar: search history ───────────────────────────────────────────────────
if st.session_state.search_history:
    with st.sidebar:
        st.header("Recent Searches")
        for item in reversed(st.session_state.search_history[-5:]):
            st.caption(f"[{item['timestamp']}] {item['query']} → {item['num_results']} results")
