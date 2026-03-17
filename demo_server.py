"""
Therapize — in-memory demo server.
Serves the full matching pipeline via FastAPI without PostgreSQL or Redis.
The Streamlit frontend connects to this at http://localhost:8000/api/v1

Run:
    python demo_server.py
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Silence noisy loggers ────────────────────────────────────────────────────
for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


app = FastAPI(title="Therapize Demo Server", version="1.0.0-demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────────────────
_pipeline_ready = False
_pipeline_error: str | None = None
_sample_therapists: list = []
_sample_embeddings: list[list[float]] = []
_embedding_pipeline = None
_query_processor = None
_hybrid_ranker = None
_modality_mapper = None


# ── Sample data (same as demo.py) ────────────────────────────────────────────
from src.models.therapist import (
    TherapistProfile, TherapistLocation, TherapyModality,
    TherapistSpecialization, InsuranceProvider, SessionFormat,
)
from src.models.query import (
    SearchRequest, SearchResponse, UserQuestionnaire,
    ExtractedQueryIntent, TherapistResult, FeedbackRequest,
)

SAMPLE_THERAPISTS = [
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/1", source_id="1",
        name="Dr. Sarah Chen", credentials=["PhD", "LCSW"],
        years_experience=12,
        modalities=[TherapyModality.CBT, TherapyModality.MINDFULNESS, TherapyModality.ACT],
        specializations=[TherapistSpecialization.ANXIETY, TherapistSpecialization.DEPRESSION, TherapistSpecialization.STRESS],
        populations_served=["adults", "young adults"],
        languages=["English", "Mandarin"],
        location=TherapistLocation(city="San Francisco", state="CA"),
        session_formats=[SessionFormat.TELEHEALTH, SessionFormat.IN_PERSON],
        accepts_insurance=[InsuranceProvider.BLUE_SHIELD, InsuranceProvider.AETNA],
        sliding_scale=True, fee_min=120, fee_max=180,
        accepting_new_clients=True, rating=4.9, review_count=47,
        profile_completeness=0.95,
        bio=(
            "I specialize in helping adults navigate anxiety, depression, and life transitions. "
            "My approach integrates Cognitive Behavioral Therapy with mindfulness practices, "
            "creating a warm and collaborative therapeutic relationship. I believe healing "
            "happens when we address both thoughts and the deeper emotional patterns underneath. "
            "I work extensively with perfectionism, overthinking, and the constant 'what-ifs' "
            "that keep people stuck in cycles of worry."
        ),
        approach_description="CBT + mindfulness integration for anxiety and depression.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/2", source_id="2",
        name="Michael Rodriguez, LMFT", credentials=["LMFT"],
        years_experience=8,
        modalities=[TherapyModality.EMDR, TherapyModality.SOMATIC, TherapyModality.TRAUMA_INFORMED],
        specializations=[TherapistSpecialization.TRAUMA, TherapistSpecialization.PTSD, TherapistSpecialization.ANXIETY],
        populations_served=["adults", "veterans", "first responders"],
        languages=["English", "Spanish"],
        location=TherapistLocation(city="Los Angeles", state="CA"),
        session_formats=[SessionFormat.BOTH],
        accepts_insurance=[InsuranceProvider.UNITED, InsuranceProvider.CIGNA],
        sliding_scale=False, fee_min=150, fee_max=200,
        accepting_new_clients=True, rating=4.8, review_count=63,
        profile_completeness=0.90,
        bio=(
            "As a trauma-informed therapist, I work with survivors of complex trauma using "
            "EMDR and somatic techniques. My work is rooted in the belief that the body holds "
            "the key to healing. I create a safe, non-judgmental space where clients can "
            "reconnect with their innate capacity for resilience. I have specialized training "
            "in treating PTSD, panic disorder, and the lingering effects of childhood trauma."
        ),
        approach_description="EMDR and somatic therapy for trauma recovery.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/3", source_id="3",
        name="Emma Thompson, LPCC", credentials=["LPCC", "NCC"],
        years_experience=15,
        modalities=[TherapyModality.GOTTMAN, TherapyModality.EFT, TherapyModality.PSYCHODYNAMIC],
        specializations=[TherapistSpecialization.COUPLES, TherapistSpecialization.RELATIONSHIP, TherapistSpecialization.FAMILY],
        populations_served=["couples", "adults", "families"],
        languages=["English"],
        location=TherapistLocation(city="Palo Alto", state="CA"),
        session_formats=[SessionFormat.IN_PERSON, SessionFormat.TELEHEALTH],
        accepts_insurance=[InsuranceProvider.BLUE_CROSS, InsuranceProvider.OPTUM],
        sliding_scale=True, fee_min=200, fee_max=280,
        accepting_new_clients=True, rating=4.7, review_count=91,
        profile_completeness=0.98,
        bio=(
            "I work with couples and individuals on relationship issues, communication patterns, "
            "and emotional intimacy using Gottman Method and Emotionally Focused Therapy. "
            "Whether you're navigating conflict, infidelity, or simply feeling disconnected, "
            "I can help you rebuild trust and deepen your connection. My couples practice is "
            "grounded in 20 years of Gottman research — the most evidence-based approach to "
            "couples therapy available."
        ),
        approach_description="Gottman Method and EFT for couples and relationship healing.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/4", source_id="4",
        name="Dr. Jessica Patel", credentials=["PsyD"],
        years_experience=10,
        modalities=[TherapyModality.DBT, TherapyModality.CBT, TherapyModality.MINDFULNESS],
        specializations=[TherapistSpecialization.PERSONALITY_DISORDERS, TherapistSpecialization.DEPRESSION, TherapistSpecialization.SELF_ESTEEM],
        populations_served=["adults", "young adults", "adolescents"],
        languages=["English", "Hindi"],
        location=TherapistLocation(city="Oakland", state="CA"),
        session_formats=[SessionFormat.TELEHEALTH],
        accepts_insurance=[InsuranceProvider.KAISER, InsuranceProvider.MAGELLAN],
        sliding_scale=True, fee_min=100, fee_max=150,
        accepting_new_clients=True, rating=4.9, review_count=28,
        profile_completeness=0.88,
        bio=(
            "I specialize in DBT for emotion dysregulation, borderline personality disorder, "
            "and chronic self-harm. Having trained at the Linehan Institute, I bring both clinical "
            "rigor and deep empathy to our work. DBT skills can transform how you relate to "
            "difficult emotions — you don't have to be at the mercy of every feeling. "
            "I offer individual DBT therapy as well as skills group coaching."
        ),
        approach_description="DBT for emotion regulation, BPD, and chronic mood issues.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/5", source_id="5",
        name="James Williams, LCSW", credentials=["LCSW"],
        years_experience=20,
        modalities=[TherapyModality.NARRATIVE, TherapyModality.HUMANISTIC, TherapyModality.ACT],
        specializations=[TherapistSpecialization.GRIEF, TherapistSpecialization.LIFE_TRANSITIONS, TherapistSpecialization.DEPRESSION],
        populations_served=["adults", "older adults", "bereaved"],
        languages=["English"],
        location=TherapistLocation(city="Sacramento", state="CA"),
        session_formats=[SessionFormat.IN_PERSON, SessionFormat.TELEHEALTH],
        accepts_insurance=[InsuranceProvider.BLUE_SHIELD, InsuranceProvider.BLUE_CROSS],
        sliding_scale=False, fee_min=130, fee_max=170,
        accepting_new_clients=True, rating=4.6, review_count=112,
        profile_completeness=0.85,
        bio=(
            "As a grief and loss specialist, I help clients process bereavement, life transitions, "
            "and the complex emotions that come with major changes. Using narrative therapy and "
            "humanistic approaches, we'll honor your experience while creating space for healing "
            "and new meaning. I've worked extensively with unexpected loss, anticipatory grief, "
            "and complicated bereavement. Grief is not a disorder — it's love with nowhere to go."
        ),
        approach_description="Narrative therapy and humanistic approaches for grief and loss.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/6", source_id="6",
        name="Dr. Amanda Foster", credentials=["PhD", "LMFT"],
        years_experience=7,
        modalities=[TherapyModality.ACT, TherapyModality.MINDFULNESS, TherapyModality.PSYCHODYNAMIC],
        specializations=[TherapistSpecialization.LGBTQ, TherapistSpecialization.ANXIETY, TherapistSpecialization.SELF_ESTEEM],
        populations_served=["LGBTQ+", "adults", "young adults", "gender non-conforming"],
        languages=["English"],
        location=TherapistLocation(city="Berkeley", state="CA"),
        session_formats=[SessionFormat.TELEHEALTH],
        accepts_insurance=[InsuranceProvider.AETNA, InsuranceProvider.CIGNA],
        sliding_scale=True, fee_min=110, fee_max=160,
        accepting_new_clients=True, rating=4.8, review_count=34,
        profile_completeness=0.92,
        bio=(
            "I work with LGBTQ+ individuals on identity, family of origin issues, and mental "
            "health. My affirming practice combines ACT with psychodynamic exploration. "
            "I believe everyone deserves a therapist who truly understands their lived experience "
            "without the need for explanation or justification. I specialize in coming out, "
            "gender transition, internalized homophobia, and navigating unsupportive families."
        ),
        approach_description="LGBTQ+-affirming therapy combining ACT and psychodynamic work.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/7", source_id="7",
        name="Robert Martinez, LCSW", credentials=["LCSW", "CADC"],
        years_experience=14,
        modalities=[TherapyModality.MOTIVATIONAL, TherapyModality.CBT, TherapyModality.MINDFULNESS],
        specializations=[TherapistSpecialization.ADDICTION, TherapistSpecialization.ANXIETY, TherapistSpecialization.DEPRESSION],
        populations_served=["adults", "people in recovery"],
        languages=["English", "Spanish"],
        location=TherapistLocation(city="San Diego", state="CA"),
        session_formats=[SessionFormat.IN_PERSON, SessionFormat.TELEHEALTH],
        accepts_insurance=[InsuranceProvider.UNITED, InsuranceProvider.BLUE_SHIELD],
        sliding_scale=True, fee_min=90, fee_max=140,
        accepting_new_clients=True, rating=4.7, review_count=56,
        profile_completeness=0.87,
        bio=(
            "Specializing in addiction and recovery, I use Motivational Interviewing and "
            "CBT to help clients overcome substance use and behavioral addictions. "
            "Recovery is not a straight line, and I meet clients exactly where they are "
            "with patience, honesty, and clinical expertise. I am in recovery myself and "
            "bring lived experience as well as clinical training to our work together."
        ),
        approach_description="Motivational Interviewing and CBT for addiction and recovery.",
    ),
    TherapistProfile(
        source="demo", source_url="https://therapize.demo/8", source_id="8",
        name="Lisa Johnson, LMFT", credentials=["LMFT"],
        years_experience=9,
        modalities=[TherapyModality.EXPOSURE, TherapyModality.CBT, TherapyModality.ACT],
        specializations=[TherapistSpecialization.OCD, TherapistSpecialization.ANXIETY, TherapistSpecialization.ADHD],
        populations_served=["adults", "adolescents", "children"],
        languages=["English"],
        location=TherapistLocation(city="San Jose", state="CA"),
        session_formats=[SessionFormat.TELEHEALTH, SessionFormat.IN_PERSON],
        accepts_insurance=[InsuranceProvider.AETNA, InsuranceProvider.OPTUM],
        sliding_scale=False, fee_min=160, fee_max=220,
        accepting_new_clients=True, rating=4.9, review_count=73,
        profile_completeness=0.93,
        bio=(
            "My practice focuses on anxiety disorders, OCD, and ADHD using evidence-based "
            "Exposure and Response Prevention (ERP) and CBT. I take a compassionate yet "
            "direct approach — we'll work together to face fears systematically and help you "
            "reclaim your life from anxiety and compulsions. OCD is treatable, and ERP is "
            "the gold-standard treatment with 60-80% response rates."
        ),
        approach_description="ERP and CBT specialist for OCD, phobias, and anxiety disorders.",
    ),
]


@app.on_event("startup")
async def startup():
    """Pre-load embeddings at startup so first search is fast."""
    global _pipeline_ready, _pipeline_error
    global _sample_therapists, _sample_embeddings
    global _embedding_pipeline, _query_processor, _hybrid_ranker, _modality_mapper

    print("  Loading embedding model (first run downloads ~90MB)...", flush=True)
    try:
        from src.pipeline.embeddings import EmbeddingPipeline
        from src.matching.query_processor import QueryProcessor
        from src.matching.hybrid_ranker import HybridRanker
        from src.matching.modality_mapper import ModalityMapper

        _embedding_pipeline = EmbeddingPipeline()
        _query_processor = QueryProcessor()
        _hybrid_ranker = HybridRanker()
        _modality_mapper = ModalityMapper()

        # Pre-compute all profile embeddings
        pairs = await _embedding_pipeline.embed_profiles(SAMPLE_THERAPISTS)
        _sample_therapists = [p for p, _ in pairs]
        _sample_embeddings = [e for _, e in pairs]

        _pipeline_ready = True
        print(f"  Pipeline ready — {len(_sample_therapists)} therapists indexed.", flush=True)
    except Exception as e:
        _pipeline_error = str(e)
        print(f"  Pipeline startup failed: {e}", flush=True)


@app.get("/api/v1/health")
async def health():
    if not _pipeline_ready:
        raise HTTPException(503, detail=_pipeline_error or "Pipeline warming up")
    return {"status": "ok", "therapists_indexed": len(_sample_therapists), "mode": "demo"}


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not _pipeline_ready:
        raise HTTPException(503, detail="Pipeline is still warming up, please retry in a moment")

    start = time.monotonic()
    query_id = request.query_id

    # 1. Extract intent
    intent = await _query_processor.extract_intent(request)

    # 2. Embed query
    query_embedding = await _embedding_pipeline.embed_query(request.free_text or "therapist")

    # 3. Modality weights
    modality_weights = _modality_mapper.get_modality_weights(intent.emotional_concerns)

    # 4. Rank
    scored = _hybrid_ranker.rank(
        candidates=_sample_therapists,
        query_text=request.free_text,
        query_embedding=query_embedding,
        candidate_embeddings=_sample_embeddings,
        recommended_modalities=modality_weights,
    )

    # 5. Build results
    results = []
    for sr in scored[:10]:
        t = sr.therapist
        fee_range = None
        if t.fee_min and t.fee_max:
            fee_range = f"${t.fee_min}–${t.fee_max}"
            if t.sliding_scale:
                fee_range += " (sliding scale)"
        elif t.sliding_scale:
            fee_range = "Sliding scale available"

        results.append(TherapistResult(
            therapist_id=t.id,
            name=t.name,
            credentials=t.credentials,
            city=t.location.city,
            session_formats=t.session_formats,
            accepts_insurance=t.accepts_insurance,
            fee_range=fee_range,
            matched_modalities=sr.matched_modalities,
            bio_excerpt=t.bio[:300] + "..." if len(t.bio) > 300 else t.bio,
            source_url=str(t.source_url),
            composite_score=sr.composite_score,
            modality_score=sr.modality_score,
            semantic_score=sr.semantic_score,
            bm25_score=sr.bm25_score,
            rating_score=sr.quality_score,
            match_explanation=sr.score_explanation,
        ))

    latency_ms = (time.monotonic() - start) * 1000

    return SearchResponse(
        query_id=query_id,
        results=results,
        total_candidates=len(_sample_therapists),
        filtered_count=len(_sample_therapists),
        extracted_intent=intent,
        cache_hit=False,
        latency_ms=latency_ms,
        llm_tokens_used=0,
    )


@app.post("/api/v1/feedback")
async def feedback(req: FeedbackRequest):
    # In demo mode we just acknowledge — no DB to write to
    return {"status": "ok", "message": "Feedback recorded (demo mode)"}


if __name__ == "__main__":
    print("\nTherapize Demo Server")
    print("=" * 40)
    print("  Backend:   http://localhost:8000")
    print("  API docs:  http://localhost:8000/docs")
    print("=" * 40)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
