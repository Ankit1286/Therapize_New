"""
Therapize — standalone demo.
Runs the full matching pipeline (LLM intent extraction + embeddings + hybrid ranking)
entirely in-memory. No PostgreSQL or Redis required.

Usage:
    python demo.py                          # interactive mode
    python demo.py "I have severe anxiety"  # single query
"""
import asyncio
import io
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Check for API key before doing anything else ─────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("\n❌  ANTHROPIC_API_KEY is not set.")
    print("    Create a .env file with:  ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)


# ── Synthetic therapist data ──────────────────────────────────────────────────
from src.models.therapist import (
    TherapistProfile, TherapistLocation, TherapyModality,
    TherapistSpecialization, InsuranceProvider, SessionFormat,
)
from src.models.query import SearchRequest, UserQuestionnaire


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


# ── Terminal colors ───────────────────────────────────────────────────────────
class C:
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    RED    = "\033[91m"
    RESET  = "\033[0m"
    LINE   = "─" * 70


def header(text: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}{C.LINE}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{C.LINE}{C.RESET}")


def section(text: str) -> None:
    print(f"\n{C.BOLD}{C.YELLOW}  {text}{C.RESET}")


def result_card(rank: int, sr, query_text: str) -> None:
    t = sr.therapist
    creds = ", ".join(t.credentials)
    formats = " / ".join(f.value.replace("_", " ") for f in t.session_formats)
    insurance = ", ".join(i.value.replace("_", " ").title() for i in t.accepts_insurance[:3])
    fee = f"${t.fee_min}–${t.fee_max}" if t.fee_min and t.fee_max else "Contact for fees"
    if t.sliding_scale:
        fee += " (sliding scale)"

    modalities_str = " · ".join(
        m.replace("_", " ").title() for m in sr.matched_modalities[:3]
    ) or ", ".join(m.value.replace("_", " ").title() for m in t.modalities[:3])

    bar_width = 20
    score_bar = int(sr.composite_score * bar_width)
    score_visual = f"{'█' * score_bar}{'░' * (bar_width - score_bar)}"

    print(f"\n  {C.BOLD}#{rank}  {t.name}{C.RESET}  {C.DIM}({creds}){C.RESET}")
    print(f"  {C.DIM}{'─' * 60}{C.RESET}")
    print(f"  {C.GREEN}Match score:{C.RESET} {score_visual} {C.BOLD}{sr.composite_score:.0%}{C.RESET}")
    print(f"  {C.DIM}  modality={sr.modality_score:.2f}  semantic={sr.semantic_score:.2f}  bm25={sr.bm25_score:.2f}  quality={sr.quality_score:.2f}{C.RESET}")
    print(f"  {C.CYAN}Location:{C.RESET}    {t.location.city}, CA  ·  {formats}")
    print(f"  {C.CYAN}Insurance:{C.RESET}   {insurance}")
    print(f"  {C.CYAN}Fees:{C.RESET}        {fee}")
    print(f"  {C.CYAN}Modalities:{C.RESET}  {modalities_str}")
    print(f"  {C.CYAN}Why matched:{C.RESET} {sr.score_explanation}")
    bio_excerpt = t.bio[:200].rsplit(" ", 1)[0] + "..."
    print(f"  {C.DIM}\"{bio_excerpt}\"{C.RESET}")


async def run_demo(query_text: str, questionnaire: UserQuestionnaire | None = None) -> None:
    from src.matching.hybrid_ranker import HybridRanker
    from src.matching.modality_mapper import ModalityMapper
    from src.matching.query_processor import QueryProcessor
    from src.pipeline.embeddings import EmbeddingPipeline

    header("Therapize — AI Therapist Matching Engine  (Demo Mode)")

    print(f"\n  {C.BOLD}Query:{C.RESET} \"{query_text}\"")
    if questionnaire and any(v is not None for v in questionnaire.model_dump().values()):
        filters = {k: v for k, v in questionnaire.model_dump().items() if v is not None}
        print(f"  {C.BOLD}Filters:{C.RESET} {filters}")

    request = SearchRequest(
        query_id=uuid4(),
        free_text=query_text,
        questionnaire=questionnaire or UserQuestionnaire(),
    )

    # ── Step 1: Intent extraction ────────────────────────────────────────
    section("Step 1/4  LLM Intent Extraction (Claude)")
    t0 = time.monotonic()
    processor = QueryProcessor()
    intent = await processor.extract_intent(request)
    llm_ms = (time.monotonic() - t0) * 1000

    print(f"  {C.DIM}Latency: {llm_ms:.0f}ms{C.RESET}")
    print(f"  Emotional concerns:  {C.GREEN}{intent.emotional_concerns}{C.RESET}")
    print(f"  Specializations:     {C.GREEN}{[s.value for s in intent.recommended_specializations]}{C.RESET}")
    print(f"  Modalities:          {C.GREEN}{[m.value for m in intent.recommended_modalities]}{C.RESET}")
    print(f"  Summary:             {C.YELLOW}\"{intent.query_summary}\"{C.RESET}")
    print(f"  Confidence:          {intent.confidence:.0%}")

    # ── Step 2: Embeddings ───────────────────────────────────────────────
    section("Step 2/4  Generating Embeddings (local, all-MiniLM-L6-v2)")
    t0 = time.monotonic()
    pipeline = EmbeddingPipeline()
    query_embedding = await pipeline.embed_query(query_text)
    profile_embeddings_pairs = await pipeline.embed_profiles(SAMPLE_THERAPISTS)
    embed_ms = (time.monotonic() - t0) * 1000

    candidate_embeddings = [emb for _, emb in profile_embeddings_pairs]
    print(f"  Embedded {len(SAMPLE_THERAPISTS)} therapist profiles + query in {embed_ms:.0f}ms")
    print(f"  Embedding dim: {len(query_embedding)}")

    # ── Step 3: Modality weights ─────────────────────────────────────────
    section("Step 3/4  Modality Mapping")
    mapper = ModalityMapper()
    modality_weights = mapper.get_modality_weights(intent.emotional_concerns)
    if modality_weights:
        for mod, weight in sorted(modality_weights.items(), key=lambda x: -x[1])[:5]:
            bar = "█" * int(weight * 10)
            print(f"  {mod:<40} {C.GREEN}{bar:<10}{C.RESET} {weight:.2f}")
    else:
        print(f"  {C.DIM}No specific modality weights — using semantic + BM25 only{C.RESET}")

    # ── Step 4: Hybrid ranking ────────────────────────────────────────────
    section("Step 4/4  Hybrid Ranking (modality + semantic + BM25 + quality)")
    t0 = time.monotonic()
    ranker = HybridRanker()
    scored = ranker.rank(
        candidates=SAMPLE_THERAPISTS,
        query_text=query_text,
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        recommended_modalities=modality_weights,
    )
    rank_ms = (time.monotonic() - t0) * 1000
    print(f"  Ranked {len(scored)} candidates in {rank_ms:.0f}ms")

    # ── Results ───────────────────────────────────────────────────────────
    header(f"Top {min(5, len(scored))} Matches")
    for i, sr in enumerate(scored[:5], 1):
        result_card(i, sr, query_text)

    total_ms = llm_ms + embed_ms + rank_ms
    print(f"\n  {C.DIM}Total pipeline: {total_ms:.0f}ms  "
          f"(LLM={llm_ms:.0f}ms  embed={embed_ms:.0f}ms  rank={rank_ms:.0f}ms){C.RESET}")
    print(f"\n{C.BOLD}{C.CYAN}{C.LINE}{C.RESET}\n")


DEMO_QUERIES = [
    "I've been having panic attacks and can't stop worrying. I need someone who takes Aetna insurance.",
    "I'm going through a divorce and feel completely lost. Telehealth only please.",
    "I think I have OCD — my intrusive thoughts are taking over my life.",
    "Struggling with grief after losing my mother. I want an in-person therapist in San Francisco.",
    "My relationship is falling apart. We need couples therapy.",
]


async def main():
    if len(sys.argv) > 1:
        # Single query from command line
        query = " ".join(sys.argv[1:])
        await run_demo(query)
        return

    # Interactive mode
    print(f"\n{C.BOLD}Therapize Demo{C.RESET} — AI Therapist Matching Engine")
    print(f"\n{C.DIM}Sample queries (press Enter to use, or type your own):{C.RESET}")
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"  {i}. {q}")

    print(f"\n  {C.DIM}Enter a number (1-{len(DEMO_QUERIES)}) or type your own query:{C.RESET}", end=" ")
    try:
        user_input = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if user_input.isdigit() and 1 <= int(user_input) <= len(DEMO_QUERIES):
        query = DEMO_QUERIES[int(user_input) - 1]
    elif user_input:
        query = user_input
    else:
        query = DEMO_QUERIES[0]

    await run_demo(query)


if __name__ == "__main__":
    asyncio.run(main())
