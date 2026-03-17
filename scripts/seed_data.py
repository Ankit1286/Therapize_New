"""
Seed script — loads sample therapist data for development/testing.

This lets you run the full system without scraping real data first.
Generates realistic synthetic therapist profiles across California.

Usage:
    python scripts/seed_data.py --count 100
"""
import asyncio
import logging
import random
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.models.therapist import (
    InsuranceProvider,
    SessionFormat,
    TherapistLocation,
    TherapistProfile,
    TherapistSpecialization,
    TherapyModality,
)
from src.pipeline.embeddings import EmbeddingPipeline
from src.storage.database import TherapistRepository, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CA_CITIES = [
    "San Francisco", "Los Angeles", "San Diego", "Sacramento",
    "Oakland", "Berkeley", "Palo Alto", "Santa Monica", "Pasadena",
    "Long Beach", "Irvine", "Santa Barbara", "San Jose", "Fresno",
]

BIOS = [
    (
        "I specialize in helping adults navigate anxiety, depression, and life transitions. "
        "My approach integrates Cognitive Behavioral Therapy with mindfulness practices, "
        "creating a warm and collaborative therapeutic relationship. I believe healing "
        "happens when we address both thoughts and the deeper emotional patterns underneath."
    ),
    (
        "As a trauma-informed therapist, I work with survivors of complex trauma using "
        "EMDR and somatic techniques. My work is rooted in the belief that the body holds "
        "the key to healing. I create a safe, non-judgmental space where clients can "
        "reconnect with their innate capacity for resilience."
    ),
    (
        "I work with couples and individuals on relationship issues, communication patterns, "
        "and emotional intimacy using Gottman Method and Emotionally Focused Therapy. "
        "Whether you're navigating conflict, infidelity, or simply feeling disconnected, "
        "I can help you rebuild trust and deepen your connection."
    ),
    (
        "My practice focuses on anxiety disorders, OCD, and phobias using evidence-based "
        "Exposure and Response Prevention (ERP) and CBT. I take a compassionate yet "
        "direct approach — we'll work together to face fears systematically and help you "
        "reclaim your life from anxiety."
    ),
    (
        "I specialize in DBT for emotion dysregulation, borderline personality disorder, "
        "and self-harm. Having trained at the Linehan Institute, I bring both clinical "
        "rigor and deep empathy to our work. DBT skills can transform how you relate to "
        "difficult emotions."
    ),
    (
        "As a grief and loss specialist, I help clients process bereavement, life transitions, "
        "and the complex emotions that come with major changes. Using narrative therapy and "
        "humanistic approaches, we'll honor your experience while creating space for healing "
        "and new meaning."
    ),
    (
        "I work with LGBTQ+ individuals on identity, family of origin issues, and mental "
        "health. My affirming practice combines ACT with psychodynamic exploration. "
        "I believe everyone deserves a therapist who truly understands their lived experience "
        "without the need for explanation or justification."
    ),
    (
        "Specializing in addiction and recovery, I use Motivational Interviewing and "
        "CBT to help clients overcome substance use and behavioral addictions. "
        "Recovery is not a straight line, and I meet clients exactly where they are "
        "with patience, honesty, and clinical expertise."
    ),
]

NAMES = [
    ("Sarah", "Chen"), ("Michael", "Rodriguez"), ("Emma", "Thompson"),
    ("David", "Kim"), ("Jessica", "Patel"), ("James", "Williams"),
    ("Amanda", "Foster"), ("Robert", "Martinez"), ("Lisa", "Johnson"),
    ("Kevin", "Brown"), ("Rachel", "Davis"), ("Thomas", "Garcia"),
    ("Nicole", "Wilson"), ("Christopher", "Anderson"), ("Michelle", "Taylor"),
    ("Daniel", "Hernandez"), ("Ashley", "Moore"), ("Matthew", "Jackson"),
    ("Stephanie", "Lee"), ("Andrew", "White"), ("Lauren", "Harris"),
    ("Ryan", "Thompson"), ("Jennifer", "Clark"), ("Brian", "Lewis"),
]

CREDENTIALS = [
    ["LMFT"], ["LCSW"], ["PhD", "LMFT"], ["PsyD"], ["MA", "LMFT"],
    ["MSW", "LCSW"], ["PhD", "LP"], ["EdD", "LMFT"], ["LMHC"], ["NCC", "LPC"],
]


def _random_therapist() -> TherapistProfile:
    first, last = random.choice(NAMES)
    city = random.choice(CA_CITIES)

    # Pick 2-4 random modalities
    all_modalities = list(TherapyModality)
    modalities = random.sample(all_modalities, k=random.randint(2, 5))

    # Pick 2-4 random specializations
    all_specs = list(TherapistSpecialization)
    specs = random.sample(all_specs, k=random.randint(2, 4))

    # Random insurance
    all_insurance = list(InsuranceProvider)
    insurance = random.sample(all_insurance, k=random.randint(1, 4))

    fee_base = random.choice([80, 100, 120, 150, 175, 200, 225])
    session_formats = random.choice([
        [SessionFormat.IN_PERSON],
        [SessionFormat.TELEHEALTH],
        [SessionFormat.BOTH],
    ])

    bio = random.choice(BIOS)

    return TherapistProfile(
        id=uuid4(),
        source="seed_data",
        source_url=f"https://example.com/therapists/{first.lower()}-{last.lower()}-{random.randint(1000,9999)}",  # type: ignore
        source_id=str(random.randint(10000, 99999)),
        name=f"{first} {last}",
        credentials=random.choice(CREDENTIALS),
        modalities=modalities,
        specializations=specs,
        location=TherapistLocation(city=city, state="CA"),
        session_formats=session_formats,
        accepts_insurance=insurance,
        sliding_scale=random.random() > 0.5,
        fee_min=fee_base,
        fee_max=fee_base + random.randint(20, 60),
        accepting_new_clients=random.random() > 0.1,
        bio=bio,
        approach_description=f"I use {', '.join(m.value.replace('_', ' ') for m in modalities[:2])} to help clients with {', '.join(s.value.replace('_', ' ') for s in specs[:2])}.",
        rating=round(random.uniform(3.5, 5.0), 1),
        review_count=random.randint(0, 50),
        profile_completeness=round(random.uniform(0.5, 1.0), 2),
    )


async def seed(count: int = 100) -> None:
    settings = get_settings()
    await init_db()

    embedding_pipeline = EmbeddingPipeline()
    repository = TherapistRepository()

    logger.info("Generating %d synthetic therapist profiles...", count)
    profiles = [_random_therapist() for _ in range(count)]

    logger.info("Generating embeddings (local sentence-transformers, no API cost)...")
    profile_embeddings = await embedding_pipeline.embed_profiles(profiles)

    logger.info("Storing to database...")
    saved = 0
    for profile, embedding in profile_embeddings:
        if not embedding:
            continue
        await repository.upsert(profile, embedding)
        saved += 1

    logger.info("Seeded %d therapists successfully!", saved)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(seed(args.count))
