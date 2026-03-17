"""
Therapist data models — the canonical schema for all therapist records.
These are the data contracts between the scraper, DB, matching engine, and API.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


class TherapyModality(str, Enum):
    """
    Standardized therapy modalities. Using an enum ensures:
    - Consistency across scrapers (different sites use different names)
    - Type safety in matching engine
    - Queryable/filterable in the DB
    """
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    EMDR = "emdr"
    ACT = "acceptance_commitment_therapy"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    MINDFULNESS = "mindfulness_based"
    SOMATIC = "somatic"
    GOTTMAN = "gottman_method"
    EFT = "emotionally_focused_therapy"
    EXPOSURE = "exposure_therapy"
    CPT = "cognitive_processing_therapy"
    NARRATIVE = "narrative_therapy"
    SOLUTION_FOCUSED = "solution_focused"
    MOTIVATIONAL = "motivational_interviewing"
    PLAY_THERAPY = "play_therapy"
    ART_THERAPY = "art_therapy"
    PSYCHOANALYTIC = "psychoanalytic"
    INTEGRATIVE = "integrative"
    CBT_I = "cbt_insomnia"
    TRAUMA_INFORMED = "trauma_informed"
    FAMILY_SYSTEMS = "family_systems"


class InsuranceProvider(str, Enum):
    BLUE_SHIELD = "blue_shield"
    BLUE_CROSS = "blue_cross"
    AETNA = "aetna"
    UNITED = "united_healthcare"
    CIGNA = "cigna"
    KAISER = "kaiser"
    MAGELLAN = "magellan"
    OPTUM = "optum"
    SELF_PAY = "self_pay"
    SLIDING_SCALE = "sliding_scale"
    OUT_OF_NETWORK = "out_of_network"


class SessionFormat(str, Enum):
    IN_PERSON = "in_person"
    TELEHEALTH = "telehealth"
    BOTH = "both"


class TherapistSpecialization(str, Enum):
    """What the therapist specializes in treating."""
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    TRAUMA = "trauma"
    PTSD = "ptsd"
    RELATIONSHIP = "relationship_issues"
    GRIEF = "grief"
    ADHD = "adhd"
    ADDICTION = "addiction"
    EATING_DISORDERS = "eating_disorders"
    OCD = "ocd"
    BIPOLAR = "bipolar"
    CHRONIC_PAIN = "chronic_pain"
    LIFE_TRANSITIONS = "life_transitions"
    CAREER = "career"
    LGBTQ = "lgbtq_affirming"
    COUPLES = "couples"
    FAMILY = "family"
    CHILDREN = "children"
    ADOLESCENTS = "adolescents"
    STRESS = "stress_burnout"
    SLEEP = "sleep_issues"
    SELF_ESTEEM = "self_esteem"
    ANGER = "anger_management"
    PERSONALITY_DISORDERS = "personality_disorders"


class TherapistLocation(BaseModel):
    city: str
    state: str = "CA"
    zip_code: Optional[str] = None
    county: Optional[str] = None
    # For geo-distance filtering
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class TherapistProfile(BaseModel):
    """
    Canonical therapist record. This is what gets stored in PostgreSQL.

    Design decision: store both raw text (for BM25) and structured fields (for filtering).
    The embedding is generated from a concatenated text blob and stored in pgvector.
    """
    id: UUID = Field(default_factory=uuid4)
    source: str = Field(..., description="Which platform this was scraped from")
    source_url: HttpUrl = Field(..., description="Original URL of the profile")
    source_id: str = Field(..., description="Platform-specific therapist ID")

    # Identity
    name: str
    credentials: list[str] = Field(default_factory=list, description="e.g., ['LMFT', 'PhD']")
    license_number: Optional[str] = None
    years_experience: Optional[int] = None

    # Clinical
    modalities: list[TherapyModality] = Field(default_factory=list)
    specializations: list[TherapistSpecialization] = Field(default_factory=list)
    populations_served: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list, description="Languages spoken")

    # Logistics
    location: TherapistLocation
    session_formats: list[SessionFormat] = Field(default_factory=list)
    accepts_insurance: list[InsuranceProvider] = Field(default_factory=list)
    sliding_scale: bool = False
    fee_min: Optional[int] = None  # in USD
    fee_max: Optional[int] = None
    accepting_new_clients: bool = True

    # Content (used for BM25 + embedding)
    bio: str = Field(default="", description="Full bio text from profile")
    approach_description: str = Field(default="", description="How they describe their approach")

    # Engagement signals
    rating: Optional[float] = None
    review_count: int = 0
    profile_completeness: float = Field(0.0, ge=0.0, le=1.0)

    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    @field_validator("fee_min", "fee_max")
    @classmethod
    def fee_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("Fee cannot be negative")
        return v

    def to_embedding_text(self) -> str:
        """
        Constructs the text blob used for embedding generation.
        This is the 'view' of the therapist that semantic search operates over.
        Order matters: put most important info first (transformer attention is position-biased).
        """
        parts = [
            f"Therapist: {self.name}",
            f"Specializations: {', '.join(s.value for s in self.specializations)}",
            f"Modalities: {', '.join(m.value for m in self.modalities)}",
            f"Approach: {self.approach_description}",
            f"Bio: {self.bio}",
            f"Populations: {', '.join(self.populations_served)}",
            f"Location: {self.location.city}, {self.location.state}",
        ]
        return " | ".join(p for p in parts if p)

    def to_bm25_document(self) -> str:
        """Text used for BM25 keyword matching — includes all searchable text."""
        return " ".join([
            self.name,
            self.bio,
            self.approach_description,
            " ".join(s.value for s in self.specializations),
            " ".join(m.value for m in self.modalities),
            " ".join(self.populations_served),
            " ".join(self.credentials),
        ])

    model_config = {"use_enum_values": False}
