"""
Query models — the request/response contracts for the search API.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.therapist import (
    InsuranceProvider,
    SessionFormat,
)


class UserQuestionnaire(BaseModel):
    """
    Structured questionnaire filled by the user before or alongside the free-text query.
    Hard filters: if provided, these are non-negotiable constraints.
    """
    city: Optional[str] = None
    zip_code: Optional[str] = None
    max_distance_miles: Optional[int] = Field(None, ge=1, le=100)
    insurance: Optional[InsuranceProvider] = None
    session_format: Optional[SessionFormat] = None
    max_budget_per_session: Optional[int] = Field(None, ge=0)
    preferred_gender: Optional[str] = None
    preferred_language: Optional[str] = None
    preferred_ethnicity: Optional[str] = None
    age_group: Optional[str] = None  # adult, adolescent, child


class SearchRequest(BaseModel):
    """
    Top-level search request. Can contain free text, questionnaire, or both.
    At least one must be provided.
    """
    query_id: UUID = Field(default_factory=uuid4)
    free_text: str = Field(
        "",
        description="Natural language description of what the user is looking for",
        max_length=2000,
    )
    questionnaire: UserQuestionnaire = Field(default_factory=UserQuestionnaire)
    session_id: Optional[str] = None   # for conversation continuity
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def has_content(self) -> bool:
        """Return True if the request has at least one searchable field (free text or any questionnaire value)."""
        return bool(self.free_text) or any(
            v is not None
            for v in self.questionnaire.model_dump().values()
        )


class ExtractedQueryIntent(BaseModel):
    """
    Structured representation of what the LLM extracted from the free-text query.
    This is an internal model — not exposed to users.

    Kept minimal intentionally: only extract what the rule-based pipeline cannot
    derive itself. Modality mapping and specialization lookup are handled
    deterministically by modality_map.json after extraction.
    """
    emotional_concerns: list[str] = Field(
        default_factory=list,
        description="Identified emotional/psychological concerns, e.g. ['anxiety', 'grief']",
    )
    inferred_filters: UserQuestionnaire = Field(
        default_factory=UserQuestionnaire,
        description="Logistical filters implied by the query (location, insurance, budget, format)",
    )
    query_summary: str = Field(
        "",
        description="One-sentence summary of what the user is looking for",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")


class TherapistResult(BaseModel):
    """A single therapist in the ranked result list."""
    therapist_id: UUID
    name: str
    credentials: list[str]
    city: Optional[str] = None
    session_formats: list[SessionFormat]
    accepts_insurance: list[InsuranceProvider]
    fee_range: Optional[str] = None  # e.g. "$120-$180"
    matched_modalities: list[str] = Field(
        description="Modalities that matched the user's concerns"
    )
    bio_excerpt: str = Field(description="Relevant excerpt from bio")
    source_url: str

    # Scoring breakdown (transparency for debugging)
    composite_score: float
    modality_score: float
    semantic_score: float
    bm25_score: float
    rating_score: float

    # Why this therapist was recommended
    match_explanation: str = Field(
        description="Structured debug explanation (internal signals)"
    )
    narrative_explanation: str = Field(
        default="",
        description="User-facing plain-English explanation of why this therapist was ranked here",
    )


class SearchResponse(BaseModel):
    query_id: UUID
    results: list[TherapistResult]
    total_candidates: int = Field(description="Total therapists before ranking")
    filtered_count: int = Field(description="After applying hard filters")
    extracted_intent: ExtractedQueryIntent
    cache_hit: bool = False
    latency_ms: float
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    filter_relaxation_note: str = Field(
        default="",
        description="Non-empty when filters were progressively relaxed to find enough results",
    )


class FeedbackRequest(BaseModel):
    query_id: UUID
    therapist_id: UUID
    rating: int = Field(..., ge=1, le=5)
    rank_position: Optional[int] = Field(None, ge=1, description="1-based rank of this therapist in the result list")
    event_type: str = Field("explicit", description="'explicit' for thumbs, 'profile_view' for link clicks")
    booked_appointment: Optional[bool] = None
    feedback_text: Optional[str] = Field(None, max_length=500)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
