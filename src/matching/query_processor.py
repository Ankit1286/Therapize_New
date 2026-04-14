"""
Query Processor — LLM-powered query understanding.

Converts free-text user queries into structured ExtractedQueryIntent.
Uses instructor + Anthropic for structured output (Pydantic) with determinism.

Design decisions:
- temperature=0 for maximum determinism
- instructor enforces Pydantic schema via Anthropic tool-use — never fails to parse
- Fallback: if LLM fails, return empty intent (graceful degradation)
- Tracing: every call logged to LangSmith for cost/quality monitoring
"""
import logging
import time

import anthropic
import instructor
from langsmith import traceable
from pydantic import ValidationError

from src.config import get_settings
from src.models.query import ExtractedQueryIntent, SearchRequest, UserQuestionnaire
from src.models.therapist import TherapyModality, TherapistSpecialization
from src.monitoring.metrics import query_processor_latency, llm_tokens_used

logger = logging.getLogger(__name__)
settings = get_settings()

EXTRACTION_SYSTEM_PROMPT = """
You are a clinical AI assistant helping match users with therapists.
Extract structured information from the user's query.

Your job:
1. Identify emotional/psychological concerns (e.g., anxiety, grief, trauma, burnout)
2. Extract any logistical preferences implied by the query (location, insurance, budget, format)
3. Rate your confidence in the extraction

Rules:
- Be conservative: only include concerns that are clearly present
- For concerns, use plain clinical language (e.g. "anxiety", "depression", "trauma")
- If no logistical filters are implied, leave them null (don't guess)
- Think like a clinical professional, not a consumer
"""


class QueryProcessor:
    """
    LLM-based query understanding with structured output extraction.

    Uses instructor + Anthropic to guarantee valid JSON matching our
    Pydantic schema via tool-use — no prompt hacking needed.
    """

    def __init__(self):
        self._client = instructor.from_anthropic(
            anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature  # Always 0 for extraction
        self.last_tokens: int = 0  # token count from the most recent LLM call

    @traceable(name="query_processor.extract_intent")
    async def extract_intent(self, request: SearchRequest) -> ExtractedQueryIntent:
        """
        Extract structured intent from a search request.

        If no free-text provided, derives intent from questionnaire only.
        Falls back gracefully if LLM call fails.
        Token count is stored in self.last_tokens after each call.
        """
        if not request.free_text:
            self.last_tokens = 0
            return self._intent_from_questionnaire(request.questionnaire)

        start = time.monotonic()
        try:
            intent, tokens = await self._call_llm(request)
            self.last_tokens = tokens
            # Merge LLM-inferred filters with explicit questionnaire filters
            # Explicit questionnaire always wins over LLM inference
            intent = self._merge_filters(intent, request.questionnaire)
            latency_ms = (time.monotonic() - start) * 1000
            query_processor_latency.observe(latency_ms)
            return intent

        except Exception as exc:
            logger.error("Query processor failed: %s — falling back to questionnaire", exc)
            self.last_tokens = 0
            return self._intent_from_questionnaire(request.questionnaire)

    async def _call_llm(self, request: SearchRequest) -> tuple[ExtractedQueryIntent, int]:
        """
        Calls Anthropic via instructor with structured output schema.

        Key: response_model=ExtractedQueryIntent enforces valid JSON via tool-use.
        No parsing failures, no prompt engineering for JSON formatting.

        Returns: (intent, tokens_used)
        """
        user_message = f"User query: {request.free_text}"
        if request.questionnaire:
            q = request.questionnaire
            filters = []
            if q.city:
                filters.append(f"Location: {q.city}")
            if q.insurance:
                filters.append(f"Insurance: {q.insurance.value}")
            if q.max_budget_per_session:
                filters.append(f"Budget: up to ${q.max_budget_per_session}/session")
            if q.session_format:
                filters.append(f"Format: {q.session_format.value}")
            if filters:
                user_message += "\n\nUser-provided filters: " + ", ".join(filters)

        response, completion = await self._client.messages.create_with_completion(
            model=self._model,
            temperature=self._temperature,  # determinism
            max_tokens=settings.llm_max_tokens,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
            response_model=ExtractedQueryIntent,  # structured output
        )

        tokens = 0
        usage = completion.usage
        if usage:
            tokens = usage.input_tokens + usage.output_tokens
            llm_tokens_used.inc(tokens)

        return response, tokens

    @staticmethod
    def _intent_from_questionnaire(q: UserQuestionnaire) -> ExtractedQueryIntent:
        """No free text — build minimal intent from questionnaire alone."""
        return ExtractedQueryIntent(
            emotional_concerns=[],
            recommended_specializations=[],
            recommended_modalities=[],
            modality_weights={},
            inferred_filters=q,
            query_summary="Therapist search based on logistical preferences",
            confidence=1.0,  # High confidence — all filters are explicit
        )

    @staticmethod
    def _merge_filters(
        intent: ExtractedQueryIntent,
        explicit: UserQuestionnaire,
    ) -> ExtractedQueryIntent:
        """
        Explicit questionnaire answers override LLM inferences.
        This prevents the LLM from misinterpreting explicit constraints.
        """
        merged = intent.inferred_filters.model_copy()
        # Explicit values take precedence
        for field, value in explicit.model_dump(exclude_none=True).items():
            setattr(merged, field, value)
        intent.inferred_filters = merged
        return intent
