"""
LangGraph search workflow — the orchestration layer.

Why LangGraph?
- Each step is a node: independently testable, debuggable
- Conditional edges handle failure gracefully (empty results → fallback)
- State is explicit: no hidden mutable state
- Full trace: every node's input/output is captured in LangSmith
- Checkpointing: can resume from any node if a step fails

Workflow:
  START
    ↓
  [extract_intent]   ← LLM call (query processor)
    ↓
  [compile_filters]  ← Rule-based (filter engine)
    ↓
  [fetch_candidates] ← DB query (SQL + vector ANN)
    ↓              ↙
  (empty?)  → [fallback_search] → ... → END
    ↓ (has results)
  [rank_candidates]  ← Hybrid ranker (BM25 + semantic + modality)
    ↓
  [build_response]   ← Format + log
    ↓
  END
"""
import logging
import time
from typing import Annotated, TypedDict

from langsmith import traceable

from src.config import get_settings
from src.matching.filter_engine import FilterEngine, FilterCriteria
from src.matching.hybrid_ranker import HybridRanker, ScoredTherapist
from src.matching.modality_mapper import ModalityMapper
from src.matching.query_processor import QueryProcessor
from src.models.query import (
    ExtractedQueryIntent,
    FeedbackRequest,
    SearchRequest,
    SearchResponse,
    TherapistResult,
)
from src.models.therapist import SessionFormat
from src.monitoring.metrics import (
    active_searches,
    cache_hits_total,
    cache_misses_total,
    candidates_after_filter,
    estimate_llm_cost,
    modality_match_score,
    search_latency,
    search_requests_total,
)
from src.monitoring.tracing import StructuredLogger
from src.storage.cache import search_cache
from src.storage.database import TherapistRepository

logger = logging.getLogger(__name__)
settings = get_settings()
structured_log = StructuredLogger(__name__)


class SearchState(TypedDict):
    """State passed between LangGraph nodes."""
    request: SearchRequest
    intent: ExtractedQueryIntent | None
    where_clause: str
    sql_params: list
    candidates: list
    candidate_embeddings: list
    scored_results: list[ScoredTherapist]
    total_candidates: int
    filtered_count: int
    query_embedding: list[float]
    error: str | None
    cache_hit: bool
    llm_tokens: int
    filter_relaxation_note: str


class SearchWorkflow:
    """
    Orchestrates the full search pipeline using LangGraph.

    Instantiate once at app startup — the graph is compiled once
    and reused for all requests (expensive to compile).
    """

    def __init__(self):
        self._query_processor = QueryProcessor()
        self._filter_engine = FilterEngine()
        self._ranker = HybridRanker()
        self._modality_mapper = ModalityMapper()
        self._repository = TherapistRepository()
        from src.pipeline.embeddings import EmbeddingPipeline
        self._embedder = EmbeddingPipeline()
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow. Called once."""
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(SearchState)

            graph.add_node("extract_intent", self._extract_intent)
            graph.add_node("embed_query", self._embed_query_node)
            graph.add_node("compile_filters", self._compile_filters)
            graph.add_node("fetch_candidates", self._fetch_candidates)
            graph.add_node("rank_candidates", self._rank_candidates)
            graph.add_node("fallback_search", self._fallback_search)
            graph.add_node("build_response", self._build_response)

            graph.set_entry_point("extract_intent")
            graph.add_edge("extract_intent", "embed_query")
            graph.add_edge("embed_query", "compile_filters")
            graph.add_edge("compile_filters", "fetch_candidates")
            graph.add_conditional_edges(
                "fetch_candidates",
                self._should_fallback,
                {"rank": "rank_candidates", "fallback": "fallback_search"},
            )
            graph.add_edge("rank_candidates", "build_response")
            graph.add_edge("fallback_search", "rank_candidates")
            graph.add_edge("build_response", END)

            return graph.compile()
        except ImportError:
            logger.warning("LangGraph not available — running pipeline sequentially")
            return None

    @traceable(name="therapize.search")
    async def run(self, request: SearchRequest) -> SearchResponse:
        """
        Main entry point. Checks cache first, then runs the workflow.
        """
        start_time = time.monotonic()
        active_searches.inc()
        search_requests_total.labels(status="started").inc()

        try:
            # ── Cache check ──────────────────────────────────────────────
            cache_key = search_cache.make_cache_key(
                request.free_text,
                request.questionnaire.model_dump_json(),
            )
            cached = await search_cache.get(cache_key)
            if cached:
                cache_hits_total.inc()
                latency_ms = (time.monotonic() - start_time) * 1000
                search_latency.observe(latency_ms)
                search_requests_total.labels(status="success").inc()
                cached["cache_hit"] = True
                cached["latency_ms"] = latency_ms
                return SearchResponse(**cached)

            cache_misses_total.inc()

            # ── Run pipeline ─────────────────────────────────────────────
            if self._graph is not None:
                result = await self._run_graph(request)
            else:
                result = await self._run_pipeline(request)

            # ── Cache result ──────────────────────────────────────────────
            await search_cache.set(cache_key, result.model_dump())

            # ── Metrics ───────────────────────────────────────────────────
            latency_ms = (time.monotonic() - start_time) * 1000
            result.latency_ms = latency_ms
            search_latency.observe(latency_ms)
            search_requests_total.labels(status="success").inc()

            structured_log.search_completed(
                query_id=str(request.query_id),
                latency_ms=latency_ms,
                num_results=len(result.results),
                cache_hit=False,
                llm_tokens=result.llm_tokens_used,
                top_score=result.results[0].composite_score if result.results else 0.0,
            )
            if latency_ms > 5000:
                structured_log.slow_query_warning(
                    str(request.query_id), latency_ms, "full_pipeline"
                )

            return result

        except Exception as exc:
            search_requests_total.labels(status="error").inc()
            structured_log.search_failed(
                str(request.query_id), str(exc), "unknown"
            )
            logger.exception("Search workflow failed for query %s", request.query_id)
            raise
        finally:
            active_searches.dec()

    async def _run_pipeline(self, request: SearchRequest) -> SearchResponse:
        """Run the pipeline sequentially (LangGraph optional)."""
        # Step 1: Extract intent
        intent = await self._query_processor.extract_intent(request)
        llm_tokens = self._query_processor.last_tokens

        # Step 2: Generate query embedding
        query_embedding = await self._embed_query(request.free_text)

        # Step 3: Compile filters → SQL
        filter_criteria = self._filter_engine.compile_filters(intent)
        where_clause, sql_params = self._filter_engine.build_sql_where(filter_criteria)

        # Step 4: Fetch candidates (SQL filter + vector ANN)
        candidates, candidate_embeddings = await self._repository.search_candidates(
            where_clause=where_clause,
            params=sql_params,
            query_embedding=query_embedding,
            top_k=settings.rerank_top_k * 5,  # fetch more than needed for BM25 re-ranking
        )

        # Progressive relaxation if too few results
        filter_relaxation_note = ""
        if len(candidates) < 3:
            candidates, candidate_embeddings, filter_relaxation_note = await self._progressive_relax(
                filter_criteria, query_embedding
            )

        filtered_count = len(candidates)
        candidates_after_filter.observe(filtered_count)

        # Step 5: Get modality weights from intent
        modality_weights = self._modality_mapper.get_modality_weights(
            intent.emotional_concerns
        )

        # Step 6: Hybrid ranking
        scored = self._ranker.rank(
            candidates=candidates,
            query_text=request.free_text,
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            recommended_modalities=modality_weights,
            preferred_city=filter_criteria.city,
        )

        top_results = scored[:settings.final_top_k]

        if top_results:
            modality_match_score.observe(top_results[0].modality_score)

        # Step 7: Build response
        results = [self._to_therapist_result(sr) for sr in top_results]

        # Step 8: Log search for feedback loop
        await self._repository.log_search(
            query_id=request.query_id,
            session_id=request.session_id,
            free_text=request.free_text,
            extracted_intent=intent.model_dump(),
            result_ids=[r.therapist_id for r in results],
            total_candidates=filtered_count,
            filtered_count=filtered_count,
            latency_ms=0,  # filled in by caller
            llm_tokens=0,
            cache_hit=False,
        )

        return SearchResponse(
            query_id=request.query_id,
            results=results,
            total_candidates=filtered_count,
            filtered_count=filtered_count,
            extracted_intent=intent,
            cache_hit=False,
            latency_ms=0,  # filled in by caller
            llm_tokens_used=llm_tokens,
            llm_cost_usd=estimate_llm_cost(llm_tokens, settings.llm_model),
            filter_relaxation_note=filter_relaxation_note,
        )

    async def _run_graph(self, request: SearchRequest) -> SearchResponse:
        """Run the pipeline through the compiled LangGraph."""
        initial_state: SearchState = {
            "request": request,
            "intent": None,
            "where_clause": "",
            "sql_params": [],
            "candidates": [],
            "candidate_embeddings": [],
            "scored_results": [],
            "total_candidates": 0,
            "filtered_count": 0,
            "query_embedding": [],
            "error": None,
            "cache_hit": False,
            "llm_tokens": 0,
            "filter_relaxation_note": "",
        }

        final_state = await self._graph.ainvoke(initial_state)

        top_results = final_state["scored_results"][:settings.final_top_k]
        results = [self._to_therapist_result(sr) for sr in top_results]

        if top_results:
            modality_match_score.observe(top_results[0].modality_score)
        candidates_after_filter.observe(final_state["filtered_count"])

        await self._repository.log_search(
            query_id=request.query_id,
            session_id=request.session_id,
            free_text=request.free_text,
            extracted_intent=final_state["intent"].model_dump() if final_state["intent"] else {},
            result_ids=[r.therapist_id for r in results],
            total_candidates=final_state["filtered_count"],
            filtered_count=final_state["filtered_count"],
            latency_ms=0,
            llm_tokens=0,
            cache_hit=False,
        )

        llm_tokens = final_state["llm_tokens"]
        return SearchResponse(
            query_id=request.query_id,
            results=results,
            total_candidates=final_state["filtered_count"],
            filtered_count=final_state["filtered_count"],
            extracted_intent=final_state["intent"],
            cache_hit=False,
            latency_ms=0,
            llm_tokens_used=llm_tokens,
            llm_cost_usd=estimate_llm_cost(llm_tokens, settings.llm_model),
            filter_relaxation_note=final_state.get("filter_relaxation_note", ""),
        )

    async def _embed_query(self, text: str) -> list[float]:
        """Generate query embedding using the shared EmbeddingPipeline instance."""
        return await self._embedder.embed_query(text or "therapist search")

    async def _progressive_relax(
        self,
        criteria: FilterCriteria,
        query_embedding: list[float],
    ) -> tuple[list, list, str]:
        """
        Progressively relax filters until at least 3 candidates are found.

        Relaxation order (preference priority, least personal first):
          1. Drop budget           — practical constraint, easiest to overlook
          2. Drop insurance        — practical constraint, self-pay always possible
          3. Drop city             — geography before personal preferences
          4. Drop language         — preference, but may be essential; explicit warning
          5. Drop gender           — most personal preference; explicit warning
          Final: bare CA search    — age_group is NEVER dropped (it is a requirement)

        Returns: (candidates, embeddings, human-readable note listing what was relaxed)
        """
        from copy import copy

        # Save original values for labels — criteria is mutated across steps
        orig_budget         = criteria.max_budget
        orig_session_format = criteria.session_format
        orig_insurance      = criteria.insurance
        orig_city           = criteria.city
        orig_gender         = criteria.preferred_gender
        orig_ethnicity      = criteria.preferred_ethnicity
        orig_language       = criteria.preferred_language
        orig_age_group      = criteria.age_group  # preserved in every step

        dropped_labels: list[str] = []

        async def _fetch(c: FilterCriteria, *, final: bool = False) -> tuple[list, list]:
            where, params = self._filter_engine.build_sql_where(c)
            top_k = settings.rerank_top_k if final else settings.rerank_top_k * 5
            return await self._repository.search_candidates(
                where_clause=where,
                params=params,
                query_embedding=query_embedding,
                top_k=top_k,
            )

        def _note() -> str:
            """Build the user-facing warning from the accumulated drop labels."""
            if not dropped_labels:
                return ""
            if len(dropped_labels) == 1:
                label_str = dropped_labels[0]
            else:
                label_str = ", ".join(dropped_labels[:-1]) + f" and {dropped_labels[-1]}"
            return (
                f"We couldn't find enough therapists with all your preferences, "
                f"so we broadened your search a little by removing {label_str}."
            )

        # ── Step 1: drop budget ───────────────────────────────────────────
        if criteria.max_budget:
            relaxed = copy(criteria)
            relaxed.max_budget = None
            candidates, embeddings = await _fetch(relaxed)
            dropped_labels.append(f"your budget limit (${orig_budget})")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at budget (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Step 2: drop session format ───────────────────────────────────
        # Relaxed before insurance so insurance-accepting (in-person) therapists
        # are reachable even when user started with telehealth.
        if criteria.session_format:
            relaxed = copy(criteria)
            relaxed.session_format = None
            candidates, embeddings = await _fetch(relaxed)
            dropped_labels.append("your session format preference")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at session_format (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Step 3: drop insurance ────────────────────────────────────────
        if criteria.insurance:
            relaxed = copy(criteria)
            relaxed.insurance = None
            candidates, embeddings = await _fetch(relaxed)
            insurance_name = orig_insurance.value.replace("_", " ").title()
            dropped_labels.append(f"your insurance filter ({insurance_name})")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at insurance (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Step 4: drop city ─────────────────────────────────────────────
        # Session format is already relaxed by this point so dropping city
        # is always safe — it simply broadens geography.
        if criteria.city:
            relaxed = copy(criteria)
            relaxed.city = None
            relaxed.zip_code = None
            candidates, embeddings = await _fetch(relaxed)
            dropped_labels.append(f"your location filter ({orig_city})")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at city (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Step 5: drop gender ───────────────────────────────────────────
        if criteria.preferred_gender:
            relaxed = copy(criteria)
            relaxed.preferred_gender = None
            candidates, embeddings = await _fetch(relaxed)
            dropped_labels.append(f"your gender preference ({orig_gender})")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at gender (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Step 6: drop ethnicity ────────────────────────────────────────
        if criteria.preferred_ethnicity:
            relaxed = copy(criteria)
            relaxed.preferred_ethnicity = None
            candidates, embeddings = await _fetch(relaxed)
            dropped_labels.append(f"your ethnicity preference ({orig_ethnicity})")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at ethnicity (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Step 7: drop language ─────────────────────────────────────────
        # Language is a functional requirement (can't do therapy without a shared
        # language) so it's kept as long as possible before the safety net.
        if criteria.preferred_language:
            relaxed = copy(criteria)
            relaxed.preferred_language = None
            candidates, embeddings = await _fetch(relaxed)
            dropped_labels.append(f"your language preference ({orig_language.title()})")
            criteria = relaxed
            if len(candidates) >= 3:
                logger.info("Progressive relax stopped at language (%d candidates)", len(candidates))
                return candidates, embeddings, _note()

        # ── Final safety net: CA + accepting_new_clients + age_group only ─
        # age_group is a requirement — never dropped.
        fallback = FilterCriteria(
            state="CA",
            accepting_new_clients=True,
            age_group=orig_age_group,
        )
        candidates, embeddings = await _fetch(fallback, final=True)
        note = _note()
        if not note:
            note = "We couldn't find enough therapists with your preferences, so we broadened your search to show you the best matches across California."
        logger.info("Progressive relax reached safety net (%d candidates)", len(candidates))
        return candidates, embeddings, note

    # LangGraph node methods
    async def _embed_query_node(self, state: SearchState) -> SearchState:
        """LangGraph node: generate and store the query embedding in state."""
        state["query_embedding"] = await self._embed_query(state["request"].free_text)
        return state

    async def _extract_intent(self, state: SearchState) -> SearchState:
        """LangGraph node: run the LLM query processor to extract structured intent from the free-text query."""
        state["intent"] = await self._query_processor.extract_intent(state["request"])
        state["llm_tokens"] = self._query_processor.last_tokens
        return state

    async def _compile_filters(self, state: SearchState) -> SearchState:
        """LangGraph node: translate extracted intent into a parameterized SQL WHERE clause."""
        criteria = self._filter_engine.compile_filters(state["intent"])
        where, params = self._filter_engine.build_sql_where(criteria)
        state["where_clause"] = where
        state["sql_params"] = params
        return state

    async def _fetch_candidates(self, state: SearchState) -> SearchState:
        """LangGraph node: execute the hybrid SQL + vector ANN query and store candidate profiles in state."""
        candidates, embeddings = await self._repository.search_candidates(
            state["where_clause"],
            state["sql_params"],
            state["query_embedding"],
            top_k=settings.rerank_top_k * 5,
        )
        state["candidates"] = candidates
        state["candidate_embeddings"] = embeddings
        state["filtered_count"] = len(candidates)
        return state

    async def _rank_candidates(self, state: SearchState) -> SearchState:
        """LangGraph node: run the hybrid ranker (modality + semantic + BM25 + quality) over the candidates."""
        modality_weights = self._modality_mapper.get_modality_weights(
            state["intent"].emotional_concerns
        )
        preferred_city = state["intent"].inferred_filters.city
        state["scored_results"] = self._ranker.rank(
            state["candidates"],
            state["request"].free_text,
            state["query_embedding"],
            state["candidate_embeddings"],
            modality_weights,
            preferred_city=preferred_city,
            emotional_concerns=state["intent"].emotional_concerns,
        )
        return state

    async def _fallback_search(self, state: SearchState) -> SearchState:
        """LangGraph node: triggered when the initial filter returns too few results; runs progressive filter relaxation."""
        criteria = self._filter_engine.compile_filters(state["intent"])
        candidates, embeddings, note = await self._progressive_relax(
            criteria,
            state["query_embedding"],
        )
        state["candidates"] = candidates
        state["candidate_embeddings"] = embeddings
        state["filtered_count"] = len(candidates)
        state["filter_relaxation_note"] = note
        return state

    async def _build_response(self, state: SearchState) -> SearchState:
        """LangGraph node: final pass-through node; response is assembled by the caller from state."""
        return state

    @staticmethod
    def _should_fallback(state: SearchState) -> str:
        """LangGraph conditional edge: route to fallback search if fewer than 3 candidates were found."""
        return "fallback" if len(state.get("candidates", [])) < 3 else "rank"

    @staticmethod
    def _to_therapist_result(scored: ScoredTherapist) -> TherapistResult:
        """Convert a ScoredTherapist (internal ranking model) to the TherapistResult returned in API responses."""
        t = scored.therapist
        fee_range = None
        if t.fee_min and t.fee_max:
            fee_range = f"${t.fee_min}-${t.fee_max}"
        elif t.fee_min:
            fee_range = f"from ${t.fee_min}"
        elif t.sliding_scale:
            fee_range = "Sliding scale available"

        return TherapistResult(
            therapist_id=t.id,
            name=t.name,
            credentials=t.credentials,
            city=t.location.city,
            session_formats=t.session_formats,
            accepts_insurance=t.accepts_insurance,
            sliding_scale=t.sliding_scale,
            fee_range=fee_range,
            matched_modalities=scored.matched_modalities,
            bio_excerpt=t.bio[:300] + "..." if len(t.bio) > 300 else t.bio,
            source_url=str(t.source_url),
            composite_score=scored.composite_score,
            modality_score=scored.modality_score,
            semantic_score=scored.semantic_score,
            bm25_score=scored.bm25_score,
            rating_score=scored.quality_score,
            match_explanation=scored.score_explanation,
            narrative_explanation=scored.narrative_explanation,
        )
