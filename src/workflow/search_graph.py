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
from src.matching.filter_engine import FilterEngine
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
        # Graph is compiled lazily on first use
        self._graph = None

    def _build_graph(self):
        """Build the LangGraph workflow. Called once."""
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(SearchState)

            graph.add_node("extract_intent", self._extract_intent)
            graph.add_node("compile_filters", self._compile_filters)
            graph.add_node("fetch_candidates", self._fetch_candidates)
            graph.add_node("rank_candidates", self._rank_candidates)
            graph.add_node("fallback_search", self._fallback_search)
            graph.add_node("build_response", self._build_response)

            graph.set_entry_point("extract_intent")
            graph.add_edge("extract_intent", "compile_filters")
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

        # Fallback: if too few results, relax filters
        if len(candidates) < 3:
            candidates, candidate_embeddings = await self._run_fallback(
                query_embedding, len(candidates)
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
        )

    async def _embed_query(self, text: str) -> list[float]:
        """Generate query embedding using local sentence-transformers."""
        from src.pipeline.embeddings import EmbeddingPipeline
        return await EmbeddingPipeline().embed_query(text or "therapist search")

    async def _run_fallback(
        self,
        query_embedding: list[float],
        current_count: int,
    ) -> tuple[list, list]:
        """
        Fallback when filters return too few results.
        Progressively relax: drop budget → drop insurance → California-wide.
        """
        logger.info("Fallback search triggered (current_count=%d)", current_count)
        # No filters except state=CA
        fallback_where = "state = $1 AND accepting_new_clients = $2 AND embedding IS NOT NULL"
        return await self._repository.search_candidates(
            where_clause=fallback_where,
            params=["CA", True],
            query_embedding=query_embedding,
            top_k=settings.rerank_top_k,
        )

    # LangGraph node methods (also usable without LangGraph)
    async def _extract_intent(self, state: SearchState) -> SearchState:
        state["intent"] = await self._query_processor.extract_intent(state["request"])
        return state

    async def _compile_filters(self, state: SearchState) -> SearchState:
        criteria = self._filter_engine.compile_filters(state["intent"])
        where, params = self._filter_engine.build_sql_where(criteria)
        state["where_clause"] = where
        state["sql_params"] = params
        return state

    async def _fetch_candidates(self, state: SearchState) -> SearchState:
        candidates, embeddings = await self._repository.search_candidates(
            state["where_clause"],
            state["sql_params"],
            state["query_embedding"],
        )
        state["candidates"] = candidates
        state["candidate_embeddings"] = embeddings
        state["filtered_count"] = len(candidates)
        return state

    async def _rank_candidates(self, state: SearchState) -> SearchState:
        modality_weights = self._modality_mapper.get_modality_weights(
            state["intent"].emotional_concerns
        )
        state["scored_results"] = self._ranker.rank(
            state["candidates"],
            state["request"].free_text,
            state["query_embedding"],
            state["candidate_embeddings"],
            modality_weights,
        )
        return state

    async def _fallback_search(self, state: SearchState) -> SearchState:
        candidates, embeddings = await self._run_fallback(
            state["query_embedding"], state["filtered_count"]
        )
        state["candidates"] = candidates
        state["candidate_embeddings"] = embeddings
        return state

    async def _build_response(self, state: SearchState) -> SearchState:
        return state

    @staticmethod
    def _should_fallback(state: SearchState) -> str:
        return "fallback" if len(state.get("candidates", [])) < 3 else "rank"

    @staticmethod
    def _to_therapist_result(scored: ScoredTherapist) -> TherapistResult:
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
        )
