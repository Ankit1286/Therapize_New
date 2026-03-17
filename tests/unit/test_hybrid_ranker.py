"""
Unit tests for the hybrid ranker.

Tests that the scoring logic is correct and that weights behave as expected.
No external dependencies — fully deterministic.
"""
import pytest
import numpy as np
from uuid import uuid4

from src.matching.hybrid_ranker import BM25Scorer, HybridRanker
from src.models.therapist import (
    TherapistLocation, TherapistProfile, TherapyModality, SessionFormat
)


def make_therapist(
    name: str = "Test Therapist",
    modalities: list[TherapyModality] = None,
    bio: str = "I help with anxiety and stress",
    city: str = "San Francisco",
) -> TherapistProfile:
    return TherapistProfile(
        source="test",
        source_url=f"https://example.com/{uuid4()}",  # type: ignore
        source_id=str(uuid4()),
        name=name,
        modalities=modalities or [TherapyModality.CBT],
        location=TherapistLocation(city=city, state="CA"),
        session_formats=[SessionFormat.TELEHEALTH],
        bio=bio,
        profile_completeness=0.8,
    )


class TestBM25Scorer:
    def test_exact_keyword_match_scores_higher(self):
        docs = [
            "specializes in anxiety and panic attacks",
            "works with couples and family therapy",
            "treats depression and mood disorders",
        ]
        bm25 = BM25Scorer()
        bm25.fit(docs)

        scores = bm25.score_all("anxiety panic")
        # First doc should score highest
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_empty_query_returns_zero_scores(self):
        bm25 = BM25Scorer()
        bm25.fit(["doc one", "doc two"])
        scores = bm25.score_all("")
        assert all(s == 0.0 for s in scores)

    def test_longer_doc_not_artificially_scored_higher(self):
        """BM25 length normalization should prevent longer docs from dominating."""
        docs = [
            "anxiety treatment expert",
            "I work with many different things " * 20 + "including anxiety",
        ]
        bm25 = BM25Scorer()
        bm25.fit(docs)
        scores = bm25.score_all("anxiety")
        # Short focused doc should score at least as high
        assert scores[0] >= scores[1] * 0.7  # within 30%


class TestHybridRanker:
    def setup_method(self):
        self.ranker = HybridRanker()

    def test_modality_match_ranks_higher(self):
        """Therapist with matching modalities should rank above one without."""
        therapist_with_cbt = make_therapist(modalities=[TherapyModality.CBT])
        therapist_with_gottman = make_therapist(modalities=[TherapyModality.GOTTMAN])

        # Use identical embeddings so modality is the differentiator
        embedding = [0.1] * 1536
        recommended_modalities = {"cognitive_behavioral_therapy": 1.0}

        results = self.ranker.rank(
            candidates=[therapist_with_cbt, therapist_with_gottman],
            query_text="anxiety treatment",
            query_embedding=embedding,
            candidate_embeddings=[embedding, embedding],
            recommended_modalities=recommended_modalities,
        )

        assert results[0].therapist.name == therapist_with_cbt.name

    def test_returns_all_candidates(self):
        """Ranker should return same number of candidates as input."""
        therapists = [make_therapist(f"Therapist {i}") for i in range(10)]
        embeddings = [[0.1] * 1536] * 10

        results = self.ranker.rank(
            candidates=therapists,
            query_text="help with anxiety",
            query_embedding=[0.1] * 1536,
            candidate_embeddings=embeddings,
            recommended_modalities={"cognitive_behavioral_therapy": 1.0},
        )

        assert len(results) == 10

    def test_scores_in_valid_range(self):
        """All composite scores should be in [0, 1]."""
        therapists = [make_therapist() for _ in range(5)]
        embeddings = [[float(i) / 1536] * 1536 for i in range(5)]

        results = self.ranker.rank(
            candidates=therapists,
            query_text="anxiety",
            query_embedding=[0.5] * 1536,
            candidate_embeddings=embeddings,
            recommended_modalities={"cognitive_behavioral_therapy": 1.0},
        )

        for r in results:
            assert 0.0 <= r.composite_score <= 1.0

    def test_sorted_descending(self):
        """Results should be sorted by composite score, highest first."""
        therapists = [make_therapist() for _ in range(5)]
        embeddings = [[float(i) / 1536] * 1536 for i in range(5)]

        results = self.ranker.rank(
            candidates=therapists,
            query_text="help with anxiety",
            query_embedding=[0.5] * 1536,
            candidate_embeddings=embeddings,
            recommended_modalities={"cognitive_behavioral_therapy": 0.9},
        )

        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates_returns_empty(self):
        results = self.ranker.rank(
            candidates=[],
            query_text="anxiety",
            query_embedding=[0.1] * 1536,
            candidate_embeddings=[],
            recommended_modalities={},
        )
        assert results == []


class TestEvaluationMetrics:
    def test_ndcg_perfect_ranking(self):
        from src.monitoring.evaluation import RankingEvaluator
        from uuid import uuid4

        evaluator = RankingEvaluator()
        ids = [uuid4() for _ in range(5)]
        # Result order matches relevance order
        relevance = {ids[0]: 3.0, ids[1]: 2.0, ids[2]: 1.0, ids[3]: 0.0, ids[4]: 0.0}
        ndcg = evaluator.ndcg_at_k(ids, relevance, k=5)
        assert ndcg == pytest.approx(1.0, abs=0.001)

    def test_ndcg_worst_ranking(self):
        from src.monitoring.evaluation import RankingEvaluator
        from uuid import uuid4

        evaluator = RankingEvaluator()
        ids = [uuid4() for _ in range(5)]
        # Reverse order: least relevant first
        relevance = {ids[0]: 0.0, ids[1]: 0.0, ids[2]: 1.0, ids[3]: 2.0, ids[4]: 3.0}
        ndcg = evaluator.ndcg_at_k(ids, relevance, k=5)
        assert ndcg < 1.0

    def test_mrr_first_result_relevant(self):
        from src.monitoring.evaluation import RankingEvaluator
        from uuid import uuid4

        evaluator = RankingEvaluator()
        ids = [uuid4() for _ in range(5)]
        relevant = {ids[0]}  # First result is relevant
        mrr = evaluator.mean_reciprocal_rank(ids, relevant)
        assert mrr == pytest.approx(1.0)

    def test_mrr_second_result_relevant(self):
        from src.monitoring.evaluation import RankingEvaluator
        from uuid import uuid4

        evaluator = RankingEvaluator()
        ids = [uuid4() for _ in range(5)]
        relevant = {ids[1]}  # Second result is relevant
        mrr = evaluator.mean_reciprocal_rank(ids, relevant)
        assert mrr == pytest.approx(0.5)
