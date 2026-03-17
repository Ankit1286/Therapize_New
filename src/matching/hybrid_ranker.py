"""
Hybrid Ranker — the core scoring pipeline.

Combines four signals into a single composite score:
  1. Modality score   (0.40 weight) — domain-specific clinical match
  2. Semantic score   (0.35 weight) — embedding cosine similarity
  3. BM25 score       (0.15 weight) — keyword overlap
  4. Quality signals  (0.10 weight) — rating + recency

Why this architecture?
- BM25 alone: misses semantic similarity ("panic" ≠ "anxiety" to BM25)
- Vector alone: misses exact terms and modality domain knowledge
- Modality alone: ignores what the therapist actually says in their bio
- Hybrid: each signal covers another's blind spots

Latency budget: ~200ms for 1000 candidates
  - SQL filter: ~20ms
  - Embedding generation: ~50ms
  - Vector ANN search: ~30ms
  - BM25: ~10ms
  - Score combination: ~5ms
  - Score combination + sort: ~5ms
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import get_settings
from src.matching.modality_mapper import ModalityMapper
from src.models.therapist import TherapistProfile

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ScoredTherapist:
    therapist: TherapistProfile
    composite_score: float = 0.0
    modality_score: float = 0.0
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    quality_score: float = 0.0
    matched_modalities: list[str] = field(default_factory=list)
    score_explanation: str = ""


class BM25Scorer:
    """
    BM25 implementation optimized for therapist profile matching.
    BM25 parameters k1=1.5, b=0.75 are standard; tuned for ~300 word bios.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def fit(self, documents: list[str]) -> None:
        """Build BM25 index from corpus of documents."""
        self._corpus = [self._tokenize(doc) for doc in documents]
        self._n_docs = len(self._corpus)
        self._avgdl = sum(len(doc) for doc in self._corpus) / max(self._n_docs, 1)

        # Document frequency
        self._doc_freqs = defaultdict(int)
        for doc in self._corpus:
            for term in set(doc):
                self._doc_freqs[term] += 1

        # Inverse document frequency (IDF)
        self._idf = {
            term: math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1
            )
            for term, df in self._doc_freqs.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        """Score a document against a query."""
        query_terms = self._tokenize(query)
        doc = self._corpus[doc_idx]
        doc_len = len(doc)

        score = 0.0
        term_freqs = defaultdict(int)
        for term in doc:
            term_freqs[term] += 1

        for term in query_terms:
            if term not in self._idf:
                continue
            tf = term_freqs[term]
            idf = self._idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
            score += idf * numerator / denominator

        return score

    def score_all(self, query: str) -> list[float]:
        """Score all documents. Returns raw BM25 scores (not normalized)."""
        return [self.score(query, i) for i in range(self._n_docs)]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()


class HybridRanker:
    """
    Multi-signal ranker combining modality, semantic, BM25, and quality scores.

    Usage pattern (called once per search request):
    1. ranker.rank(candidates, query_text, query_embedding, recommended_modalities)
    2. Returns ScoredTherapist list sorted by composite_score descending
    """

    def __init__(self):
        self._modality_mapper = ModalityMapper()

    def rank(
        self,
        candidates: list[TherapistProfile],
        query_text: str,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        recommended_modalities: dict[str, float],
    ) -> list[ScoredTherapist]:
        """
        Full ranking pipeline.

        Args:
            candidates: Filtered therapist profiles
            query_text: Original user query text (for BM25)
            query_embedding: Query vector (from local sentence-transformers model)
            candidate_embeddings: Per-therapist embeddings (pre-computed, stored in pgvector)
            recommended_modalities: {modality_name: weight} from ModalityMapper

        Returns:
            List of ScoredTherapist sorted by composite_score, descending
        """
        if not candidates:
            return []

        n = len(candidates)
        query_vec = np.array(query_embedding, dtype=np.float32)

        # ── 1. Modality Scores ────────────────────────────────────────────
        modality_scores = []
        matched_modalities_list = []
        for therapist in candidates:
            score = self._modality_mapper.score_therapist_modalities(
                therapist.modalities, recommended_modalities
            )
            modality_scores.append(score)
            # Track which modalities matched (for explainability)
            therapist_mod_values = {m.value for m in therapist.modalities}
            matched = [
                m for m, w in recommended_modalities.items()
                if m in therapist_mod_values and w > 0.5
            ]
            matched_modalities_list.append(matched)

        # ── 2. Semantic Scores (cosine similarity) ─────────────────────────
        semantic_scores = self._compute_semantic_scores(
            query_vec, candidate_embeddings
        )

        # ── 3. BM25 Scores ────────────────────────────────────────────────
        bm25_scores = self._compute_bm25_scores(candidates, query_text)

        # ── 4. Quality Scores ─────────────────────────────────────────────
        quality_scores = [self._compute_quality_score(t) for t in candidates]

        # ── 5. Normalize each signal to [0, 1] ───────────────────────────
        modality_scores = self._normalize(modality_scores)
        semantic_scores = self._normalize(semantic_scores)
        bm25_scores = self._normalize(bm25_scores)
        quality_scores = self._normalize(quality_scores)

        # ── 6. Weighted Combination ───────────────────────────────────────
        w_mod = settings.weight_modality
        w_sem = settings.weight_semantic
        w_bm25 = settings.weight_bm25
        w_qual = settings.weight_recency + settings.weight_rating

        results = []
        for i, therapist in enumerate(candidates):
            composite = (
                w_mod * modality_scores[i]
                + w_sem * semantic_scores[i]
                + w_bm25 * bm25_scores[i]
                + w_qual * quality_scores[i]
            )
            explanation = self._build_explanation(
                therapist, modality_scores[i], semantic_scores[i],
                bm25_scores[i], matched_modalities_list[i]
            )
            results.append(ScoredTherapist(
                therapist=therapist,
                composite_score=round(composite, 4),
                modality_score=round(modality_scores[i], 4),
                semantic_score=round(semantic_scores[i], 4),
                bm25_score=round(bm25_scores[i], 4),
                quality_score=round(quality_scores[i], 4),
                matched_modalities=matched_modalities_list[i],
                score_explanation=explanation,
            ))

        results.sort(key=lambda x: x.composite_score, reverse=True)

        logger.debug(
            "Ranked %d candidates. Top score: %.4f (modality=%.2f, semantic=%.2f)",
            n,
            results[0].composite_score if results else 0,
            results[0].modality_score if results else 0,
            results[0].semantic_score if results else 0,
        )

        return results

    def _compute_semantic_scores(
        self,
        query_vec: np.ndarray,
        candidate_embeddings: list[list[float]],
    ) -> list[float]:
        """
        Batch cosine similarity computation using numpy for speed.
        Much faster than computing one-by-one for 1000 candidates.
        """
        if not candidate_embeddings:
            return []

        candidate_matrix = np.array(candidate_embeddings, dtype=np.float32)

        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        candidate_matrix = candidate_matrix / (candidate_norms + 1e-8)

        # Dot product = cosine similarity (after normalization)
        scores = candidate_matrix @ query_norm
        return scores.tolist()

    def _compute_bm25_scores(
        self,
        candidates: list[TherapistProfile],
        query_text: str,
    ) -> list[float]:
        """Build a BM25 index over candidate documents and score against query."""
        documents = [t.to_bm25_document() for t in candidates]
        bm25 = BM25Scorer()
        bm25.fit(documents)
        return bm25.score_all(query_text)

    @staticmethod
    def _compute_quality_score(therapist: TherapistProfile) -> float:
        """
        Quality signal combining rating and profile completeness.
        Uses log-dampened rating count to avoid favoring therapists with
        many low-quality reviews over those with few high-quality ones.
        """
        rating_score = 0.0
        if therapist.rating is not None:
            # Normalize 5-star rating to [0, 1]
            normalized_rating = (therapist.rating - 1) / 4
            # Dampen by review count confidence
            confidence = math.log1p(therapist.review_count) / math.log1p(100)
            rating_score = normalized_rating * min(1.0, confidence)

        completeness_score = therapist.profile_completeness
        return 0.6 * rating_score + 0.4 * completeness_score

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """Min-max normalization to [0, 1]. Handles all-zero edge case."""
        if not scores:
            return scores
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)  # all tied — equal ranking contribution
        return [(s - min_s) / (max_s - min_s) for s in scores]

    @staticmethod
    def _build_explanation(
        therapist: TherapistProfile,
        modality_score: float,
        semantic_score: float,
        bm25_score: float,
        matched_modalities: list[str],
    ) -> str:
        """Human-readable explanation for UI/debugging."""
        parts = []
        if matched_modalities:
            readable_modalities = [m.replace("_", " ").title() for m in matched_modalities[:3]]
            parts.append(f"Specializes in {', '.join(readable_modalities)}")
        if semantic_score > 0.7:
            parts.append("strong profile match for your specific concerns")
        if bm25_score > 0.6:
            parts.append("uses relevant therapeutic language")
        if not parts:
            parts.append("matches your search criteria")
        return "; ".join(parts).capitalize() + "."
