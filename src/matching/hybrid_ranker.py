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

# Plain-English names for modalities — keys match TherapyModality enum values.
_MODALITY_NAMES: dict[str, str] = {
    "cognitive_behavioral_therapy":  "Cognitive Behavioral Therapy (CBT)",
    "dialectical_behavior_therapy":  "Dialectical Behavior Therapy (DBT)",
    "emdr":                          "EMDR",
    "acceptance_commitment_therapy": "Acceptance and Commitment Therapy (ACT)",
    "psychodynamic":                 "psychodynamic therapy",
    "humanistic":                    "humanistic therapy",
    "mindfulness_based":             "mindfulness-based therapy",
    "somatic":                       "somatic therapy",
    "gottman_method":                "Gottman Method",
    "emotionally_focused_therapy":   "Emotionally Focused Therapy (EFT)",
    "exposure_therapy":              "exposure therapy",
    "cognitive_processing_therapy":  "Cognitive Processing Therapy (CPT)",
    "narrative_therapy":             "narrative therapy",
    "solution_focused":              "solution-focused therapy (SFBT)",
    "motivational_interviewing":     "motivational interviewing",
    "play_therapy":                  "play therapy",
    "art_therapy":                   "art therapy",
    "psychoanalytic":                "psychoanalytic therapy",
    "integrative":                   "integrative therapy",
    "cbt_insomnia":                  "CBT for Insomnia (CBT-I)",
    "trauma_informed":               "trauma-informed therapy",
    "family_systems":                "family systems therapy",
}


# Descriptive sentences per modality — embedded once at startup for bio-corroboration.
# Describes what the therapist actually does in practice, not just the name.
_MODALITY_DESCRIPTIONS: dict[str, str] = {
    "cognitive_behavioral_therapy": (
        "Identifying and restructuring negative thought patterns and cognitive distortions "
        "to change behaviour and improve emotional regulation."
    ),
    "dialectical_behavior_therapy": (
        "Teaching distress tolerance, emotion regulation, mindfulness, and interpersonal "
        "effectiveness skills to people with intense emotional responses."
    ),
    "emdr": (
        "Processing traumatic memories through bilateral stimulation such as eye movements "
        "to reduce their emotional charge and distress."
    ),
    "acceptance_commitment_therapy": (
        "Developing psychological flexibility by accepting difficult thoughts and feelings "
        "and committing to values-based action."
    ),
    "psychodynamic": (
        "Exploring unconscious patterns, early experiences, and relational dynamics "
        "to understand how the past shapes current emotions and behaviour."
    ),
    "humanistic": (
        "Supporting personal growth and self-actualisation through empathy, unconditional "
        "positive regard, and the therapeutic relationship."
    ),
    "mindfulness_based": (
        "Using present-moment awareness and non-judgmental attention to reduce stress, "
        "anxiety, and depressive relapse."
    ),
    "somatic": (
        "Working with bodily sensations, posture, and movement to release trauma and "
        "stress stored in the body."
    ),
    "gottman_method": (
        "Evidence-based couples therapy focused on building friendship, managing conflict, "
        "and creating shared meaning in relationships."
    ),
    "emotionally_focused_therapy": (
        "Restructuring emotional responses and attachment bonds in couples and individuals "
        "to build secure, loving relationships."
    ),
    "exposure_therapy": (
        "Gradually and systematically confronting feared situations or memories to reduce "
        "avoidance and anxiety responses."
    ),
    "cognitive_processing_therapy": (
        "Helping trauma survivors identify and challenge unhelpful beliefs about safety, "
        "trust, power, esteem, and intimacy formed after traumatic events."
    ),
    "narrative_therapy": (
        "Externalising problems and re-authoring personal stories to help clients build "
        "preferred identities separate from their difficulties."
    ),
    "solution_focused": (
        "Focusing on strengths, resources, and future goals rather than analysing problems "
        "and past causes."
    ),
    "motivational_interviewing": (
        "Evoking intrinsic motivation to change by exploring ambivalence and strengthening "
        "commitment to behavioural goals."
    ),
    "play_therapy": (
        "Using play as a natural medium for children to express feelings, process experiences, "
        "and develop coping skills."
    ),
    "art_therapy": (
        "Using creative art-making to express emotions, process trauma, and promote "
        "self-awareness and healing."
    ),
    "psychoanalytic": (
        "Deep exploration of unconscious conflicts, drives, and early developmental "
        "experiences through free association and dream analysis."
    ),
    "integrative": (
        "Drawing flexibly from multiple therapeutic traditions to tailor treatment to "
        "each individual client's needs."
    ),
    "cbt_insomnia": (
        "Addressing the thoughts and behaviours that perpetuate chronic insomnia through "
        "sleep restriction, stimulus control, and cognitive restructuring."
    ),
    "trauma_informed": (
        "Recognising the pervasive impact of trauma and prioritising physical, psychological, "
        "and emotional safety in all aspects of care."
    ),
    "family_systems": (
        "Understanding individuals within the context of their family relationships and "
        "working to change dysfunctional interaction patterns."
    ),
}


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
    narrative_explanation: str = ""


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
        # Pre-compute modality description embeddings once at startup.
        # Used for bio-corroboration: checks whether a matched modality is
        # genuinely reflected in the therapist's bio text.
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embedding_model)
        items = list(_MODALITY_DESCRIPTIONS.items())
        vecs = _model.encode([d for _, d in items], normalize_embeddings=True)
        self._modality_embeddings: dict[str, np.ndarray] = {
            k: vecs[i] for i, (k, _) in enumerate(items)
        }
        logger.info("HybridRanker: pre-computed embeddings for %d modalities", len(items))

    def rank(
        self,
        candidates: list[TherapistProfile],
        query_text: str,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
        recommended_modalities: dict[str, float],
        preferred_city: Optional[str] = None,
        emotional_concerns: Optional[list[str]] = None,
    ) -> list[ScoredTherapist]:
        """
        Full ranking pipeline.

        Args:
            candidates: Filtered therapist profiles
            query_text: Original user query text (for BM25)
            query_embedding: Query vector (from local sentence-transformers model)
            candidate_embeddings: Per-therapist embeddings (pre-computed, stored in pgvector)
            recommended_modalities: {modality_name: weight} from ModalityMapper
            preferred_city: When set, adds a 5th location signal (weight=0.15) that
                boosts therapists in that city. Reduces other weights proportionally.
            emotional_concerns: Concerns extracted from the user's query (e.g. ["anxiety", "grief"]).
                Used only for the user-facing narrative explanation, not for scoring.

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
        for i, therapist in enumerate(candidates):
            raw_score = self._modality_mapper.score_therapist_modalities(
                therapist.modalities, recommended_modalities
            )
            therapist_mod_values = {m.value for m in therapist.modalities}
            matched = [
                m for m, w in recommended_modalities.items()
                if m in therapist_mod_values and w > 0.5
            ]
            matched_modalities_list.append(matched)

            # Bio-corroboration: discount modality score if the bio doesn't
            # actually reflect the matched modalities (catches checkbox inflation).
            if matched and candidate_embeddings:
                bio_vec = np.array(candidate_embeddings[i], dtype=np.float32)
                bio_vec /= (np.linalg.norm(bio_vec) + 1e-8)
                corroboration = self._corroboration_factor(
                    matched, bio_vec, self._modality_embeddings
                )
            else:
                corroboration = 1.0

            # Breadth penalty: lightly discount therapists who list many modalities.
            penalty = self._breadth_penalty(len(therapist.modalities))

            modality_scores.append(raw_score * corroboration * penalty)

        # ── 2. Semantic Scores (cosine similarity) ─────────────────────────
        semantic_scores = self._compute_semantic_scores(
            query_vec, candidate_embeddings
        )

        # ── 3. BM25 Scores ────────────────────────────────────────────────
        bm25_scores = self._compute_bm25_scores(candidates, query_text)

        # ── 4. Quality Scores ─────────────────────────────────────────────
        quality_scores = [self._compute_quality_score(t) for t in candidates]

        # ── 5. Location Scores (binary: 1.0 if city matches, else 0.0) ───
        location_scores: list[float] = []
        if preferred_city:
            preferred_city_lower = preferred_city.lower()
            for therapist in candidates:
                therapist_city = (therapist.location.city or "").lower()
                location_scores.append(1.0 if therapist_city == preferred_city_lower else 0.0)

        # ── 6. Normalize each signal to [0, 1] ───────────────────────────
        modality_scores = self._normalize(modality_scores)
        semantic_scores = self._normalize(semantic_scores)
        bm25_scores = self._normalize(bm25_scores)
        quality_scores = self._normalize(quality_scores)
        # location_scores are already binary [0, 1] — no normalization needed

        # ── 7. Weighted Combination ───────────────────────────────────────
        if preferred_city:
            w_mod, w_sem, w_bm25, w_qual, w_loc = 0.35, 0.30, 0.13, 0.07, 0.15
        else:
            w_mod = settings.weight_modality
            w_sem = settings.weight_semantic
            w_bm25 = settings.weight_bm25
            w_qual = settings.weight_recency + settings.weight_rating
            w_loc = 0.0

        results = []
        for i, therapist in enumerate(candidates):
            loc_score = location_scores[i] if preferred_city else 0.0
            composite = (
                w_mod * modality_scores[i]
                + w_sem * semantic_scores[i]
                + w_bm25 * bm25_scores[i]
                + w_qual * quality_scores[i]
                + w_loc * loc_score
            )
            explanation = self._build_explanation(
                therapist, modality_scores[i], semantic_scores[i],
                bm25_scores[i], quality_scores[i], matched_modalities_list[i],
                recommended_modalities, preferred_city, loc_score,
            )
            try:
                narrative = self._build_narrative(
                    therapist, modality_scores[i], semantic_scores[i],
                    bm25_scores[i], quality_scores[i], matched_modalities_list[i],
                    recommended_modalities, preferred_city, loc_score,
                    emotional_concerns,
                )
            except Exception:
                logger.exception("_build_narrative failed for therapist %s", therapist.id)
                narrative = ""
            results.append(ScoredTherapist(
                therapist=therapist,
                composite_score=round(composite, 4),
                modality_score=round(modality_scores[i], 4),
                semantic_score=round(semantic_scores[i], 4),
                bm25_score=round(bm25_scores[i], 4),
                quality_score=round(quality_scores[i], 4),
                matched_modalities=matched_modalities_list[i],
                score_explanation=explanation,
                narrative_explanation=narrative,
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
    def _corroboration_factor(
        matched_modalities: list[str],
        bio_vec: np.ndarray,
        modality_embeddings: dict[str, np.ndarray],
    ) -> float:
        """Measure how well matched modalities are reflected in the therapist's bio.

        Computes average cosine similarity between each matched modality's description
        embedding and the therapist's bio embedding, then maps to [0.5, 1.0]:
          - 0.5 = no bio evidence for any matched modality
          - 1.0 = bio strongly reflects all matched modalities

        Returns 1.0 (no penalty) when there are no matched modalities or no cached
        modality embeddings, so unmatched therapists are not affected.
        """
        if not matched_modalities:
            return 1.0
        sims = [
            float(np.dot(bio_vec, modality_embeddings[m]))
            for m in matched_modalities if m in modality_embeddings
        ]
        if not sims:
            return 1.0
        avg_sim = sum(sims) / len(sims)
        return 0.5 + 0.5 * max(0.0, avg_sim)

    @staticmethod
    def _breadth_penalty(n_modalities: int) -> float:
        """Light penalty for therapists who list many modalities.

        Therapists with <= 5 modalities receive no penalty (multiplier = 1.0).
        Those with 20 listed modalities receive ~0.66×.
        Uses log scaling so the penalty grows slowly and doesn't over-punish.
        """
        TYPICAL = 5
        if n_modalities <= TYPICAL:
            return 1.0
        return math.log(TYPICAL + 3) / math.log(n_modalities + 3)

    @staticmethod
    def _build_narrative(
        therapist: TherapistProfile,
        modality_score: float,
        semantic_score: float,
        bm25_score: float,
        quality_score: float,
        matched_modalities: list[str],
        recommended_modalities: dict[str, float],
        preferred_city: Optional[str] = None,
        location_score: float = 0.0,
        emotional_concerns: Optional[list[str]] = None,
    ) -> str:
        """
        Human-readable explanation of why this therapist was ranked where they were.

        Covers up to five dimensions in plain English:
        1. Opening — who the therapist is and how their bio connects to the user
        2. Specialisation match — which approaches matched, why they help with the concerns
        3. Bio relevance — how closely the therapist's profile language fits the user's description
        4. Location context — whether the therapist is in the user's preferred city (if set)
        5. Credibility — rating and review count when notable

        Scores are normalised relative to the candidate pool: 1.0 = best, 0.0 = worst.
        """
        # Brief explanations of why a modality helps with common concern types.
        # Used to add one sentence of clinical context to the narrative.
        _CONCERN_MODALITY_WHY: dict[str, dict[str, str]] = {
            "anxiety": {
                "cognitive_behavioral_therapy": "CBT is one of the most well-researched treatments for anxiety, helping to identify and reframe the thought patterns that fuel worry.",
                "acceptance_commitment_therapy": "ACT helps with anxiety by building tolerance for uncertainty and redirecting focus toward what matters most.",
                "exposure_therapy": "Exposure therapy directly addresses anxiety by gradually reducing avoidance, which is what keeps anxiety going.",
                "mindfulness_based": "Mindfulness-based approaches help break the cycle of anxious rumination by building present-moment awareness.",
                "dialectical_behavior_therapy": "DBT provides concrete skills for managing overwhelming anxiety, including distress tolerance and emotion regulation.",
            },
            "depression": {
                "cognitive_behavioral_therapy": "CBT for depression targets the negative thought patterns and behavioural withdrawal that maintain low mood.",
                "psychodynamic": "Psychodynamic therapy can uncover the deeper roots of depression, including unresolved grief, relational patterns, and self-critical beliefs.",
                "acceptance_commitment_therapy": "ACT helps with depression by loosening the grip of self-critical thoughts and reconnecting with personal values.",
                "mindfulness_based": "Mindfulness-based approaches are particularly effective for preventing depressive relapse by changing the relationship with difficult thoughts.",
                "humanistic": "Humanistic therapy addresses depression by rebuilding self-worth and reconnecting with a sense of purpose.",
            },
            "trauma": {
                "emdr": "EMDR is specifically designed for trauma — it helps the brain process distressing memories so they lose their emotional charge.",
                "cognitive_processing_therapy": "CPT directly addresses the unhelpful beliefs about safety, trust, and self-blame that often follow traumatic experiences.",
                "somatic": "Somatic therapy is particularly valuable for trauma held in the body — physical tension, hypervigilance, and dissociation.",
                "trauma_informed": "A trauma-informed approach ensures that all aspects of therapy prioritise safety and avoid re-traumatisation.",
                "psychodynamic": "Psychodynamic therapy can help explore how past trauma shapes current relationships and emotional responses.",
            },
            "grief": {
                "humanistic": "Humanistic therapy creates a safe, non-judgmental space to sit with grief at your own pace.",
                "psychodynamic": "Psychodynamic work can help process complicated grief, especially when loss connects to earlier experiences.",
                "acceptance_commitment_therapy": "ACT helps with grief by making room for painful feelings rather than fighting them, while staying connected to what still matters.",
                "narrative_therapy": "Narrative therapy helps people honour what was lost and find ways to carry that story forward.",
            },
            "stress": {
                "mindfulness_based": "Mindfulness-based approaches directly target the physiological and cognitive patterns of chronic stress.",
                "acceptance_commitment_therapy": "ACT helps reduce stress-driven burnout by clarifying values and addressing overcommitment patterns.",
                "cognitive_behavioral_therapy": "CBT addresses the perfectionism and all-or-nothing thinking that often drives chronic stress.",
                "somatic": "Somatic work helps discharge the physical toll of prolonged stress held in the nervous system.",
            },
        }

        def _mod_name(m: str) -> str:
            return _MODALITY_NAMES.get(m, m.replace("_", " ").title())

        def _why_sentence(concerns: list[str], modality: str) -> str:
            """Return a 'why it helps' sentence for the best matching concern-modality pair."""
            for concern in concerns:
                why = _CONCERN_MODALITY_WHY.get(concern.lower(), {}).get(modality)
                if why:
                    return why
            return ""

        sentences: list[str] = []
        concerns = emotional_concerns or []
        concerns_phrase = " and ".join(concerns[:3]) if concerns else ""

        # ── 1. Opening — connect the therapist to the user ────────────────
        # Pull the first meaningful sentence from the bio if available.
        bio_opening = ""
        if therapist.bio:
            first_sentence = therapist.bio.split(".")[0].strip()
            if 20 < len(first_sentence) < 180:
                bio_opening = first_sentence + "."

        if bio_opening and semantic_score >= 0.5:
            sentences.append(bio_opening)

        # ── 2. Specialisation match ───────────────────────────────────────
        therapist_mods = {m.value for m in therapist.modalities}
        wanted_mods = {m for m, w in recommended_modalities.items() if w > 0.3}
        missed = wanted_mods - therapist_mods

        if matched_modalities and modality_score >= 0.5:
            named = [_mod_name(m) for m in matched_modalities[:3]]
            mod_str = ", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 else named[0]
            if concerns_phrase:
                sentences.append(
                    f"They practise {mod_str} — approaches well suited to {concerns_phrase}."
                )
            else:
                sentences.append(
                    f"They practise {mod_str}, which aligns closely with what you described."
                )
            # Add a "why it helps" sentence for the primary matched modality
            why = _why_sentence(concerns, matched_modalities[0])
            if why:
                sentences.append(why)
            if missed and modality_score < 0.8:
                missed_named = [_mod_name(m) for m in sorted(missed)[:2]]
                sentences.append(
                    f"They don't list {' or '.join(missed_named)}, which was also flagged for your concerns — "
                    f"worth asking about in a consultation."
                )
        elif matched_modalities:
            named = [_mod_name(m) for m in matched_modalities[:2]]
            sentences.append(
                f"They have some experience with {' and '.join(named)}, "
                f"though their coverage of recommended approaches is partial."
            )
        else:
            wanted_named = [_mod_name(m) for m in sorted(wanted_mods)[:3]]
            if wanted_named:
                sentences.append(
                    f"Their profile doesn't list the specific approaches most recommended for your concerns "
                    f"({', '.join(wanted_named)}), but their overall profile language is a strong fit."
                )
            else:
                sentences.append(
                    "Ranked based on how closely their overall profile matches what you described."
                )

        # ── 3. Bio / language relevance ───────────────────────────────────
        sem_high = semantic_score >= 0.6
        bm25_high = bm25_score >= 0.5
        bio_low = semantic_score < 0.35 and bm25_score < 0.3

        if sem_high and bm25_high:
            sentences.append(
                "Reading their bio, the themes and language closely reflect what you shared — "
                "the way they describe their work suggests a genuine fit."
            )
        elif sem_high:
            sentences.append(
                "Their therapeutic style and the way they describe their work aligns well with what you're going through, "
                "even if they use somewhat different language."
            )
        elif bm25_high:
            sentences.append(
                "Their bio touches on several of the themes you mentioned directly."
            )
        elif bio_low:
            sentences.append(
                "This match is driven more by their clinical specialisation than by direct language overlap with your description — "
                "reading their full profile is especially worthwhile here."
            )

        # ── 4. Location ───────────────────────────────────────────────────
        if preferred_city:
            if location_score == 1.0:
                sentences.append(f"They are based in {preferred_city}, matching your location.")
            else:
                therapist_city = therapist.location.city or "another city"
                formats = {f.value for f in therapist.session_formats}
                if "telehealth" in formats or "both" in formats:
                    sentences.append(
                        f"They are based in {therapist_city} rather than {preferred_city}, "
                        f"but offer online sessions so distance isn't a barrier."
                    )
                else:
                    sentences.append(
                        f"They are based in {therapist_city}, not {preferred_city} — "
                        f"confirm whether they see clients in your area before booking."
                    )

        # ── 5. Credibility ────────────────────────────────────────────────
        if therapist.rating and therapist.review_count >= 5:
            star = round(therapist.rating, 1)
            if star >= 4.5:
                sentences.append(
                    f"They are highly rated at {star}\u2605 across {therapist.review_count} client reviews."
                )
            elif star >= 4.0:
                sentences.append(
                    f"Rated {star}\u2605 from {therapist.review_count} reviews."
                )

        return " ".join(sentences)

    @staticmethod
    def _build_explanation(
        therapist: TherapistProfile,
        modality_score: float,
        semantic_score: float,
        bm25_score: float,
        quality_score: float,
        matched_modalities: list[str],
        recommended_modalities: dict[str, float],
        preferred_city: Optional[str] = None,
        location_score: float = 0.0,
    ) -> str:
        """
        Detailed debug explanation covering all ranking signals.

        Scores are normalized to [0,1] relative to the candidate pool:
        1.0 = best in batch, 0.0 = worst in batch.

        Format:
          [SCORES] composite breakdown with raw values
          [MODALITY] which modalities matched/missed and why
          [SEMANTIC] cosine similarity interpretation
          [BM25] keyword overlap interpretation
          [QUALITY] profile completeness / rating signal
          [LOCATION] city match boost (only when preferred_city is set)
        """
        if preferred_city:
            w_mod, w_sem, w_bm25, w_qual, w_loc = 0.35, 0.30, 0.13, 0.07, 0.15
        else:
            w_mod, w_sem, w_bm25, w_qual, w_loc = 0.40, 0.35, 0.15, 0.10, 0.0
        composite = (
            w_mod * modality_score
            + w_sem * semantic_score
            + w_bm25 * bm25_score
            + w_qual * quality_score
            + w_loc * location_score
        )

        lines = []

        # ── Composite breakdown ───────────────────────────────────────────
        lines.append(
            f"[SCORES] composite={composite:.3f} | "
            f"modality={modality_score:.3f}×{w_mod} | "
            f"semantic={semantic_score:.3f}×{w_sem} | "
            f"bm25={bm25_score:.3f}×{w_bm25} | "
            f"quality={quality_score:.3f}×{w_qual}"
        )

        # ── Modality signal ───────────────────────────────────────────────
        therapist_mods = {m.value for m in therapist.modalities}
        wanted_mods = {m for m, w in recommended_modalities.items() if w > 0.3}
        hit = therapist_mods & wanted_mods
        miss = wanted_mods - therapist_mods
        all_therapist_mods = ", ".join(sorted(therapist_mods)) or "none listed"
        if hit:
            lines.append(
                f"[MODALITY] matched={', '.join(sorted(hit))} | "
                f"missed={', '.join(sorted(miss)) or 'none'} | "
                f"therapist has: {all_therapist_mods}"
            )
        else:
            lines.append(
                f"[MODALITY] no modality overlap — "
                f"query wants: {', '.join(sorted(wanted_mods)) or 'none'} | "
                f"therapist has: {all_therapist_mods}"
            )

        # ── Semantic signal ───────────────────────────────────────────────
        if semantic_score >= 0.7:
            sem_label = "high — profile language closely mirrors query"
        elif semantic_score >= 0.4:
            sem_label = "moderate — partial thematic overlap with query"
        else:
            sem_label = "low — profile embedding diverges from query vector"
        lines.append(f"[SEMANTIC] {sem_label} (normalized={semantic_score:.3f})")

        # ── BM25 signal ───────────────────────────────────────────────────
        if bm25_score >= 0.6:
            bm25_label = "high — strong exact-term overlap"
        elif bm25_score >= 0.3:
            bm25_label = "moderate — some shared vocabulary"
        else:
            bm25_label = "low — few query terms appear in bio"
        lines.append(f"[BM25] {bm25_label} (normalized={bm25_score:.3f})")

        # ── Quality signal ────────────────────────────────────────────────
        rating_str = f"{therapist.rating:.1f}★ ({therapist.review_count} reviews)" \
            if therapist.rating else "no rating"
        lines.append(
            f"[QUALITY] {rating_str} | "
            f"completeness={therapist.profile_completeness:.2f} | "
            f"normalized={quality_score:.3f}"
        )

        # ── Location signal (only when preferred_city is set) ─────────────
        if preferred_city:
            therapist_city = therapist.location.city or "unknown"
            match_label = "matched" if location_score == 1.0 else "no match"
            lines.append(
                f"[LOCATION] preferred={preferred_city} | "
                f"therapist={therapist_city} | "
                f"{match_label} (boost={w_loc}×{location_score:.0f})"
            )

        return " || ".join(lines)
