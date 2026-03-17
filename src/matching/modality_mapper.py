"""
Modality Mapper — the domain intelligence layer.

Maps extracted emotional concerns to evidence-based therapy modalities.
This is the heart of the matching engine: it converts user language into
clinical concepts that can be matched against therapist profiles.

No LLM needed here — this is a deterministic lookup with fuzzy matching fallback.
Determinism is critical: same input must always produce same modality weights.
"""
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from src.models.therapist import TherapyModality

logger = logging.getLogger(__name__)

KNOWLEDGE_FILE = Path(__file__).parent / "knowledge" / "modality_map.json"


@lru_cache(maxsize=1)
def _load_knowledge_base() -> dict:
    """Load once and cache in memory. ~50KB, no need to reload on every call."""
    with open(KNOWLEDGE_FILE) as f:
        return json.load(f)


class ModalityMapper:
    """
    Deterministic, rule-based mapper from concerns to modalities.

    Design notes:
    - No LLM calls here: deterministic, zero cost, zero latency
    - Uses alias normalization to handle natural language variation
    - Returns weighted dict so the ranker can do nuanced scoring
    - Fallback: if concern is unknown, log it for dataset expansion
    """

    def __init__(self):
        self._kb = _load_knowledge_base()
        self._aliases: dict[str, str] = self._kb.get("_aliases", {})
        self._concerns: dict[str, dict[str, float]] = {
            k: v for k, v in self._kb.items()
            if not k.startswith("_")
        }

    def normalize_concern(self, concern: str) -> str:
        """
        Normalize free-text concern to a canonical key.
        e.g., "panic attacks" → "panic_disorder", "can't sleep" → "insomnia"
        """
        concern_lower = concern.lower().strip()
        # Direct alias lookup
        if concern_lower in self._aliases:
            return self._aliases[concern_lower]
        # Slug-ify and try direct match
        slug = re.sub(r"[^a-z0-9]+", "_", concern_lower).strip("_")
        if slug in self._concerns:
            return slug
        # Partial match: if any known concern is a substring
        for known in self._concerns:
            if known in slug or slug in known:
                return known
        logger.warning("Unknown concern '%s' — no modality mapping found", concern)
        return concern  # return as-is; caller handles None gracefully

    def get_modality_weights(
        self,
        concerns: list[str],
        concern_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Given a list of concerns, return aggregated modality weights.

        Args:
            concerns: List of concern strings (normalized or raw)
            concern_weights: How prominent each concern is (1.0 = primary concern)
                             If None, all concerns treated equally.

        Returns:
            Dict mapping modality name → aggregate weight (0-1)

        Example:
            concerns = ["anxiety", "insomnia"]
            → {
                "cognitive_behavioral_therapy": 0.9,  (avg of 1.0 + 0.8)
                "acceptance_commitment_therapy": 0.85,
                "mindfulness_based": 0.75,
                "cbt_insomnia": 0.5,   (only from insomnia)
                ...
              }
        """
        if not concerns:
            return {}

        if concern_weights is None:
            concern_weights = {c: 1.0 for c in concerns}

        # Aggregate modality scores across all concerns
        accumulated: dict[str, list[float]] = {}

        for concern in concerns:
            normalized = self.normalize_concern(concern)
            modality_scores = self._concerns.get(normalized, {})
            concern_weight = concern_weights.get(concern, 1.0)

            for modality, strength in modality_scores.items():
                weighted_score = strength * concern_weight
                if modality not in accumulated:
                    accumulated[modality] = []
                accumulated[modality].append(weighted_score)

        if not accumulated:
            return {}

        # Aggregate: use max to avoid penalizing therapists who cover many concerns
        # (vs average which would dilute scores when one concern has no match)
        result = {
            modality: max(scores)
            for modality, scores in accumulated.items()
        }

        # Normalize to [0, 1]
        max_score = max(result.values()) if result else 1.0
        return {k: v / max_score for k, v in result.items()}

    def score_therapist_modalities(
        self,
        therapist_modalities: list[TherapyModality],
        recommended_weights: dict[str, float],
    ) -> float:
        """
        Score a therapist based on how well their modalities match the recommendations.

        Uses a weighted intersection: therapists who have the most strongly
        recommended modalities score higher.

        Returns: float in [0, 1]
        """
        if not therapist_modalities or not recommended_weights:
            return 0.0

        therapist_modality_values = {m.value for m in therapist_modalities}

        # Sum of weights for modalities the therapist has
        matched_weight = sum(
            weight
            for modality, weight in recommended_weights.items()
            if modality in therapist_modality_values
        )

        # Normalize by the maximum possible score (if therapist had all recommended)
        max_possible = sum(recommended_weights.values())
        if max_possible == 0:
            return 0.0

        return min(1.0, matched_weight / max_possible)

    def get_all_concerns(self) -> list[str]:
        """Returns all known concern keys (useful for autocomplete/validation)."""
        return list(self._concerns.keys())

    def get_top_modalities_for_concerns(
        self,
        concerns: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Returns top-k modalities sorted by weight. Useful for UI explainability."""
        weights = self.get_modality_weights(concerns)
        sorted_modalities = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_modalities[:top_k]
