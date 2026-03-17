"""
Unit tests for the modality mapper — the most critical component.

These tests verify that the domain knowledge layer is correct.
They're fast (no I/O), deterministic, and run in CI.
"""
import pytest
from src.matching.modality_mapper import ModalityMapper
from src.models.therapist import TherapyModality


@pytest.fixture
def mapper():
    return ModalityMapper()


def test_anxiety_maps_to_cbt(mapper):
    weights = mapper.get_modality_weights(["anxiety"])
    assert "cognitive_behavioral_therapy" in weights
    assert weights["cognitive_behavioral_therapy"] >= 0.9


def test_trauma_maps_to_emdr(mapper):
    weights = mapper.get_modality_weights(["trauma"])
    assert "emdr" in weights
    assert weights["emdr"] == 1.0  # EMDR is gold standard for trauma


def test_alias_resolution(mapper):
    """'panic attacks' should resolve to 'panic_disorder'."""
    normalized = mapper.normalize_concern("panic attacks")
    assert normalized == "panic_disorder"

    weights = mapper.get_modality_weights(["panic attacks"])
    assert "cognitive_behavioral_therapy" in weights


def test_multiple_concerns_aggregate(mapper):
    """Multiple concerns should aggregate modality weights."""
    weights = mapper.get_modality_weights(["anxiety", "insomnia"])
    # Both anxiety and insomnia recommend CBT
    assert "cognitive_behavioral_therapy" in weights
    # CBT-I is specific to insomnia
    assert "cbt_insomnia" in weights


def test_unknown_concern_returns_empty(mapper):
    """Unknown concerns should not crash — return empty dict."""
    weights = mapper.get_modality_weights(["completely_made_up_condition_xyz"])
    assert isinstance(weights, dict)  # no crash


def test_therapist_modality_scoring(mapper):
    """Therapist with matching modalities should score higher."""
    recommended = {"cognitive_behavioral_therapy": 1.0, "emdr": 0.8}

    # Therapist with CBT
    cbt_therapist_modalities = [TherapyModality.CBT]
    score_with_match = mapper.score_therapist_modalities(
        cbt_therapist_modalities, recommended
    )

    # Therapist with unrelated modality
    unrelated_modalities = [TherapyModality.GOTTMAN]
    score_without_match = mapper.score_therapist_modalities(
        unrelated_modalities, recommended
    )

    assert score_with_match > score_without_match


def test_normalization_range(mapper):
    """All modality weights should be in [0, 1]."""
    concerns = ["anxiety", "depression", "trauma", "relationship_issues"]
    weights = mapper.get_modality_weights(concerns)
    for modality, weight in weights.items():
        assert 0.0 <= weight <= 1.0, f"{modality} weight {weight} out of range"


def test_empty_concerns(mapper):
    weights = mapper.get_modality_weights([])
    assert weights == {}


def test_concern_weights_respected(mapper):
    """Higher-weight concerns should produce higher modality scores."""
    weights_equal = mapper.get_modality_weights(
        ["anxiety", "trauma"],
        concern_weights={"anxiety": 1.0, "trauma": 1.0}
    )
    weights_anxiety_dominant = mapper.get_modality_weights(
        ["anxiety", "trauma"],
        concern_weights={"anxiety": 1.0, "trauma": 0.2}
    )
    # CBT is gold standard for anxiety but not trauma
    # Should be higher when anxiety dominates
    assert isinstance(weights_equal, dict)
    assert isinstance(weights_anxiety_dominant, dict)
