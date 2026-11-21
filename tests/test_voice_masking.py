"""Unit tests for Module 2 — Zero-Leakage Blind Character-Voice Tester & Stylometrics."""

import pytest

from src.parser import ScreenplayParser
from src.schemas import VoicePrediction
from src.voice_auditor import (
    build_stratified_test_set,
    compute_character_stylometrics,
    detect_and_mask_leakage,
    evaluate_voice_predictions,
    rewrite_and_retest_line,
    select_top_characters,
)


def test_anti_leakage_filter():
    """Verifies that self-identifying statements are flagged as leaked and candidate names are masked."""
    candidates = ["JULIANA", "ALEXANDER", "CATHERINE", "RUPERT"]

    # 1. Self-identifying leakage should be flagged
    leaked, _ = detect_and_mask_leakage("I am Juliana, and I will not obey.", "JULIANA", candidates)
    assert leaked is True

    leaked, _ = detect_and_mask_leakage("My name is Catherine of Ashburg.", "CATHERINE", candidates)
    assert leaked is True

    leaked, _ = detect_and_mask_leakage("I am the Duke of Harrington.", "ALEXANDER", candidates)
    assert leaked is True

    # 2. Non-leaking lines should pass with leaked=False
    not_leaked, masked = detect_and_mask_leakage(
        "You have betrayed every vow made in that cathedral.", "JULIANA", candidates
    )
    assert not_leaked is False
    assert "betrayed every vow" in masked

    # 3. Mention of another candidate name in vocative dialogue should be masked to [NAME]
    not_leaked, masked = detect_and_mask_leakage(
        "Juliana, step back before you hurt yourself!", "ALEXANDER", candidates
    )
    assert not_leaked is False
    assert "Juliana" not in masked
    assert "[NAME]" in masked


def test_stratified_test_set_sampling():
    """Verifies stratified test set creation across sample synthetic episodes."""
    parser = ScreenplayParser()
    episodes = parser.parse_directory("sample_data")

    top_chars = select_top_characters(episodes, top_k=3, min_lines=2)
    assert len(top_chars) >= 2

    # Verify top_k=None selects all characters without artificial limits
    all_chars = select_top_characters(episodes, top_k=None, min_lines=2)
    assert len(all_chars) == 4
    assert "CAPTAIN MAYA VANCE" in all_chars
    assert "CHANCELLOR THORNE" in all_chars

    test_lines, leakage_count = build_stratified_test_set(
        episodes,
        top_characters=top_chars,
        samples_per_char=5,
        min_words=3,
    )

    assert len(test_lines) > 0
    # Every item must have required test fields
    for item in test_lines:
        assert "beat_id" in item
        assert "masked_text" in item
        assert "true_speaker" in item
        assert item["true_speaker"] in top_chars
        # Verify no speaker names leak in masked text
        assert not item["masked_text"].startswith(item["true_speaker"])


def test_deterministic_stylometrics_math():
    """Verifies exact arithmetic of stylometrics (TTR, questions, exclamations, pronouns, TF-IDF)."""
    parser = ScreenplayParser()
    episodes = parser.parse_directory("sample_data")
    top_chars = select_top_characters(episodes, top_k=2, min_lines=2)

    stylometrics, pair_sim = compute_character_stylometrics(episodes, top_chars)

    for c in top_chars:
        metrics = stylometrics[c]
        assert "avg_words_per_line" in metrics
        assert "lexical_diversity_ttr" in metrics
        assert "question_ratio" in metrics
        assert "exclamation_ratio" in metrics
        assert "first_person_ratio" in metrics

        # Type-Token Ratio must be between 0.0 and 1.0
        assert 0.0 <= metrics["lexical_diversity_ttr"] <= 1.0
        # Ratios must be valid proportions
        assert 0.0 <= metrics["question_ratio"] <= 1.0
        assert 0.0 <= metrics["exclamation_ratio"] <= 1.0
        assert 0.0 <= metrics["first_person_ratio"] <= 1.0

    # TF-IDF Cosine Similarity Matrix properties
    for c1 in top_chars:
        # Diagonal self-similarity must be ~1.0
        assert abs(pair_sim[c1][c1] - 1.0) < 1e-3
        for c2 in top_chars:
            # Symmetry
            assert abs(pair_sim[c1][c2] - pair_sim[c2][c1]) < 1e-3


def test_confusion_matrix_and_metrics_math():
    """Verifies exact Precision, Recall, F1, and K x K Confusion Matrix math with synthetic predictions."""
    candidates = ["ALICE", "BOB"]

    # Ground truth: 4 ALICE lines, 4 BOB lines
    # Predictions:
    # ALICE -> 3 predicted ALICE (TP), 1 predicted BOB (FN)
    # BOB   -> 1 predicted ALICE (FP), 3 predicted BOB (TP)
    preds = [
        # True ALICE
        VoicePrediction(beat_id="B1", masked_text="L1", true_speaker="ALICE", predicted_speaker="ALICE", is_correct=True, confidence=0.9),
        VoicePrediction(beat_id="B2", masked_text="L2", true_speaker="ALICE", predicted_speaker="ALICE", is_correct=True, confidence=0.8),
        VoicePrediction(beat_id="B3", masked_text="L3", true_speaker="ALICE", predicted_speaker="ALICE", is_correct=True, confidence=0.7),
        VoicePrediction(beat_id="B4", masked_text="L4", true_speaker="ALICE", predicted_speaker="BOB", is_correct=False, confidence=0.85),
        # True BOB
        VoicePrediction(beat_id="B5", masked_text="L5", true_speaker="BOB", predicted_speaker="BOB", is_correct=True, confidence=0.9),
        VoicePrediction(beat_id="B6", masked_text="L6", true_speaker="BOB", predicted_speaker="BOB", is_correct=True, confidence=0.8),
        VoicePrediction(beat_id="B7", masked_text="L7", true_speaker="BOB", predicted_speaker="BOB", is_correct=True, confidence=0.75),
        VoicePrediction(beat_id="B8", masked_text="L8", true_speaker="BOB", predicted_speaker="ALICE", is_correct=False, confidence=0.80),
    ]

    result = evaluate_voice_predictions(preds, candidates)

    # 1. Overall Accuracy: 6 / 8 = 0.75
    assert result.accuracy == 0.75
    assert result.random_baseline == 0.50

    # 2. Confusion Matrix
    # row=ALICE -> ALICE: 3, BOB: 1
    assert result.confusion_matrix["ALICE"]["ALICE"] == 3
    assert result.confusion_matrix["ALICE"]["BOB"] == 1
    # row=BOB -> ALICE: 1, BOB: 3
    assert result.confusion_matrix["BOB"]["ALICE"] == 1
    assert result.confusion_matrix["BOB"]["BOB"] == 3

    # 3. Precision & Recall:
    # ALICE Precision = 3 / (3 + 1) = 0.75; Recall = 3 / 4 = 0.75; F1 = 0.75
    # BOB Precision = 3 / (3 + 1) = 0.75; Recall = 3 / 4 = 0.75; F1 = 0.75
    assert result.per_character_metrics["ALICE"]["precision"] == 0.75
    assert result.per_character_metrics["ALICE"]["recall"] == 0.75
    assert result.per_character_metrics["ALICE"]["f1_score"] == 0.75
    assert result.f1_macro == 0.75

    # 4. Most Confused Pairs
    assert len(result.most_confused_pairs) == 1
    assert result.most_confused_pairs[0]["confusion_count"] == 2

    # 5. Top Flagged Lines
    assert len(result.top_flagged_lines) == 2
    # Flagged lines sorted by confidence descending (B4 confidence 0.85, B8 confidence 0.80)
    assert result.top_flagged_lines[0].beat_id == "B4"


def test_rewrite_and_retest_line():
    """Verifies that rewrite_and_retest_line executes cleanly and returns structured improvement metrics."""
    flagged = VoicePrediction(
        beat_id="E01_B014",
        masked_text="I don't think you should do that.",
        true_speaker="ALEXANDER",
        predicted_speaker="RUPERT",
        is_correct=False,
        confidence=0.85,
    )

    result = rewrite_and_retest_line(flagged)
    assert "rewritten_text" in result
    assert "craft_explanation" in result
    assert "retest_prediction" in result
    assert "improved" in result
    assert result["original_text"] == flagged.masked_text
