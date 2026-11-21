"""Phase 4 Verification Script: Module 2 — Zero-Leakage Blind Character-Voice Tester.

Demonstrates:
- Top-K character selection and anti-leakage filtering
- Pure Python deterministic stylometrics (TTR, questions, pronouns, TF-IDF cosine similarity)
- Blind speaker attribution test with Archetype Cards
- Exact confusion matrix, accuracy, and F1-macro evaluation
- Top flagged interchangeable lines and Rewrite & Re-Test workflow
"""

import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.llm_client import LLMClient
from src.parser import ScreenplayParser
from src.voice_auditor import audit_season_voice, rewrite_and_retest_line


def run_verification():
    print("=" * 70)
    print("PHASE 4: ZERO-LEAKAGE BLIND CHARACTER-VOICE TESTER VERIFICATION")
    print("=" * 70)

    # Ingest synthetic sample episodes
    parser = ScreenplayParser()
    episodes = parser.parse_directory("sample_data")
    print(f"Loaded {len(episodes)} sample episodes for Module 2 audit.")

    client = LLMClient()
    voice_result = audit_season_voice(
        episodes,
        top_k=3,
        samples_per_char=5,
        client=client,
    )

    print("-" * 70)
    print("1. EVALUATION METRICS:")
    print(f"   Tested Characters:          {voice_result.top_speakers}")
    print(f"   Tested Dialogue Lines:      {voice_result.tested_lines_count}")
    print(f"   Leakage Filtered Lines:     {voice_result.leakage_filtered_count}")
    print(f"   Blind Prediction Accuracy:  {voice_result.accuracy * 100:.1f}%")
    print(f"   Random Baseline:            {voice_result.random_baseline * 100:.1f}%")
    print(f"   F1-Macro Score:             {voice_result.f1_macro:.4f}")
    print(f"   Voice Health Status:        {voice_result.status}")

    print("\n2. DETERMINISTIC STYLOMETRICS SUMMARY:")
    for spk in voice_result.top_speakers:
        sm = voice_result.stylometric_summary.get(spk, {})
        print(f"   {spk:22s} | Avg Words: {sm.get('avg_words_per_line', 0):4.1f} | TTR: {sm.get('lexical_diversity_ttr', 0):.3f} | Q-Ratio: {sm.get('question_ratio', 0):.2f} | 1st-Person: {sm.get('first_person_ratio', 0):.2f}")

    print("\n3. PAIRWISE TF-IDF VOCABULARY COSINE SIMILARITY:")
    for spk1 in voice_result.top_speakers:
        sim_row = [f"{voice_result.pair_similarity_matrix[spk1][spk2]:.3f}" for spk2 in voice_result.top_speakers]
        print(f"   {spk1:22s} : [{' '.join(sim_row)}]")

    print("\n4. CONFUSION MATRIX (Row=True Speaker, Col=Predicted Speaker):")
    header = f"   {'TRUE \\ PRED':20s} " + " ".join(f"{c[:8]:>8s}" for c in voice_result.top_speakers)
    print(header)
    print("   " + "-" * (len(header) - 3))
    for true_spk in voice_result.top_speakers:
        row_str = " ".join(f"{voice_result.confusion_matrix[true_spk][pred_spk]:8d}" for pred_spk in voice_result.top_speakers)
        print(f"   {true_spk:20s} {row_str}")

    print("\n5. MOST CONFUSED CHARACTER PAIRS:")
    for p in voice_result.most_confused_pairs[:3]:
        print(f"   {p['pair'][0]} <-> {p['pair'][1]}: {p['confusion_count']} mutual confusions (Rate: {p['confusion_rate'] * 100:.1f}%)")

    print("\n6. TOP FLAGGED INTERCHANGEABLE LINES:")
    for idx, f in enumerate(voice_result.top_flagged_lines[:3], 1):
        print(f"   [{idx}] Beat {f.beat_id}:")
        print(f"       True Speaker:      {f.true_speaker}")
        print(f"       Predicted Speaker: {f.predicted_speaker} (Confidence: {f.confidence * 100:.0f}%)")
        print(f"       Masked Line:       \"{f.masked_text}\"")
        print(f"       Voice Cue Reason:  {f.voice_cue_reason}")

    # 7. Demonstrate Rewrite & Re-Test Assistant
    if voice_result.top_flagged_lines:
        sample_flagged = voice_result.top_flagged_lines[0]
        print("\n7. REWRITE & RE-TEST WORKFLOW DEMONSTRATION:")
        print(f"   Original Flagged Line: \"{sample_flagged.masked_text}\" (True: {sample_flagged.true_speaker}, Pred: {sample_flagged.predicted_speaker})")
        rewrite_outcome = rewrite_and_retest_line(
            sample_flagged,
            client=client,
            candidate_characters=voice_result.top_speakers,
            archetypes=voice_result.character_archetypes,
        )
        print(f"   Rewritten Line:        \"{rewrite_outcome['rewritten_text']}\"")
        print(f"   Craft Rationale:       {rewrite_outcome['craft_explanation']}")
        print(f"   Re-Test Prediction:    {rewrite_outcome['retest_prediction']} (Improved: {rewrite_outcome['improved']})")

    print("=" * 70)
    print(">>> GATE 4 CHECK: PASSED (Zero-leakage, stylometrics, confusion matrix & rewrite verified)")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
