#!/usr/bin/env bash
# ============================================================
# MELD ERC – Full Experiment Runner
# Run from /tmp2/b11902128/NLP/
# Each stage can be run independently.
# ============================================================

set -e
SPLIT="test"   # change to dev for validation runs
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

run_py() {
    python "$@"
}

echo "========================================================"
echo " MELD ERC Experiment Pipeline"
echo " Split: $SPLIT"
echo "========================================================"
export CUDA_VISIBLE_DEVICES=1
# ── Stage 0: Dry-run sanity check ────────────────────────────
echo ""
echo "[Stage 0] Dry-run sanity check (T1, M1) ..."
# run_py "$SCRIPTS_DIR/run_text_conditions.py" --conditions T1 M1 --split "$SPLIT" --dry_run

# ── Stage 1: Text conditions T1 / T2 / T3 ────────────────────
echo ""
echo "[Stage 1] Text conditions: T1, T2, T3 ..."
run_py "$SCRIPTS_DIR/run_text_conditions.py" --conditions T1 T2 T3 --split "$SPLIT" --batch_size 16 --max_new_tokens 16

# ── Stage 2: Masked conditions M1 / M2 / M3 ──────────────────
echo ""
echo "[Stage 2] Masked conditions: M1, M2, M3 ..."
run_py "$SCRIPTS_DIR/run_text_conditions.py" --conditions M1 M2 M3 --split "$SPLIT" --batch_size 16 --max_new_tokens 16

# ── Stage 3: Audio conditions A1 / A2 / A3 ───────────────────
echo ""
echo "[Stage 3] Audio conditions: A1, A2, A3 ..."
run_py "$SCRIPTS_DIR/run_audio_conditions.py" --conditions A1 A2 A3 --split "$SPLIT" --batch_size 16 --max_new_tokens 16

# ── Stage 4: Audio conditions A1_mllm / A2_mllm / A3_mllm ───────────────────
echo ""
echo "[Stage 4] Audio conditions: A1_mllm, A2_mllm, A3_mllm ..."
python run_audio_mllm.py --conditions A1_mllm A2_mllm A3_mllm --split "$SPLIT"

# ── Stage 5: Baselines ────────────────────────────────────────
echo ""
echo "[Stage 5] Baselines: B1, B2, B3 ..."
run_py "$SCRIPTS_DIR/run_baselines.py" --baselines B1 B2 B3 --split "$SPLIT"

# ── Stage 6: Evaluation ───────────────────────────────────────
echo ""
echo "[Stage 6] Unified evaluation ..."
run_py "$SCRIPTS_DIR/evaluate.py" --split "$SPLIT"

echo ""
echo "========================================================"
echo " Done! Results in /tmp2/b11902128/NLP/results/"
echo " Eval   in /tmp2/b11902128/NLP/eval/"
echo "========================================================"