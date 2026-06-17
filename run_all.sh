#!/usr/bin/env bash
# ============================================================
# MELD ERC – Full Experiment Runner
# Run from /tmp2/b11902128/NLP_final/
#
# ── 換模型只改這裡 ──────────────────────────────────────────
#
# 純文字 / 音訊特徵 LLM (T, M, A 系列):
#   "meta-llama/Llama-3.2-3B-Instruct"
#   "meta-llama/Llama-3.1-8B-Instruct"
#   "Qwen/Qwen2-Audio-7B-Instruct"
#   "mistralai/Voxtral-Mini-3B-2507"
#   "nvidia/Nemotron-Mini-4B-Instruct"
#   "mistralai/Mistral-Nemo-Instruct-2407"
TEXT_MODEL="meta-llama/Llama-3.2-3B-Instruct"
#
# MLLM (A_mllm 系列): 只有 "qwen" 和 "voxtral" 支援
# 純文字模型請設成空字串 "" 跳過 Stage 4
MLLM_MODEL=""   # "" = 跳過
#
# ── 其他設定 ───────────────────────────────────────────────
SPLIT="test"
BATCH_SIZE=16
GPU=0            # 要用哪張 GPU
# ─────────────────────────────────────────────────────────────

set -e
export CUDA_VISIBLE_DEVICES=$GPU
export TRANSFORMERS_VERBOSITY=error
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# 從 model id 取最後一段當資料夾名稱 (e.g. Voxtral-Mini-3B-2507)
MODEL_SLUG="${TEXT_MODEL##*/}"
OUT_DIR="./data/${MODEL_SLUG}"

echo "========================================================"
echo " MELD ERC Experiment Pipeline"
echo " Text model : $TEXT_MODEL"
echo " MLLM model : $MLLM_MODEL"
echo " Output dir : $OUT_DIR"
echo " Split      : $SPLIT  |  GPU: $GPU"
echo "========================================================"

# ── Stage 1: Text conditions T1/T2/T3 + COT/DEF/FS ──────────
echo ""
echo "[Stage 1] Text conditions: T1 T2 T3 COT DEF FS ..."
python "$SCRIPTS_DIR/run_text_conditions.py" \
    --conditions T1 T2 T3 COT DEF FS \
    --split "$SPLIT" \
    --model_id "$TEXT_MODEL" \
    --output_dir "$OUT_DIR" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens 64

# ── Stage 2: Masked conditions M1/M2/M3 + MCOT/MDEF/MFS ─────
echo ""
echo "[Stage 2] Masked conditions: M1 M2 M3 MCOT MDEF MFS ..."
python "$SCRIPTS_DIR/run_text_conditions.py" \
    --conditions M1 M2 M3 MCOT MDEF MFS \
    --split "$SPLIT" \
    --model_id "$TEXT_MODEL" \
    --output_dir "$OUT_DIR" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens 64

# ── Stage 3: Audio (librosa features) A1/A2/A3 ───────────────
echo ""
echo "[Stage 3] Audio conditions (librosa): A1 A2 A3 ..."
python "$SCRIPTS_DIR/run_audio_conditions.py" \
    --conditions A1 A2 A3 \
    --split "$SPLIT" \
    --model_id "$TEXT_MODEL" \
    --output_dir "$OUT_DIR" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens 16

# ── Stage 4: MLLM audio A1/A2/A3 (只有 qwen / voxtral 支援) ─
if [ -n "$MLLM_MODEL" ]; then
    echo ""
    echo "[Stage 4] MLLM audio ($MLLM_MODEL): A1 A2 A3 ..."
    python "$SCRIPTS_DIR/run_mllm_audio.py" \
        --model "$MLLM_MODEL" \
        --conditions A1 A2 A3 \
        --split "$SPLIT" \
        --output_dir "$OUT_DIR"
else
    echo ""
    echo "[Stage 4] MLLM_MODEL is empty — skipping MLLM audio stage."
fi

# ── Stage 5: Baselines ────────────────────────────────────────
echo ""
echo "[Stage 5] Baselines: B1 B2 B3 ..."
python "$SCRIPTS_DIR/run_baselines.py" \
    --baselines B1 B2 B3 \
    --split "$SPLIT"

# ── Stage 6: Evaluation ───────────────────────────────────────
echo ""
echo "[Stage 6] Evaluation ..."
python "$SCRIPTS_DIR/evaluate.py" \
    --split "$SPLIT" \
    --results_dir "$OUT_DIR"

echo ""
echo "========================================================"
echo " Done!  Results → $OUT_DIR"
echo "========================================================"