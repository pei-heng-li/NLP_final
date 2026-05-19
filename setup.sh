# MELD ERC — Setup & Usage Notes
# ============================================================
# Environment: meow2, conda env "nlp" (Python 3.10)
# Base path:   /tmp2/b11902128/NLP/
# ============================================================

# ── 1. Copy scripts ──────────────────────────────────────────
# Put all four .py files and run_all.sh in /tmp2/b11902128/NLP/
export CUDA_VISIBLE_DEVICES=1
# ── 2. Extra dependencies ────────────────────────────────────
pip install scikit-learn librosa soundfile audioread ffmpeg-python

# librosa needs ffmpeg on PATH to read .mp4; install if missing:
# conda install -c conda-forge ffmpeg -y

# ── 3. Directory structure expected ──────────────────────────
# /tmp2/b11902128/NLP/
# ├── MELD.Raw/
# │   ├── train_sent_emo.csv
# │   ├── dev_sent_emo.csv
# │   ├── test_sent_emo.csv
# │   ├── train_splits/          ← dia{N}_utt{M}.mp4
# │   ├── dev_splits_complete/
# │   └── output_repeated_splits_test/
# ├── run_text_conditions.py
# ├── run_audio_conditions.py
# ├── run_baselines.py
# ├── evaluate.py
# └── run_all.sh

# ── 4. Run individual stages ─────────────────────────────────

# Stage 1–2: Text + Masked conditions
# python run_text_conditions.py --conditions T1 T2 T3 M1 M2 M3 --split test
python run_text_conditions.py --conditions T1 T2 T3 --split test

# Stage 3: Audio conditions  (feature extraction cached automatically)
# python run_audio_conditions.py --conditions A1 A2 A3 --split test

# Stage 4: Baselines
# B3 trains RoBERTa-base (~15 min on 1 GPU); model cached in roberta_meld/
# python run_baselines.py --baselines B1 B2 B3 --split test

# Stage 5: Evaluation
python evaluate.py --split test

# ── 5. Run everything at once ─────────────────────────────────
bash run_all.sh

# ── 6. Quick dry-run ──────────────────────────────────────────
# python run_text_conditions.py --conditions T1 T2 M1 --split test --dry_run

# ── 6. Output files ───────────────────────────────────────────
# results/{COND}_{split}.jsonl   ← raw predictions
# eval/summary_{split}.csv       ← macro/weighted F1 table
# eval/derived_{split}.json      ← masking drop, context recovery, etc.
# eval/change_rates_{split}.json ← prediction change rates
# eval/neutral_collapse_{split}.json
# eval/cm_{COND}_{split}.csv     ← confusion matrices
# eval/per_class_f1_{split}.csv
# eval/report_{COND}_{split}.txt ← sklearn classification report