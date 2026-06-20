# Do LLMs Understand Conversational Emotion or Rely on Lexical Cues?
### A Case Study on MELD

A diagnostic study of Emotion Recognition in Conversation (ERC) that asks whether
open-weight LLMs *infer* emotion from dialogue and prosody, or simply *match*
explicit emotion words to labels. We evaluate six open-weight instruction-tuned
models on MELD across a grid of conditions that independently vary three
information sources — explicit lexical cues, dialogue context, and acoustic
information — using a lexicon-driven masking protocol with a built-in control
group.

Po-Ching Chen, Pei-heng Li, Yi-Hsuan Su · National Taiwan University

---

## Overview

The core idea is **controlled subtraction**. For every test utterance we build a
masked counterpart in which lexicon-listed emotion words are replaced by
`[MASK]`, then measure how performance changes as we add or remove each
information source. Because not every utterance contains a maskable word, the
design yields a built-in **no-mask control group** (utterances whose text is
identical across conditions), which separates genuine lexical-removal effects
from prompt-framing artifacts.

Key findings:
- **Substantial lexical reliance** — masking emotion words drops Macro-F1 /
  Weighted-F1 by ~0.06 on average; dialogue context does not repair it.
- **Valence collapse to surprise** — masking-induced errors do not retreat to
  the majority class *neutral* but collapse disproportionately toward
  *surprise* (~39% of all flips).
- **Persistent Error dominates** — most errors are wrong with *and* without
  context, so the "does context help?" question concerns only a minority of
  examples.
- **Audio repairs anger** — native audio's benefit is concentrated almost
  entirely on *anger*, near-zero in text for Qwen2-Audio but recovered an
  order of magnitude with the raw waveform.

---

## Dataset: MELD

We use the [Multimodal EmotionLines Dataset (MELD)](https://affective-meld.github.io/),
derived from the TV series *Friends*. It contains 1,433 dialogues (~13,000
utterances), each annotated with a speaker, dialogue position, one of seven
emotion labels (*anger, disgust, fear, joy, neutral, sadness, surprise*), and an
aligned audio/video clip. We run all experiments on the official test split
(280 dialogues / 2,610 utterances).

### Download and placement

```bash
# Download and extract the raw MELD release
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
tar -xvf MELD.Raw.tar.gz

# Extract the per-split archives inside MELD.Raw/
cd MELD.Raw
tar -xzf dev.tar.gz
tar -xzf test.tar.gz
tar -xzf train.tar.gz
cd ..
```

After extraction, the project root should contain a `MELD.Raw/` directory with
the per-split CSV annotation files (`train_sent_emo.csv`, `dev_sent_emo.csv`,
`test_sent_emo.csv`) and the corresponding video clip folders. The scripts read
annotations from `./MELD.Raw/` by default.

---

## Environment

No `sudo` required; tested on a single GPU (e.g. RTX 4090, 24 GB).

```bash
# Core dependencies
pip install pandas
pip install transformers accelerate torch

# Some MLLMs (e.g. Voxtral) require a recent transformers build
pip install git+https://github.com/huggingface/transformers

# Audio feature extraction (A-series) and decoding
pip install librosa soundfile
# ffmpeg is required to extract audio from MELD .mp4 clips
#   (install via your system package manager, e.g. conda install ffmpeg)
```

Additional notes:
- **Voxtral** uses the `mistral_common` tekken tokenizer; install it if you
  evaluate Voxtral (`pip install mistral_common`).
- Models are downloaded automatically from the Hugging Face Hub on first use;
  gated checkpoints (e.g. Llama) require `huggingface-cli login`.

---

## Repository structure

```
NLP_final/
├── README.md
├── run_all.sh                  # one-shot pipeline runner (set the model at the top)
├── run_text_conditions.py      # T1/T2/T3, M1/M2/M3, COT/DEF/FS, MCOT/MDEF/MFS
├── run_audio_conditions.py     # A1/A2/A3 (librosa textualized features)
├── run_mllm_audio.py           # A1_mllm/A2_mllm/A3_mllm (native audio; qwen/voxtral)
├── run_baselines.py            # B1 majority / B2 lexicon / B3 fine-tuned RoBERTa
├── evaluate.py                 # metrics, confusion matrices, per-class reports
├── emotion_lexicon.json        # masking lexicon (6 emotion categories + neutral)
├── MELD.Raw/                   # dataset (downloaded, see above)
├── data/                       # per-model results        (created after running)
│   └── <Model-Slug>/eval/      # evaluated metrics per condition
├── audio_features/             # cached librosa features   (created after running)
└── roberta_meld/               # B3 RoBERTa checkpoint      (created by run_baselines)
```

`data/`, `audio_features/`, and `roberta_meld/` do not exist on a fresh clone —
they are produced by running the pipeline.

---

## Running the experiments

The full pipeline is driven by `run_all.sh`. **To switch models, edit only the
variables at the top of the script:**

```bash
# ── change the model here ──────────────────────────────────
TEXT_MODEL="meta-llama/Llama-3.2-3B-Instruct"   # text / audio-feature LLM
MLLM_MODEL=""        # "qwen" or "voxtral" for native audio; "" to skip Stage 4
SPLIT="test"
BATCH_SIZE=16
GPU=0
```

Then run:

```bash
bash run_all.sh
```

The script executes six stages and writes results to `./data/<Model-Slug>/`:

| Stage | Script | Conditions |
|-------|--------|------------|
| 1 | `run_text_conditions.py` | T1 T2 T3 COT DEF FS |
| 2 | `run_text_conditions.py` | M1 M2 M3 MCOT MDEF MFS |
| 3 | `run_audio_conditions.py` | A1 A2 A3 (textualized librosa features) |
| 4 | `run_mllm_audio.py` | A1_mllm A2_mllm A3_mllm (only if `MLLM_MODEL` is set) |
| 5 | `run_baselines.py` | B1 B2 B3 |
| 6 | `evaluate.py` | metrics over all of the above |

Stage 4 runs only for the two audio-capable models (Qwen2-Audio, Voxtral); set
`MLLM_MODEL=""` for text-only models to skip it. The six evaluated models are:

```
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Llama-3.1-8B-Instruct
nvidia/Nemotron-Mini-4B-Instruct
nvidia/Mistral-NeMo-Minitron-8B-Instruct
Qwen/Qwen2-Audio-7B-Instruct          # MLLM_MODEL="qwen"
mistralai/Voxtral-Mini-3B-2507        # MLLM_MODEL="voxtral"
```

### Running stages individually

Each script can also be called directly, e.g.:

```bash
python run_text_conditions.py \
    --conditions T1 M1 \
    --split test \
    --model_id meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./data/Llama-3.2-3B-Instruct \
    --batch_size 16
```

---

## Experimental design

We organize conditions into four families that each isolate one information
source:

- **T-series (unmasked text)** — `T1` utterance only · `T2` + dialogue history ·
  `T3` + speaker focus.
- **M-series (masked text)** — `M1`/`M2`/`M3` mirror `T1`/`T2`/`T3` but with the
  target utterance's emotion words replaced by `[MASK]`. Only the *target* is
  masked; dialogue history stays unmasked.
- **Prompting variants** — `COT/MCOT` (chain-of-thought), `DEF/MDEF`
  (definition-grounded), `FS/MFS` (few-shot), crossed with unmasked/masked text.
- **Acoustic conditions** — `A1/A2/A3` feed *textualized* librosa features to a
  text LLM; `A1_mllm/A2_mllm/A3_mllm` feed the *raw waveform* to an audio MLLM.

Masking is driven by `emotion_lexicon.json`: multi-word phrases are matched
longest-first, then single tokens, case-insensitively, replacing matches with
`[MASK]` while preserving punctuation. The `Is_Masked` flag (true iff the masked
utterance differs from the original) defines the **masked subset** vs. the
**no-mask control subset**.

### Metrics

Every condition reports **Macro-F1** (unweighted mean over the seven classes)
and **Weighted-F1** (support-weighted). We additionally derive:
- **Masking Drop** = F1(T_i) − F1(M_i)
- **Context Recovery** = F1(M_j) − F1(M_1), for j ∈ {2,3}

---

## Baselines

| | Baseline | Macro-F1 | Weighted-F1 |
|---|----------|:--------:|:-----------:|
| B1 | Majority label (always *neutral*) | 0.09 | 0.31 |
| B2 | Lexicon-only (rule-based keyword match) | 0.24 | 0.42 |
| B3 | Fine-tuned RoBERTa (`roberta-base`, in-domain) | 0.46 | 0.63 |

B1/B2 bound performance from below; B3 (trained 5 epochs, lr 2e-5, batch 32,
model selection by dev Macro-F1) is a supervised in-domain reference. B2 reuses
the same lexicon as the masking procedure, so it measures what pure keyword
spotting can achieve.

---

## Results summary

The best zero-shot Weighted-F1 (0.59) matches but does not exceed B3's 0.63,
and one zero-shot condition ties B3 on Macro-F1 (0.46) — supervised in-domain
training remains valuable. All six models' simplest `T1` condition beats the
lexicon-only baseline, yet adding dialogue context, speaker focus, or elaborate
prompting generally does **not** help and often hurts. Removing lexical cues
(`M`-series) lowers performance consistently across all models, and the induced
errors collapse disproportionately toward *surprise*. Native audio helps only
the two MLLMs, and almost entirely on *anger*. See the paper for the full tables
and per-emotion analysis.

---

## Citation / Course

NLP 2026 Final Project, National Taiwan University. See the accompanying paper
for full methodology, results, and discussion.