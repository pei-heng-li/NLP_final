"""
MELD ERC Experiment: Audio Conditions (A1, A2, A3)

Pipeline:
  1. Extract audio features from .mp4 files using librosa (MFCC, pitch, energy, ZCR)
  2. Build text prompts that describe the audio features
  3. Run LLM inference

Usage:
    python run_audio_conditions.py --conditions A1 A2 A3 --split test
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS     = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]
EMOTION_SET  = set(EMOTIONS)
EMOTION_OPTS = ", ".join(EMOTIONS)
# MODEL_ID     = "meta-llama/Llama-3.2-1B-Instruct"
# MODEL_ID     = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_ID     = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_ROOT    = Path("./MELD.Raw")
OUT_ROOT     = Path("./results")
FEAT_CACHE   = Path("./audio_features")

# Audio split dirs per data split
AUDIO_DIRS = {
    "train": DATA_ROOT / "train_splits",
    "dev":   DATA_ROOT / "dev_splits_complete",
    "test":  DATA_ROOT / "output_repeated_splits_test",
}

# Emotion keywords: loaded from emotion_lexicon.json (same directory as this script)
LEXICON_PATH = Path(__file__).parent / "emotion_lexicon.json"
with open(LEXICON_PATH, encoding="utf-8") as _f:
    EMOTION_KEYWORDS: dict[str, list[str]] = json.load(_f)

ALL_KEYWORDS: set[str] = set()
MULTI_WORD_PHRASES: list[str] = []
for _kws in EMOTION_KEYWORDS.values():
    for _kw in _kws:
        if " " in _kw:
            MULTI_WORD_PHRASES.append(_kw)
        else:
            ALL_KEYWORDS.add(_kw)
MULTI_WORD_PHRASES.sort(key=len, reverse=True)


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_split(split: str) -> pd.DataFrame:
    fname = {"train": "train_sent_emo.csv",
             "dev":   "dev_sent_emo.csv",
             "test":  "test_sent_emo.csv"}[split]
    df = pd.read_csv(DATA_ROOT / fname)
    df.columns = df.columns.str.strip()
    df["Emotion"] = df["Emotion"].str.strip().str.lower()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    return df


# ── Masking ────────────────────────────────────────────────────────────────────
def mask_utterance(utterance: str) -> str:
    """Multi-word phrases first, then token-level masking."""
    text = utterance
    for phrase in MULTI_WORD_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("[MASK]", text)
    tokens = text.split()
    masked = []
    for tok in tokens:
        if "[MASK]" in tok:
            masked.append(tok)
        else:
            clean = re.sub(r"[^a-z]", "", tok.lower())
            if clean in ALL_KEYWORDS:
                trail = re.sub(r"^[a-zA-Z\[\]]+", "", tok)
                masked.append("[MASK]" + trail)
            else:
                masked.append(tok)
    return " ".join(masked)


# ── Audio Feature Extraction ───────────────────────────────────────────────────
def get_audio_path(split: str, dia_id: int, utt_id: int) -> Path | None:
    audio_dir = AUDIO_DIRS[split]
    p = audio_dir / f"dia{dia_id}_utt{utt_id}.mp4"
    return p if p.exists() else None


def extract_features(audio_path: Path) -> dict | None:
    """
    Extract acoustic features using librosa.
    Returns a dict of summary statistics, or None on failure.
    """
    try:
        import librosa
        # Load audio (convert mp4 audio track via soundfile / ffmpeg backend)
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        if len(y) == 0:
            return None

        duration = librosa.get_duration(y=y, sr=sr)

        # MFCC (13 coefficients) – capture timbre / spectral shape
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1).tolist()
        mfcc_std  = mfcc.std(axis=1).tolist()

        # Pitch (F0) via pyin
        f0, voiced_flag, _ = librosa.pyin(y, fmin=60, fmax=600,
                                           sr=sr, fill_na=None)
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean  = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std   = float(np.nanstd(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        voiced_ratio = float(voiced_flag.mean())    if voiced_flag is not None else 0.0

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(rms.mean())
        energy_std  = float(rms.std())

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(zcr.mean())

        # Speech rate proxy: syllable-like onsets per second
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_rate = len(onset_frames) / max(duration, 0.01)  # onsets/sec

        return {
            "duration":     round(duration, 3),
            "mfcc_mean":    [round(v, 4) for v in mfcc_mean],
            "mfcc_std":     [round(v, 4) for v in mfcc_std],
            "pitch_mean":   round(pitch_mean, 2),
            "pitch_std":    round(pitch_std,  2),
            "voiced_ratio": round(voiced_ratio, 4),
            "energy_mean":  round(energy_mean, 6),
            "energy_std":   round(energy_std,  6),
            "zcr_mean":     round(zcr_mean,    6),
            "onset_rate":   round(onset_rate,  4),
        }
    except Exception as e:
        print(f"    [WARN] Feature extraction failed for {audio_path}: {e}")
        return None


def extract_all_features(df: pd.DataFrame, split: str) -> dict[str, dict | None]:
    """
    Extract and cache audio features for all utterances.
    Key: "dia{N}_utt{M}"
    """
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = FEAT_CACHE / f"{split}_features.json"

    if cache_file.exists():
        print(f"  Loading cached features from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Extracting audio features for {split} split …")
    features = {}
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        key = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
        path = get_audio_path(split, row["Dialogue_ID"], row["Utterance_ID"])
        if path is None:
            features[key] = None
            print(f"    [{i+1}/{n}] MISSING: {key}", end="\r")
        else:
            features[key] = extract_features(path)
            print(f"    [{i+1}/{n}] {key}", end="\r")
    print()

    with open(cache_file, "w") as f:
        json.dump(features, f, indent=2)
    print(f"  Cached to {cache_file}")
    return features


# ── Naturalise Audio Features ──────────────────────────────────────────────────
def describe_audio(feat: dict | None) -> str:
    """
    Convert numeric audio features into a natural language description
    suitable for LLM input. Returns None if features are unavailable.
    """
    if feat is None:
        return None

    lines = []

    # Duration
    dur = feat.get("duration", 0)
    if dur < 1.0:
        lines.append(f"The utterance is very short ({dur:.2f}s).")
    elif dur < 3.0:
        lines.append(f"The utterance is {dur:.2f}s long.")
    else:
        lines.append(f"The utterance is relatively long ({dur:.2f}s).")

    # Pitch
    p_mean  = feat.get("pitch_mean", 0)
    p_std   = feat.get("pitch_std",  0)
    v_ratio = feat.get("voiced_ratio", 0)
    if v_ratio < 0.2:
        lines.append("The speech has little voicing (whisper-like or very quiet).")
    elif p_mean > 250:
        lines.append(f"The pitch is high (mean ~{p_mean:.0f} Hz), suggesting excitement or surprise.")
    elif p_mean > 160:
        lines.append(f"The pitch is moderate (mean ~{p_mean:.0f} Hz).")
    elif p_mean > 0:
        lines.append(f"The pitch is low (mean ~{p_mean:.0f} Hz), possibly calm or serious.")
    if p_std > 60:
        lines.append("Pitch variation is high, indicating expressive or emotional delivery.")
    elif p_std > 20:
        lines.append("Pitch variation is moderate.")
    else:
        lines.append("Pitch variation is low, suggesting a flat or monotone delivery.")

    # Energy / loudness
    e_mean = feat.get("energy_mean", 0)
    if e_mean > 0.05:
        lines.append("The speech energy is high, suggesting loud or emphatic delivery.")
    elif e_mean > 0.01:
        lines.append("The speech energy is moderate.")
    else:
        lines.append("The speech energy is low, suggesting quiet or subdued delivery.")

    # Speech rate (onset rate)
    onset_rate = feat.get("onset_rate", 0)
    if onset_rate > 8:
        lines.append("The speech rate is fast.")
    elif onset_rate > 4:
        lines.append("The speech rate is moderate.")
    else:
        lines.append("The speech rate is slow.")

    # ZCR (breathiness / noisiness)
    zcr = feat.get("zcr_mean", 0)
    if zcr > 0.1:
        lines.append("The zero-crossing rate is high, suggesting breathy or noisy speech.")

    return " ".join(lines)


# ── Context Builder ────────────────────────────────────────────────────────────
def build_context(df: pd.DataFrame, dia_id: int, utt_id: int) -> str:
    dia = df[df["Dialogue_ID"] == dia_id].sort_values("Utterance_ID")
    prior = dia[dia["Utterance_ID"] < utt_id]
    lines = [f'{r["Speaker"]}: "{r["Utterance"]}"' for _, r in prior.iterrows()]
    return "\n".join(lines)


# ── Prompt Builders ────────────────────────────────────────────────────────────
def prompt_A1(speaker: str, audio_desc: str) -> str:
    return (
        "This is a single-choice question.\n\n"
        "You will be given acoustic features of a spoken utterance.\n"
        "Your task is to determine the emotion of the audio information.\n\n"
        f"Audio description: {audio_desc}\n\n"
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_A2(speaker: str, masked_utt: str, audio_desc: str) -> str:
    return (
        "This is a single-choice question.\n\n"
        "You will be given a masked utterance and acoustic features of a spoken utterance.\n"
        "Some emotion-bearing words in the utterance have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the audio information.\n\n"

        f'Masked utterance: "{masked_utt}"\n'
        f"Audio description: {audio_desc}\n\n"
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_A3(speaker: str, masked_utt: str, audio_desc: str, context: str) -> str:
    ctx_block = f"Conversation:\n{context}\n\n" if context else ""
    return (
        "This is a single-choice question.\n\n"
        "You will be given a conversation, a masked utterance, and acoustic features.\n"
        "Some emotion-bearing words in the target utterance have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the audio information.\n\n"
        f"{ctx_block}"

        f'Masked utterance: "{masked_utt}"\n'
        f"Audio description: {audio_desc}\n\n"
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )


def build_prompt_audio(condition: str, row: pd.Series, df: pd.DataFrame,
                       features: dict) -> tuple[str, bool]:
    """
    Returns (prompt_str, has_audio).
    has_audio=False means the audio file was missing; the prompt will note this.
    """
    speaker  = row["Speaker"]
    utt      = row["Utterance"]
    dia_id   = row["Dialogue_ID"]
    utt_id   = row["Utterance_ID"]
    key      = f"dia{dia_id}_utt{utt_id}"
    feat     = features.get(key)
    audio_desc = describe_audio(feat)
    has_audio  = audio_desc is not None
    # Fallback description when audio is missing — model still gets a prompt,
    # but the record will be flagged has_audio=False for downstream filtering.
    if audio_desc is None:
        audio_desc = "Audio information is unavailable for this utterance."
    masked = mask_utterance(utt)

    if condition == "A1":
        return prompt_A1(speaker, audio_desc), has_audio
    elif condition == "A2":
        return prompt_A2(speaker, masked, audio_desc), has_audio
    elif condition == "A3":
        ctx = build_context(df, dia_id, utt_id)
        return prompt_A3(speaker, masked, audio_desc, ctx), has_audio
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ── Response Parsing ───────────────────────────────────────────────────────────
def parse_prediction(text: str) -> str:
    text_lower = text.strip().lower()
    for line in text_lower.splitlines():
        line = line.strip().rstrip(".")
        if line in EMOTION_SET:
            return line
    for emo in EMOTIONS:
        if emo in text_lower:
            return emo
    return "neutral"


# ── Model Loading & Inference ──────────────────────────────────────────────────
def load_model():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def run_inference(tokenizer, model, prompts: list[str],
                  batch_size: int = 16, max_new_tokens: int = 16) -> list[str]:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size,
        return_full_text=False,
    )
    results = []
    total = len(prompts)
    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        outputs = pipe(chat_prompts)
        for out in outputs:
            generated = out[0]["generated_text"]
            results.append(generated if isinstance(generated, str) else str(generated))
        print(f"  [{min(i+batch_size, total)}/{total}] done", end="\r")
    print()
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["A1","A2","A3"],
                        choices=["A1","A2","A3"])
    parser.add_argument("--split", default="test", choices=["train","dev","test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print 3 sample prompts per condition without running model")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.split} split …")
    df = load_split(args.split)
    print(f"  {len(df)} utterances loaded.")

    # Extract / load audio features first
    features = extract_all_features(df, args.split)

    if not args.dry_run:
        tokenizer, model = load_model()

    for cond in args.conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {cond}  |  split: {args.split}")
        print(f"{'='*60}")

        out_path = OUT_ROOT / f"{cond}_{args.split}.jsonl"
        if out_path.exists():
            print(f"  Output already exists: {out_path}. Skipping.")
            continue

        prompts = []
        has_audio_flags = []
        for _, row in df.iterrows():
            prompt, has_audio = build_prompt_audio(cond, row, df, features)
            prompts.append(prompt)
            has_audio_flags.append(has_audio)

        if args.dry_run:
            for i in range(min(3, len(prompts))):
                print(f"\n--- Sample {i} ---\n{prompts[i]}\n")
            continue

        raw_outputs = run_inference(
            tokenizer, model, prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        records = []
        for idx, (row_tuple, raw) in enumerate(zip(df.itertuples(), raw_outputs)):
            generated = raw.strip()
            pred = parse_prediction(generated)
            key  = f"dia{row_tuple.Dialogue_ID}_utt{row_tuple.Utterance_ID}"
            records.append({
                "Sr_No":        getattr(row_tuple, "Sr_No", idx),
                "Dialogue_ID":  row_tuple.Dialogue_ID,
                "Utterance_ID": row_tuple.Utterance_ID,
                "Speaker":      row_tuple.Speaker,
                "Utterance":    row_tuple.Utterance,
                "gold":         row_tuple.Emotion,
                "audio_key":    key,
                "has_audio":    has_audio_flags[idx],
                "raw_output":   generated,
                "prediction":   pred,
                "condition":    cond,
                "split":        args.split,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()