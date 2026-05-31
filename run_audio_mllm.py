"""
MELD ERC Experiment: Audio Conditions (A1, A2, A3) with Qwen2-Audio-7B-Instruct
真正的 MLLM — 直接吃音訊檔，不需要手動提特徵。

Conditions:
  A1: audio only          → model listens to the audio clip, no text
  A2: masked utterance + audio
  A3: dialogue context + masked utterance + audio

Usage:
    python run_audio_mllm.py --conditions A1 A2 A3 --split test
    python run_audio_mllm.py --conditions A1 --split test --dry_run

Install:
    pip install git+https://github.com/huggingface/transformers
    pip install librosa soundfile
"""

import argparse
import json
import re
import warnings
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS     = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]
EMOTION_SET  = set(EMOTIONS)
EMOTION_OPTS = ", ".join(EMOTIONS)
MODEL_ID     = "Qwen/Qwen2-Audio-7B-Instruct"
DATA_ROOT    = Path("./MELD.Raw")
OUT_ROOT     = Path("./results")
LEXICON_PATH = Path(__file__).parent / "emotion_lexicon.json"

AUDIO_DIRS = {
    "train": DATA_ROOT / "train_splits",
    "dev":   DATA_ROOT / "dev_splits_complete",
    "test":  DATA_ROOT / "output_repeated_splits_test",
}

# Qwen2-Audio expects 16kHz mono float32 numpy array
SAMPLE_RATE = 16000

# ── Lexicon / Masking ──────────────────────────────────────────────────────────
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


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_split(split: str) -> pd.DataFrame:
    fname = {"train": "train_sent_emo.csv",
             "dev":   "dev_sent_emo.csv",
             "test":  "test_sent_emo.csv"}[split]
    df = pd.read_csv(DATA_ROOT / fname)
    df.columns = df.columns.str.strip()
    df["Emotion"] = df["Emotion"].str.strip().str.lower()
    return df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)


# ── Audio Loading ──────────────────────────────────────────────────────────────
def get_audio_path(split: str, dia_id: int, utt_id: int) -> Path | None:
    p = AUDIO_DIRS[split] / f"dia{dia_id}_utt{utt_id}.mp4"
    return p if p.exists() else None


def load_audio(path: Path) -> np.ndarray | None:
    """Load audio file → 16kHz mono float32 numpy array."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Failed to load audio {path}: {e}")
        return None


# ── Context Builder ────────────────────────────────────────────────────────────
def build_context(df: pd.DataFrame, dia_id: int, utt_id: int) -> str:
    dia = df[df["Dialogue_ID"] == dia_id].sort_values("Utterance_ID")
    prior = dia[dia["Utterance_ID"] < utt_id]
    lines = [f'{r["Speaker"]}: "{r["Utterance"]}"' for _, r in prior.iterrows()]
    return "\n".join(lines)


# ── Prompt / Conversation Builders ────────────────────────────────────────────
# Speaker info intentionally omitted — no training data, speaker name is noise.
# Qwen2-Audio content format: list of {"type": "audio", "audio": np.ndarray}
# and {"type": "text", "text": "..."} items.

def build_conversation_A1(utt: str, audio: np.ndarray) -> list[dict]:
    text = (
        "This is a single-choice question.\n\n"
        "You will hear a spoken utterance.\n"
        "Your task is to determine the emotion of the speaker when they said the utterance.\n\n"
        f'Target utterance: "{utt}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )
    return [{"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text",  "text": text},
    ]}]


def build_conversation_A2(masked_utt: str, audio: np.ndarray) -> list[dict]:
    """A2: masked utterance + audio. Prompt mirrors M1 format."""
    text = (
        "This is a single-choice question.\n\n"
        "You will hear a spoken utterance and be given a masked version of it.\n"
        "Some emotion-bearing words in the utterance have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the speaker when they said the utterance.\n\n"
        f'Masked utterance: "{masked_utt}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )
    return [{"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text",  "text": text},
    ]}]


def build_conversation_A3(masked_utt: str, audio: np.ndarray,
                           context: str) -> list[dict]:
    """A3: dialogue context + masked utterance + audio. Prompt mirrors M2 format."""
    ctx_block = f"Conversation:\n{context}\n\n" if context else ""
    text = (
        "This is a single-choice question.\n\n"
        "You will hear a spoken utterance, be given a conversation and a masked version of it.\n"
        "Some emotion-bearing words in the target utterance have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the speaker when they said the target utterance.\n\n"
        f"{ctx_block}"
        f'Masked utterance: "{masked_utt}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )
    return [{"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text",  "text": text},
    ]}]


def build_conversation_no_audio(condition: str, masked_utt: str,
                                 context: str) -> list[dict]:
    """Fallback when audio is missing — text-only, flagged has_audio=False."""
    if condition == "A1":
        text = (
            "This is a single-choice question.\n\n"
            "Audio information is unavailable for this utterance.\n"
            "Your task is to determine the emotion of the speaker.\n\n"
            f'Target utterance: "{masked_utt}"\n\n'
            f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
            "Answer with only one label."
        )
    elif condition == "A2":
        text = (
            "This is a single-choice question.\n\n"
            "Audio information is unavailable for this utterance.\n"
            "Some emotion-bearing words in the utterance have been replaced with [MASK].\n"
            "Your task is to determine the emotion of the speaker.\n\n"
            f'Masked utterance: "{masked_utt}"\n\n'
            f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
            "Answer with only one label."
        )
    else:  # A3
        ctx_block = f"Conversation:\n{context}\n\n" if context else ""
        text = (
            "This is a single-choice question.\n\n"
            "Audio information is unavailable for this utterance.\n"
            "Some emotion-bearing words in the target utterance have been replaced with [MASK].\n"
            "Your task is to determine the emotion of the speaker of the last utterance.\n\n"
            f"{ctx_block}"
            f'Masked utterance: "{masked_utt}"\n\n'
            f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
            "Answer with only one label."
        )
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


# Inflection map: model may output adjective/adverb forms instead of noun labels
_INFLECTION_MAP = {
    "angry":       "anger",
    "angrily":     "anger",
    "furious":     "anger",
    "surprised":   "surprise",
    "surprising":  "surprise",
    "shocked":     "surprise",
    "joyful":      "joy",
    "joyfully":    "joy",
    "happy":       "joy",
    "happily":     "joy",
    "cheerful":    "joy",
    "excited":     "joy",
    "sad":         "sadness",
    "sadly":       "sadness",
    "unhappy":     "sadness",
    "sorrowful":   "sadness",
    "fearful":     "fear",
    "scared":      "fear",
    "afraid":      "fear",
    "frightened":  "fear",
    "disgusted":   "disgust",
    "disgusting":  "disgust",
    "revolted":    "disgust",
}

# ── Response Parsing ───────────────────────────────────────────────────────────
def parse_prediction(text: str) -> str:
    """
    Robust parser: handles lowercase, Title Case, UPPER CASE outputs,
    inflected adjective forms (angry→anger, sad→sadness, etc.),
    and verbose formats like "The emotion is: Angry".
    """
    text_lower = text.strip().lower()

    # 1. exact match on each line
    for line in text_lower.splitlines():
        line = line.strip().rstrip(".,:").strip()
        if line in EMOTION_SET:
            return line
        if line in _INFLECTION_MAP:
            return _INFLECTION_MAP[line]

    # 2. word-boundary search for canonical labels
    for emo in EMOTIONS:
        if re.search(rf"\b{emo}\b", text_lower):
            return emo

    # 3. word-boundary search for inflected forms
    for inflected, canonical in _INFLECTION_MAP.items():
        if re.search(rf"\b{inflected}\b", text_lower):
            return canonical

    return "neutral"


# ── Model Loading ──────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return processor, model


# ── Single-sample Inference ────────────────────────────────────────────────────
# def run_one(processor, model, conversation: list[dict]) -> str:
#     """
#     Run inference for a single conversation.

#     Qwen2-Audio processor.apply_chat_template does NOT handle numpy arrays
#     in the content dict — it only handles audio_url strings for placeholder
#     insertion. We separate the audio arrays out, build a template-friendly
#     conversation (with audio_url placeholders), then pass arrays separately
#     to processor(audios=...).
#     """
#     # Separate audio arrays from the conversation structure.
#     # Replace {"type":"audio","audio":array} with {"type":"audio","audio_url":"placeholder"}
#     # so apply_chat_template inserts the correct <|AUDIO|> token.
#     audios = []
#     clean_conv = []
#     for msg in conversation:
#         if isinstance(msg["content"], list):
#             clean_content = []
#             for ele in msg["content"]:
#                 if ele["type"] == "audio":
#                     audios.append(ele["audio"])
#                     # placeholder url — only used for token insertion, not fetched
#                     clean_content.append({"type": "audio",
#                                           "audio_url": "placeholder.wav"})
#                 else:
#                     clean_content.append(ele)
#             clean_conv.append({"role": msg["role"], "content": clean_content})
#         else:
#             clean_conv.append(msg)

#     text_prompt = processor.apply_chat_template(
#         clean_conv,
#         add_generation_prompt=True,
#         tokenize=False,
#     )

#     if audios:
#         inputs = processor(
#             text=text_prompt,
#             audios=audios,
#             sampling_rate=SAMPLE_RATE,
#             return_tensors="pt",
#             padding=True,
#         )
#     else:
#         inputs = processor(
#             text=text_prompt,
#             return_tensors="pt",
#             padding=True,
#         )

#     inputs = {k: v.to(model.device) for k, v in inputs.items()
#               if isinstance(v, torch.Tensor)}

#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=16,
#             do_sample=False,
#         )

#     input_len = inputs["input_ids"].shape[1]
#     generated_ids = output_ids[:, input_len:]
#     generated = processor.batch_decode(
#         generated_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )[0]
#     return generated.strip()

def run_one(processor, model, conversation: list[dict]) -> str:
    """
    繞過 apply_chat_template 內部 audios 衝突的乾淨版本
    """
    audios = []
    pure_text_conv = []
    
    for msg in conversation:
        if isinstance(msg["content"], list):
            text_parts = []
            for ele in msg["content"]:
                if ele["type"] == "audio":
                    # 1. 收集真正的音訊 numpy array
                    audios.append(ele["audio"])
                    # 2. 直接手動寫入 Qwen2-Audio 辨識音訊用的特殊 Token
                    text_parts.append("<|AUDIO|>")
                elif ele["type"] == "text":
                    text_parts.append(ele["text"])
            
            # 把原本 list 形式的 content 改成純文字字串
            pure_text_conv.append({
                "role": msg["role"],
                "content": "\n".join(text_parts)
            })
        else:
            pure_text_conv.append(msg)

    # 讓 apply_chat_template 把它當成「純文字對話」處理，這樣絕對不會產生 conflict kwargs
    text_prompt = processor.apply_chat_template(
        pure_text_conv,
        add_generation_prompt=True,
        tokenize=False,
    )

    # 丟給 processor，這次明確只給 text 和 audio (注意是單數 audio)
    if audios:
        inputs = processor(
            text=text_prompt,
            audio=audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=text_prompt,
            return_tensors="pt",
            padding=True,
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    generated = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return generated.strip()
    
# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["A1_mllm", "A2_mllm", "A3_mllm"],
                        choices=["A1_mllm", "A2_mllm", "A3_mllm"])
    parser.add_argument("--split", default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Print 3 sample conversations without running model")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.split} split …")
    df = load_split(args.split)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"  {len(df)} utterances loaded.")

    if not args.dry_run:
        processor, model = load_model()

    for cond in args.conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {cond}  (Qwen2-Audio MLLM)  |  split: {args.split}")
        print(f"{'='*60}")

        out_path = OUT_ROOT / f"{cond}_{args.split}.jsonl"
        if out_path.exists():
            print(f"  Output already exists: {out_path}. Skipping.")
            continue

        records = []
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            speaker  = row["Speaker"]
            utt      = row["Utterance"]
            dia_id   = row["Dialogue_ID"]
            utt_id   = row["Utterance_ID"]
            masked   = mask_utterance(utt)
            key      = f"dia{dia_id}_utt{utt_id}"

            # Load audio
            audio_path = get_audio_path(args.split, dia_id, utt_id)
            audio_arr  = load_audio(audio_path) if audio_path else None
            has_audio  = audio_arr is not None

            # Build conversation (no speaker info passed)
            # cond is "A1_mllm" / "A2_mllm" / "A3_mllm"
            base = cond.split("_")[0]   # "A1" / "A2" / "A3"
            if has_audio:
                if base == "A1":
                    conv = build_conversation_A1(utt, audio_arr)
                elif base == "A2":
                    conv = build_conversation_A2(masked, audio_arr)
                else:  # A3
                    ctx  = build_context(df, dia_id, utt_id)
                    conv = build_conversation_A3(masked, audio_arr, ctx)
            else:
                ctx = build_context(df, dia_id, utt_id) if base == "A3" else ""
                conv = build_conversation_no_audio(base, masked, ctx)

            if args.dry_run:
                if i < 3:
                    print(f"\n--- Sample {i} (has_audio={has_audio}) ---")
                    for msg in conv:
                        content = msg["content"] if isinstance(msg["content"], list) else [{"type":"text","text":msg["content"]}]
                        for ele in content:
                            if ele["type"] == "text":
                                print(ele["text"])
                            else:
                                arr = ele.get("audio")
                                shape = arr.shape if hasattr(arr, "shape") else "N/A"
                                print(f"[AUDIO array: shape={shape}]")
                continue

            # Inference
            generated = run_one(processor, model, conv)
            pred = parse_prediction(generated)

            records.append({
                "Sr_No":        getattr(row, "Sr No.", i),
                "Dialogue_ID":  dia_id,
                "Utterance_ID": utt_id,
                "Speaker":      speaker,   # stored for analysis, not used in prompt
                "Utterance":    utt,
                "gold":         row["Emotion"],
                "audio_key":    key,
                "has_audio":    has_audio,
                "raw_output":   generated,
                "prediction":   pred,
                "condition":    cond,
                "split":        args.split,
                "model":        MODEL_ID,
            })

            if (i + 1) % 50 == 0 or (i + 1) == n:
                print(f"  [{i+1}/{n}] {utt[:40]!r} → {pred}")

        if args.dry_run:
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()