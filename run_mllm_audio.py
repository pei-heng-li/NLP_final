"""
MELD ERC Experiment: MLLM Audio Conditions (A1, A2, A3)
真正的 MLLM — 直接吃音訊檔，不需要手動提特徵。

支援模型:
  qwen  → Qwen/Qwen2-Audio-7B-Instruct   (condition suffix: _mllm)
  voxtral → mistralai/Voxtral-Mini-3B-2507 (condition suffix: _voxtral)

Conditions:
  A1: audio only
  A2: masked utterance + audio
  A3: dialogue context + masked utterance + audio

Output condition names: A1_{suffix} / A2_{suffix} / A3_{suffix}

Usage:
    python run_mllm_audio.py --model qwen    --conditions A1 A2 A3 --split test
    python run_mllm_audio.py --model voxtral --conditions A1 A2 A3 --split test
    python run_mllm_audio.py --model voxtral --conditions A1 --split test --dry_run

Install:
    pip install transformers>=4.53.0 torch librosa soundfile
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
from transformers import AutoProcessor
from transformers import Qwen2AudioForConditionalGeneration
from transformers import VoxtralForConditionalGeneration

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "qwen":    "Qwen/Qwen2-Audio-7B-Instruct",
    "voxtral": "mistralai/Voxtral-Mini-3B-2507",
}
SUFFIXES = {
    "qwen":    "mllm",
    "voxtral": "voxtral",
}

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS     = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]
EMOTION_SET  = set(EMOTIONS)
EMOTION_OPTS = ", ".join(EMOTIONS)
DATA_ROOT    = Path("./MELD.Raw")
OUT_ROOT  = Path("./data")  # overridden at runtime by --output_dir
LEXICON_PATH = Path(__file__).parent / "emotion_lexicon.json"
SAMPLE_RATE  = 16000  # Qwen2-Audio expects 16kHz numpy array

AUDIO_DIRS = {
    "train": DATA_ROOT / "train_splits",
    "dev":   DATA_ROOT / "dev_splits_complete",
    "test":  DATA_ROOT / "output_repeated_splits_test",
}

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
    df = pd.read_csv(DATA_ROOT / fname, encoding="cp1252")
    df.columns = df.columns.str.strip()
    df["Emotion"] = df["Emotion"].str.strip().str.lower()
    return df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)


# ── Audio Loading ──────────────────────────────────────────────────────────────
def get_audio_path(split: str, dia_id: int, utt_id: int) -> Path | None:
    p = AUDIO_DIRS[split] / f"dia{dia_id}_utt{utt_id}.mp4"
    return p if p.exists() else None


def load_audio_array(path: Path) -> np.ndarray | None:
    """Load audio → 16kHz mono float32 numpy array (for Qwen2-Audio).
    Uses ffmpeg → soundfile to handle mp4 without librosa's deprecated audioread."""
    import subprocess, io, soundfile as sf
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-t", "30",
            "-f", "wav", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
        )
        y, _ = sf.read(io.BytesIO(result.stdout), dtype="float32")
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


# ── Conversation Builders ──────────────────────────────────────────────────────
# audio_input: np.ndarray (Qwen) 或 str path (Voxtral)
# key:         "audio" (Qwen) 或 "path" (Voxtral)

def _audio_ele(audio_input, model_key: str) -> dict:
    if model_key == "qwen":
        return {"type": "audio", "audio": audio_input}
    else:  # voxtral
        return {"type": "audio", "path": audio_input}


def build_conversation_A1(utt: str, audio_input, model_key: str) -> list[dict]:
    text = (
        "This is a single-choice question.\n\n"
        "You will hear a spoken utterance.\n"
        "Your task is to determine the emotion of the speaker when they said the utterance.\n\n"
        f'Target utterance: "{utt}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )
    return [{"role": "user", "content": [
        _audio_ele(audio_input, model_key),
        {"type": "text", "text": text},
    ]}]


def build_conversation_A2(masked_utt: str, audio_input, model_key: str) -> list[dict]:
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
        _audio_ele(audio_input, model_key),
        {"type": "text", "text": text},
    ]}]


def build_conversation_A3(masked_utt: str, audio_input, model_key: str,
                           context: str) -> list[dict]:
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
        _audio_ele(audio_input, model_key),
        {"type": "text", "text": text},
    ]}]


def build_conversation_no_audio(base: str, utt: str, masked_utt: str,
                                 context: str) -> list[dict]:
    """Fallback when audio file is missing — text-only."""
    if base == "A1":
        text = (
            "This is a single-choice question.\n\n"
            "Audio information is unavailable for this utterance.\n"
            "Your task is to determine the emotion of the speaker.\n\n"
            f'Target utterance: "{utt}"\n\n'
            f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
            "Answer with only one label."
        )
    elif base == "A2":
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


# ── Inflection Map / Parsing ───────────────────────────────────────────────────
_INFLECTION_MAP = {
    "angry": "anger", "angrily": "anger", "furious": "anger",
    "surprised": "surprise", "surprising": "surprise", "shocked": "surprise",
    "joyful": "joy", "joyfully": "joy", "happy": "joy",
    "happily": "joy", "cheerful": "joy", "excited": "joy",
    "sad": "sadness", "sadly": "sadness", "unhappy": "sadness", "sorrowful": "sadness",
    "fearful": "fear", "scared": "fear", "afraid": "fear", "frightened": "fear",
    "disgusted": "disgust", "disgusting": "disgust", "revolted": "disgust",
}


def parse_prediction(text: str) -> str:
    text_lower = text.strip().lower()
    for line in text_lower.splitlines():
        line = line.strip().rstrip(".,:").strip()
        if line in EMOTION_SET:
            return line
        if line in _INFLECTION_MAP:
            return _INFLECTION_MAP[line]
    for emo in EMOTIONS:
        if re.search(rf"\b{emo}\b", text_lower):
            return emo
    for inflected, canonical in _INFLECTION_MAP.items():
        if re.search(rf"\b{inflected}\b", text_lower):
            return canonical
    return "neutral"


# ── Model Loading ──────────────────────────────────────────────────────────────
def load_model(model_key: str):
    model_id = MODELS[model_key]
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    if model_key == "qwen":
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:  # voxtral
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return processor, model


# ── Single-sample Inference ────────────────────────────────────────────────────
def run_one_qwen(processor, model, conversation: list[dict]) -> str:
    """
    Qwen2-Audio: 繞過 apply_chat_template 內部 audios 衝突。
    手動抽出 numpy arrays，插入 <|AUDIO|> token，再呼叫 processor()。
    """
    audios = []
    pure_text_conv = []
    for msg in conversation:
        if isinstance(msg["content"], list):
            text_parts = []
            for ele in msg["content"]:
                if ele["type"] == "audio":
                    audios.append(ele["audio"])
                    text_parts.append("<|AUDIO|>")
                elif ele["type"] == "text":
                    text_parts.append(ele["text"])
            pure_text_conv.append({"role": msg["role"], "content": "\n".join(text_parts)})
        else:
            pure_text_conv.append(msg)

    text_prompt = processor.apply_chat_template(
        pure_text_conv, add_generation_prompt=True, tokenize=False
    )
    if audios:
        inputs = processor(text=text_prompt, audio=audios,
                           sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=text_prompt, return_tensors="pt", padding=True)

    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    generated = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return generated.strip()


# Module-level cache so we don't reload mc_tok for every sample
_mc_tok_voxtral = None

def _get_mc_tok():
    global _mc_tok_voxtral
    if _mc_tok_voxtral is None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        _mc_tok_voxtral = MistralTokenizer.from_hf_hub(MODELS["voxtral"])
    return _mc_tok_voxtral


def run_one_voxtral(processor, model, conversation: list[dict]) -> str:
    """
    Voxtral MLLM inference.

    Pipeline:
      mp4 → librosa → numpy array
        ├─ mistral_common.encode_chat_completion → token_ids
        └─ processor.feature_extractor          → input_features (Whisper mel)
      model.generate(input_ids, input_features)
    """
    from mistral_common.protocol.instruct.chunk import AudioChunk, RawAudio, TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.audio import Audio

    mc_tok = _get_mc_tok()

    chunks = []
    audio_arrays = []   # collect numpy arrays in order for feature_extractor

    for msg in conversation:
        if not isinstance(msg["content"], list):
            continue
        for ele in msg["content"]:
            if ele["type"] == "audio" and "path" in ele:
                try:
                    # Use ffmpeg to decode mp4 → raw pcm, then read with soundfile
                    # This avoids librosa's deprecated audioread fallback for mp4.
                    import subprocess, io, soundfile as sf
                    # Cap at 30s to guard against corrupted mp4s (e.g. dia38_utt4)
                    cmd = [
                        "ffmpeg", "-y", "-i", ele["path"],
                        "-t", "30",
                        "-f", "wav", "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "pipe:1",
                    ]
                    result = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
                    )
                    y, _ = sf.read(io.BytesIO(result.stdout), dtype="float32")
                    y = y.astype(np.float32)
                    audio_arrays.append(y)
                    audio_obj = Audio(audio_array=y, sampling_rate=SAMPLE_RATE, format="wav")
                    raw = RawAudio(data=audio_obj.to_base64("wav"), format="wav")
                    chunks.append(AudioChunk(input_audio=raw))
                except Exception as e:
                    print(f"  [WARN] audio load failed on {ele['path']}: {e}")
            elif ele["type"] == "text":
                chunks.append(TextChunk(text=ele["text"]))

    req = ChatCompletionRequest(messages=[UserMessage(content=chunks)])
    encoded = mc_tok.encode_chat_completion(req)
    input_ids = torch.tensor([encoded.tokens], dtype=torch.long).to(model.device)

    model_inputs = {"input_ids": input_ids}

    if audio_arrays:
        # WhisperFeatureExtractor: (batch_audio, n_mels=128, T=3000)
        feat = processor.feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        # feat.input_features shape: (n_audios, 128, 3000)
        # audio_tower.conv1 expects (n_audios, 128, 3000) — no extra batch dim
        model_inputs["input_features"] = feat.input_features.to(
            dtype=torch.bfloat16, device=model.device
        )

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, max_new_tokens=16, do_sample=False)

    generated_ids = output_ids[0, input_ids.shape[1]:]
    eos_id = mc_tok.instruct_tokenizer.tokenizer.eos_id
    ids = generated_ids.tolist()
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    return mc_tok.instruct_tokenizer.tokenizer.decode(ids).strip()


def run_one(processor, model, conversation: list[dict], model_key: str) -> str:
    if model_key == "qwen":
        return run_one_qwen(processor, model, conversation)
    else:
        return run_one_voxtral(processor, model, conversation)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Which MLLM backend to use")
    parser.add_argument("--conditions", nargs="+", default=["A1", "A2", "A3"],
                        choices=["A1", "A2", "A3"])
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Print 3 sample conversations without running model")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    suffix    = SUFFIXES[args.model]
    model_id  = MODELS[args.model]
    global OUT_ROOT
    if args.output_dir:
        OUT_ROOT = args.output_dir
    else:
        OUT_ROOT = Path(f"./data/{model_id.split('/')[-1]}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.split} split …")
    df = load_split(args.split)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"  {len(df)} utterances loaded.")

    if not args.dry_run:
        processor, model = load_model(args.model)

    for base in args.conditions:
        cond = f"{base}_{suffix}"
        print(f"\n{'='*60}")
        print(f"  Condition: {cond}  ({model_id})  |  split: {args.split}")
        print(f"{'='*60}")

        out_path = OUT_ROOT / f"{cond}_{args.split}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"  Output already exists: {out_path}. Skipping.")
            continue

        records = []
        n = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            utt    = row["Utterance"]
            dia_id = row["Dialogue_ID"]
            utt_id = row["Utterance_ID"]
            masked = mask_utterance(utt)
            key    = f"dia{dia_id}_utt{utt_id}"

            audio_path = get_audio_path(args.split, dia_id, utt_id)
            has_audio  = audio_path is not None

            if has_audio:
                if args.model == "qwen":
                    audio_input = load_audio_array(audio_path)
                    has_audio   = audio_input is not None
                else:  # voxtral: pass path string
                    audio_input = str(audio_path)

            if has_audio:
                ctx = build_context(df, dia_id, utt_id) if base == "A3" else ""
                if base == "A1":
                    conv = build_conversation_A1(utt, audio_input, args.model)
                elif base == "A2":
                    conv = build_conversation_A2(masked, audio_input, args.model)
                else:
                    conv = build_conversation_A3(masked, audio_input, args.model, ctx)
            else:
                ctx  = build_context(df, dia_id, utt_id) if base == "A3" else ""
                conv = build_conversation_no_audio(base, utt, masked, ctx)

            if args.dry_run:
                if i < 3:
                    print(f"\n--- Sample {i} (has_audio={has_audio}) ---")
                    for msg in conv:
                        content = msg["content"]
                        if isinstance(content, list):
                            for ele in content:
                                if ele["type"] == "text":
                                    print(ele["text"])
                                elif ele["type"] == "audio":
                                    val = ele.get("path") or f"array shape={ele['audio'].shape}"
                                    print(f"[AUDIO: {val}]")
                        else:
                            print(content)
                continue

            generated = run_one(processor, model, conv, args.model)
            pred = parse_prediction(generated)

            records.append({
                "Sr_No":        getattr(row, "Sr No.", i),
                "Dialogue_ID":  dia_id,
                "Utterance_ID": utt_id,
                "Speaker":      row["Speaker"],
                "Utterance":    utt,
                "gold":         row["Emotion"],
                "audio_key":    key,
                "has_audio":    has_audio,
                "raw_output":   generated,
                "prediction":   pred,
                "condition":    cond,
                "split":        args.split,
                "model":        model_id,
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