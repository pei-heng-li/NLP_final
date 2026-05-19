"""
MELD ERC Experiment: Text Conditions (T1, T2, T3, M1, M2, M3)
Run on meow2 under /tmp2/b11902128/NLP/
Usage:
    python run_text_conditions.py --conditions T1 T2 T3 M1 M2 M3 --split test
"""

import argparse
import json
import os
import re
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]
EMOTION_SET = set(EMOTIONS)
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_ROOT = Path("./MELD.Raw")
OUT_ROOT  = Path("./results")

# Emotion keywords: loaded from emotion_lexicon.json (same directory as this script)
LEXICON_PATH = Path(__file__).parent / "emotion_lexicon.json"
with open(LEXICON_PATH, encoding="utf-8") as _f:
    EMOTION_KEYWORDS: dict[str, list[str]] = json.load(_f)

# Single-word keywords (flat set) and multi-word phrases handled separately
ALL_KEYWORDS: set[str] = set()
MULTI_WORD_PHRASES: list[str] = []
for _kws in EMOTION_KEYWORDS.values():
    for _kw in _kws:
        if " " in _kw:
            MULTI_WORD_PHRASES.append(_kw)
        else:
            ALL_KEYWORDS.add(_kw)
# Sort longest first so overlapping phrases match greedily
MULTI_WORD_PHRASES.sort(key=len, reverse=True)


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_split(split: str) -> pd.DataFrame:
    """Load train/dev/test CSV and return a cleaned DataFrame."""
    fname = {"train": "train_sent_emo.csv",
             "dev":   "dev_sent_emo.csv",
             "test":  "test_sent_emo.csv"}[split]
    df = pd.read_csv(DATA_ROOT / fname)
    df.columns = df.columns.str.strip()
    # Normalise emotion labels to lowercase
    df["Emotion"] = df["Emotion"].str.strip().str.lower()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    return df


# ── Masking ────────────────────────────────────────────────────────────────────
def mask_utterance(utterance: str) -> str:
    """
    Replace emotion-bearing words/phrases with [MASK].
    Multi-word phrases (e.g. "oh my god") are matched first (longest first),
    then single tokens are matched against ALL_KEYWORDS.
    Punctuation attached to [MASK] (e.g. "[MASK],") is preserved.
    """
    # Step 1: replace multi-word phrases (case-insensitive)
    text = utterance
    for phrase in MULTI_WORD_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("[MASK]", text)

    # Step 2: token-level masking for single words
    tokens = text.split()
    masked = []
    for tok in tokens:
        # Already masked (possibly with punctuation attached, e.g. "[MASK],")
        if "[MASK]" in tok:
            masked.append(tok)
        else:
            clean = re.sub(r"[^a-z]", "", tok.lower())
            if clean in ALL_KEYWORDS:
                # Preserve trailing punctuation (e.g. "sad," → "[MASK],")
                trail = re.sub(r"^[a-zA-Z\[\]]+", "", tok)
                masked.append("[MASK]" + trail)
            else:
                masked.append(tok)
    return " ".join(masked)


# ── Context Builder ────────────────────────────────────────────────────────────
def build_context(df: pd.DataFrame, dia_id: int, utt_id: int,
                  use_masked_target: bool = False) -> tuple[str, str]:
    """
    Return (context_str, target_utterance_str).
    Context = all prior turns in the same dialogue.
    """
    dia = df[df["Dialogue_ID"] == dia_id].sort_values("Utterance_ID")
    prior = dia[dia["Utterance_ID"] < utt_id]
    target_row = dia[dia["Utterance_ID"] == utt_id].iloc[0]

    context_lines = []
    for _, row in prior.iterrows():
        context_lines.append(f'{row["Speaker"]}: "{row["Utterance"]}"')
    context_str = "\n".join(context_lines)

    target_utt = target_row["Utterance"]
    if use_masked_target:
        target_utt = mask_utterance(target_utt)

    return context_str, target_utt


# ── Prompt Builders ────────────────────────────────────────────────────────────
EMOTION_OPTS = ", ".join(EMOTIONS)

def prompt_T1(speaker: str, utterance: str) -> str:
    return (
        "This is a single-choice question.\n\n"
        "You will be given a target utterance from a conversation.\n"
        "Your task is to determine the emotion of the speaker when they said the target utterance.\n\n"
        f"Target speaker: {speaker}\n"
        f'Target utterance: "{utterance}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_T2(speaker: str, utterance: str, context: str) -> str:
    ctx_block = f"Conversation:\n{context}\n\n" if context else ""
    return (
        "This is a single-choice question.\n\n"
        "You will be given a conversation and a target utterance.\n"
        "Your task is to determine the emotion of the target speaker when they said the target utterance.\n\n"
        f"{ctx_block}"
        f"Target speaker: {speaker}\n"
        f'Target utterance: "{utterance}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_T3(speaker: str, utterance: str, context: str) -> str:
    ctx_block = f"Conversation:\n{context}\n\n" if context else ""
    return (
        "This is a single-choice question.\n\n"
        "You will be given a conversation and a target utterance.\n"
        "Your task is to determine the emotion of the target speaker when they said the target utterance.\n\n"
        f"{ctx_block}"
        f"Target speaker: {speaker}\n"
        f'Target utterance: "{utterance}"\n\n'
        f"Note: Focus on the emotional state of {speaker} specifically.\n\n"
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_M1(speaker: str, masked_utt: str) -> str:
    return (
        "This is a single-choice question.\n\n"
        "You will be given a target utterance from a conversation.\n"
        "Some emotion-bearing words have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the speaker when they said the target utterance.\n\n"
        f"Target speaker: {speaker}\n"
        f'Target utterance: "{masked_utt}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_M2(speaker: str, masked_utt: str, context: str) -> str:
    ctx_block = f"Conversation:\n{context}\n\n" if context else ""
    return (
        "This is a single-choice question.\n\n"
        "You will be given a conversation and a target utterance.\n"
        "Some emotion-bearing words in the target utterance have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the target speaker when they said the target utterance.\n\n"
        f"{ctx_block}"
        f"Target speaker: {speaker}\n"
        f'Target utterance: "{masked_utt}"\n\n'
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )

def prompt_M3(speaker: str, masked_utt: str, context: str) -> str:
    ctx_block = f"Conversation:\n{context}\n\n" if context else ""
    return (
        "This is a single-choice question.\n\n"
        "You will be given a conversation and a target utterance.\n"
        "Some emotion-bearing words in the target utterance have been replaced with [MASK].\n"
        "Your task is to determine the emotion of the target speaker when they said the target utterance.\n\n"
        f"{ctx_block}"
        f"Target speaker: {speaker}\n"
        f'Target utterance: "{masked_utt}"\n\n'
        f"Note: Focus on the emotional state of {speaker} specifically.\n\n"
        f"Choose one emotion from the following options:\n{EMOTION_OPTS}\n\n"
        "Answer with only one label."
    )


# ── Build prompt for a row ─────────────────────────────────────────────────────
def build_prompt(condition: str, row: pd.Series, df: pd.DataFrame) -> tuple[str, dict]:
    """Return (prompt_str, metadata) where metadata holds context/masked_utterance for saving."""
    speaker  = row["Speaker"]
    utt      = row["Utterance"]
    dia_id   = row["Dialogue_ID"]
    utt_id   = row["Utterance_ID"]
    masked   = mask_utterance(utt)

    if condition == "T1":
        prompt = prompt_T1(speaker, utt)
        meta   = {"context": None, "masked_utterance": None}
    elif condition == "T2":
        ctx, _ = build_context(df, dia_id, utt_id, use_masked_target=False)
        prompt = prompt_T2(speaker, utt, ctx)
        meta   = {"context": ctx, "masked_utterance": None}
    elif condition == "T3":
        ctx, _ = build_context(df, dia_id, utt_id, use_masked_target=False)
        prompt = prompt_T3(speaker, utt, ctx)
        meta   = {"context": ctx, "masked_utterance": None}
    elif condition == "M1":
        prompt = prompt_M1(speaker, masked)
        meta   = {"context": None, "masked_utterance": masked}
    elif condition == "M2":
        ctx, _ = build_context(df, dia_id, utt_id, use_masked_target=False)
        prompt = prompt_M2(speaker, masked, ctx)
        meta   = {"context": ctx, "masked_utterance": masked}
    elif condition == "M3":
        ctx, _ = build_context(df, dia_id, utt_id, use_masked_target=False)
        prompt = prompt_M3(speaker, masked, ctx)
        meta   = {"context": ctx, "masked_utterance": masked}
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return prompt, meta


# ── Response Parsing ───────────────────────────────────────────────────────────
def parse_prediction(text: str) -> str:
    """Extract the first valid emotion label from model output."""
    text_lower = text.strip().lower()
    # Try exact match on first word/line
    for line in text_lower.splitlines():
        line = line.strip().rstrip(".")
        if line in EMOTION_SET:
            return line
    # Fallback: find first emotion keyword anywhere
    for emo in EMOTIONS:
        if emo in text_lower:
            return emo
    return "neutral"  # ultimate fallback


# ── Model Inference ────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Fix: set pad_token so batched pipeline works
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
    """Run batched inference; returns list of generated-only strings."""
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size,
        return_full_text=False,   # return generated tokens only, not the prompt
    )

    results = []
    total = len(prompts)
    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]
        # Format as chat messages
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
                        default=["T1","T2","T3","M1","M2","M3"],
                        choices=["T1","T2","T3","M1","M2","M3"])
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

        # Build prompts
        prompts = []
        metas   = []
        for _, row in df.iterrows():
            p, m = build_prompt(cond, row, df)
            prompts.append(p)
            metas.append(m)

        if args.dry_run:
            for i in range(min(3, len(prompts))):
                print(f"\n--- Sample {i} ---\n{prompts[i]}\n")
            continue

        # Inference
        raw_outputs = run_inference(
            tokenizer, model, prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        # Parse and save
        records = []
        for idx, (row_tuple, raw) in enumerate(zip(df.itertuples(), raw_outputs)):
            # pipeline return_full_text=False → raw is already generated-only
            generated = raw.strip()
            pred = parse_prediction(generated)
            meta = metas[idx]
            records.append({
                "Sr_No":            getattr(row_tuple, "Sr_No", idx),
                "Dialogue_ID":      row_tuple.Dialogue_ID,
                "Utterance_ID":     row_tuple.Utterance_ID,
                "Speaker":          row_tuple.Speaker,
                "Utterance":        row_tuple.Utterance,
                "masked_utterance": meta["masked_utterance"],
                "context":          meta["context"],
                "gold":             row_tuple.Emotion,
                "raw_output":       generated,
                "prediction":       pred,
                "condition":        cond,
                "split":            args.split,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()