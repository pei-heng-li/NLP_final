"""
MELD ERC Baselines
  B1: Always predict 'neutral'
  B2: Rule-based keyword matching → else 'neutral'
  B3: RoBERTa-base fine-tuned on MELD train set

Usage:
    python run_baselines.py --baselines B1 B2 B3 --split test
"""

import argparse
import json
import re
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS   = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]
ID2LABEL   = {i: e for i, e in enumerate(EMOTIONS)}
LABEL2ID   = {e: i for i, e in enumerate(EMOTIONS)}
DATA_ROOT  = Path("/tmp2/b11902128/NLP/MELD.Raw")
OUT_ROOT   = Path("/tmp2/b11902128/NLP/results")
MODEL_DIR  = Path("/tmp2/b11902128/NLP/roberta_meld")
LEXICON_PATH = Path(__file__).parent / "emotion_lexicon.json"

# B2 keyword → emotion (loaded from shared lexicon, multi-word phrases skipped)
with open(LEXICON_PATH, encoding="utf-8") as _f:
    _EMOTION_KEYWORDS: dict[str, list[str]] = json.load(_f)

KEYWORD_MAP: dict[str, str] = {}
for _emo, _kws in _EMOTION_KEYWORDS.items():
    if _emo == "neutral":   # neutral keywords → fallback, don't map explicitly
        continue
    for _kw in _kws:
        if " " not in _kw:  # single-word only; multi-word handled separately
            KEYWORD_MAP[_kw] = _emo

# Multi-word phrases sorted longest-first for greedy matching
MULTI_WORD_PHRASES: list[tuple[str, str]] = sorted(
    [(_kw, _emo) for _emo, _kws in _EMOTION_KEYWORDS.items()
     for _kw in _kws if " " in _kw and _emo != "neutral"],
    key=lambda x: len(x[0]), reverse=True,
)


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_split(split: str) -> pd.DataFrame:
    fname = {"train": "train_sent_emo.csv",
             "dev":   "dev_sent_emo.csv",
             "test":  "test_sent_emo.csv"}[split]
    df = pd.read_csv(DATA_ROOT / fname)
    df.columns = df.columns.str.strip()
    df["Emotion"] = df["Emotion"].str.strip().str.lower()
    return df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)


def save_results(records: list[dict], condition: str, split: str):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{condition}_{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


# ── B1: Always neutral ─────────────────────────────────────────────────────────
def run_B1(df: pd.DataFrame, split: str):
    print("\nRunning B1: always neutral …")
    records = []
    for _, row in df.iterrows():
        records.append({
            "Dialogue_ID":  row["Dialogue_ID"],
            "Utterance_ID": row["Utterance_ID"],
            "Speaker":      row["Speaker"],
            "Utterance":    row["Utterance"],
            "gold":         row["Emotion"],
            "prediction":   "neutral",
            "condition":    "B1",
            "split":        split,
        })
    save_results(records, "B1", split)


# ── B2: Rule-based keyword matching ───────────────────────────────────────────
def b2_predict(utterance: str) -> str:
    text_lower = utterance.lower()
    # Check multi-word phrases first (longest-first greedy)
    for phrase, emo in MULTI_WORD_PHRASES:
        if phrase in text_lower:
            return emo
    # Then single-word tokens
    tokens = re.findall(r"[a-z]+", text_lower)
    for tok in tokens:
        if tok in KEYWORD_MAP:
            return KEYWORD_MAP[tok]
    return "neutral"


def run_B2(df: pd.DataFrame, split: str):
    print("\nRunning B2: rule-based keyword …")
    records = []
    for _, row in df.iterrows():
        pred = b2_predict(row["Utterance"])
        records.append({
            "Dialogue_ID":  row["Dialogue_ID"],
            "Utterance_ID": row["Utterance_ID"],
            "Speaker":      row["Speaker"],
            "Utterance":    row["Utterance"],
            "gold":         row["Emotion"],
            "prediction":   pred,
            "condition":    "B2",
            "split":        split,
        })
    save_results(records, "B2", split)


# ── B3: RoBERTa fine-tuned on MELD ────────────────────────────────────────────
class MELDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            df["Utterance"].tolist(),
            truncation=True, padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = [LABEL2ID[e] for e in df["Emotion"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}


def train_roberta(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    roberta_name = "roberta-base"
    print(f"\nFine-tuning {roberta_name} on MELD train …")
    tokenizer = AutoTokenizer.from_pretrained(roberta_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        roberta_name,
        num_labels=len(EMOTIONS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = MELDDataset(train_df, tokenizer)
    dev_dataset   = MELDDataset(dev_df,   tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"  Model saved to {MODEL_DIR}")
    return MODEL_DIR


def run_B3(df: pd.DataFrame, split: str, do_train: bool = False):
    print("\nRunning B3: RoBERTa fine-tuned …")

    if do_train or not MODEL_DIR.exists():
        train_df = load_split("train")
        dev_df   = load_split("dev")
        train_roberta(train_df, dev_df)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    utterances = df["Utterance"].tolist()
    batch_size = 64
    all_preds = []
    for i in range(0, len(utterances), batch_size):
        batch = utterances[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend([ID2LABEL[p] for p in preds])
        print(f"  [{min(i+batch_size, len(utterances))}/{len(utterances)}]", end="\r")
    print()

    records = []
    for (_, row), pred in zip(df.iterrows(), all_preds):
        records.append({
            "Dialogue_ID":  row["Dialogue_ID"],
            "Utterance_ID": row["Utterance_ID"],
            "Speaker":      row["Speaker"],
            "Utterance":    row["Utterance"],
            "gold":         row["Emotion"],
            "prediction":   pred,
            "condition":    "B3",
            "split":        split,
        })
    save_results(records, "B3", split)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+",
                        default=["B1","B2","B3"],
                        choices=["B1","B2","B3"])
    parser.add_argument("--split", default="test", choices=["train","dev","test"])
    parser.add_argument("--retrain_b3", action="store_true",
                        help="Force re-training of B3 even if checkpoint exists")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_split(args.split)
    print(f"Loaded {len(df)} utterances ({args.split}).")

    for b in args.baselines:
        out_path = OUT_ROOT / f"{b}_{args.split}.jsonl"
        if out_path.exists():
            print(f"  {b} output exists, skipping.")
            continue
        if b == "B1":
            run_B1(df, args.split)
        elif b == "B2":
            run_B2(df, args.split)
        elif b == "B3":
            run_B3(df, args.split, do_train=args.retrain_b3)


if __name__ == "__main__":
    main()