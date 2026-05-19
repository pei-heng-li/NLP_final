"""
MELD ERC Unified Evaluation
Computes: Macro F1, Weighted F1, Masking Drop, Context Recovery,
          Prediction Change Rate, Confusion Matrix, per-class F1

Usage:
    python evaluate.py --split test
    python evaluate.py --split test --conditions T1 T2 COT DEF FS MCOT MDEF MFS
    python evaluate.py --split test --results_dir data/llama_1B_instruct
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report
)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS  = ["surprise", "anger", "neutral", "joy", "sadness", "fear", "disgust"]

OUT_ROOT  = Path("./results")
EVAL_ROOT = Path("./eval")

ALL_CONDITIONS = [
    "T1","T2","T3",
    "COT","DEF","FS",
    "M1","M2","M3",
    "MCOT","MDEF","MFS",
    "A1","A2","A3",
    "B1","B2","B3",
]


# ── Load Results ───────────────────────────────────────────────────────────────
def load_condition(condition: str, split: str, out_root: Path) -> pd.DataFrame | None:
    path = out_root / f"{condition}_{split}.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Normalise labels
    df["gold"]       = df["gold"].str.strip().str.lower()
    df["prediction"] = df["prediction"].str.strip().str.lower()
    return df


# ── Core Metrics ───────────────────────────────────────────────────────────────
def compute_f1(df: pd.DataFrame) -> dict:
    gold  = df["gold"].tolist()
    preds = df["prediction"].tolist()
    macro    = f1_score(gold, preds, average="macro",    labels=EMOTIONS, zero_division=0)
    weighted = f1_score(gold, preds, average="weighted", labels=EMOTIONS, zero_division=0)
    per_cls  = f1_score(gold, preds, average=None,       labels=EMOTIONS, zero_division=0)
    return {
        "macro_f1":    round(macro,    4),
        "weighted_f1": round(weighted, 4),
        "per_class":   {e: round(v, 4) for e, v in zip(EMOTIONS, per_cls)},
        "n":           len(df),
    }


def prediction_change_rate(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Fraction of utterances where prediction changed between two conditions."""
    merged = df_a[["Dialogue_ID","Utterance_ID","prediction"]].merge(
        df_b[["Dialogue_ID","Utterance_ID","prediction"]],
        on=["Dialogue_ID","Utterance_ID"],
        suffixes=("_a","_b"),
    )
    if len(merged) == 0:
        return float("nan")
    changed = (merged["prediction_a"] != merged["prediction_b"]).sum()
    return round(changed / len(merged), 4)


def compute_confusion(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    gold  = df["gold"].tolist()
    preds = df["prediction"].tolist()
    cm = confusion_matrix(gold, preds, labels=EMOTIONS)
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
    cm_df.index.name = f"True \\ Pred ({condition})"
    return cm_df


# ── Summary Table ──────────────────────────────────────────────────────────────
def build_summary(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for cond, res in results.items():
        row = {"Condition": cond,
               "N":           res["n"],
               "Macro F1":    res["macro_f1"],
               "Weighted F1": res["weighted_f1"]}
        for emo in EMOTIONS:
            row[f"F1_{emo}"] = res["per_class"].get(emo, 0.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Condition")


def masking_drop(results: dict, cond_a: str, cond_b: str) -> float | str:
    """F1(cond_a) - F1(cond_b); positive = cond_a better."""
    if cond_a not in results or cond_b not in results:
        return "N/A"
    return round(results[cond_a]["macro_f1"] - results[cond_b]["macro_f1"], 4)


def context_recovery(results: dict, cond_masked: str, cond_ctx: str) -> float | str:
    """F1(ctx) - F1(masked); positive = context helps."""
    if cond_masked not in results or cond_ctx not in results:
        return "N/A"
    return round(results[cond_ctx]["macro_f1"] - results[cond_masked]["macro_f1"], 4)


# ── Neutral Collapse Analysis ──────────────────────────────────────────────────
def neutral_collapse_rate(df: pd.DataFrame) -> dict:
    """
    For each minority emotion, compute fraction predicted as 'neutral'.
    Highlights disgust/fear/sadness → neutral collapse.
    """
    minority = ["disgust", "fear", "sadness", "surprise", "anger", "joy"]
    rates = {}
    for emo in minority:
        sub = df[df["gold"] == emo]
        if len(sub) == 0:
            rates[emo] = float("nan")
        else:
            rates[emo] = round((sub["prediction"] == "neutral").sum() / len(sub), 4)
    return rates


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train","dev","test"])
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of conditions to evaluate; default: all found")
    parser.add_argument("--results_dir", type=Path, default=OUT_ROOT,
                        help=f"Directory containing *_{{split}}.jsonl files; default: {OUT_ROOT}")
    args = parser.parse_args()

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    target_conditions = args.conditions if args.conditions else ALL_CONDITIONS
    out_root = args.results_dir

    # ── Load all available results ─────────────────────────────────────────────
    dfs: dict[str, pd.DataFrame] = {}
    results: dict[str, dict]     = {}

    print(f"\nLoading results for split: {args.split}")
    print(f"Results directory: {out_root}")
    for cond in target_conditions:
        df = load_condition(cond, args.split, out_root)
        if df is None:
            print(f"  [SKIP] {cond}: no output file found")
            continue
        dfs[cond] = df
        results[cond] = compute_f1(df)
        print(f"  [OK]   {cond}: macro_f1={results[cond]['macro_f1']:.4f}, "
              f"weighted_f1={results[cond]['weighted_f1']:.4f}, n={results[cond]['n']}")

    if not results:
        print("No results found. Run experiments first.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    summary = build_summary(results)
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (Macro F1 / Weighted F1 / Per-class F1)")
    print('='*70)
    print(summary.to_string())

    summary.to_csv(EVAL_ROOT / f"summary_{args.split}.csv")
    print(f"\nSaved → {EVAL_ROOT}/summary_{args.split}.csv")

    # ── Derived metrics ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DERIVED METRICS")
    print('='*70)

    derived = {}
    # Masking Drop: T1 → M1 (lexical cue reliance)
    derived["Masking_Drop_T1-M1"]      = masking_drop(results, "T1", "M1")
    # Context Recovery: M1 → M2, M1 → M3
    derived["Context_Recovery_M1-M2"]  = context_recovery(results, "M1", "M2")
    derived["Context_Recovery_M1-M3"]  = context_recovery(results, "M1", "M3")
    # Context Effect on T: T1 → T2, T1 → T3
    derived["Context_Gain_T1-T2"]      = masking_drop(results, "T2", "T1")
    derived["Context_Gain_T1-T3"]      = masking_drop(results, "T3", "T1")
    # Audio Effect: M1 → A2, M2 → A3
    derived["Audio_Gain_M1-A2"]        = masking_drop(results, "A2", "M1")
    derived["Audio_Gain_M2-A3"]        = masking_drop(results, "A3", "M2")
    # Prompting strategies compared to direct classification with dialogue.
    derived["Prompt_Gain_T2-COT"]       = masking_drop(results, "COT", "T2")
    derived["Prompt_Gain_T2-DEF"]       = masking_drop(results, "DEF", "T2")
    derived["Prompt_Gain_T2-FS"]        = masking_drop(results, "FS", "T2")
    # Masking effect under the same prompting strategy.
    derived["Masking_Drop_COT-MCOT"]     = masking_drop(results, "COT", "MCOT")
    derived["Masking_Drop_DEF-MDEF"]     = masking_drop(results, "DEF", "MDEF")
    derived["Masking_Drop_FS-MFS"]       = masking_drop(results, "FS", "MFS")
    # Prompting strategies compared to masked direct classification with dialogue.
    derived["Prompt_Gain_M2-MCOT"]       = masking_drop(results, "MCOT", "M2")
    derived["Prompt_Gain_M2-MDEF"]       = masking_drop(results, "MDEF", "M2")
    derived["Prompt_Gain_M2-MFS"]        = masking_drop(results, "MFS", "M2")

    for k, v in derived.items():
        print(f"  {k:<35} = {v}")

    with open(EVAL_ROOT / f"derived_{args.split}.json", "w") as f:
        json.dump(derived, f, indent=2)

    # ── Prediction Change Rates ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PREDICTION CHANGE RATES")
    print('='*70)

    pairs = [("T1","M1"), ("T2","M2"), ("T3","M3"),
             ("T1","T2"), ("T1","T3"), ("M1","A2"),
             ("T2","COT"), ("T2","DEF"), ("T2","FS"),
             ("COT","DEF"), ("COT","FS"), ("DEF","FS"),
             ("COT","MCOT"), ("DEF","MDEF"), ("FS","MFS"),
             ("M2","MCOT"), ("M2","MDEF"), ("M2","MFS"),
             ("MCOT","MDEF"), ("MCOT","MFS"), ("MDEF","MFS")]
    change_rates = {}
    for a, b in pairs:
        if a in dfs and b in dfs:
            rate = prediction_change_rate(dfs[a], dfs[b])
            change_rates[f"{a}→{b}"] = rate
            print(f"  {a} → {b}: {rate:.4f} ({rate*100:.1f}% changed)")

    with open(EVAL_ROOT / f"change_rates_{args.split}.json", "w") as f:
        json.dump(change_rates, f, indent=2)

    # ── Neutral Collapse ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("NEUTRAL COLLAPSE RATES (fraction of class predicted as 'neutral')")
    print('='*70)

    collapse_all = {}
    for cond in sorted(dfs.keys()):
        rates = neutral_collapse_rate(dfs[cond])
        collapse_all[cond] = rates
        row_str = "  ".join([f"{e}={v:.3f}" for e, v in rates.items()])
        print(f"  {cond:<5} | {row_str}")

    with open(EVAL_ROOT / f"neutral_collapse_{args.split}.json", "w") as f:
        json.dump(collapse_all, f, indent=2)

    # ── Confusion Matrices ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONFUSION MATRICES")
    print('='*70)

    for cond in sorted(dfs.keys()):
        cm = compute_confusion(dfs[cond], cond)
        cm_path = EVAL_ROOT / f"cm_{cond}_{args.split}.csv"
        cm.to_csv(cm_path)
        print(f"\n  [{cond}]")
        print(cm.to_string())

    # ── Per-class F1 heatmap table ─────────────────────────────────────────────
    cls_rows = []
    for cond, res in results.items():
        row = {"Condition": cond}
        row.update(res["per_class"])
        cls_rows.append(row)
    cls_df = pd.DataFrame(cls_rows).set_index("Condition")
    cls_df.to_csv(EVAL_ROOT / f"per_class_f1_{args.split}.csv")
    print(f"\n{'='*70}")
    print("PER-CLASS F1")
    print('='*70)
    print(cls_df.to_string())

    # ── Full classification report per condition ───────────────────────────────
    for cond, df in dfs.items():
        report = classification_report(
            df["gold"], df["prediction"],
            labels=EMOTIONS, zero_division=0,
        )
        rpt_path = EVAL_ROOT / f"report_{cond}_{args.split}.txt"
        with open(rpt_path, "w") as f:
            f.write(f"Condition: {cond}  |  split: {args.split}\n\n")
            f.write(report)

    print(f"\nAll evaluation files saved to {EVAL_ROOT}/")


if __name__ == "__main__":
    main()
