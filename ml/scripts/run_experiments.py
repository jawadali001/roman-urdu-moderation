"""Run a small, fixed set of experiments and print one comparison table.

Two groups, each answering one question:

  data   Does adding RUACD help on RUHSOLD, and how well does a model trained on
         one source work on the other (cross-dataset)?
           ruhsold_only, ruacd_only, both_cap10k (the baseline), both_cap3k
  model  How much do class weights and regularization matter?
           class_weight {balanced, sqrt, none} x C {1, 4, 16}, on both_cap10k

Everything is scored on the VALIDATION set only. The test set is not touched,
so it stays usable for one final, honest number. Because several variants are
compared on the same validation set, small differences (within the confidence
intervals) should not be over-read.

Run from the repo root:

    python ml/scripts/run_experiments.py                 # both groups (several minutes)
    python ml/scripts/run_experiments.py --groups data   # the quick group (about 1-2 min)

Output: a printed table and ml/artifacts/experiments_validation.json
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from tqdm import tqdm

from rumod.data.schema import REPO_ROOT
from rumod.data.split import cap_per_label
from rumod.evaluation.metrics import evaluate_by_source
from rumod.models.predict import ModerationModel

DATA_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"
SEED = 42

BOTH = ("ruhsold", "ruacd")

# (name, sources used for training, RUACD cap per class or None, model params)
DATA_GROUP = [
    ("ruhsold_only", ("ruhsold",), None, {}),
    ("ruacd_only", ("ruacd",), 10_000, {}),
    ("both_cap10k", BOTH, 10_000, {}),
    ("both_cap3k", BOTH, 3_000, {}),
]
MODEL_GROUP = [
    (f"cw={cw}_C={c}", BOTH, 10_000, {"class_weight": None if cw == "none" else cw, "C": c})
    for cw in ("balanced", "sqrt", "none")
    for c in (1.0, 4.0, 16.0)
]


def build_train(
    train: pd.DataFrame, sources: tuple[str, ...], ruacd_cap: int | None
) -> pd.DataFrame:
    parts = []
    for source in sources:
        part = train[train["source"] == source]
        if source == "ruacd" and ruacd_cap:
            part = cap_per_label(part, ruacd_cap, SEED)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize(report: dict) -> dict:
    summary = {}
    for source, rep in report.items():
        tv = rep["toxic_vs_clean"]
        summary[source] = {
            "macro_f1": rep["macro_f1"],
            "macro_f1_ci": rep["macro_f1_ci"],
            "toxic_f1": tv["f1"],
            "toxic_recall": tv["recall"],
            "false_alarm_rate": tv["false_alarm_rate"],
            "f1_per_class": {k: v["f1"] for k, v in rep["per_class"].items() if v["support"] > 0},
        }
    return summary


def print_table(rows: list[dict]) -> None:
    head = (
        f"{'experiment':<20}{'train':>7} | {'RUHSOLD macro-F1 [95% CI]':<27}"
        f"{'toxic F1':>9}{'recall':>8}{'false alarms':>13} | "
        f"{'RUACD F1':>9}{'false alarms':>13}"
    )
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        h, a = r["ruhsold"], r["ruacd"]
        ci = f"{h['macro_f1']:.3f} [{h['macro_f1_ci'][0]:.3f}, {h['macro_f1_ci'][1]:.3f}]"
        print(f"{r['name']:<20}{r['train_rows']:>7,} | {ci:<27}{h['toxic_f1']:>9.3f}"
              f"{h['toxic_recall']:>8.3f}{h['false_alarm_rate']:>13.1%} | "
              f"{a['macro_f1']:>9.3f}{a['false_alarm_rate']:>13.1%}")
    print("\nfalse alarms = share of genuinely clean messages that get flagged.")
    print("ruacd_only cannot predict profane/religious_hate/sexual_hate, so its RUHSOLD macro-F1 "
          "is low by construction; compare its toxic F1 instead.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline experiments on validation data")
    parser.add_argument("--groups", nargs="+", choices=["data", "model"], default=["data", "model"])
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    validation = pd.read_csv(DATA_DIR / "validation.csv", keep_default_na=False)

    experiments = []
    if "data" in args.groups:
        experiments += DATA_GROUP
    if "model" in args.groups:
        experiments += MODEL_GROUP

    rows = []
    bar = tqdm(experiments, desc="experiments", unit="exp", dynamic_ncols=True)
    for name, sources, cap, params in bar:
        bar.set_postfix_str(name)
        subset = build_train(train, sources, cap)
        started = time.time()
        model = ModerationModel.fit(subset["text"], subset["label"], **params)
        labels = sorted(set(model.labels) | set(validation["label"]))
        report = evaluate_by_source(
            validation, model.predict_labels(validation["text"]), labels, n_boot=args.n_boot
        )
        row = {"name": name, "train_rows": len(subset), "sources": list(sources),
               "ruacd_cap": cap, "params": {k: str(v) for k, v in params.items()},
               **summarize(report)}
        rows.append(row)
        tqdm.write(f"  finished {name} in {time.time() - started:.0f}s")

    print_table(rows)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "experiments_validation.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()