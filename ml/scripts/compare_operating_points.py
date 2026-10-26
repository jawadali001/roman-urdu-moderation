"""Compare training setups without picking a threshold.

For each setup we train a model and score the validation set, then report, per
source, the ROC-AUC of the toxic score and the recall we get when at most 2%,
5% or 10% of clean messages may be flagged.

Why: the experiments table compares models at one fixed threshold, so a model
that is simply more cautious looks different from one that is better. Comparing
recall at the same false-alarm rate removes that effect.

Validation set only. The test set is not touched.

Run from the repo root:

    python ml/scripts/compare_operating_points.py
    python ml/scripts/compare_operating_points.py --variants ruhsold_only both_cap3k

Output: a printed table and ml/artifacts/operating_points_validation.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

from rumod.data.schema import REPO_ROOT
from rumod.evaluation.operating_points import DEFAULT_TARGETS, operating_summary, toxic_scores
from rumod.models.predict import ModerationModel

DATA_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"

# name -> (training sources, RUACD cap per class)
VARIANTS = {
    "ruhsold_only": (("ruhsold",), None),
    "both_cap1k": (("ruhsold", "ruacd"), 1_000),
    "both_cap3k": (("ruhsold", "ruacd"), 3_000),
    "both_cap10k": (("ruhsold", "ruacd"), 10_000),
}


def build_train(train: pd.DataFrame, sources, cap) -> pd.DataFrame:
    from rumod.data.split import cap_per_label

    parts = []
    for source in sources:
        part = train[train["source"] == source]
        if source == "ruacd" and cap:
            part = cap_per_label(part, cap, 42)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def print_table(results: dict) -> None:
    pct = [f"R@{t:.0%}" for t in DEFAULT_TARGETS]
    head = (f"{'variant':<14}{'rows':>7} | {'RUHSOLD AUC':>11}" + "".join(f"{p:>8}" for p in pct)
            + f"{'thr@5%':>8} | {'RUACD AUC':>9}" + "".join(f"{p:>8}" for p in pct))
    print("\n" + head)
    print("-" * len(head))
    for name, r in results.items():
        h, a = r["ruhsold"], r["ruacd"]
        thr = h["at"][1]["threshold"]
        thr_text = f"{thr:.2f}" if thr is not None else "-"
        print(f"{name:<14}{r['train_rows']:>7,} | {h['auc']:>11.3f}"
              + "".join(f"{p['recall']:>8.3f}" for p in h["at"]) + f"{thr_text:>8} | "
              + f"{a['auc']:>9.3f}" + "".join(f"{p['recall']:>8.3f}" for p in a["at"]))
    print("\nR@x% = share of toxic messages caught when at most x% of clean messages are flagged.")
    print("thr@5% = toxic-score threshold (1 - P(clean)) giving a 5% false-alarm rate on RUHSOLD.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold-free comparison of training setups")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    validation = pd.read_csv(DATA_DIR / "validation.csv", keep_default_na=False)

    results = {}
    bar = tqdm(args.variants, desc="variants", unit="variant", dynamic_ncols=True)
    for name in bar:
        bar.set_postfix_str(name)
        sources, cap = VARIANTS[name]
        subset = build_train(train, sources, cap)
        model = ModerationModel.fit(subset["text"], subset["label"])
        scores = pd.Series(toxic_scores(model, validation["text"]), index=validation.index)
        entry: dict[str, object] = {"train_rows": len(subset)}
        for source, part in validation.groupby("source"):
            y_toxic = (part["label"] != "clean").to_numpy()
            entry[source] = operating_summary(y_toxic, scores.loc[part.index].to_numpy())
        results[name] = entry
        tqdm.write(f"  finished {name}")

    print_table(results)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "operating_points_validation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()