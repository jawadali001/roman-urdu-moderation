"""Check the hand-labeled files and report how close you are to the targets.

    python ml/scripts/check_labeled_batch.py

Prints counts only (no tweet text), so the output is safe to share.
"""
from __future__ import annotations

from rumod.data.real_set import (
    ADMIN_LABELS,
    EVAL_LABELS,
    TARGET_PER_CLASS,
    load_labeled_batches,
    summarize,
    validate,
)


def main() -> None:
    df = load_labeled_batches()
    summary = summarize(df)

    print(f"rows: {summary['rows']:,}   labeled: {summary['labeled']:,}   "
          f"not yet labeled: {summary['unlabeled']:,}")
    print(f"\n{'label':<16}{'all rows':>9}{'random rows':>13}   target {TARGET_PER_CLASS}")
    for label in EVAL_LABELS:
        n, n_random = summary["counts"][label], summary["counts_random_only"][label]
        mark = "ok" if n >= TARGET_PER_CLASS else f"needs {TARGET_PER_CLASS - n} more"
        print(f"{label:<16}{n:>9}{n_random:>13}   {mark}")
    for label in ADMIN_LABELS:
        print(f"{label:<16}{summary['counts'][label]:>9}   (not scored)")
    print(f"\nclean messages marked 'looks risky but clean': {summary['hard_negatives']}")

    problems = validate(df)
    print("\nproblems:" if problems else "\nno problems found")
    for problem in problems:
        print(f"  - {problem}")

    if summary["shortfall"]:
        most = max(summary["shortfall"].values())
        print(f"\nTo fill the gaps, sample another batch, for example:\n"
              f"  python ml/scripts/sample_for_labeling.py --batch 2 --random 100 "
              f"--per-class {max(40, most * 2)}")


if __name__ == "__main__":
    main()