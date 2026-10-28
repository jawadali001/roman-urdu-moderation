"""Agreement between our labels and RUHSOLD's official labels.

The study sample is stratified (the same number of tweets per official class),
which is not how classes occur naturally. Raw agreement and kappa on that sample
therefore describe the sample. We also report prevalence-weighted versions: each
row counts in proportion to how common its official class is in RUHSOLD, which
estimates what agreement would look like on natural data. Confidence intervals
come from a stratified bootstrap (resampling inside each official class).

Kappa is reported for the 5 classes and for toxic-vs-clean only. A low 5-class
kappa with a high toxic-vs-clean kappa means we agree on what is toxic and only
draw the boundaries between toxic classes differently.
"""
from __future__ import annotations

import numpy as np

from rumod.utils.progress import track

CLEAN = "clean"
SKIPPED_LABELS = ("unsure", "unusable", "")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def confusion_matrix(official, mine, labels, weights=None) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    rows = [index[o] for o in official]
    cols = [index[m] for m in mine]
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    np.add.at(matrix, (rows, cols), w)
    return matrix


def kappa_from_matrix(matrix: np.ndarray) -> float:
    """Cohen's kappa from a (possibly weighted) confusion matrix."""
    total = matrix.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(matrix) / total
    expected = float(matrix.sum(axis=0) @ matrix.sum(axis=1)) / total**2
    if expected == 1:
        return float("nan")
    return float((observed - expected) / (1 - expected))


def _stats(official, mine, labels, prior) -> dict:
    counts = {label: int((official == label).sum()) for label in labels}
    weights = np.array([prior[o] / counts[o] for o in official])
    plain = confusion_matrix(official, mine, labels)
    weighted = confusion_matrix(official, mine, labels, weights)
    return {
        "agreement": float((official == mine).mean()),
        "kappa": kappa_from_matrix(plain),
        "weighted_agreement": float(np.trace(weighted) / weighted.sum()),
        "weighted_kappa": kappa_from_matrix(weighted),
    }


def _bootstrap(official, mine, labels, prior, n_boot, seed, progress, desc) -> dict:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(official == label) for label in labels]
    groups = [g for g in groups if len(g)]
    samples = {key: [] for key in ("agreement", "kappa", "weighted_agreement", "weighted_kappa")}
    for _ in track(range(n_boot), desc, enabled=progress):
        picked = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        for key, value in _stats(official[picked], mine[picked], labels, prior).items():
            samples[key].append(value)
    return {
        key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for key, v in samples.items()
    }


def _block(official, mine, labels, prior, n_boot, seed, progress, desc) -> dict:
    point = _stats(official, mine, labels, prior)
    intervals = _bootstrap(official, mine, labels, prior, n_boot, seed, progress, desc)
    return {key: {"value": point[key], "ci": intervals[key]} for key in point}


def split_comparable(official, mine) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop rows we labeled unsure/unusable or left blank; report how many."""
    official, mine = np.asarray(official), np.asarray(mine)
    keep = ~np.isin(mine, SKIPPED_LABELS)
    skipped = {label: int((mine == label).sum()) for label in ("unsure", "unusable", "")}
    return official[keep], mine[keep], {"skipped": skipped}


def agreement_report(
    official, mine, labels: list[str], prior: dict, n_boot: int = 1000, seed: int = 42,
    progress: bool = False,
) -> dict:
    """Full report. `prior` maps each official label to its natural share (sums to 1)."""
    official, mine = np.asarray(official), np.asarray(mine)
    binary_prior = {CLEAN: prior[CLEAN], "toxic": 1 - prior[CLEAN]}
    to_binary = lambda arr: np.where(arr == CLEAN, CLEAN, "toxic")  # noqa: E731

    counts = confusion_matrix(official, mine, labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_share = np.nan_to_num(counts / counts.sum(axis=1, keepdims=True))

    per_class = {}
    for i, label in enumerate(labels):
        n = int(counts[i].sum())
        agreed = int(counts[i, i])
        others = counts[i].copy()
        others[i] = 0
        alt = int(others.argmax()) if others.sum() else None
        per_class[label] = {
            "n": n, "agree": agreed,
            "agreement": agreed / n if n else None,
            "ci": list(wilson_interval(agreed, n)),
            "main_alternative": labels[alt] if alt is not None else None,
            "main_alternative_share": float(others[alt] / n) if alt is not None else None,
        }

    return {
        "n": int(len(official)),
        "five_class": _block(
            official, mine, labels, prior, n_boot, seed, progress, "bootstrap (5 classes)"
        ),
        "toxic_vs_clean": _block(
            to_binary(official), to_binary(mine), [CLEAN, "toxic"], binary_prior,
            n_boot, seed, progress, "bootstrap (toxic vs clean)",
        ),
        "per_class": per_class,
        "confusion": {"labels": list(labels), "counts": counts.astype(int).tolist(),
                      "row_share": row_share.tolist()},
    }


def format_agreement(report: dict) -> str:
    def line(name, block):
        parts = []
        for key, title in (("agreement", "agreement"), ("kappa", "kappa"),
                           ("weighted_agreement", "weighted agreement"),
                           ("weighted_kappa", "weighted kappa")):
            b = block[key]
            parts.append(f"{title} {b['value']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]")
        return f"{name}\n  " + "\n  ".join(parts)

    out = [f"compared {report['n']} tweets (your label vs the official RUHSOLD label)\n",
           line("5 classes", report["five_class"]), "",
           line("toxic vs clean only", report["toxic_vs_clean"]),
           "\nper official class: how often your label matched",
           f"{'official class':<16}{'n':>4}{'matched':>9}  {'95% CI':<14}most common other label"]
    for label, c in report["per_class"].items():
        ci = f"[{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]" if c["agreement"] is not None else "-"
        rate = f"{c['agreement']:.0%}" if c["agreement"] is not None else "-"
        alt = (f"{c['main_alternative']} ({c['main_alternative_share']:.0%})"
               if c["main_alternative"] else "-")
        out.append(f"{label:<16}{c['n']:>4}{rate:>9}  {ci:<14}{alt}")
    labels = report["confusion"]["labels"]
    out.append("\nconfusion (rows = official, columns = yours), counts")
    out.append(" " * 16 + "".join(f"{lab[:8]:>9}" for lab in labels))
    for lab, row in zip(labels, report["confusion"]["counts"], strict=True):
        out.append(f"{lab:<16}" + "".join(f"{v:>9}" for v in row))
    out.append("\nweighted = counted by how common each official class is in RUHSOLD "
               "(the sample itself has the same number per class).")
    return "\n".join(out)