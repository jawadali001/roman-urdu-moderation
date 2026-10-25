"""Download the raw source datasets into data/raw/.

Sources
-------
* RUHSOLD  (Hugging Face: community-datasets/roman_urdu_hate_speech)
           coarse (binary) and fine-grained labels, official train/validation/test splits.
* RUACD    (GitHub: shaheerakr/roman-urdu-abusive-comment-detector)
           one CSV with columns `comment`, `rating` (TRUE = abusive).

Usage (from the repo root)
--------------------------
    python ml/src/rumod/data/download.py                 # download everything
    python ml/src/rumod/data/download.py --only ruacd    # just one source
    python ml/src/rumod/data/download.py --force         # re-download existing files

Raw files are never edited by hand. Cleaning happens later in clean.py and the
results go to data/interim/. data/raw/ is gitignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ml/src/rumod/data/download.py -> parents[4] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "raw"

RUHSOLD_HF_ID = "community-datasets/roman_urdu_hate_speech"
RUHSOLD_CONFIGS = ("Coarse_Grained", "Fine_Grained")

RUACD_URL = (
    "https://raw.githubusercontent.com/shaheerakr/"
    "roman-urdu-abusive-comment-detector/master/dataset.csv"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(folder: Path, entry: dict) -> None:
    """Keep a small record of what was downloaded, when, and from where."""
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[entry["file"]] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2))


def download_ruacd(force: bool = False) -> None:
    folder = RAW_DIR / "ruacd"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "dataset.csv"

    if target.exists() and not force:
        print(f"[ruacd] {target.relative_to(REPO_ROOT)} already exists, skipping (use --force)")
    else:
        print(f"[ruacd] downloading {RUACD_URL}")
        response = requests.get(RUACD_URL, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)

    df = pd.read_csv(target)
    print(f"[ruacd] rows={len(df):,} columns={list(df.columns)}")
    print(f"[ruacd] rating counts:\n{df['rating'].value_counts().to_string()}")

    write_manifest(
        folder,
        {
            "file": target.name,
            "source": RUACD_URL,
            "rows": int(len(df)),
            "sha256": sha256_of(target),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


def download_ruhsold(force: bool = False) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("The 'datasets' package is missing. Run: pip install datasets")

    folder = RAW_DIR / "ruhsold"
    folder.mkdir(parents=True, exist_ok=True)

    for config in RUHSOLD_CONFIGS:
        expected = [folder / f"{config}_{split}.csv" for split in ("train", "validation", "test")]
        if all(p.exists() for p in expected) and not force:
            print(f"[ruhsold] {config}: files already exist, skipping (use --force)")
            continue

        print(f"[ruhsold] downloading config {config}")
        dataset = load_dataset(RUHSOLD_HF_ID, config)

        for split, split_ds in dataset.items():
            target = folder / f"{config}_{split}.csv"
            split_ds.to_pandas().to_csv(target, index=False)

            label_feature = split_ds.features.get("label")
            label_names = getattr(label_feature, "names", None)
            print(f"[ruhsold] {config}/{split}: rows={len(split_ds):,} label_names={label_names}")

            write_manifest(
                folder,
                {
                    "file": target.name,
                    "source": f"hf://datasets/{RUHSOLD_HF_ID} ({config})",
                    "rows": int(len(split_ds)),
                    "label_names": label_names,
                    "sha256": sha256_of(target),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets into data/raw/")
    parser.add_argument("--only", choices=["ruhsold", "ruacd"], help="download a single source")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    if args.only in (None, "ruhsold"):
        download_ruhsold(force=args.force)
    if args.only in (None, "ruacd"):
        download_ruacd(force=args.force)

    print("\nDone. Raw files are in data/raw/ (gitignored).")


if __name__ == "__main__":
    main()