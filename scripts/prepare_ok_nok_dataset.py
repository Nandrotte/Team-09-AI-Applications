"""Extract the workshop OK/NOK ZIP into a reproducible, bottle-grouped split."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


def group_key(filename: str) -> str:
    """Group alternate views of the same bottle; filenames encode `_id_<id>`."""
    stem = Path(filename).stem
    match = re.match(r"^(.*?_id_[^_]+)", stem, re.IGNORECASE)
    return match.group(1).lower() if match else stem.lower()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=root.parent / "ai_applications_ok_nok_classification_training_set.zip")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "ok_nok")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.zip.is_file():
        parser.error(f"dataset ZIP not found: {args.zip}")

    with zipfile.ZipFile(args.zip) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
        grouped: dict[str, dict[str, list[zipfile.ZipInfo]]] = defaultdict(lambda: defaultdict(list))
        for member in members:
            parts = Path(member.filename).parts
            if len(parts) < 3 or parts[-2].lower() not in {"ok", "nok"}:
                continue
            label = parts[-2].lower()
            grouped[label][group_key(parts[-1])].append(member)

        rng = random.Random(args.seed)
        assignment: dict[tuple[str, str], str] = {}
        for label, groups in grouped.items():
            keys = sorted(groups)
            rng.shuffle(keys)
            n = len(keys)
            n_test = max(1, round(n * 0.15))
            n_val = max(1, round(n * 0.15))
            for i, key in enumerate(keys):
                split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
                assignment[(label, key)] = split

        # Only replace this generated output folder; the ZIP and source images stay untouched.
        if args.output.exists():
            shutil.rmtree(args.output)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for label, groups in grouped.items():
            for key, images in groups.items():
                split = assignment[(label, key)]
                for member in images:
                    destination = args.output / split / label / Path(member.filename).name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    counts[split][label] += 1
        print("Split counts (images; all views of a bottle stay together):")
        for split in ("train", "val", "test"):
            print(f"{split}: OK={counts[split]['ok']} NOK={counts[split]['nok']}")
        print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
