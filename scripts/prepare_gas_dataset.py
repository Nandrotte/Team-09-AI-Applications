"""Create deterministic train/validation/test splits from a YOLO export."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def read_names(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            raw = line.split(":", 1)[1].strip().strip("[]")
            return [item.strip().strip("'\"") for item in raw.split(",")]
    raise ValueError(f"could not find names in {data_yaml}")


def split_dataset(
    source: Path,
    output: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    group_by_source: bool = False,
) -> None:
    source_images = source / "train" / "images"
    source_labels = source / "train" / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(
            f"expected YOLO folders at {source_images} and {source_labels}"
        )

    images = sorted(
        path for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    pairs = [(image, source_labels / f"{image.stem}.txt") for image in images]
    missing = [(image, label) for image, label in pairs if not label.is_file()]
    if missing:
        raise ValueError(f"{len(missing)} images do not have matching label files")
    if not images:
        raise ValueError("no images found in the source train/images folder")

    if group_by_source:
        # Roboflow names augmented copies "<photo>_jpg.rf.<hash>.jpg"; keep all
        # copies of one photo in the same split to avoid train/val leakage.
        groups: dict[str, list] = {}
        for pair in pairs:
            groups.setdefault(pair[0].name.split(".rf.")[0], []).append(pair)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        train_end = round(len(keys) * train_ratio)
        val_end = train_end + round(len(keys) * val_ratio)
        key_splits = {
            "train": keys[:train_end],
            "val": keys[train_end:val_end],
            "test": keys[val_end:],
        }
        splits = {
            split: [pair for key in split_keys for pair in groups[key]]
            for split, split_keys in key_splits.items()
        }
    else:
        shuffled = pairs[:]
        random.Random(seed).shuffle(shuffled)
        train_end = round(len(shuffled) * train_ratio)
        val_end = train_end + round(len(shuffled) * val_ratio)
        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }
    names = read_names(source / "data.yaml")
    output.mkdir(parents=True, exist_ok=True)

    for split, split_pairs in splits.items():
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    (output / "data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                f"nc: {len(names)}",
                f"names: [{', '.join(repr(name) for name in names)}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created dataset at {output}")
    for split, split_pairs in splits.items():
        print(f"{split}: {len(split_pairs)} images")
    print(f"classes: {names}")


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=repository_root / "data" / "raw" / "roboflow_gas_tank_yolo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "data" / "processed" / "gasbottles",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="keep augmented copies of one photo in the same split "
        "(use with --output data/processed/gasbottles_grouped)",
    )
    args = parser.parse_args()
    if not args.source.is_absolute():
        args.source = repository_root / args.source
    if not args.output.is_absolute():
        args.output = repository_root / args.output
    try:
        split_dataset(
            args.source, args.output, 0.8, 0.1, args.seed, args.group_by_source
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
