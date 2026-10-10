"""Check an Ultralytics YOLO dataset before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_config(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key == "names":
            names = [
                item.strip().strip("'\"")
                for item in value.strip("[]").split(",")
                if item.strip()
            ]
        else:
            values[key] = value.strip("'\"")
    return values, names


def check_split(dataset: Path, split: str, image_path: Path, class_count: int) -> tuple[int, int]:
    images = sorted(
        path
        for path in image_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    errors = 0
    annotations = 0
    label_path = dataset / split / "labels"

    print(f"{split}: {len(images)} images")
    for image in images:
        label = label_path / f"{image.stem}.txt"
        if not label.is_file():
            print(f"  ERROR missing label: {label}")
            errors += 1
            continue
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                print(f"  ERROR {label}:{line_number}: expected 5 values")
                errors += 1
                continue
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                print(f"  ERROR {label}:{line_number}: values must be numeric")
                errors += 1
                continue
            if not 0 <= class_id < class_count:
                print(f"  ERROR {label}:{line_number}: invalid class id {class_id}")
                errors += 1
            if any(not 0 <= value <= 1 for value in coordinates):
                print(f"  ERROR {label}:{line_number}: coordinates must be between 0 and 1")
                errors += 1
            annotations += 1
    return errors, annotations


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=repository_root / "data" / "processed" / "gasbottles" / "data.yaml",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"Dataset configuration not found: {args.data}", file=sys.stderr)
        return 1

    dataset = args.data.parent
    config, names = read_config(args.data)
    if not names:
        print("ERROR: data.yaml does not contain a names list.")
        return 1

    print(f"Dataset: {dataset}")
    print(f"Classes: {names}")
    total_errors = 0
    total_annotations = 0

    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if split_value is None:
            print(f"WARNING: no {split} split in data.yaml")
            continue
        image_path = Path(split_value)
        if not image_path.is_absolute():
            image_path = dataset / image_path
        if not image_path.is_dir():
            print(f"ERROR: {split} image directory does not exist: {image_path}")
            total_errors += 1
            continue
        errors, annotations = check_split(dataset, split, image_path, len(names))
        total_errors += errors
        total_annotations += annotations

    print(f"Total annotations: {total_annotations}")
    if total_errors:
        print(f"Dataset check failed with {total_errors} problem(s).")
        return 1
    print("Dataset check passed. No invalid annotations were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
