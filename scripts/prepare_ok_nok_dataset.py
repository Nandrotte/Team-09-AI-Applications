"""Build the OK/NOK image-classification dataset (WS5, second iteration).

Sources
* the lecturer's archive `ai_applications_ok_nok_classification_training_set.zip`
  (OK/NOK crops, mostly from the line videos, plus photos taken at the plant);
* optionally our own bottle crops from the pipeline (results/pipeline/<rec>/crops),
  labelled with the workbook `classification` column through the synchroniser's
  bottle -> ground-truth mapping. These are exactly the crops the final application
  feeds to the classifier ("gather training data in the format you will present it").

Fixes compared with the first version
* Grouping: all images of one physical bottle (same recording + same bottle id, any
  camera, any frame) stay in the same split. Before, `14_55_front_id_66` and
  `14_55_back_left_id_66` could land in train and test -> leakage between views.
* Hold-out recordings (default 14_32) are removed completely, so the per-bottle video
  evaluation (classify_bottles.py) runs on bottles the model has never seen.

The val split (15% of the groups, stratified per class) is used for early stopping
and to choose the NOK threshold. NOK images are oversampled in train to counter the
~4.4:1 class imbalance.

    py scripts/prepare_ok_nok_dataset.py --pipeline-recordings 14_55 --holdout 14_32
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import RAW_PRIMAGAZ, RESULTS, ROOT  # noqa: E402
from gass.ground_truth import load_ground_truth  # noqa: E402
from gass.gt_mapping import load_mapping  # noqa: E402
from gass.reference_frames import ZIP_NAME  # noqa: E402

GROUP_RE = re.compile(r"^(\d\d_\d\d)_(?:(?:front|back_left|back_right|top)_)?id_([a-z0-9]+?)(?:tbc)?(?:_|\.|$)", re.I)
REC_RE = re.compile(r"^(\d\d_\d\d)_")


def group_of(name: str) -> tuple[str, str]:
    """(recording, group key). Images without a bottle id are their own group."""
    stem = Path(name).name.lower()
    match = GROUP_RE.match(stem)
    if match:
        return match.group(1), f"{match.group(1)}_id_{match.group(2)}"
    rec = REC_RE.match(stem)
    return (rec.group(1) if rec else "plant"), stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=RAW_PRIMAGAZ / ZIP_NAME)
    parser.add_argument("--pipeline-recordings", nargs="*", default=["14_55"],
                        help="recordings whose pipeline crops are added (needs sync + crops)")
    parser.add_argument("--holdout", nargs="*", default=["14_32"], help="recordings excluded from training")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "ok_nok")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--nok-oversample", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=320,
                        help="images are stored at this size (the classifier uses 224 px; keeps training fast)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    holdout = set(args.holdout)

    items = []  # (label, group, recording, source, payload)
    skipped_holdout = 0
    with zipfile.ZipFile(args.zip) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) < 2 or parts[-2].lower() not in {"ok", "nok"} or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            recording, group = group_of(parts[-1])
            if recording in holdout:
                skipped_holdout += 1
                continue
            items.append((parts[-2].lower(), group, recording, "zip", name))

    for rec in args.pipeline_recordings:
        if rec in holdout:
            continue
        folder = RESULTS / "pipeline" / rec
        mapping, _ = load_mapping(folder)
        gt = load_ground_truth(rec).set_index("id")
        for crop in sorted((folder / "crops").glob("*/*.jpg")):
            gt_id = mapping.get(int(crop.parent.name))
            if gt_id is None or gt_id not in gt.index:
                continue
            label = "nok" if gt.loc[gt_id, "classification"] == "nok" else "ok"
            items.append((label, f"{rec}_id_{gt_id}", rec, "pipeline", crop))

    groups = defaultdict(lambda: {"label": None, "items": []})
    for label, group, *_ in items:
        groups[group]["label"] = "nok" if label == "nok" or groups[group]["label"] == "nok" else "ok"
    for item in items:
        groups[item[1]]["items"].append(item)

    rng = random.Random(args.seed)
    split_of = {}
    for label in ("ok", "nok"):
        keys = sorted(g for g, v in groups.items() if v["label"] == label)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * args.val_fraction))
        for i, key in enumerate(keys):
            split_of[key] = "val" if i < n_val else "train"

    if args.output.exists():
        shutil.rmtree(args.output)
    counts = defaultdict(lambda: defaultdict(int))
    with zipfile.ZipFile(args.zip) as archive:
        for label, group, recording, source, payload in items:
            split = split_of[group]
            copies = args.nok_oversample if (split == "train" and label == "nok") else 1
            stem = (Path(payload).stem if source == "zip" else f"{recording}_{payload.parent.name}_{payload.stem}")
            data = archive.read(payload) if source == "zip" else payload.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            scale = min(1.0, args.max_side / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            for c in range(copies):
                out = args.output / split / label / f"{source}_{stem}{'' if c == 0 else f'_dup{c}'}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[split][f"{label}_images"] += 1
            counts[split][f"{label}_from_{source}"] += 1
    for split in ("train", "val"):
        counts[split]["groups_ok"] = sum(1 for g, s in split_of.items() if s == split and groups[g]["label"] == "ok")
        counts[split]["groups_nok"] = sum(1 for g, s in split_of.items() if s == split and groups[g]["label"] == "nok")
    report = {"holdout_recordings": sorted(holdout), "skipped_holdout_images": skipped_holdout,
              "pipeline_recordings": args.pipeline_recordings, "nok_oversample_train": args.nok_oversample,
              "counts": {k: dict(v) for k, v in counts.items()}}
    (args.output / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
