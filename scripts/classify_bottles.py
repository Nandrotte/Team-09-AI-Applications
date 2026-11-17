"""Step 4 of the pipeline (WS5): one OK/NOK decision per synchronised bottle.

All crops of a bottle (several frames from each of the 3 cameras) are classified;
the bottle score is the mean NOK probability of its `--top` most suspicious crops
(damage or dirt is often visible from one side only, so averaging over every crop
would hide it). The bottle is NOK when the score >= the threshold chosen on the
validation set (models/ok_nok_threshold.json).

Each bottle is compared once with its own workbook row: `classification` (lecturer's
visual label, the ground truth for a visual classifier) and, for information,
`Primagaz_status` (operator decision, which also includes non-visual reasons).

    py scripts/classify_bottles.py --recording 14_32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import RESULTS, ROOT  # noqa: E402
from gass.ground_truth import load_ground_truth  # noqa: E402
from gass.gt_mapping import load_mapping  # noqa: E402
from gass.metrics import binary_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "ok_nok_best.pt")
    parser.add_argument("--threshold", type=float, default=None, help="default: models/ok_nok_threshold.json")
    parser.add_argument("--top", type=int, default=3, help="mean of the N highest crop scores per bottle")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    folder = args.dir or RESULTS / "pipeline" / args.recording
    info = json.loads((ROOT / "models" / "ok_nok_threshold.json").read_text(encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else info["nok_threshold"]
    model = YOLO(str(args.weights))
    nok_index = next(i for i, n in model.names.items() if str(n).lower() == "nok")

    crops = defaultdict(list)
    with (folder / "crops_index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            crops[int(row["bottle_id"])].append(folder / row["path"])

    scores = {}
    crop_rows = []
    for bottle_id, paths in sorted(crops.items()):
        probs = []
        for result, path in zip(model.predict([str(p) for p in paths], imgsz=info.get("imgsz", 224),
                                              device=args.device, verbose=False), paths):
            p = float(result.probs.data[nok_index])
            probs.append(p)
            crop_rows.append({"bottle_id": bottle_id, "crop": path.name, "nok_probability": round(p, 4)})
        top = sorted(probs, reverse=True)[:args.top]
        scores[bottle_id] = {"score": sum(top) / len(top), "mean_all": sum(probs) / len(probs), "crops": len(probs)}

    gt = load_ground_truth(args.recording).set_index("id")
    mapping, mapping_source = load_mapping(folder)
    rows, pred, truth_visual, truth_status, pred_mean = [], [], [], [], []
    for bottle_id, s in sorted(scores.items()):
        decision = s["score"] >= threshold
        gt_id = mapping.get(bottle_id)
        row = {"bottle_id": bottle_id, "gt_id": gt_id, "score": round(s["score"], 4),
               "mean_all": round(s["mean_all"], 4), "crops": s["crops"], "decision": "NOK" if decision else "OK"}
        if gt_id is not None and gt_id in gt.index:
            row["gt_classification"] = gt.loc[gt_id, "classification"]
            row["gt_primagaz_status"] = gt.loc[gt_id, "primagaz_status"]
            pred.append(decision)
            pred_mean.append(s["mean_all"] >= threshold)
            truth_visual.append(row["gt_classification"] == "nok")
            truth_status.append(row["gt_primagaz_status"] == "NOK")
        rows.append(row)

    with (folder / "classification_per_bottle.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bottle_id", "gt_id", "score", "mean_all", "crops", "decision",
                                                    "gt_classification", "gt_primagaz_status"])
        writer.writeheader()
        writer.writerows(rows)
    with (folder / "classification_per_crop.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bottle_id", "crop", "nok_probability"])
        writer.writeheader()
        writer.writerows(crop_rows)

    report = {
        "recording": args.recording, "weights": str(args.weights.relative_to(ROOT)), "threshold": threshold,
        "aggregation": f"mean of top-{args.top} crop NOK probabilities", "mapping": mapping_source,
        "bottles_evaluated": len(pred),
        "vs_classification (visual ground truth)": binary_metrics(pred, truth_visual),
        "vs_classification, mean over all crops": binary_metrics(pred_mean, truth_visual),
        "vs_primagaz_status (operator, for information)": binary_metrics(pred, truth_status),
    }
    (folder / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
