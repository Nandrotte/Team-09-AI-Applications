"""Step 5 of the pipeline (WS6): read tare weight and year from the bottle crops.

For every synchronised bottle, all its crops (several frames x 3 cameras, made by
extract_bottle_crops.py) are read with EasyOCR. Text segments on the same line are
joined before the regular expressions are applied (the print "10.8 KG-2034" is often
split into "10.8" / "KG" / "-2034"), and the candidates of all crops are combined
with a confidence-weighted vote (gass/ocr_parse.py).

Each bottle is then compared once with its own workbook row (mapping from the
synchroniser), using the metrics in gass/metrics.py: coverage, mean absolute error
in kg, share within 0.1 kg for tare; coverage, exact year, error in years and
pass/fail decision accuracy for the year.

    py scripts/ocr_bottles.py --recording 14_55
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import RESULTS  # noqa: E402
from gass.ground_truth import load_ground_truth  # noqa: E402
from gass.gt_mapping import load_mapping  # noqa: E402
from gass.metrics import tare_metrics, tare_score, year_metrics, year_score  # noqa: E402
from gass.ocr_parse import parse_text, vote  # noqa: E402


def join_lines(segments: list[tuple]) -> list[tuple[str, float]]:
    """Group EasyOCR segments (box, text, conf) into lines and join them left-to-right."""
    items = []
    for box, text, conf in segments:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"y": sum(ys) / 4, "h": max(ys) - min(ys), "x": min(xs), "text": text, "conf": float(conf)})
    items.sort(key=lambda i: i["y"])
    lines: list[list[dict]] = []
    for item in items:
        if lines and abs(item["y"] - lines[-1][-1]["y"]) <= 0.7 * max(item["h"], lines[-1][-1]["h"], 1):
            lines[-1].append(item)
        else:
            lines.append([item])
    joined = []
    for line in lines:
        if len(line) < 2:
            continue
        line.sort(key=lambda i: i["x"])
        joined.append((" ".join(i["text"] for i in line), min(i["conf"] for i in line)))
    return joined


def preprocess(crop, max_side: int, band: tuple[float, float] = (0.0, 1.0)):
    """Keep the shoulder band where the ink-jet print is, limit the size, grey + CLAHE."""
    h = crop.shape[0]
    crop = crop[int(h * band[0]):int(h * band[1])]
    h, w = crop.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--mag-ratio", type=float, default=1.5)
    parser.add_argument("--band", type=float, nargs=2, default=[0.30, 0.85],
                        help="vertical part of the crop that is read (the print sits at 48-72%% of the height)")
    parser.add_argument("--rotation", action="store_true",
                        help="also try +-15 degree recognition variants (4x slower, no clear gain on 14_55)")
    parser.add_argument("--per-camera", type=int, default=3,
                        help="crops per camera and bottle (0 = all); 5 gave no gain over 3 on 14_55")
    parser.add_argument("--min-year", type=int, default=2025, help="certification valid if year >= this")
    parser.add_argument("--min-tare-support", type=float, default=0.0,
                        help="answer 'unread' when the winning tare has less vote support (tuned on 14_55)")
    parser.add_argument("--min-year-support", type=float, default=0.0,
                        help="answer 'unread' when the winning year has less vote support (tuned on 14_55)")
    parser.add_argument("--limit-bottles", type=int, default=0, help="only the first N bottles (quick tests)")
    parser.add_argument("--out-suffix", default="", help="suffix for output files (experiments)")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    import easyocr  # imported here so --help works without the dependency

    folder = args.dir or RESULTS / "pipeline" / args.recording
    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, download_enabled=False, verbose=False)
    crops = defaultdict(list)
    with (folder / "crops_index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            crops[int(row["bottle_id"])].append(row)

    start = time.time()
    raw_rows, predictions = [], {}
    bottle_ids = sorted(crops)[:args.limit_bottles] if args.limit_bottles else sorted(crops)
    for bottle_id in bottle_ids:
        readings = []
        rows = crops[bottle_id]
        if args.per_camera:
            by_cam = defaultdict(list)
            for row in rows:
                by_cam[row["camera"]].append(row)
            rows = [r for cam_rows in by_cam.values() for r in cam_rows[:args.per_camera]]
        for row in rows:
            # raw OCR output is cached per crop + settings, so the parser can be improved
            # and re-evaluated without running the (slow) OCR again
            cache = folder / "ocr_cache" / f"{row['path'].replace('/', '_').replace(chr(92), '_')}" \
                f"_{args.max_side}_{args.mag_ratio}_{args.band[0]}-{args.band[1]}_{int(args.rotation)}.json"
            if cache.is_file():
                segments = json.loads(cache.read_text(encoding="utf-8"))
            else:
                image = preprocess(cv2.imread(str(folder / row["path"])), args.max_side, tuple(args.band))
                segments = [([[float(x), float(y)] for x, y in box], text, float(conf)) for box, text, conf in
                            reader.readtext(image, rotation_info=[15, 345] if args.rotation else None,
                                            mag_ratio=args.mag_ratio, width_ths=0.8, ycenter_ths=0.6)]
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(segments), encoding="utf-8")
            texts = [(text, float(conf)) for _, text, conf in segments] + join_lines(segments)
            for text, conf in texts:
                parsed = parse_text(text)
                if parsed["tare"] or parsed["year"]:
                    readings.append({**parsed, "conf": conf})
                    raw_rows.append({"bottle_id": bottle_id, "camera": row["camera"], "frame": row["frame"],
                                     "text": text, "conf": round(conf, 3), "tare": parsed["tare"],
                                     "year": parsed["year"]})
        result = vote(readings)
        if result["tare_support"] < args.min_tare_support:
            result["tare"] = None
        if result["year_support"] < args.min_year_support:
            result["year"] = None
        predictions[bottle_id] = result
        if bottle_id % 10 == 0:
            print(f"  bottle {bottle_id}: {predictions[bottle_id]} ({time.time() - start:.0f}s)", flush=True)

    with (folder / f"ocr_raw_readings{args.out_suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bottle_id", "camera", "frame", "text", "conf", "tare", "year"])
        writer.writeheader()
        writer.writerows(raw_rows)

    gt = load_ground_truth(args.recording).set_index("id")
    mapping, mapping_source = load_mapping(folder)
    tare_pairs, tare_pairs_scale, year_pairs, rows_out = [], [], [], []
    for bottle_id, pred in sorted(predictions.items()):
        gt_id = mapping.get(bottle_id)
        row = {"bottle_id": bottle_id, "gt_id": gt_id, **pred}
        if gt_id is not None and gt_id in gt.index:
            truth = gt.loc[gt_id]
            row.update(gt_tarra=truth["tarra"], gt_year=truth["year"], gt_pr_tare_kg=truth["pr_tare_kg"])
            if not math.isnan(truth["tarra"]) and not truth["tarra_uncertain"]:
                tare_pairs.append((pred["tare"], float(truth["tarra"])))
            if not math.isnan(truth["pr_tare_kg"]):
                tare_pairs_scale.append((pred["tare"], float(truth["pr_tare_kg"])))
            if not math.isnan(truth["year"]) and not truth["year_uncertain"]:
                year_pairs.append((pred["year"], int(truth["year"])))
        rows_out.append(row)

    with (folder / f"ocr_predictions{args.out_suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["bottle_id", "gt_id", "tare", "tare_support", "year", "year_support", "n_readings",
                  "gt_tarra", "gt_year", "gt_pr_tare_kg"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    report = {
        "recording": args.recording, "engine": "EasyOCR (en) + regex + weighted vote",
        "settings": {"per_camera": args.per_camera, "band": args.band, "max_side": args.max_side,
                     "mag_ratio": args.mag_ratio, "rotation": args.rotation,
                     "min_tare_support": args.min_tare_support, "min_year_support": args.min_year_support},
        "bottles": len(predictions), "mapping": mapping_source, "seconds": round(time.time() - start, 1),
        "tare_vs_lector_reading": tare_metrics(tare_pairs),
        "tare_vs_primagaz_scale": tare_metrics(tare_pairs_scale),
        "year": year_metrics(year_pairs, args.min_year),
        "tare_score_partial_credit": tare_score(tare_pairs),
        "year_score": year_score(year_pairs, args.min_year),
        "note": "Ground-truth values marked '?' or uncertain in the workbook are excluded.",
    }
    (folder / f"ocr_report{args.out_suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
