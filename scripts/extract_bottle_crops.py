"""Step 3 of the pipeline: cut out a few crops per synchronised bottle and camera.

For every bottle in bottles_<mode>.csv we pick up to --per-camera frames (largest,
most confident boxes, spread over the bottle's time window) from each camera and
save the crops under results/pipeline/<rec>/crops/<bottle_id>/<camera>_<frame>.jpg.
These crops are the input for the colour check, the OK/NOK classifier and OCR,
i.e. the same format the final application will see.

The colour of each bottle is estimated and compared with the ground-truth colour
sequence (sequence alignment), which checks the synchronisation as suggested by
the lecturer. Results: crops_index.csv and colour_check.json.

    py scripts/extract_bottle_crops.py --recording 14_55
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import CAMERAS, RESULTS, video_path  # noqa: E402
from gass.colour import cyan_fraction  # noqa: E402
from gass.evaluation import align_sequences  # noqa: E402
from gass.ground_truth import load_ground_truth  # noqa: E402
from gass.sync import read_bottles  # noqa: E402


def pick_frames(rows: list[dict], n: int) -> list[dict]:
    """Spread n picks over time, each the best (area x conf) box in its time slice."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["frame"])
    slices = [rows[i * len(rows) // n:(i + 1) * len(rows) // n] for i in range(n)]
    picks = []
    for part in slices:
        if part:
            picks.append(max(part, key=lambda r: r["area"] * r["conf"]))
    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "robust"])
    parser.add_argument("--per-camera", type=int, default=5)
    parser.add_argument("--pad", type=float, default=0.03, help="padding as fraction of box size")
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()

    folder = args.dir or RESULTS / "pipeline" / args.recording
    bottles = read_bottles(folder / f"bottles_{args.mode}.csv")
    crop_dir = folder / "crops"
    if crop_dir.exists():
        shutil.rmtree(crop_dir)

    wanted: dict[str, dict[int, list[tuple[int, tuple]]]] = defaultdict(lambda: defaultdict(list))
    for camera in CAMERAS:
        tracks = defaultdict(list)
        with (folder / f"tracks_{camera}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                tracks[int(row["track_id"])].append({"frame": int(row["frame"]), "conf": float(row["conf"]),
                                                     "box": box, "area": (box[2] - box[0]) * (box[3] - box[1])})
        for b in bottles:
            ids = [int(t) for t in b[f"tracks_{camera}"].split()]
            offset = int(b[f"offset_{camera}"])
            lo, hi = b["start_frame"] + offset, b["end_frame"] + offset
            rows = [r for t in ids for r in tracks.get(t, []) if lo <= r["frame"] <= hi]
            for pick in pick_frames(rows, args.per_camera):
                wanted[camera][pick["frame"]].append((b["bottle_id"], pick["box"]))

    index_rows = []
    crops_per_bottle = defaultdict(list)
    for camera in CAMERAS:
        cap = cv2.VideoCapture(str(video_path(args.recording, camera)))
        frames_needed = wanted[camera]
        last = max(frames_needed) if frames_needed else -1
        frame_no = 0
        while frame_no <= last:
            if frame_no in frames_needed:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                for bottle_id, (x1, y1, x2, y2) in frames_needed[frame_no]:
                    px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                    xa, ya = int(max(0, x1 - px)), int(max(0, y1 - py))
                    xb, yb = int(min(w, x2 + px)), int(min(h, y2 + py))
                    crop = frame[ya:yb, xa:xb]
                    out = crop_dir / f"{bottle_id:03d}" / f"{camera}_{frame_no:05d}.jpg"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    index_rows.append({"bottle_id": bottle_id, "camera": camera, "frame": frame_no,
                                       "path": str(out.relative_to(folder)), "x1": xa, "y1": ya, "x2": xb, "y2": yb})
                    crops_per_bottle[bottle_id].append(cyan_fraction(crop))
            elif not cap.grab():
                break
            frame_no += 1
        cap.release()
        print(f"{camera}: {sum(len(v) for v in frames_needed.values())} crops")

    with (folder / "crops_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    colours = {}
    for b in bottles:  # median cyan fraction over the bottle's crops (keeps memory low)
        fractions = sorted(crops_per_bottle.get(b["bottle_id"], [])) or [0.0]
        median = fractions[len(fractions) // 2]
        colours[b["bottle_id"]] = ("cyan" if median >= 0.25 else "black", round(median, 3))
    gt = load_ground_truth(args.recording)
    truth = ["cyan" if c.startswith("cyan") else "black" for c in gt["color"]]
    pred = [colours[b["bottle_id"]][0] for b in bottles]
    pairs, stats = align_sequences(pred, truth)
    mapping = {bottles[p]["bottle_id"]: int(gt["id"].iloc[t]) for p, t in pairs if p is not None and t is not None}
    report = {"recording": args.recording, "mode": args.mode, **stats,
              "aligned_id_equals_gt_id": sum(1 for k, v in mapping.items() if k == v),
              "colours": {str(k): v for k, v in colours.items()},
              "alignment_bottle_to_gt_id": {str(k): v for k, v in mapping.items()}}
    (folder / "colour_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("colours", "alignment_bottle_to_gt_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
