"""Render a demo video: the 3 synchronised cameras side by side, one ID per bottle.

Each panel shows the box of the bottle that is currently being inspected with its
global ID. When the later pipeline steps have run, the bottle's OK/NOK decision,
tare weight and year are shown too, next to the ground truth for comparison.

    py scripts/render_demo_video.py --recording 14_32 --start-s 0 --duration-s 60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import CAMERAS, RESULTS, video_path  # noqa: E402
from gass.ground_truth import load_ground_truth  # noqa: E402
from gass.gt_mapping import load_mapping  # noqa: E402
from gass.sync import read_bottles  # noqa: E402


def read_optional(path: Path, key: str = "bottle_id") -> dict[int, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(r[key]): r for r in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--mode", default="strict", choices=["strict", "robust"])
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()

    folder = args.dir or RESULTS / "pipeline" / args.recording
    meta = json.loads((folder / "tracks_meta.json").read_text(encoding="utf-8"))
    fps = meta["front"]["fps"]
    bottles = read_bottles(folder / f"bottles_{args.mode}.csv")
    offsets = {c: int(bottles[0][f"offset_{c}"]) for c in CAMERAS} if bottles else {c: 0 for c in CAMERAS}
    classification = read_optional(folder / "classification_per_bottle.csv")
    ocr = read_optional(folder / "ocr_predictions.csv")
    try:
        mapping, _ = load_mapping(folder)
    except FileNotFoundError:
        mapping = {}
    gt = load_ground_truth(args.recording).set_index("id")

    # camera -> frame -> list of (box, bottle_id)
    boxes = {c: defaultdict(list) for c in CAMERAS}
    for camera in CAMERAS:
        track_to_bottle = {}
        for b in bottles:
            for t in b[f"tracks_{camera}"].split():
                track_to_bottle[int(t)] = b
        with (folder / f"tracks_{camera}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                b = track_to_bottle.get(int(row["track_id"]))
                frame = int(row["frame"])
                if b and b["start_frame"] + offsets[camera] - 10 <= frame <= b["end_frame"] + offsets[camera] + 10:
                    boxes[camera][frame].append(((float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                                                 b["bottle_id"]))
    stride = meta["front"]["stride"]

    caps = {c: cv2.VideoCapture(str(video_path(args.recording, c))) for c in CAMERAS}
    start = int(args.start_s * fps)
    for camera, cap in caps.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + offsets[camera])
    pw = args.panel_width
    ph = int(pw * meta["front"]["height"] / meta["front"]["width"])
    out_path = folder / f"demo_{args.recording}_{int(args.start_s)}s.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw * 3, ph + 90))
    scale = pw / meta["front"]["width"]
    current = None
    for ref in range(start, start + int(args.duration_s * fps)):
        panels = []
        for camera in CAMERAS:
            ok, frame = caps[camera].read()
            if not ok:
                frame = np.zeros((meta[camera]["height"], meta[camera]["width"], 3), np.uint8)
            panel = cv2.resize(frame, (pw, ph))
            cam_frame = ref + offsets[camera]
            near = cam_frame - (cam_frame % stride)
            for (x1, y1, x2, y2), bid in boxes[camera].get(near, []):
                colour = (0, 0, 255) if classification.get(bid, {}).get("decision") == "NOK" else (0, 200, 0)
                cv2.rectangle(panel, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), colour, 2)
                cv2.putText(panel, f"ID {bid}", (int(x1 * scale) + 4, int(y1 * scale) + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.putText(panel, camera, (8, ph - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panels.append(panel)
        for b in bottles:
            if b["start_frame"] <= ref <= b["end_frame"]:
                current = b["bottle_id"]
        bar = np.zeros((90, pw * 3, 3), np.uint8)
        found = sum(1 for b in bottles if b["start_frame"] <= ref)
        line1 = f"{args.recording}  t={ref / fps:6.1f}s   bottles so far: {found}"
        cv2.putText(bar, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if current is not None:
            c, o = classification.get(current, {}), ocr.get(current, {})
            gt_id = mapping.get(current)
            line2 = (f"bottle ID {current}: {c.get('decision', '?')} (score {c.get('score', '-')})"
                     f"  tare {o.get('tare') or '-'} kg  year {o.get('year') or '-'}")
            if gt_id in gt.index:
                g = gt.loc[gt_id]
                line2 += (f"   | truth: id {gt_id} {g['classification'].upper()} tare {g['tarra']} year "
                          f"{'' if g['year'] != g['year'] else int(g['year'])} {g['color']}")
            cv2.putText(bar, line2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        writer.write(np.vstack([np.hstack(panels), bar]))
    writer.release()
    for cap in caps.values():
        cap.release()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
