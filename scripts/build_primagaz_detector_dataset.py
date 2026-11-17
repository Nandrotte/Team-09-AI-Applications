"""Build a YOLO detection dataset from real Primagaz frames (WS2/WS4 improvement).

Why: the first detector was trained on web photos (Roboflow) and misses many bottles
on the real line cameras, which breaks tracking and synchronisation.

How the boxes are obtained without manual annotation:
1. The lecturer's OK/NOK archive contains crops named
   `<rec>_<camera>_id_<bottle>_<frame>.jpg`. The frame number refers to the front
   camera timeline; we estimate the per-camera frame offset with template matching
   (14_55: back_left +35, back_right +5).
2. Each crop is located again in its frame with template matching. Because the crop
   is an unscaled cut-out, the match position + crop size is an exact bounding box.
   Matches with a score below --min-score are dropped.
3. Other bottles that are clearly visible in the same frame are added from the old
   detector (confidence >= --extra-conf, little overlap with the matched box), so
   partially visible neighbours are not learned as background.

The split is made per bottle id in time blocks (first 70% of ids train, next 15% val,
last 15% test), so one physical bottle never appears in two splits.

    py scripts/build_primagaz_detector_dataset.py --recording 14_55
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import RAW_PRIMAGAZ, ROOT, video_path  # noqa: E402
from gass.reference_frames import ZIP_NAME  # noqa: E402

PATTERN = re.compile(r"^(\d\d_\d\d)_(front|back_left|back_right)_id_(\d+)_(\d+)\.jpg$", re.I)


def read_frame(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


def locate(frame, crop, scale: float = 0.25):
    """Best template-match position (x, y) in full-resolution pixels and its score."""
    small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    small_crop = cv2.resize(crop, None, fx=scale, fy=scale)
    if small_crop.shape[0] > small_frame.shape[0] or small_crop.shape[1] > small_frame.shape[1]:
        return None, -1.0
    _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(small_frame, small_crop, cv2.TM_CCOEFF_NORMED))
    # refine at full resolution in a small window around the coarse match
    x0, y0 = int(loc[0] / scale), int(loc[1] / scale)
    pad = int(8 / scale)
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb = min(frame.shape[1], x0 + crop.shape[1] + pad)
    yb = min(frame.shape[0], y0 + crop.shape[0] + pad)
    window = frame[ya:yb, xa:xb]
    if window.shape[0] < crop.shape[0] or window.shape[1] < crop.shape[1]:
        return None, -1.0
    _, score, _, fine = cv2.minMaxLoc(cv2.matchTemplate(window, crop, cv2.TM_CCOEFF_NORMED))
    return (xa + fine[0], ya + fine[1]), float(score)


def estimate_offset(cap, samples, search: int) -> int:
    """Frame offset of this camera relative to the frame numbers in the file names."""
    offsets = []
    for crop, frame_no in samples:
        best = (-1.0, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - search))
        for f in range(max(0, frame_no - search), frame_no + search + 1):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=0.2, fy=0.2)
            tmpl = cv2.resize(crop, None, fx=0.2, fy=0.2)
            score = cv2.minMaxLoc(cv2.matchTemplate(small, tmpl, cv2.TM_CCOEFF_NORMED))[1]
            best = max(best, (score, f - frame_no))
        if best[0] > 0.9:
            offsets.append(best[1])
    if not offsets:
        raise RuntimeError("could not estimate camera offset")
    return int(statistics.median(offsets))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def is_neighbour_bottle(box, width: int, height: int) -> bool:
    """Extra boxes are only trusted when they look like a neighbouring bottle that is
    partly in view: tall and touching the left or right edge of the frame. This drops
    false positives of the old detector on background objects (boxes, railings)."""
    x1, y1, x2, y2 = box
    tall = (y2 - y1) >= 0.6 * height
    at_edge = x1 <= 0.02 * width or x2 >= 0.98 * width
    return tall and at_edge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", default="14_55")
    parser.add_argument("--old-model", type=Path, default=ROOT / "models" / "gas_tank_best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "primagaz_detector")
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--extra-conf", type=float, default=0.6)
    parser.add_argument("--width", type=int, default=1280, help="saved image width (labels are normalised)")
    parser.add_argument("--search", type=int, default=60, help="frames searched when estimating offsets")
    args = parser.parse_args()

    with zipfile.ZipFile(RAW_PRIMAGAZ / ZIP_NAME) as archive:
        crops = defaultdict(list)  # camera -> [(name, bottle, frame)]
        for name in archive.namelist():
            match = PATTERN.match(name.rsplit("/", 1)[-1])
            if match and match.group(1) == args.recording:
                crops[match.group(2)].append((name, int(match.group(3)), int(match.group(4))))
        bottles = sorted({b for items in crops.values() for _, b, _ in items})
        n = len(bottles)
        split_of = {b: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test")
                    for i, b in enumerate(bottles)}

        if args.output.exists():
            shutil.rmtree(args.output)
        old = YOLO(str(args.old_model))
        report = {"recording": args.recording, "offsets": {}, "images": defaultdict(int),
                  "matched_boxes": 0, "extra_boxes": 0, "rejected": 0}
        for camera, items in sorted(crops.items()):
            cap = cv2.VideoCapture(str(video_path(args.recording, camera)))
            items.sort(key=lambda t: t[2])
            step = max(1, len(items) // 6)
            samples = [(cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR), frame)
                       for name, _, frame in items[::step][:6]]
            offset = estimate_offset(cap, samples, args.search)
            report["offsets"][camera] = offset
            print(f"{camera}: offset {offset:+d} frames, {len(items)} crops")
            for name, bottle, frame_no in items:
                crop = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                frame = read_frame(cap, frame_no + offset)
                if frame is None or crop is None:
                    report["rejected"] += 1
                    continue
                pos, score = locate(frame, crop)
                if pos is None or score < args.min_score:
                    report["rejected"] += 1
                    continue
                h, w = frame.shape[:2]
                boxes = [(pos[0], pos[1], pos[0] + crop.shape[1], pos[1] + crop.shape[0])]
                report["matched_boxes"] += 1
                pred = old.predict(frame, conf=args.extra_conf, classes=[1], device=0, verbose=False)[0]
                for box in pred.boxes.xyxy.tolist():
                    if all(iou(box, other) < 0.3 for other in boxes) and is_neighbour_bottle(box, w, h):
                        boxes.append(tuple(box))
                        report["extra_boxes"] += 1
                split = split_of[bottle]
                stem = f"{args.recording}_{camera}_{frame_no + offset:05d}"
                img_dir = args.output / "images" / split
                lbl_dir = args.output / "labels" / split
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                scale = args.width / w
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), cv2.resize(frame, (args.width, int(h * scale))),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                with (lbl_dir / f"{stem}.txt").open("w", encoding="utf-8") as handle:
                    for x1, y1, x2, y2 in boxes:
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        handle.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
                report["images"][split] += 1
            cap.release()

    (args.output / "data.yaml").write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: gas-bottle\n", encoding="utf-8")
    report["images"] = dict(report["images"])
    report["split_by_bottle_id"] = {s: [b for b in bottles if split_of[b] == s][::max(1, n // 20)]
                                    for s in ("train", "val", "test")}
    (args.output / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "split_by_bottle_id"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
