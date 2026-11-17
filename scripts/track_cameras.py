"""Step 1 of the pipeline: detect + track bottles in every camera video of a recording.

For each camera this writes `results/pipeline/<recording>/tracks_<camera>.csv` with one
row per tracked box (frame, track_id, conf, x1, y1, x2, y2) plus a small JSON with the
video metadata. Later steps (sync, classification, OCR) reuse these files, so the
expensive detection only runs once.

    py scripts/track_cameras.py --recording 14_55
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import CAMERAS, ROOT, RESULTS, video_path  # noqa: E402


def track_camera(model_path: Path, video: Path, out_csv: Path, args) -> dict:
    capture = cv2.VideoCapture(str(video))
    meta = {
        "video": str(video.relative_to(ROOT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "stride": args.stride, "conf": args.conf, "imgsz": args.imgsz,
        "model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
    }
    capture.release()

    model = YOLO(str(model_path))
    if args.class_id is None:  # bottle class: 'gas-tank' in the web-photo model, 0 in the Primagaz model
        args.class_id = next((i for i, n in model.names.items() if "tank" in n or "bottle" in n), 0)
    meta["class_id"] = args.class_id
    stream = model.track(source=str(video), stream=True, persist=True, tracker=args.tracker,
                         classes=[args.class_id], conf=args.conf, imgsz=args.imgsz,
                         device=args.device, vid_stride=args.stride, verbose=False)
    start = time.time()
    rows = 0
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "track_id", "conf", "x1", "y1", "x2", "y2"])
        for index, result in enumerate(stream):
            frame = index * args.stride
            if args.max_frames and frame >= args.max_frames:
                break
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            for box, track_id, conf in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.conf.tolist()):
                writer.writerow([frame, track_id, round(conf, 3), *(round(v, 1) for v in box)])
                rows += 1
            if index % 500 == 0:
                print(f"  {video.stem}: frame {frame}/{meta['frame_count']} ({time.time() - start:.0f}s)", flush=True)
    meta["rows"] = rows
    meta["seconds"] = round(time.time() - start, 1)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, help="e.g. 14_55")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    default_model = ROOT / "models" / "primagaz_detector.pt"
    if not default_model.is_file():
        default_model = ROOT / "models" / "gas_tank_best.pt"
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--class-id", type=int, default=None, help="bottle class (default: auto)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or RESULTS / "pipeline" / args.recording
    out_dir.mkdir(parents=True, exist_ok=True)
    all_meta = {}
    for camera in args.cameras:
        video = video_path(args.recording, camera)
        print(f"{camera}: tracking {video.name}")
        all_meta[camera] = track_camera(args.model, video, out_dir / f"tracks_{camera}.csv", args)
        print(f"{camera}: {all_meta[camera]['rows']} boxes in {all_meta[camera]['seconds']}s")
    (out_dir / "tracks_meta.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
