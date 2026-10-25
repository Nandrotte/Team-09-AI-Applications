"""Track gas bottles in one video and log when each one crosses a virtual line.

Writes a crossings CSV (one row per bottle, numbered 1..n in order of crossing).
Use scripts/sync_bottles.py to merge the CSVs of several cameras into global IDs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def load_global_map(path: Path, camera: str) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["track_id"]): int(row["global_id"])
            for row in csv.DictReader(handle)
            if row["camera"] == camera and row["track_id"]
        }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera", help="camera name (default: video file name)")
    parser.add_argument("--model", type=Path, default=root / "models" / "gas_tank_best.pt")
    parser.add_argument("--classifier", type=Path, help="optional OK/NOK classifier weights")
    parser.add_argument("--classify-every", type=int, default=15,
                        help="classify each tracked bottle every N frames (default 15)")
    parser.add_argument("--class-id", type=int, default=1, help="gas-tank class (default 1)")
    parser.add_argument("--axis", choices=["x", "y"], default="x",
                        help="x: bottles move horizontally, line is vertical")
    parser.add_argument("--line", type=float, default=0.5,
                        help="line position as fraction of frame width/height")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "tracking")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--global-map", type=Path,
                        help="global_ids.csv from sync_bottles.py; shows global IDs in the video")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    camera = args.camera or args.video.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gmap = load_global_map(args.global_map, camera) if args.global_map else {}

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    line_px = int((width if args.axis == "x" else height) * args.line)

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(args.out_dir / f"{camera}_tracked.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    model = YOLO(str(args.model))
    classifier = YOLO(str(args.classifier)) if args.classifier else None
    last_pos: dict[int, float] = {}
    crossed: dict[int, int] = {}  # track_id -> sequence number
    class_scores: dict[int, list[float]] = {}  # mean NOK probability over sampled frames
    last_classified_frame: dict[int, int] = {}
    rows = []
    stream = model.track(source=str(args.video), stream=True, persist=True,
                         tracker="bytetrack.yaml", classes=[args.class_id],
                         conf=args.conf, device=args.device, verbose=False)
    for frame_idx, result in enumerate(stream):
        if args.max_frames and frame_idx >= args.max_frames:
            break
        frame = result.orig_img.copy() if writer else None
        if frame is not None:
            if args.axis == "x":
                cv2.line(frame, (line_px, 0), (line_px, height), (0, 255, 255), 2)
            else:
                cv2.line(frame, (0, line_px), (width, line_px), (0, 255, 255), 2)
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            classes = boxes.cls.int().tolist() if boxes.cls is not None else [args.class_id] * len(boxes)
            for box, track_id, detected_class in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), classes):
                if detected_class != args.class_id:
                    continue
                x1, y1, x2, y2 = box
                if (classifier is not None
                        and (track_id not in last_classified_frame
                             or frame_idx - last_classified_frame[track_id] >= max(1, args.classify_every))):
                    crop = result.orig_img[max(0, int(y1)):min(height, int(y2)),
                                            max(0, int(x1)):min(width, int(x2))]
                    if crop.size:
                        prediction = classifier.predict(crop, imgsz=320, device=args.device, verbose=False)[0]
                        names = prediction.names
                        nok_index = next((i for i, name in names.items() if str(name).lower() == "nok"), None)
                        if nok_index is not None:
                            scores = class_scores.setdefault(track_id, [0.0, 0.0])
                            scores[0] += float(prediction.probs.data[nok_index].item())
                            scores[1] += 1.0
                            last_classified_frame[track_id] = frame_idx
                pos = (x1 + x2) / 2 if args.axis == "x" else (y1 + y2) / 2
                prev = last_pos.get(track_id)
                last_pos[track_id] = pos
                if (prev is not None and track_id not in crossed
                        and (prev - line_px) * (pos - line_px) <= 0 and prev != pos):
                    crossed[track_id] = len(crossed) + 1
                    rows.append({"camera": camera, "seq": crossed[track_id],
                                 "track_id": track_id, "frame": frame_idx,
                                 "time_s": round(frame_idx / fps, 3)})
                if frame is not None:
                    label = f"G{gmap[track_id]}" if track_id in gmap else (
                        f"#{crossed[track_id]}" if track_id in crossed else f"t{track_id}")
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    if track_id in class_scores and class_scores[track_id][1]:
                        nok_p = class_scores[track_id][0] / class_scores[track_id][1]
                        decision = "NOK" if nok_p >= 0.5 else "OK"
                        cv2.putText(frame, f"{decision} {nok_p:.2f}",
                                    (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255) if decision == "NOK" else (0, 200, 0), 2)
                    scale = max(0.9, width / 500)
                    cv2.putText(frame, label, (int(x1) + 6, max(int(40 * scale), int(y1) + int(40 * scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), max(2, int(scale * 2)))
        if writer is not None:
            writer.write(frame)
    if writer is not None:
        writer.release()

    out_csv = args.out_dir / f"{camera}_crossings.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=["camera", "seq", "track_id", "frame", "time_s",
                                                  "decision", "nok_probability", "classified_frames"])
        out.writeheader()
        for row in rows:
            scores = class_scores.get(row["track_id"])
            if scores and scores[1]:
                p_nok = scores[0] / scores[1]
                row.update(decision="NOK" if p_nok >= 0.5 else "OK",
                           nok_probability=round(p_nok, 4), classified_frames=int(scores[1]))
            else:
                row.update(decision="", nok_probability="", classified_frames=0)
        out.writerows(rows)
    print(f"{camera}: {len(rows)} bottles crossed the line -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
