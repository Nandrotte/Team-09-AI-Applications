"""Run bottle detection and OK/NOK classification on a synchronized 3-view clip."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=root / "models" / "gas_tank_best.pt")
    parser.add_argument("--classifier", type=Path, default=root / "models" / "ok_nok_best.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "classification_video" / "multiview")
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=5, help="process every Nth video frame")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not args.detector.is_file() or not args.classifier.is_file():
        parser.error("detector or classifier weights are missing")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Layout of the supplied synchronized composite: front on top, left/right below.
    views = {
        "front": (round(width * 0.187), 0, round(width * 0.569), round(height * 0.382)),
        "left": (0, round(height * 0.384), round(width * 0.498), height),
        "right": (round(width * 0.500), round(height * 0.384), width, height),
    }
    detectors = {name: YOLO(str(args.detector)) for name in views}
    classifier = YOLO(str(args.classifier))
    names = classifier.names
    nok_index = next((i for i, name in names.items() if str(name).lower() == "nok"), None)
    if nok_index is None:
        parser.error("classifier weights do not contain a NOK class")

    tracks: dict[tuple[str, int], dict[str, float | int]] = defaultdict(
        lambda: {"nok_sum": 0.0, "classifications": 0, "detections": 0,
                 "first_frame": -1, "last_frame": -1}
    )
    last_classified: dict[tuple[str, int], int] = {}
    total_frames = 0
    sampled_frames = 0
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            if frame_index % max(1, args.stride) == 0:
                sampled_frames += 1
                for view, (x0, y0, x1, y1) in views.items():
                    image = frame[y0:y1, x0:x1]
                    if image.size == 0:
                        continue
                    result = detectors[view].track(
                        image, persist=True, tracker="bytetrack.yaml", classes=[args.class_id],
                        conf=args.conf, device=args.device, verbose=False,
                    )[0]
                    boxes = result.boxes
                    if boxes is None or boxes.id is None:
                        continue
                    for box, track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
                        key = (view, int(track_id))
                        state = tracks[key]
                        state["detections"] += 1
                        if state["first_frame"] < 0:
                            state["first_frame"] = frame_index
                        state["last_frame"] = frame_index
                        if frame_index - last_classified.get(key, -10**9) < 10:
                            continue
                        x1b, y1b, x2b, y2b = map(int, box)
                        crop = image[max(0, y1b):min(image.shape[0], y2b),
                                     max(0, x1b):min(image.shape[1], x2b)]
                        if crop.size == 0:
                            continue
                        prediction = classifier.predict(crop, imgsz=320, device=args.device, verbose=False)[0]
                        state["nok_sum"] += float(prediction.probs.data[nok_index].item())
                        state["classifications"] += 1
                        last_classified[key] = frame_index
            frame_index += 1
    finally:
        cap.release()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "track_classifications.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["view", "track_id", "first_time_s", "last_time_s", "detections",
                  "classified_frames", "mean_nok_probability", "decision"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (view, track_id), state in sorted(tracks.items()):
            n = int(state["classifications"])
            p_nok = float(state["nok_sum"]) / n if n else None
            first = int(state["first_frame"])
            last = int(state["last_frame"])
            writer.writerow({
                "view": view, "track_id": track_id,
                "first_time_s": round(first / fps, 2), "last_time_s": round(last / fps, 2),
                "detections": state["detections"], "classified_frames": n,
                "mean_nok_probability": round(p_nok, 4) if p_nok is not None else "",
                "decision": ("NOK" if p_nok >= 0.5 else "OK") if p_nok is not None else "",
            })
    metadata = {
        "video": str(args.video), "size": f"{width}x{height}", "fps": fps,
        "duration_s": round(total_frames / fps, 2), "sampled_frames": sampled_frames,
        "stride": args.stride, "tracks_by_view": {view: sum(1 for v, _ in tracks if v == view) for view in views},
        "note": "Unlabeled video: classifications are predictions, not accuracy metrics. Track IDs are local to each view.",
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "tracks_csv": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
