"""Demo: simulate several cameras from ONE video and run tracking + synchronisation.

Each simulated camera is a different time window and crop of the same footage, so
this only demonstrates the pipeline (per-camera tracking, offset estimation, shared
global IDs). It is not a test with independent camera views.

Output (under results/demo/): cam*.mp4, cam*_tracked.mp4 and demo_global_ids.mp4
(all cameras side by side, showing the same global ID G<n> per bottle).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# name, start second, crop (x0, x1, y0, y1 as fractions of the frame)
CAMERAS = [
    ("camA", 71.0, (0.0, 1.0, 0.0, 1.0)),
    ("camB", 72.5, (0.1, 1.0, 0.0, 1.0)),
    ("camC", 74.0, (0.0, 0.9, 0.05, 1.0)),
]


def cut(video: Path, out: Path, start: float, end: float, crop) -> None:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = int(crop[0] * width), int(crop[1] * width)
    y0, y1 = int(crop[2] * height), int(crop[3] * height)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (x1 - x0, y1 - y0))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    for _ in range(int((end - start) * fps)):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame[y0:y1, x0:x1])
    writer.release()
    capture.release()


def run(*cmd: str) -> None:
    subprocess.run([sys.executable, *cmd], check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path,
                        default=root / "testvideo" / "Bolivia FLEXSPEED for Side-Entry Valves 2014.mp4")
    parser.add_argument("--end", type=float, default=93.0)
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "demo")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scripts = root / "scripts"
    csvs = []
    for name, start, crop in CAMERAS:
        clip = args.out_dir / f"{name}.mp4"
        cut(args.video, clip, start, args.end, crop)
        run(str(scripts / "track_bottles.py"), "--video", str(clip), "--camera", name,
            "--out-dir", str(args.out_dir), "--device", args.device)
        csvs.append(str(args.out_dir / f"{name}_crossings.csv"))
    global_ids = args.out_dir / "global_ids.csv"
    run(str(scripts / "sync_bottles.py"), *csvs, "--output", str(global_ids))
    for name, _, _ in CAMERAS:
        run(str(scripts / "track_bottles.py"), "--video", str(args.out_dir / f"{name}.mp4"),
            "--camera", name, "--out-dir", str(args.out_dir), "--device", args.device,
            "--save-video", "--global-map", str(global_ids))

    captures = [cv2.VideoCapture(str(args.out_dir / f"{name}_tracked.mp4")) for name, _, _ in CAMERAS]
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    size = (640, 360)
    writer = cv2.VideoWriter(str(args.out_dir / "demo_global_ids.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[0] * len(captures), size[1]))
    while True:
        frames = []
        for (name, _, _), capture in zip(CAMERAS, captures):
            ok, frame = capture.read()
            frame = cv2.resize(frame, size) if ok else np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(frame, name, (10, size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            frames.append(frame)
        if all(not c.isOpened() or c.get(cv2.CAP_PROP_POS_FRAMES) >= c.get(cv2.CAP_PROP_FRAME_COUNT)
               for c in captures) and not any(f.any() for f in frames):
            break
        writer.write(np.hstack(frames))
        if all(c.get(cv2.CAP_PROP_POS_FRAMES) >= c.get(cv2.CAP_PROP_FRAME_COUNT) for c in captures):
            break
    writer.release()
    print(f"Demo video: {args.out_dir / 'demo_global_ids.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
