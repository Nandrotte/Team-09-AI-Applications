"""Run the full prototype on one recording, step by step.

    1. track_cameras.py      detect + track bottles in front / back_left / back_right
    2. sync_cameras.py       one global, increasing ID per bottle (+ evaluation)
    3. extract_bottle_crops  crops per bottle and camera (+ colour check vs. workbook)
    4. classify_bottles.py   OK/NOK per bottle (+ evaluation)
    5. ocr_bottles.py        tare weight and year per bottle (+ evaluation)
    6. score_recording.py    one ranking score from all step reports
    7. render_demo_video.py  3-camera demo video with IDs and results

    py scripts/run_pipeline.py --recording 14_32
    py scripts/run_pipeline.py --recording 14_32 --from-step 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--detector", default=str(ROOT / "models" / "primagaz_detector.pt"))
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--to-step", type=int, default=7)
    parser.add_argument("--demo-start-s", type=float, default=0.0)
    parser.add_argument("--demo-duration-s", type=float, default=60.0)
    args = parser.parse_args()
    rec = ["--recording", args.recording]
    steps = {
        1: ["track_cameras.py", *rec, "--model", args.detector],
        2: ["sync_cameras.py", *rec],
        3: ["extract_bottle_crops.py", *rec],
        4: ["classify_bottles.py", *rec],
        5: ["ocr_bottles.py", *rec],
        6: ["score_recording.py", *rec],
        7: ["render_demo_video.py", *rec, "--start-s", str(args.demo_start_s),
            "--duration-s", str(args.demo_duration_s)],
    }
    for number in range(args.from_step, args.to_step + 1):
        command = [sys.executable, str(SCRIPTS / steps[number][0]), *steps[number][1:]]
        print(f"\n=== step {number}: {' '.join(steps[number])}", flush=True)
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            print(f"step {number} failed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
