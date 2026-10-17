"""Merge per-camera crossing CSVs into one global, monotonically increasing bottle ID.

The first CSV is the reference camera. For every other camera the time offset is
estimated, crossings are matched to the reference within --tolerance seconds, and
bottles seen by only some cameras still get their own ID. IDs are numbered in time
order on the reference timeline, so they only ever increase.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_crossings(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["time_s"] = float(row["time_s"])
    return sorted(rows, key=lambda row: row["time_s"])


def estimate_offset(ref: list[dict], other: list[dict], tolerance: float) -> float:
    """Offset (other - reference) that makes the most crossings line up.

    Every pairwise time difference is a candidate, so a missed or extra bottle
    does not shift the estimate the way an index-by-index comparison would.
    """
    best_offset, best_score = 0.0, (-1, 0.0)
    for ref_row in ref:
        for other_row in other:
            offset = other_row["time_s"] - ref_row["time_s"]
            matches = match(ref, other, offset, tolerance)
            error = sum(abs(other[j]["time_s"] - ref[i]["time_s"] - offset)
                        for i, j in matches.items())
            score = (len(matches), -error)
            if score > best_score:
                best_offset, best_score = offset, score
    return best_offset


def match(ref: list[dict], other: list[dict], offset: float, tolerance: float) -> dict[int, int]:
    """Monotonic nearest-neighbour matching; maps reference index to other index."""
    matches: dict[int, int] = {}
    j = 0
    for i, ref_row in enumerate(ref):
        target = ref_row["time_s"] + offset
        while j < len(other) and other[j]["time_s"] < target - tolerance:
            j += 1
        if j < len(other) and abs(other[j]["time_s"] - target) <= tolerance:
            matches[i] = j
            j += 1
    return matches


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("crossings", type=Path, nargs="+",
                        help="crossings CSVs; the first one is the reference camera")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="max seconds between the same bottle in two cameras")
    parser.add_argument("--output", type=Path,
                        default=root / "results" / "tracking" / "global_ids.csv")
    args = parser.parse_args()

    cameras = [read_crossings(path) for path in args.crossings]
    names = [rows[0]["camera"] if rows else path.stem
             for rows, path in zip(cameras, args.crossings)]
    ref = cameras[0]

    # Each bottle: time on the reference timeline plus the row seen per camera.
    bottles = [{"time": row["time_s"], "seen": {0: row}} for row in ref]
    for cam_idx in range(1, len(cameras)):
        other = cameras[cam_idx]
        offset = estimate_offset(ref, other, args.tolerance)
        matches = match(ref, other, offset, args.tolerance)
        print(f"{names[cam_idx]}: offset {offset:+.2f}s vs {names[0]}, "
              f"matched {len(matches)}/{len(other)} bottles")
        for i, j in matches.items():
            bottles[i]["seen"][cam_idx] = other[j]
        matched_other = set(matches.values())
        for j, row in enumerate(other):
            if j not in matched_other:
                bottles.append({"time": row["time_s"] - offset, "seen": {cam_idx: row}})
    bottles.sort(key=lambda bottle: bottle["time"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["global_id", "camera", "track_id", "seq", "frame", "time_s", "seen_by_all"])
        for global_id, bottle in enumerate(bottles, start=1):
            complete = len(bottle["seen"]) == len(cameras)
            for cam_idx, row in sorted(bottle["seen"].items()):
                out.writerow([global_id, names[cam_idx], row["track_id"], row["seq"],
                              row["frame"], row["time_s"], int(complete)])
    full = sum(len(b["seen"]) == len(cameras) for b in bottles)
    print(f"{len(bottles)} global bottles, {full} seen by all {len(cameras)} cameras -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
