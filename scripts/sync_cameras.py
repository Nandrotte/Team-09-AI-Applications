"""Step 2 of the pipeline: merge the per-camera tracks into one ID per bottle.

Reads `results/pipeline/<rec>/tracks_<camera>.csv` (from track_cameras.py), estimates
the frame offset of every camera against the front camera (or uses --offsets),
applies the lecturer's synchronisation rule (strict) and our robust variant, and
scores both against the ground truth:

* number of bottles vs. the workbook,
* per-bottle scores against the frame-level reference labels (when available).

Output: bottles_<mode>.csv, sync_report.json in the same folder.

    py scripts/sync_cameras.py --recording 14_55
    py scripts/sync_cameras.py --recording 14_55 --offsets back_left=35 back_right=5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import CAMERAS, RESULTS  # noqa: E402
from gass.evaluation import evaluate_against_reference  # noqa: E402
from gass.ground_truth import load_ground_truth  # noqa: E402
from gass.reference_frames import load_reference_frames  # noqa: E402
from gass.sync import (build_combos, estimate_offset_purity, load_biggest_boxes, read_bottles,  # noqa: E402
                       sync_robust, sync_strict, write_bottles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS), help="first = reference camera")
    parser.add_argument("--offsets", nargs="*", default=None,
                        help="camera=frames, e.g. back_left=35 (default: estimate automatically)")
    parser.add_argument("--k", type=int, default=6, help="consecutive samples needed for a new bottle")
    parser.add_argument("--min-area", type=float, default=0.08,
                        help="ignore boxes smaller than this fraction of the frame")
    parser.add_argument("--max-missing", type=int, default=1, help="robust mode: cameras allowed to miss")
    parser.add_argument("--main-mode", default="strict", choices=["strict", "robust"],
                        help="mode used by the next pipeline steps (default: the lecturer's rule)")
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()

    folder = args.dir or RESULTS / "pipeline" / args.recording
    meta = json.loads((folder / "tracks_meta.json").read_text(encoding="utf-8"))
    ref_cam = args.cameras[0]
    stride = meta[ref_cam]["stride"]
    fps = meta[ref_cam]["fps"]
    frame_area = meta[ref_cam]["width"] * meta[ref_cam]["height"]
    biggest = {c: load_biggest_boxes(folder / f"tracks_{c}.csv", args.min_area * frame_area) for c in args.cameras}

    offsets = {ref_cam: 0}
    offset_support = {}
    if args.offsets:
        offsets.update({k: int(v) for k, v in (item.split("=") for item in args.offsets)})
    else:
        for camera in args.cameras[1:]:
            offsets[camera], offset_support[camera] = estimate_offset_purity(biggest[ref_cam], biggest[camera],
                                                                             stride=stride)
    print("offsets (camera frame - reference frame):", offsets)

    n_frames = min(meta[c]["frame_count"] - offsets.get(c, 0) for c in args.cameras)
    combos = build_combos(biggest, offsets, args.cameras, range(0, n_frames, stride), stride)

    gt = load_ground_truth(args.recording)
    try:
        reference = load_reference_frames(args.recording)
    except FileNotFoundError:
        reference = {}
    report = {"recording": args.recording, "main_mode": args.main_mode, "offsets": offsets, "offset_purity": offset_support,
              "k": args.k, "stride": stride, "min_area": args.min_area,
              "ground_truth_bottles": int(len(gt)), "reference_bottles": len(reference), "modes": {}}
    for mode in ("strict", "robust"):
        bottles = sync_strict(combos, args.k) if mode == "strict" else sync_robust(combos, args.k, args.max_missing)
        path = folder / f"bottles_{mode}.csv"
        write_bottles(path, bottles, args.cameras, offsets, fps)
        result = {"bottles_found": len(bottles), "count_error": len(bottles) - int(len(gt))}
        if reference:
            result["reference"] = evaluate_against_reference(read_bottles(path), reference)
            mapping = result["reference"].pop("mapping")
            if mode == args.main_mode:
                (folder / "bottle_to_gt_reference.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        report["modes"][mode] = result
        ref = result.get("reference", {})
        print(f"{mode:6s}: {len(bottles)} bottles (ground truth {len(gt)})"
              + (f" | recall {ref['recall']} one-to-one {ref['one_to_one_rate']} exact-id {ref['exact_id_rate']}"
                 f" merged {len(ref['merged'])} split {len(ref['split_ids'])} false {len(ref['false_bottles'])}"
                 if ref else ""))
    (folder / "sync_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
