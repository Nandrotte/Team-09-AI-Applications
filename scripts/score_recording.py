"""Combine the step reports of one recording into a single ranking score (WS6).

    score = 100 x (0.25 sync + 0.30 visual F2 + 0.25 tare + 0.20 year)

* sync   = bottles matched exactly once / max(workbook bottles, bottles found)
           (frame-level reference labels when available, else colour alignment)
* visual = F2 of NOK on the lecturer's visual classification (recall weighs 4x)
* tare   = mean partial credit 1 - |error| / 0.5 kg per bottle, unread = 0
* year   = 1 exact, 0.5 wrong year but right pass/fail decision, 0 otherwise

    py scripts/score_recording.py --recording 14_32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import RESULTS  # noqa: E402
from gass.metrics import RANKING_WEIGHTS, binary_metrics, ranking_score, sync_score  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--min-year", type=int, default=2025, help="certification valid if year >= this")
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()
    folder = args.dir or RESULTS / "pipeline" / args.recording

    sync = load(folder / "sync_report.json")
    robust = sync.get("modes", {}).get(sync.get("main_mode", "strict"), {})
    n_truth = sync.get("ground_truth_bottles", 0)
    if "reference" in robust:
        one_to_one = robust["reference"]["one_to_one"]
        sync_source = "reference frames"
    else:
        colour = load(folder / "colour_check.json")
        one_to_one = colour.get("aligned", 0)
        sync_source = "colour alignment (upper bound: aligned pairs)"
    parts = {"sync": sync_score(one_to_one, n_truth, robust.get("bottles_found", 0))}
    classification = load(folder / "classification_report.json")
    parts["visual_f2"] = classification.get("vs_classification (visual ground truth)", {}).get("f2", 0.0)
    ocr = load(folder / "ocr_report.json")
    parts["tare"] = ocr.get("tare_score_partial_credit", 0.0)
    parts["year"] = ocr.get("year_score", 0.0)
    report = {"recording": args.recording, "weights": RANKING_WEIGHTS, "parts": parts,
              "sync_source": sync_source, "ranking_score": ranking_score(parts)}

    # Final push-out decision as the installation would make it, compared with the operator.
    cls_rows, ocr_rows = folder / "classification_per_bottle.csv", folder / "ocr_predictions.csv"
    if cls_rows.is_file() and ocr_rows.is_file():
        with ocr_rows.open(newline="", encoding="utf-8") as handle:
            years = {r["bottle_id"]: r["year"] for r in csv.DictReader(handle)}
        pred, truth = [], []
        with cls_rows.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if not r.get("gt_primagaz_status"):
                    continue
                year = years.get(r["bottle_id"])
                expired = bool(year) and int(float(year)) < args.min_year
                pred.append(r["decision"] == "NOK" or expired)
                truth.append(r["gt_primagaz_status"] == "NOK")
        report["final_decision_vs_primagaz_status"] = {
            "rule": f"push out if visually NOK or read year < {args.min_year}",
            **binary_metrics(pred, truth)}
    (folder / "ranking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
