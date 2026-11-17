"""Fine-tune the bottle detector on real Primagaz frames (WS2 second iteration).

Starts from the first (web-photo) detector, trains on the dataset made by
build_primagaz_detector_dataset.py, evaluates on the held-out bottle ids (test
split) and copies the best weights to models/primagaz_detector.pt.

    py scripts/train_primagaz_detector.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass import ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "primagaz_detector" / "data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "gas_tank_best.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10, help="early stopping (lecturer feedback: overfitting)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="primagaz_detector")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    model.train(data=str(args.data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(ROOT / "results" / "training"),
                name=args.name, exist_ok=False, seed=42, workers=0, single_cls=True, plots=True)
    run_dir = Path(model.trainer.save_dir)
    target = ROOT / "models" / "primagaz_detector.pt"
    shutil.copy2(run_dir / "weights" / "best.pt", target)

    metrics = YOLO(str(target)).val(data=str(args.data), split="test", imgsz=args.imgsz, device=args.device,
                                    project=str(ROOT / "results" / "test"), name=args.name, exist_ok=False,
                                    workers=0, plots=True)
    summary = {
        "weights": str(target.relative_to(ROOT)),
        "train_run": str(run_dir.relative_to(ROOT)),
        "test_precision": round(float(metrics.box.mp), 4),
        "test_recall": round(float(metrics.box.mr), 4),
        "test_mAP50": round(float(metrics.box.map50), 4),
        "test_mAP50_95": round(float(metrics.box.map), 4),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
