"""Fine-tune and evaluate a small YOLO image classifier for OK/NOK bottles."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "ok_nok")
    parser.add_argument("--weights", default="yolo26n-cls.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="ok_nok_yolo26n_cls")
    args = parser.parse_args()
    if not (args.data / "train").is_dir() or not (args.data / "val").is_dir():
        parser.error("dataset not found; run scripts/prepare_ok_nok_dataset.py first")
    model = YOLO(args.weights)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(root / "results" / "training"),
                name=args.name, exist_ok=False, seed=42, workers=0, patience=10)
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    target = root / "models" / "ok_nok_best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)

    # Evaluate the held-out, bottle-grouped test split (never used for fitting).
    evaluator = YOLO(str(target))
    metrics = evaluator.val(data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch,
                            device=args.device, project=str(root / "results" / "test"),
                            name=args.name, exist_ok=False, workers=0)
    summary = {key: float(value) for key, value in metrics.results_dict.items() if isinstance(value, (int, float))}
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"model": str(target), "test_metrics": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
