"""Fine-tune the OK/NOK image classifier and choose its decision threshold (WS5).

* YOLO26n-cls fine-tuned on data/processed/ok_nok (prepare_ok_nok_dataset.py).
* Early stopping on the validation split (lecturer feedback: the loss curves showed
  overfitting without it).
* The NOK threshold is not fixed at 0.5: it is chosen on the validation images to
  maximise F2 (recall counts 4x as much as precision), because missing a damaged or
  dirty bottle is worse than an extra manual check. The threshold is stored next to
  the weights and used by classify_bottles.py.

    py scripts/train_ok_nok.py
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
from gass.metrics import binary_metrics  # noqa: E402


def nok_probabilities(model: YOLO, folder: Path, imgsz: int, device: str) -> tuple[list[float], list[bool]]:
    nok_index = next(i for i, n in model.names.items() if str(n).lower() == "nok")
    probs, truth = [], []
    for label in ("ok", "nok"):
        files = sorted((folder / label).glob("*.jpg"))
        for start in range(0, len(files), 64):
            for result in model.predict([str(f) for f in files[start:start + 64]], imgsz=imgsz,
                                        device=device, verbose=False):
                probs.append(float(result.probs.data[nok_index]))
                truth.append(label == "nok")
    return probs, truth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "ok_nok")
    parser.add_argument("--weights", default=str(ROOT / "models" / "pretrained" / "yolo26n-cls.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="ok_nok_yolo26n_cls")
    args = parser.parse_args()
    if not (args.data / "train").is_dir() or not (args.data / "val").is_dir():
        parser.error("dataset not found; run scripts/prepare_ok_nok_dataset.py first")

    model = YOLO(args.weights)
    model.train(data=str(args.data), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(ROOT / "results" / "training"),
                name=args.name, exist_ok=False, seed=42, workers=0, plots=True)
    run_dir = Path(model.trainer.save_dir)
    target = ROOT / "models" / "ok_nok_best.pt"
    shutil.copy2(run_dir / "weights" / "best.pt", target)

    best = YOLO(str(target))
    probs, truth = nok_probabilities(best, args.data / "val", args.imgsz, args.device)
    candidates = []
    for step in range(5, 96):
        threshold = step / 100
        m = binary_metrics([p >= threshold for p in probs], truth)
        candidates.append((m["f2"], -abs(threshold - 0.5), threshold, m))
    f2, _, threshold, val_metrics = max(candidates)
    summary = {"weights": str(target.relative_to(ROOT)), "train_run": str(run_dir.relative_to(ROOT)),
               "imgsz": args.imgsz, "nok_threshold": threshold, "threshold_rule": "max F2 on validation images",
               "val_at_threshold": val_metrics,
               "val_at_0.5": binary_metrics([p >= 0.5 for p in probs], truth)}
    (ROOT / "models" / "ok_nok_threshold.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "threshold.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
