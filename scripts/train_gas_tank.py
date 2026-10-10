"""Train and evaluate a small YOLO gas-tank detector on the local dataset."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=        repository_root / "data" / "processed" / "gasbottles" / "data.yaml",
    )
    parser.add_argument(
        "--model",
        default=str(repository_root / "models" / "yolo26n.pt"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default="gas_tank_yolo26n")
    args = parser.parse_args()
    if not args.data.is_absolute():
        args.data = repository_root / args.data
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = repository_root / model_path

    try:
        if not args.data.is_file():
            raise FileNotFoundError(f"Dataset not found: {args.data}")
        torch: Any = importlib.import_module("torch")
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available in the active PyTorch installation. "
                    "Install a CUDA-enabled PyTorch build first."
                )
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        ultralytics: Any = importlib.import_module("ultralytics")
        model = ultralytics.YOLO(str(model_path))
        training_results = repository_root / "results" / "training"
        test_results = repository_root / "results" / "test"
        print(f"Training {model_path} on {args.device}")
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=42,
            project=str(training_results),
            name=args.name,
            exist_ok=False,
            workers=0,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        print(f"Best weights: {best}")
        repository_model = repository_root / "models" / "gas_tank_best.pt"
        repository_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, repository_model)
        print(f"Best model copied to: {repository_model}")
        print("Evaluating on the held-out test split...")
        metrics = model.val(
            data=str(args.data),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=True,
            project=str(test_results),
            name=args.name,
            exist_ok=False,
            workers=0,
        )
        print(f"Training results: {results.save_dir}")
        print(f"Test results: {metrics.save_dir}")
        print(f"Test mAP50: {metrics.box.map50:.4f}")
        print(f"Test mAP50-95: {metrics.box.map:.4f}")
        print(f"Test precision: {metrics.box.mp:.4f}")
        print(f"Test recall: {metrics.box.mr:.4f}")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
