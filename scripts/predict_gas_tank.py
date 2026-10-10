"""Run the trained gas-tank detector on an image or directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=repository_root / "models" / "gas_tank_best.pt",
        help="Trained model weights (default: models/gas_tank_best.pt)",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.model.is_absolute():
        args.model = repository_root / args.model
    if not args.source.is_absolute():
        args.source = repository_root / args.source

    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not args.source.exists():
        parser.error(f"source does not exist: {args.source}")
    model = YOLO(str(args.model))
    prediction_results = model.predict(
        source=str(args.source),
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(repository_root / "results" / "predictions"),
        name="gas_tank",
    )
    output_dir = prediction_results[0].save_dir if prediction_results else (
        repository_root / "results" / "predictions" / "gas_tank"
    )
    print(f"Predictions saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
