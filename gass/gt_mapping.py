"""Which ground-truth id belongs to which synchronised bottle.

Preferred: the frame-level reference labels (exact). Otherwise: the colour-sequence
alignment made by extract_bottle_crops.py. Downstream evaluations (classifier, OCR)
use this mapping, so every bottle is evaluated once, against its own workbook row.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_mapping(folder: Path) -> tuple[dict[int, int], str]:
    ref = folder / "bottle_to_gt_reference.json"
    if ref.is_file():
        return {int(k): int(v) for k, v in json.loads(ref.read_text(encoding="utf-8")).items()}, "reference frames"
    colour = folder / "colour_check.json"
    if colour.is_file():
        data = json.loads(colour.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in data["alignment_bottle_to_gt_id"].items()}, "colour alignment"
    raise FileNotFoundError(f"no bottle -> ground truth mapping in {folder}")
