"""Frame-level reference labels hidden in the lecturer's OK/NOK image archive.

Many images in `ai_applications_ok_nok_classification_training_set.zip` are named
`<recording>_<camera>_id_<bottle id>_<frame>.jpg`, e.g. `14_55_front_id_66_5994.jpg`:
bottle 66 of the 14_55 ground truth is in view of the front camera at frame 5994.
The same frame numbers appear for all three cameras, which shows that the cropped
videos of one recording are already frame-synchronised.

These labels let us score the tracker/synchroniser per bottle instead of only
comparing totals.
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict

from . import RAW_PRIMAGAZ

ZIP_NAME = "ai_applications_ok_nok_classification_training_set.zip"
PATTERN = re.compile(r"^(\d\d_\d\d)_(front|back_left|back_right|top)_id_(\d+)_(\d+)\.(?:jpg|png)$", re.I)


def load_reference_frames(recording: str) -> dict[int, list[int]]:
    """Bottle id -> sorted frame numbers (all cameras merged) for one recording."""
    path = RAW_PRIMAGAZ / ZIP_NAME
    frames: dict[int, set[int]] = defaultdict(set)
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = PATTERN.match(name.rsplit("/", 1)[-1])
            if match and match.group(1) == recording:
                frames[int(match.group(3))].add(int(match.group(4)))
    return {bottle: sorted(values) for bottle, values in sorted(frames.items())}
