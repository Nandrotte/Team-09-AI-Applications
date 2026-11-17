"""Bottle colour (black vs. cyan/green) from a crop, used to verify synchronisation.

The lecturer suggests checking that the sequence of colours found in the video
matches the `color` column of the ground truth. Primagaz bottles here are either
black or cyan (green-blue). We look at the body of the bottle (centre of the crop)
and measure which fraction of pixels is clearly saturated green/cyan.
"""

from __future__ import annotations

import cv2
import numpy as np


def cyan_fraction(crop: np.ndarray) -> float:
    h, w = crop.shape[:2]
    body = crop[int(h * 0.35):int(h * 0.9), int(w * 0.25):int(w * 0.75)]
    if body.size == 0:
        return 0.0
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (hue >= 35) & (hue <= 100) & (sat >= 70) & (val >= 50)
    return float(mask.mean())

