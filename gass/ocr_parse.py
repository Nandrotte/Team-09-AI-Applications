"""Turn raw OCR text into tare weight and certification year, and vote per bottle.

Lecturer tips (WS6 slides): use regular expressions on the text segments that were
found, and vote over multiple images of a single bottle.

The ink-jet print on the shoulder looks like `10.8 KG-2034 P15Y ...`: tare weight in
kg followed by the year. Sticker weights (12.5 butane / 10.5 propane) are *net*
weights and must not be used as tare, so we only accept a number that is directly
followed by KG and lies in a plausible tare range.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Characters OCR often confuses inside numbers.
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                           "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9"})

TARE_RE = re.compile(r"(?<![\d.])(\d{1,2})\s?[.,·:]\s?(\d{1,2})\s*(?:K\s*[G6C]|KG|kg)", re.I)
# Seen on the Primagaz videos: "1064KG", "107 KG" (decimal point lost), "10 9KC" (space).
TARE_NO_SEP_RE = re.compile(r"(?<![\d.])(\d{2})(\d{1,2})\s*K\s*[G6C]", re.I)
TARE_SPACE_RE = re.compile(r"(?<![\d.])(\d{2})\s(\d)\s*K\s*[G6C]", re.I)
YEAR_AFTER_KG_RE = re.compile(r"K\s*[G6C]\s*[-–—~=._ ]?\s*(2\s?0\s?[1-4]\s?\d)", re.I)
YEAR_RE = re.compile(r"(?<!\d)(20[1-4]\d)(?!\d)")
TARE_RANGE = (8.0, 22.0)
STICKER_NET_WEIGHTS = {12.5, 10.5}
YEAR_RANGE = (2010, 2049)


def normalise(text: str) -> str:
    """Upper-case and fix letter/digit confusions only in number-like tokens."""
    text = text.upper().replace("K6", "KG").replace("KC", "KG")
    tokens = re.split(r"(\s+)", text)
    fixed = []
    for token in tokens:
        if sum(ch.isdigit() for ch in token) >= max(1, len(token) // 2):
            token = token.translate(DIGIT_FIX)
        fixed.append(token)
    return "".join(fixed)


def parse_text(text: str) -> dict:
    """Candidates found in one OCR text (one crop)."""
    norm = normalise(text)
    tares = []
    for whole, frac in TARE_RE.findall(norm):
        value = float(f"{whole}.{frac}")
        if TARE_RANGE[0] <= value <= TARE_RANGE[1]:
            tares.append(round(value, 2))
    weak_tare = False
    if not tares:  # fall back to readings where the decimal separator was lost
        for pattern in (TARE_SPACE_RE, TARE_NO_SEP_RE):
            for whole, frac in pattern.findall(norm):
                value = float(f"{whole}.{frac}")
                # "125KG" / "105KG" without a separator is almost always the net-weight sticker
                # (12.5 kg butane, 10.5 kg propane), which must not be used as tare
                if value in STICKER_NET_WEIGHTS:
                    continue
                if TARE_RANGE[0] <= value <= TARE_RANGE[1]:
                    tares.append(round(value, 2))
                    weak_tare = True
    years = [int(y.replace(" ", "")) for y in YEAR_AFTER_KG_RE.findall(norm)]
    strong_year = bool(years)
    if not years:
        years = [int(y) for y in YEAR_RE.findall(norm)]
    years = [y for y in years if YEAR_RANGE[0] <= y <= YEAR_RANGE[1]]
    # "10.8 KG-2034": tare directly followed by a year is the ink-jet print, not a sticker
    strong_tare = bool(re.search(TARE_RE.pattern + r"\s*[-–—~=._ ]?\s*2\s?0\s?[1-4]\s?\d", norm, re.I))
    strong_tare = strong_tare or "TARE" in norm or "TARA" in norm
    return {"tare": tares, "year": years, "strong_year": strong_year, "strong_tare": strong_tare and not weak_tare,
            "weak_tare": weak_tare, "text": norm}


def vote(readings: list[dict]) -> dict:
    """Weighted vote over all readings of one bottle.

    Each reading is parse_text() output plus 'conf' (OCR confidence 0..1).
    Tare values are voted after rounding to 0.1 kg (the workbook precision).
    """
    tare_votes, year_votes = defaultdict(float), defaultdict(float)
    for r in readings:
        weight = max(0.05, float(r.get("conf", 0.5)))
        for t in r["tare"]:
            tare_votes[round(t, 1)] += weight * (1.0 if r.get("strong_tare") else 0.25 if r.get("weak_tare") else 0.5)
        for y in r["year"]:
            year_votes[y] += weight * (1.0 if r.get("strong_year") else 0.5)
    result = {"tare": None, "tare_support": 0.0, "year": None, "year_support": 0.0,
              "n_readings": len(readings)}
    if tare_votes:
        result["tare"], result["tare_support"] = max(tare_votes.items(), key=lambda kv: kv[1])
    if year_votes:
        result["year"], result["year_support"] = max(year_votes.items(), key=lambda kv: kv[1])
    result["tare_support"] = round(result["tare_support"], 3)
    result["year_support"] = round(result["year_support"], 3)
    return result
