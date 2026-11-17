"""Load the Primagaz ground-truth workbooks into one clean table.

The workbooks differ per recording (extra `Sequence`/`Time.1` columns, '?' for
unreadable values, '2025?' for uncertain reads, 'black ' with a trailing space).
`load_ground_truth` normalises all of that so the rest of the code can rely on:

    id (int), color (str), classification ('ok'/'nok'), primagaz_status ('OK'/'NOK'),
    year (float or NaN), year_uncertain (bool), tarra (float or NaN),
    tarra_uncertain (bool), pr_tare_kg (float or NaN)
"""

from __future__ import annotations

import math
import re

import pandas as pd

from . import RAW_PRIMAGAZ


def _parse_number(value) -> tuple[float, bool]:
    """'10.7' -> (10.7, False); '15.4?' -> (15.4, True); '?' -> (nan, True)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan, True
    text = str(value).strip().replace(",", ".")
    uncertain = "?" in text
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return math.nan, True
    return float(match.group()), uncertain


def workbook_path(recording: str):
    matches = sorted((RAW_PRIMAGAZ / recording).glob("*ground_truth*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"no ground-truth workbook in {RAW_PRIMAGAZ / recording}")
    return matches[0]


def load_ground_truth(recording: str) -> pd.DataFrame:
    raw = pd.read_excel(workbook_path(recording))
    raw = raw[raw["id"].notna()].copy()
    rows = []
    for _, row in raw.iterrows():
        year, year_unc = _parse_number(row.get("year"))
        tarra, tarra_unc = _parse_number(row.get("tarra"))
        pr_tare = row.get("Pr_Tare_KG")
        rows.append({
            "recording": recording,
            "id": int(row["id"]),
            "time": row.get("Time"),
            "color": str(row.get("color", "")).strip().lower(),
            "classification": str(row.get("classification", "")).strip().lower(),
            "primagaz_status": str(row.get("Primagaz_status", "")).strip().upper(),
            "year": year,
            "year_uncertain": year_unc,
            "tarra": tarra,
            "tarra_uncertain": tarra_unc,
            "pr_tare_kg": float(pr_tare) if pd.notna(pr_tare) else math.nan,
        })
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
