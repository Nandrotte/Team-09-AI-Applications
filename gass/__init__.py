"""Shared code for the Primagaz gas-bottle (GASS) prototype of team 09."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PRIMAGAZ = ROOT / "data" / "raw" / "primagaz"
RESULTS = ROOT / "results"
CAMERAS = ("front", "back_left", "back_right")


def video_path(recording: str, camera: str) -> Path:
    """Path of one cropped camera video, e.g. data/raw/primagaz/14_55/14_55_front_cropped.mp4."""
    folder = RAW_PRIMAGAZ / recording
    for suffix in (".mp4", ".MP4"):
        candidate = folder / f"{recording}_{camera}_cropped{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no video for {recording}/{camera} in {folder}")
