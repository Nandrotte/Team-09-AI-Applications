"""Multi-camera synchronisation: one monotonically increasing ID per bottle.

Implements the lecturer's method (WS5 slides, "Synchronising 3 video's"):

    For each (sampled) frame, take the track id of the *biggest* bounding box in
    every camera and combine them, e.g. [3, 55, 23]. When the same combination is
    seen in K consecutive frames, a new bottle is found. [3, None, 23] does not
    count, and a track id that already belonged to an earlier bottle may not
    start a new bottle.

`strict` mode is exactly that rule. `robust` mode keeps the idea but tolerates the
failure modes we measured on the Primagaz videos:
* one camera may miss the bottle for a while (at most `max_missing` cameras None);
* a track that is fragmented in one camera (new track id, same bottle) is merged
  with the neighbouring segment when the other cameras keep the same track ids.

All cameras are put on one reference timeline first: camera frame = reference frame
+ offset[camera] (14_55: front 0, back_left +35, back_right +5).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

Combo = tuple  # one track id (or None) per camera


@dataclass
class Bottle:
    bottle_id: int
    start: int                      # reference frame of first supporting sample
    end: int                        # reference frame of last supporting sample
    tracks: dict = field(default_factory=dict)   # camera -> set of track ids
    frames: int = 0                 # number of supporting samples

    @property
    def middle(self) -> int:
        return (self.start + self.end) // 2


def load_biggest_boxes(csv_path: Path, min_area: float = 0.0) -> dict[int, tuple[int, tuple]]:
    """frame -> (track id, box) of the largest box in that frame."""
    best: dict[int, tuple[float, int, tuple]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < min_area:
                continue
            frame = int(row["frame"])
            if frame not in best or area > best[frame][0]:
                best[frame] = (area, int(row["track_id"]), box)
    return {frame: (track, box) for frame, (_, track, box) in best.items()}


def change_events(biggest: dict[int, tuple[int, tuple]], min_gap: int = 20) -> list[int]:
    """Frames where the biggest-box track id changes (a new bottle takes over the view)."""
    events, last_track, last_event = [], None, -10**9
    for frame in sorted(biggest):
        track = biggest[frame][0]
        if track != last_track and frame - last_event >= min_gap:
            events.append(frame)
            last_event = frame
        last_track = track
    return events


def estimate_offset(ref_events: list[int], cam_events: list[int], search: int = 200, tolerance: int = 6) -> tuple[int, int]:
    """Offset (camera frame - reference frame) that lines up the most change events."""
    best = (-1, 0)
    for offset in range(-search, search + 1):
        cam = set()
        for e in cam_events:
            cam.update(range(e - offset - tolerance, e - offset + tolerance + 1))
        score = sum(1 for e in ref_events if e in cam)
        if score > best[0] or (score == best[0] and abs(offset) < abs(best[1])):
            best = (score, offset)
    return best[1], best[0]


def estimate_offset_purity(ref: dict[int, tuple[int, tuple]], cam: dict[int, tuple[int, tuple]],
                           search: int = 120, stride: int = 2) -> tuple[int, float]:
    """Offset (camera frame - reference frame) at which the biggest-box track ids of the
    two cameras agree best.

    For a candidate offset, every sampled reference frame pairs a reference track id
    with a camera track id. At the right offset each reference track co-occurs with
    (almost) one camera track; at a wrong offset bottle boundaries are shifted and the
    pairs get mixed. Purity = share of pairs that belong to the dominant camera track
    of their reference track. Changes of the biggest box are less reliable (each
    camera sees a bottle enter at a different moment), purity uses the whole visit.
    """
    best = (-1.0, 0)
    ref_frames = sorted(ref)
    for offset in range(-search, search + 1):
        pairs: dict[int, dict[int, int]] = {}
        total = 0
        for frame in ref_frames:
            target = frame + offset
            hit = cam.get(target) or cam.get(target - 1) if stride > 1 else cam.get(target)
            if hit is None:
                continue
            counts = pairs.setdefault(ref[frame][0], {})
            counts[hit[0]] = counts.get(hit[0], 0) + 1
            total += 1
        if total == 0:
            continue
        purity = sum(max(c.values()) for c in pairs.values()) / total
        # also reward overlap, so an offset that pairs only a few frames cannot win
        score = purity * min(1.0, total / max(1, len(ref_frames) * 0.5))
        if score > best[0]:
            best = (score, offset)
    return best[1], round(best[0], 4)


def build_combos(biggest: dict[str, dict], offsets: dict[str, int], cameras: list[str],
                 ref_frames: range, stride: int) -> list[tuple[int, Combo]]:
    """Combination of biggest-box track ids per reference frame.

    Camera frames are sampled with `stride`; when ref+offset is not sampled the
    nearest sampled frame (within stride) is used.
    """
    combos = []
    for ref in ref_frames:
        combo = []
        for camera in cameras:
            target = ref + offsets.get(camera, 0)
            hit = None
            for delta in sorted(range(-stride + 1, stride), key=abs):
                if target + delta in biggest[camera]:
                    hit = biggest[camera][target + delta][0]
                    break
            combo.append(hit)
        combos.append((ref, tuple(combo)))
    return combos


def sync_strict(combos: list[tuple[int, Combo]], k: int) -> list[Bottle]:
    bottles: list[Bottle] = []
    used: set[tuple[int, int]] = set()   # (camera index, track id) already assigned
    run_combo, run_len, run_start = None, 0, 0
    for ref, combo in combos:
        if combo == run_combo:
            run_len += 1
        else:
            run_combo, run_len, run_start = combo, 1, ref
        if None in combo:
            continue
        if bottles and bottles[-1].tracks and all(
                tid in bottles[-1].tracks.get(i, set()) for i, tid in enumerate(combo)):
            bottles[-1].end = ref           # still the same bottle
            bottles[-1].frames += 1
            continue
        if run_len == k:
            if any((i, tid) in used for i, tid in enumerate(combo)):
                continue                    # note 2: a track id may not start a second bottle
            bottle = Bottle(len(bottles) + 1, run_start, ref, {i: {tid} for i, tid in enumerate(combo)}, k)
            bottles.append(bottle)
            used.update((i, tid) for i, tid in enumerate(combo))
    return bottles


def sync_robust(combos: list[tuple[int, Combo]], k: int, max_missing: int = 1) -> list[Bottle]:
    n_cams = len(combos[0][1]) if combos else 0
    # 1) split the timeline into segments of compatible combinations
    segments: list[dict] = []
    current = None
    for ref, combo in combos:
        if sum(t is None for t in combo) > max_missing:
            current = None
            continue
        if current is not None and all(
                t is None or current["ids"][i] is None or current["ids"][i] == t
                for i, t in enumerate(combo)):
            current["end"] = ref
            current["frames"] += 1
            current["ids"] = [current["ids"][i] if t is None else t for i, t in enumerate(combo)]
            for i, t in enumerate(combo):
                if t is not None:
                    current["tracks"].setdefault(i, set()).add(t)
            continue
        current = {"start": ref, "end": ref, "frames": 1, "ids": list(combo),
                   "tracks": {i: {t} for i, t in enumerate(combo) if t is not None}}
        segments.append(current)
    segments = [s for s in segments if s["frames"] >= k]

    # 2) merge neighbouring segments that share a track id in a majority of the cameras
    merged: list[dict] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            shared = sum(bool(prev["tracks"].get(i, set()) & seg["tracks"].get(i, set())) for i in range(n_cams))
            if shared >= n_cams // 2 + 1:
                prev["end"] = seg["end"]
                prev["frames"] += seg["frames"]
                for i, ids in seg["tracks"].items():
                    prev["tracks"].setdefault(i, set()).update(ids)
                continue
        merged.append(seg)
    return [Bottle(n + 1, s["start"], s["end"], s["tracks"], s["frames"]) for n, s in enumerate(merged)]


def write_bottles(path: Path, bottles: list[Bottle], cameras: list[str], offsets: dict[str, int], fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bottle_id", "start_frame", "end_frame", "mid_frame", "start_s", "end_s", "samples",
                         *[f"tracks_{c}" for c in cameras], *[f"offset_{c}" for c in cameras]])
        for b in bottles:
            writer.writerow([b.bottle_id, b.start, b.end, b.middle, round(b.start / fps, 2), round(b.end / fps, 2),
                             b.frames, *[" ".join(map(str, sorted(b.tracks.get(i, set())))) for i in range(len(cameras))],
                             *[offsets.get(c, 0) for c in cameras]])


def read_bottles(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("bottle_id", "start_frame", "end_frame", "mid_frame", "samples"):
            row[key] = int(row[key])
    return rows
