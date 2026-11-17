"""Scoring of the synchroniser against the ground truth.

Two kinds of ground truth are used:
* the workbook (xlsx): number of bottles, their order and colour;
* frame-level reference labels (see reference_frames.py): for most bottles we know
  frames where that bottle id is in view, on the front-camera timeline.
"""

from __future__ import annotations

import statistics


def evaluate_against_reference(bottles: list[dict], reference: dict[int, list[int]], tolerance: int = 0) -> dict:
    """Per-bottle scores. `bottles` rows need bottle_id/start_frame/end_frame."""
    def covering(frame: int) -> list[int]:
        return [b["bottle_id"] for b in bottles
                if b["start_frame"] - tolerance <= frame <= b["end_frame"] + tolerance]

    matches, splits, missed = {}, [], []
    for gt_id, frames in reference.items():
        centre = int(statistics.median(frames))
        hits = covering(centre)
        if not hits:
            missed.append(gt_id)
            continue
        matches[gt_id] = hits[0]
        touched = {bid for f in frames for bid in covering(f)}
        if len(touched) > 1:
            splits.append(gt_id)

    by_pred: dict[int, list[int]] = {}
    for gt_id, pred in matches.items():
        by_pred.setdefault(pred, []).append(gt_id)
    merges = {pred: ids for pred, ids in by_pred.items() if len(ids) > 1}

    first = min(min(f) for f in reference.values())
    last = max(max(f) for f in reference.values())
    in_range = [b for b in bottles if b["end_frame"] >= first - 60 and b["start_frame"] <= last + 60]
    false_bottles = [b["bottle_id"] for b in in_range if b["bottle_id"] not in by_pred]

    exact = sum(1 for gt_id, pred in matches.items() if gt_id == pred)
    one_to_one = sum(1 for gt_id, pred in matches.items() if len(by_pred[pred]) == 1)
    n_ref = len(reference)
    return {
        "reference_bottles": n_ref,
        "found": len(matches),
        "recall": round(len(matches) / n_ref, 4) if n_ref else None,
        "one_to_one": one_to_one,
        "one_to_one_rate": round(one_to_one / n_ref, 4) if n_ref else None,
        "exact_id": exact,
        "exact_id_rate": round(exact / n_ref, 4) if n_ref else None,
        "missed_ids": missed,
        "merged": {str(k): v for k, v in merges.items()},
        "split_ids": splits,
        "false_bottles": false_bottles,
        "predicted_in_range": len(in_range),
        # predicted bottle -> ground-truth id, only where the match is one-to-one
        "mapping": {str(pred): gt_ids[0] for pred, gt_ids in by_pred.items() if len(gt_ids) == 1},
    }


def align_sequences(pred: list[str], truth: list[str], gap: int = -1, mismatch: int = -1, match: int = 2):
    """Needleman-Wunsch alignment of two label sequences (e.g. colours).

    Returns (pairs, stats) where pairs are (pred index or None, truth index or None).
    """
    n, m = len(pred), len(truth)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
    for j in range(1, m + 1):
        score[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (match if pred[i - 1] == truth[j - 1] else mismatch)
            score[i][j] = max(diag, score[i - 1][j] + gap, score[i][j - 1] + gap)
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + (match if pred[i - 1] == truth[j - 1] else mismatch):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and score[i][j] == score[i - 1][j] + gap:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    aligned = [(p, t) for p, t in pairs if p is not None and t is not None]
    agree = sum(1 for p, t in aligned if pred[p] == truth[t])
    return pairs, {
        "predicted": n, "truth": m, "aligned": len(aligned),
        "colour_agreement": round(agree / len(aligned), 4) if aligned else None,
        "extra_predicted": sum(1 for p, t in pairs if t is None),
        "missing_predicted": sum(1 for p, t in pairs if p is None),
    }
