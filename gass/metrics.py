"""Performance metrics that follow the lecturer's feedback.

WS7 feedback on OCR metrics: precision/recall do not fit a predicted *number*, and a
prediction of 5.2 kg should not be scored the same as 5000 kg when the truth is 5 kg.
So we report *how* wrong a reading is, and we treat tare and year differently:

* tare (kg): what matters is the size of the error, because the filling station uses
  it to compute the gas mass -> coverage, mean absolute error, share within 0.1 kg.
* year: what matters is the decision it drives (certification still valid or not)
  -> coverage, exact-year accuracy, mean absolute error in years and accuracy of the
  pass/fail decision.

Coverage (how many bottles got a reading at all) is always reported next to the
error, so a method cannot look good by only answering easy bottles.
"""

from __future__ import annotations

import math
import statistics


def tare_metrics(pairs: list[tuple[float | None, float]]) -> dict:
    """pairs: (predicted tare or None, true tare) for bottles with a known true tare."""
    n = len(pairs)
    read = [(p, t) for p, t in pairs if p is not None]
    errors = [abs(p - t) for p, t in read]
    out = {"bottles": n, "read": len(read), "coverage": round(len(read) / n, 4) if n else None}
    if errors:
        out.update({
            "mae_kg": round(statistics.mean(errors), 3),
            "median_abs_error_kg": round(statistics.median(errors), 3),
            "rmse_kg": round(math.sqrt(statistics.mean(e * e for e in errors)), 3),
            "exact_0.1kg": round(sum(e < 0.05 for e in errors) / len(errors), 4),
            "within_0.2kg": round(sum(e <= 0.2 + 1e-9 for e in errors) / len(errors), 4),
            "gross_errors_over_1kg": sum(e > 1.0 for e in errors),
            # usable = read AND within 0.1 kg, over all bottles: one number for ranking
            "usable_rate": round(sum(e <= 0.1 + 1e-9 for e in errors) / n, 4),
        })
    return out


def year_metrics(pairs: list[tuple[int | None, int]], min_valid_year: int = 2025) -> dict:
    """pairs: (predicted year or None, true year). Decision: valid if year >= min_valid_year."""
    n = len(pairs)
    read = [(p, t) for p, t in pairs if p is not None]
    out = {"bottles": n, "read": len(read), "coverage": round(len(read) / n, 4) if n else None,
           "decision_rule": f"certified if year >= {min_valid_year}"}
    if read:
        errors = [abs(p - t) for p, t in read]
        decision_ok = [(p >= min_valid_year) == (t >= min_valid_year) for p, t in read]
        out.update({
            "exact": round(sum(e == 0 for e in errors) / len(read), 4),
            "mae_years": round(statistics.mean(errors), 3),
            "decision_accuracy": round(sum(decision_ok) / len(read), 4),
            "wrong_expired_decisions": sum(1 for (p, t), ok in zip(read, decision_ok) if not ok),
            "usable_rate": round(sum(e == 0 for e in errors) / n, 4),
        })
    return out


def tare_score(pairs: list[tuple[float | None, float]], tolerance_kg: float = 0.5) -> float:
    """Partial credit per bottle: 1 - |error| / tolerance (floored at 0); unread = 0.

    Truth 5.0 kg: reading 5.0 -> 1.0, 5.2 -> 0.6, 5000 -> 0.0. This scores *how* wrong
    a reading is, as asked in the WS7 feedback.
    """
    if not pairs:
        return 0.0
    return round(sum(0.0 if p is None else max(0.0, 1 - abs(p - t) / tolerance_kg) for p, t in pairs) / len(pairs), 4)


def year_score(pairs: list[tuple[int | None, int]], min_valid_year: int = 2025) -> float:
    """1 for the exact year, 0.5 for a wrong year with the right pass/fail decision, else 0."""
    if not pairs:
        return 0.0
    total = 0.0
    for p, t in pairs:
        if p is None:
            continue
        if p == t:
            total += 1.0
        elif (p >= min_valid_year) == (t >= min_valid_year):
            total += 0.5
    return round(total / len(pairs), 4)


def sync_score(one_to_one: int, n_truth: int, n_predicted: int) -> float:
    """Bottles matched exactly once, divided by max(truth, predicted): punishes misses and extras."""
    denominator = max(n_truth, n_predicted)
    return round(one_to_one / denominator, 4) if denominator else 0.0


RANKING_WEIGHTS = {"sync": 0.25, "visual_f2": 0.30, "tare": 0.25, "year": 0.20}


def ranking_score(parts: dict[str, float], weights: dict[str, float] = RANKING_WEIGHTS) -> float:
    """One number (0-100) to rank solutions; parts are the 0-1 component scores."""
    return round(100 * sum(weights[k] * parts.get(k, 0.0) for k in weights), 1)


def binary_metrics(pred: list[bool], truth: list[bool], beta: float = 2.0) -> dict:
    """NOK = positive class. F2 weights recall higher (missing a bad bottle is worse)."""
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum(t and not p for p, t in zip(pred, truth))
    tn = sum(not p and not t for p, t in zip(pred, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "nok_precision": round(precision, 4), "nok_recall": round(recall, 4),
            f"f{beta:g}": round(fbeta, 4),
            "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "accuracy": round((tp + tn) / len(pred), 4) if pred else None}
