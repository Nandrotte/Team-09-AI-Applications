# Progress of the full solution (status 17 Nov 2026)

All numbers are per bottle on session **14_32**, which no model was trained on (173 synchronised bottles, workbook as ground truth). One command runs everything: `py scripts/run_pipeline.py --recording 14_32`.

| Step | Status | Result on 14_32 | Main open point |
|---|---|---|---|
| Detection (WS2) | done, v2 on plant frames | held-out frames 14_55: mAP50 0.99 (v1: 0.32) | trained on one session only |
| Sync, one ID per bottle (WS4) | done, lecturer's rule + automatic time offsets | 173/174 bottles, colour sequence 100%, ID = workbook id for 172 | top camera not used |
| OK/NOK visual (WS5) | done, per bottle | NOK recall 5/6, precision 5/6, 1 false alarm on 167 | only 6 NOK bottles to test on |
| OCR tare (WS6) | working | read 35%, of which 98% within 0.1 kg | coverage |
| OCR year (WS6) | working | read 49%, 89% exact, 96% right decision | coverage, 9↔3 confusions |
| Ranking score (WS6) | defined + computed | **67.3 / 100** | weights to agree with Primagaz |

**Final push-out decision** (visually NOK, or read year < 2025) against the operator's `Primagaz_status`: 5 of 15 pushed bottles caught, 5 extra alarms (3.2% of the OK bottles). Visual inspection alone catches 3 of 15. The operator also removes bottles for reasons that are not visible (tare missing, filling problems). Stopping bottles whose tare cannot be read would catch more of them, but needs higher OCR coverage first, or it would stop about two thirds of the line.

**Demo:** `py scripts/render_demo_video.py --recording 14_32 --start-s 100 --duration-s 60` shows the three cameras with the same ID per bottle, the decision, tare and year, next to the ground truth. The video is not in Git because of the NDA.

**Who explains what:** detector and dataset (WS2), synchronisation (WS4), classifier (WS5), OCR and metrics (WS6). Assign one owner per item in the GitHub issues before the presentation.

## Questions

- **Primagaz:** What are the other push-out reasons, and is a false alarm acceptable when it avoids a missed bad bottle?
- **Lecturer:** Is `year ≥ 2025` the right certification rule? Do you agree with the ranking weights?
