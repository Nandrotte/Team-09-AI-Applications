# WS6 – OCR and performance criteria (second iteration)

## OCR method (`scripts/ocr_bottles.py`, `gass/ocr_parse.py`)

1. **Input:** the synchronised bottle crops (3 frames × 3 cameras per bottle), cut to the shoulder band where the ink-jet print sits. We measured that band at 48–72% of the crop height and read 30–85%.
2. **OCR:** EasyOCR (English), grey + CLAHE. Text segments on one line are joined before parsing, because "10.8 KG-2034" often comes out as "10.8" / "KG" / "-2034".
3. **Regex** (lecturer tip):
   - **Tare:** a number followed by `KG` in the 8–22 kg range. Variants seen on the videos are accepted ("1064KG", "107 KG", "10 9KC"), with a lower weight.
   - **Year:** the value after `KG-`, otherwise a plausible `20xx`.
   - **Stickers:** "125KG"/"105KG" without a decimal point are the net-weight sticker (12.5 kg butane / 10.5 kg propane) and are ignored, as the lecturer warned.
4. **Vote** (lecturer tip) over all readings of the bottle, weighted by OCR confidence. A tare directly followed by a year (the ink-jet pattern) counts double.

## Metrics: *how* wrong, not just right/wrong (WS7 feedback)

Precision and recall do not fit a predicted number, so tare and year are scored differently. Coverage is always shown next to the error, so a method cannot look good by only answering easy bottles.

- **Tare:** coverage, mean absolute error (kg), share within 0.1 kg. Ranking credit per bottle is `max(0, 1 − |error| / 0.5 kg)`: reading 5.2 for 5.0 gets 0.6, reading 5000 gets 0.
- **Year:** coverage, exact year, error in years, and accuracy of the decision it drives (certified if year ≥ 2025). Ranking credit: 1 for the exact year, 0.5 for a wrong year with the right decision, 0 otherwise.

Workbook values marked `?` or uncertain are excluded.

| Session | Tare read | Within 0.1 kg | MAE | >1 kg errors | Year read | Exact | Decision OK |
|---|---|---|---|---|---|---|---|
| 14_55 (dev, used for tuning) | 32/78 (41%) | 91% | 0.28 kg | 1 | 40/66 (61%) | 88% | 100% |
| **14_32 (unseen)** | **46/132 (35%)** | **98%** | **0.12 kg** | **1** | **55/113 (49%)** | **89%** | **96%** |

Before the sticker rule, 14_32 had 6 errors over 1 kg, all of them "125kg"/"10Skg" sticker reads. A minimum vote support, tuned on 14_55, removed more correct reads than wrong ones, so it is off. In production, a weakly supported reading could go to manual check instead.

**Limit:** coverage. Half of the prints are not read in any of the 9 crops (motion blur, curved surface, print on the side facing away). Next steps: more frames per bottle near the moment the print faces a camera, the top camera, and a fine-tuned text recogniser.

## Ranking criterion: one score per solution

    score = 100 × (0.25·sync + 0.30·visual + 0.25·tare + 0.20·year)

| Part | Definition (per bottle, on a session the solution was not trained on) | Why this weight |
|---|---|---|
| sync | bottles matched exactly once ÷ max(workbook bottles, bottles found) | every other result depends on the right ID |
| visual | F2 of NOK vs. `classification` | a missed damaged bottle is a safety risk, so recall weighs 4× |
| tare | mean tare credit (above) | wrong tare means wrongly filled bottles (economic) |
| year | mean year credit (above) | expired bottles must be stopped, but the decision matters more than the digit |

**Our score on 14_32:** sync 0.994, visual 0.833, tare 0.338, year 0.451, giving **67.3 / 100**. `scripts/score_recording.py` computes it from the step reports.

As a gate before ranking, a solution must keep up with the line (≥ 1 bottle per 4 s per line).
