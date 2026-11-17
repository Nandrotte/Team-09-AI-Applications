# WS5 – OK/NOK classifier (second iteration)

**NOK** = big visible damage, really dirty, or an old/different bottle type. Small dents, tare and certification year are out of scope here; tare and year are handled by OCR in WS6.

## What we fixed after the first iteration

| Problem in v1 | Fix in v2 |
|---|---|
| Views of one bottle (`14_55_front_id_66`, `14_55_back_left_id_66`) could land in train *and* test. | Grouping on recording + bottle id, so every image of a bottle stays in one split. |
| Tested on images from the video the model was trained on (14_55). | Session **14_32 is held out completely** (25 lecturer images removed). The test is per bottle on that video. |
| Training data in a different format than the application sees. | Added **our own pipeline crops** of 14_55, labelled with the workbook `classification` through the synchroniser, the same crops the classifier gets in the full pipeline. |
| Fixed 0.5 threshold, 60% NOK recall. | Threshold chosen on validation for **F2** (recall weighs 4×): 0.32. NOK oversampled ×3 against the 4.4:1 imbalance. |
| No early stopping. | Early stopping (patience 8): training stopped after 12 epochs and kept epoch 4 (val loss 0.045). After that the train loss kept falling while the val loss rose: the overfitting the lecturer pointed out. |
| One crop decides. | **Per bottle:** mean of the 3 most suspicious crops out of 15 (5 frames × 3 cameras), because damage or dirt is often visible from one side only. |

**Data:** 2,184 OK + 256 NOK training images (275 bottle groups), 306 OK + 50 NOK validation images (48 groups). Model: YOLO26n-cls, image size 224.

## Results

**Validation images** (bottles not used for fitting, threshold 0.32): NOK recall 0.98, precision 0.96, false-alarm rate 0.7%.

**Per bottle on the unseen session 14_32** (173 synchronised bottles, ground truth = lecturer's `classification`):

| Aggregation | NOK recall | NOK precision | F2 | False alarms (OK bottles) |
|---|---|---|---|---|
| **top-3 crops (chosen)** | **5/6 = 0.83** | **5/6 = 0.83** | **0.83** | **1 / 167 = 0.6%** |
| mean of all crops | 1/6 = 0.17 | 1/1 | 0.20 | 0 / 167 |

Against the operator's `Primagaz_status` the visual recall is only 3/15. Operators also push out bottles for non-visual reasons (unreadable tare, expired year, filling problems), so `classification` is the right ground truth for this classifier, as the lecturer said.

## Limits

- Only 6 NOK bottles in the test session, so the confidence interval is wide. One extra miss would be 67% recall.
- The missed bottle (id 46) scores 0.06; it needs more varied NOK examples.
- The false alarm (id 149) scores 0.39, just above the threshold.

**Reproduce:** `py scripts/prepare_ok_nok_dataset.py`, `py scripts/train_ok_nok.py`, `py scripts/classify_bottles.py --recording 14_32`
