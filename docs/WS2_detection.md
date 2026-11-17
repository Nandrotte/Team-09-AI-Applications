# WS2 – Bottle detection (second iteration)

## What changed

The first detector (YOLO26n on 1,155 Roboflow web photos) reported mAP50 0.98 on its own validation split. We tested that model on **real Primagaz frames** and it reached only **mAP50 0.32** (recall 0.38). This domain gap is why tracking and synchronisation broke in WS4/WS5: bottles were missed, and railings and boxes were detected as bottles.

## New detector: trained on plant frames without manual labelling

- **Data:** the lecturer's OK/NOK archive holds 916 crops named `14_55_<camera>_id_<bottle>_<frame>.jpg`. These are unscaled cut-outs of the line videos.
  - We locate every crop again in its frame with template matching (all 916 found, score ≥ 0.9). That gives an exact bounding box.
  - While doing this we measured the camera offsets: back_left runs **+35 frames** and back_right **+5 frames** behind front.
  - Neighbouring bottles that are partly in view were added from the old detector, but only boxes that are tall and touching the frame edge (30 kept, 26 false boxes rejected).
- **Split:** by bottle id in time blocks (first 70% of ids train, 15% val, 15% test), so one physical bottle never sits in two splits. That gives 634 / 141 / 141 frames.
- **Model:** fine-tuned from the first detector, 40 epochs, image size 640, early stopping (patience 10). The validation loss kept decreasing with the training loss (0.69 → 0.24), so there is no overfitting.

## Results (held-out bottles, 141 frames)

| Detector | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Web-photo model (WS2 v1) | 0.388 | 0.379 | 0.324 | 0.159 |
| **Primagaz model (v2)** | **0.993** | **0.972** | **0.990** | **0.949** |

Curves and the confusion matrix are in `results/training/primagaz_detector/` and `results/test/primagaz_detector/`. Frames themselves are not in Git (NDA).

## Limits

- All training frames come from one session (14_55). The effect of other sessions is measured downstream, on session 14_32, through the synchronisation results (see WS4).
- Boxes come from the lecturer's crops plus filtered old-detector boxes. A few partly visible neighbours may still be unlabelled.

**Reproduce:** `py scripts/build_primagaz_detector_dataset.py`, then `py scripts/train_primagaz_detector.py`.
