# Workshop 5 — OK/NOK progress

## Scope

Goal 1 only. OCR (tare weight and certification year) is out of scope.

## Data and implementation

- The supplied archive contains **1,005 OK** and **226 NOK** images.
- `scripts/prepare_ok_nok_dataset.py` builds a deterministic, bottle-grouped 70/15/15 split (seed 42), preventing alternate views of one bottle from appearing in both training and evaluation.
- Generated split: **837 train** (685 OK / 152 NOK), **206 validation** (162 OK / 44 NOK), **188 test** (158 OK / 30 NOK).
- `scripts/train_ok_nok.py` fine-tunes a small pretrained YOLO classifier, evaluates the held-out test split, and saves `models/ok_nok_best.pt` plus metrics under `results/`.
- `scripts/track_bottles.py` accepts the classifier, samples detected bottle crops, averages NOK scores by track, and writes the decision and score to the crossing CSV. `--save-video` overlays the current track decision.

## Results and limits

Trained `yolo26n-cls.pt` for 15 epochs on CPU (early stopping selected epoch 5). On the held-out test split, top-1 accuracy was **93.1%** (175/188 images). The confusion matrix gives **NOK recall 60.0%** (18/30), **NOK precision 94.7%** (18/19), and **OK recall 99.4%** (157/158). The accuracy is high partly because OK is the majority class; missing 12 of 30 NOK bottles is a material limitation.

The detector + tracker + classifier path was run over 400 frames of the repository's existing factory video. It produced one crossing row with an NOK score of 0.7125 and wrote an annotated video to `results/demo/workshop5/`. This video has no ground-truth OK/NOK labels and is unrelated to the supplied image set, so this confirms the integration runs but does not measure video accuracy.

The supplied synchronized three-view clip (`Workshop  test video.mp4`, 59.85 s, 23.98 FPS) was processed by `scripts/test_ok_nok_multiview_video.py`, sampling every fifth frame. The panels yielded 14 front, 22 left, and 18 right view-specific track segments. The classifier predicted 53 OK and one NOK segment; the NOK prediction was in the left view (mean NOK probability 0.6002 from one classified frame). These are not confirmed unique bottle identities: IDs are local to each view, and overlapping bottles can create duplicate or fragmented tracks. No labels were supplied for this clip, so this is a qualitative run rather than an accuracy evaluation. Per-track scores and times are in `results/classification_video/multiview/track_classifications.csv`.

### Ground-truth check for the shortened clip

The supplied clip is the first 60 seconds of the 14:55 recording. The workbook has seven entries for 14:55, all labeled `classification=ok` (visual label); their operator statuses are five OK and two NOK. The left-view NOK segment at 29.61 s is therefore a possible false alert against the visual labels. It received only one classifier sample. Because detections are fragmented into view-local track IDs and are not linked to the seven ground-truth bottle IDs, this comparison is not a valid per-bottle confusion matrix or accuracy score. A reliable metric needs identity/time matching for each bottle.

The archive is imbalanced (about 4.4 OK images per NOK image) and the split has only 30 NOK test images. Treat metrics as an initial estimate. This classifier targets large visible damage, severe dirt, and old/different bottle types. Small dents/scratches, tare-weight visibility, and certification year require additional data or OCR and are not claimed as solved. The classifier is trained on supplied still images, so performance on cropped bottles from the real line still needs a labeled video evaluation.

## Files

- `scripts/prepare_ok_nok_dataset.py` — extract and split the supplied ZIP.
- `scripts/train_ok_nok.py` — train and evaluate.
- `scripts/track_bottles.py` — optional track-level classifier integration.
