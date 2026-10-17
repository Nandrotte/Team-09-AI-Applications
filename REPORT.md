# Team 09 – Progress report: gas bottle tracking & OK/NOK

Repository: https://github.com/Nandrotte/Team-09-AI-Applications

## Summary (read this first)

| Item | Status |
|---|---|
| Synchronised tracking with one ID per bottle across the 3/4 videos | **Working on a simulated 3-camera demo, not yet on real footage.** Demo video: `results/demo/demo_global_ids.mp4` (3 cameras side by side, same G<n> label per bottle). 17 global bottles, 13 seen by all 3 cameras. `scripts/track_bottles.py` (per video) and `scripts/sync_bottles.py` (global IDs) exist. The demo uses ONE public factory video (`testvideo/`) cut into 3 time windows/crops (`scripts/demo_multicam.py`), so it shows the pipeline, not independent camera views. Limits: that video has a handheld camera, and the detector (trained on web photos) also boxes a person and misses some bottles. Real footage and a retrained detector are still needed. |
| Bottle detector (YOLO26n, class `gas-tank`) | **Done, first version.** Trained 20 epochs; mAP50 0.98 on the validation split, but see the caveat on leakage below. |
| OK/NOK classification | **Not started.** Architecture proposed below, and the data we need is identified. |
| OCR | Not started (out of scope for this report). |

What we can honestly show today: a trained detector and its curves, plus tested tracking/sync scripts. What we cannot show: IDs across real videos. The plan and open issues below describe how we get there.

## 1. What exists (the first experiments)

- **Data:** a public Roboflow Universe set "gas tank yolo" (1155 images, 2 classes: `bubble`, `gas-tank`), stored in [data/raw/roboflow_gas_tank_yolo/](data/raw/roboflow_gas_tank_yolo/). Script `scripts/prepare_gas_dataset.py` makes a seeded 80/10/10 split (871 / 109 / 109 images).
- **Model:** YOLO26n fine-tuned with `scripts/train_gas_tank.py` (20 epochs, image size 416, batch 8, CUDA). Weights: `models/gas_tank_best.pt`.
- **Result (run `results/training/gas_tank_yolo26n-2`, epoch 20, validation split):** precision 0.957, recall 0.940, mAP50 0.981, mAP50-95 0.625. Curves and confusion matrix are in `results/training/gas_tank_yolo26n-2/` and `results/test/gas_tank_yolo26n/`.
- **Earlier sanity check:** the untrained COCO model on a Primagaz photo (`results/predictions/detections.csv`) labelled the bottle as "toilet" at 0.15 confidence. This is why a fine-tuned detector was needed.
- **Predictions on a video:** `results/predictions/gas_tank/preview_*.jpg` are frames from `testvideo/Bolivia FLEXSPEED...mp4`. That is a stand-in YouTube video, **not** the Primagaz line footage, so it says nothing about real performance.

### Known weaknesses of these numbers (be honest about them)

1. **Probable train/val leakage** (fixed in code, not yet retrained: `py scripts/prepare_gas_dataset.py --group-by-source --output data/processed/gasbottles_grouped` gives a split with no shared source photos). The Roboflow export contains 3 augmented copies per source photo (file names like `100_jpg.rf.<hash>.jpg`). The split script shuffles the copies, so copies of the same photo end up in train and val/test (e.g. `100_jpg.rf.297e...` is in val while `100_jpg.rf.b3a9...` is in train). The 0.98 mAP50 is therefore optimistic. Retraining on the grouped split will give the honest number.
2. **Domain gap.** Training images are generic web photos of gas tanks, not the filling-station camera. The `bubble` class is irrelevant to us.
3. **No evaluation on Primagaz footage yet**, because that footage has no boxes.

## 2. Synchronised tracking – approach (scripts written, untested on real footage)

**Goal:** every bottle gets one monotonically increasing ID that is the same in every camera video.

**Proposed architecture**
1. Detect bottles per frame with the YOLO detector (done as a first version; must be retrained on Primagaz frames).
2. Track within one video with an off-the-shelf tracker (Ultralytics `model.track(..., tracker="bytetrack.yaml")`). This gives a per-video track ID, not yet a global one.
3. Convert track events to a **bottle sequence per camera**: the n-th bottle that crosses a fixed line (virtual gate) in the video. Because the bottles move on a conveyor in one direction, order is preserved.
4. Synchronise the cameras by aligning these sequences (the n-th bottle in camera A = the n-th bottle in camera B), with a time offset estimated from the first crossing and corrected when a bottle is missed or double-counted. Global ID = position in the aligned sequence, so it is monotonic by construction.
5. Check against the shared log file (per-bottle rows with tare weight and status) as ground truth: the number of bottles and their order must match.

**Why sequence-based and not appearance-based:** all bottles look almost the same, so re-identification by appearance will be unreliable. Order on the conveyor is the stronger signal.

**Usage**
```
py scripts/track_bottles.py --video cam1.mp4 --camera cam1 --axis x --line 0.5
(repeat per camera)
py scripts/sync_bottles.py results/tracking/cam1_crossings.csv results/tracking/cam2_crossings.csv ...
py scripts/track_bottles.py --video cam1.mp4 --camera cam1 --save-video --global-map results/tracking/global_ids.csv
```
The first CSV is the reference camera. The line position and direction must be tuned per camera. The sync offset is estimated by choosing the time shift that matches the most crossings, so one missed bottle does not shift later IDs.

**Main risks:** missed or duplicate detections shifting all following IDs (needs the resynchronisation in step 4), different frame rates / start times per video, bottles that are pushed out (NOK) and leave the line.

## 3. OK/NOK classification – first thoughts

- **Tool/architecture:** crop each tracked bottle from the detector box, then classify the crop. Start with a YOLO classification head or a small pretrained CNN fine-tuned on crops (binary OK/NOK), and take a decision per bottle by aggregating over frames (e.g. majority or mean probability across the track). If damage location matters, move to detection of defect types.
- **Data needed:** crops of OK and NOK bottles from the real line, with varied position and lighting.
- **Data we have / can get:**
  - the per-bottle log on AWS (`Primagaz_status`: OK = sent to filling station, NOK = pushed out by operator) gives **weak labels for free** once the bottle IDs are aligned with the videos (this is the reason the tracking comes first);
  - the AI engineer's own columns (`id/color/year/tarra`, `classification`) as extra reference, not as ground truth;
  - the extra video set `14_43`;
  - some local NOK photos taken at the Primagaz plant (private, not in Git).
- **Caveat:** `NOK` in the log means "operator pushed it out", which is not necessarily "visibly defective". The labels are noisy and the reasons for pushing out should be asked to the client.

## 4. Open issues and next steps

Open questions/bugs/tasks should be filed in the GitHub issue tracker and linked here. They are listed below so they can be filed as-is at https://github.com/Nandrotte/Team-09-AI-Applications/issues (no issues are filed yet):

1. Implement per-video tracking with ByteTrack and a virtual counting line.
2. Align bottle sequences across the 3/4 videos and produce the demo video with global IDs.
3. Validate the aligned IDs against the AWS log file (count and order).
4. Fix the dataset split to be grouped by source photo (leakage).
5. Annotate Primagaz frames (bottle boxes) and retrain/evaluate the detector on them.
6. Build the OK/NOK crop dataset from log status + aligned IDs; train a first classifier.
7. Ask Primagaz why bottles are pushed out (NOK definition).
8. `README.md` is Dutch and `03_dataset_check.py` naming is inconsistent with the other scripts; tidy up.

## 5. Who can explain what

Fill in names before submitting – every item above needs at least one owner who can explain it: detector training / dataset split (___), tracking and synchronisation approach (___), OK/NOK plan and data (___), repository and scripts (___).
