# Team 09 – Primagaz gas-bottle inspection (AI Applications)

A software-only prototype that follows every gas bottle on the Primagaz conveyor through the front, back-left and back-right cameras. For each bottle it:
- gives **one increasing ID**, the same in all cameras;
- decides **OK/NOK** (damage, dirt, old type);
- reads the **tare weight** and **certification year**.

Every step is scored per bottle against the Primagaz workbooks.

Group 9: Hendrik Beeken, Nand Rotté, Andres Rafael Rivas Barreto.

## Reports (one page per workshop)

| Workshop | Report |
|---|---|
| WS1 Strategy, test setup, metrics | [docs/WS1_strategy.md](docs/WS1_strategy.md) |
| WS2 Bottle detection | [docs/WS2_detection.md](docs/WS2_detection.md) |
| WS4 Tracking over multiple videos | [docs/WS4_multicam_tracking.md](docs/WS4_multicam_tracking.md) |
| WS5 OK/NOK classifier | [docs/WS5_ok_nok.md](docs/WS5_ok_nok.md) |
| WS6 OCR and ranking criteria | [docs/WS6_ocr_and_criteria.md](docs/WS6_ocr_and_criteria.md) |
| Progress of the full solution | [docs/full_solution_progress.md](docs/full_solution_progress.md) |

## Pipeline

```text
videos (front, back_left, back_right)
  1 track_cameras.py         Primagaz YOLO detector + ByteTrack per camera
  2 sync_cameras.py          time offsets + lecturer's biggest-box rule -> one ID per bottle
  3 extract_bottle_crops.py  5 crops per camera and bottle (+ colour check vs. workbook)
  4 classify_bottles.py      OK/NOK per bottle (top-3 crop score, recall-oriented threshold)
  5 ocr_bottles.py           EasyOCR + regex + vote -> tare (kg) and year
  6 score_recording.py       one ranking score
  7 render_demo_video.py     3-camera demo video with IDs and results
```

Run everything for one recording:

```bash
py scripts/run_pipeline.py --recording 14_32
```

Shared code lives in [`gass/`](gass/): ground-truth loading, synchronisation, metrics, OCR parsing and the reference labels.

## Data (not in Git)

The Primagaz videos, workbooks and images are under NDA and this repository is public. Download them from the course data link into:

```text
data/raw/primagaz/
  ai_applications_ok_nok_classification_training_set.zip
  14_55/14_55_front_cropped.mp4, ..._back_left_..., ..._back_right_..., 14_55_ground_truth_ai_app.xlsx
  14_32/...
```

Weights trained on that data (`models/primagaz_detector.pt`, `models/ok_nok_best.pt`) are also kept local. Rebuild them with:

```bash
py scripts/build_primagaz_detector_dataset.py --recording 14_55
```

```bash
py scripts/train_primagaz_detector.py
```

```bash
py scripts/run_pipeline.py --recording 14_55 --to-step 3
```

```bash
py scripts/prepare_ok_nok_dataset.py --pipeline-recordings 14_55 --holdout 14_32
```

```bash
py scripts/train_ok_nok.py
```

Only code, aggregated metrics (`results/**/**report*.json`) and training curves are committed.

## Setup

Python 3.13+ with a CUDA GPU (tested on a 6 GB laptop GPU).

```bash
py -m pip install -r requirements.txt
```

EasyOCR downloads its models on first use. If Python reports an SSL error, download `craft_mlt_25k.zip` and `english_g2.zip` from the EasyOCR releases and unzip them into `~/.EasyOCR/model/`.

## First iteration (WS2 v1, kept for comparison)

- `scripts/prepare_gas_dataset.py`, `check_gas_dataset.py`, `train_gas_tank.py`, `predict_gas_tank.py`: a YOLO26n detector trained on a public Roboflow set of web photos (`data/raw/roboflow_gas_tank_yolo/`).
- It reached mAP50 0.98 on web photos but only 0.32 on Primagaz frames, which is why it was replaced (see WS2).
- Check the dataset licence in that folder before sharing it outside the course.
