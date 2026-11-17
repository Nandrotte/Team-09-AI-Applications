# WS1 – Strategy (second iteration)

**Group 9** – Hendrik Beeken, Nand Rotté, Andres Rafael Rivas Barreto
**Repository:** https://github.com/Nandrotte/Team-09-AI-Applications

## How we solve the task

Primagaz films every bottle on the conveyor with fixed cameras (front, back-left, back-right, top) before the filling carousel. Our software-only prototype processes those videos per bottle:

1. **Detect** bottles in every frame (YOLO26n, fine-tuned on plant frames).
2. **Track** them per camera (ByteTrack).
3. **Synchronise** the cameras into one increasing ID per bottle. We first align the videos in time (frame offset per camera) and then apply the lecturer's rule: the biggest box in each camera, stable for K frames, is the bottle under inspection.
4. **Classify** each bottle OK/NOK (big damage, very dirty, old type) from its crops in all cameras.
5. **Read** the tare weight and certification year with OCR, using regular expressions and a vote over all crops of the bottle.
6. **Decide** per bottle: push out when it is visually NOK, the tare is unreadable, or the certification year has expired.

**Existing technology:** Ultralytics YOLO (detection, classification), ByteTrack, OpenCV, EasyOCR.
**Self-made AI components:**
- a Primagaz detector, trained on frames that are labelled automatically from the lecturer's crops;
- the multi-camera synchroniser;
- the OK/NOK classifier with a recall-oriented threshold;
- the OCR parser and voting;
- the evaluation code.

## Test setup

Offline on the recorded sessions (13_15 … 14_55), with the workbooks as ground truth. Detector and classifier are trained on 14_55 and the lecturer's images, and **evaluated on another session (14_32)** that they never saw. Every bottle is scored once, against its own workbook row. In a real installation the same pipeline would run on live camera streams. On a 6 GB laptop GPU, detection and tracking of all three cameras run at 0.9× real time (every 2nd frame) and OCR needs about 5.7 s per bottle. A desktop GPU per line, or fewer frames per bottle, is needed for real time.

## How we measure

| Component | Metric (per bottle) |
|---|---|
| Synchronisation | bottles found vs. workbook, one-to-one matches, ID = workbook ID, colour sequence agreement |
| OK/NOK | NOK recall (most important), precision, F2, false-alarm rate |
| Tare | coverage, mean absolute error (kg), share within 0.1 kg |
| Year | coverage, exact year, error in years, pass/fail decision accuracy |

These metrics are combined into one ranking score, described in [WS6_ocr_and_criteria.md](WS6_ocr_and_criteria.md).

## Questions

- **Primagaz:** Why are bottles pushed out besides damage, dirt, old type, missing tare and expired year? Which error is worse for you, a missed NOK bottle or a false alarm?
- **Lecturer:** Should the certification rule be `year >= 2025` or `year > current year`?
