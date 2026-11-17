# WS4 – Tracking over multiple videos (second iteration)

**Goal:** every bottle gets one monotonically increasing ID (1 … x) that is the same in the front, back-left and back-right videos and equals the workbook `id`.

## Method

1. **Detect + track per camera:** Primagaz detector (WS2 v2) + ByteTrack on every 2nd frame (`scripts/track_cameras.py`).
2. **Align the videos in time:** the recordings do not start at the same moment. For each camera we pick the frame offset at which its biggest-box track IDs agree best with the front camera ("purity", `estimate_offset_purity`). No ground truth is needed for this. Result for 14_55: back_left +46 and back_right +16. The lecturer's crops imply +35/+5; both work, because a bottle stays in view for about 100 frames.
3. **Lecturer's rule (strict):** per sampled frame, combine the track ID of the biggest box of each camera, e.g. `[3, 55, 23]`.
   - 6 identical combinations in a row make a new bottle.
   - A combination with `None` does not count.
   - A track ID that already belongs to a bottle cannot start a new one.
4. We also built a **robust variant** that tolerates one missing camera and merges fragmented tracks. It merged neighbouring bottles and shifted the IDs, so we keep the lecturer's rule.

## Evaluation (per bottle, `scripts/sync_cameras.py`, `scripts/extract_bottle_crops.py`)

- **Reference frames:** the lecturer's crop names tell at which frame which bottle id is in view (107 bottles of 14_55). We check whether every bottle is found exactly once and gets the right ID.
- **Workbook:** number of bottles, plus the colour of every bottle compared with the `color` column after sequence alignment, as the lecturer suggests.

| Session | Detector | Bottles found / workbook | One-to-one | ID = workbook id | Colour agreement |
|---|---|---|---|---|---|
| 14_55 | web photos (v1) | 24 / 112 | 19% | 0% | – |
| 14_55 | Primagaz (v2) | **110 / 112** | **99.1%** | **99.1%** | **100%** (110/110) |
| 14_55 | v2 + lecturer offsets | 110 / 112 | 100% | 100% | 100% |
| **14_32 (unseen session)** | Primagaz (v2) | **173 / 174** | – | 172/173* | **100%** (173/173) |

\*14_32 has no reference frames, so the ID check uses the colour alignment.

The 14_55 videos end about 3 s after bottle 110, so bottles 111–112 are at most partly in view and have no reference frames. In 14_32 the only missed bottle is id 173, one of the last two bottles in the video; all other bottles 1–172 get exactly their workbook id.

## What made the difference

Synchronisation failed in the first iteration because the web-photo detector missed most bottles: with v1 the same code finds only 24 bottles. Once detection is reliable, the simple biggest-box rule is enough, provided the videos are aligned in time first.

## Limits and next steps

- Offline processing: 3 × 23 frames/s on a laptop GPU (RTX, 6 GB) at stride 2. That is about real time for one camera, so a live installation needs a GPU per line or a larger stride.
- The top camera is not used yet.
- Bottles that the operator removes between camera view and filling are visible in the video but missing from the filling log. The workbook used here is per camera passage, so this does not affect the scores above.

**Reproduce:** `py scripts/run_pipeline.py --recording 14_32 --to-step 3`
