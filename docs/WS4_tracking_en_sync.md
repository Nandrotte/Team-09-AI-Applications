# WS4 – Tracken en één ID per fles over de 3 camera's

**Doel:** elke fles krijgt één oplopend ID (1, 2, 3, …), hetzelfde in front, back_left en back_right, en gelijk aan de `id` in de Excel.

Drie scripts doen dit. Daarna volgt de demovideo.

## Stap 1 – `scripts/track_cameras.py`: detecteren en tracken per camera

- Leest de 3 video's van een opname (`data/raw/primagaz/<opname>/<opname>_<camera>_cropped.mp4`).
- Per camera: detector (WS2) + **ByteTrack** op elk 2e frame (`--stride 2`). Zo krijgt dezelfde fles binnen één video hetzelfde track-ID.
- Track-ID's zijn **per camera**: ID 55 in back_left heeft niets te maken met ID 55 in front. Die koppeling maakt stap 2.
- Uitvoer in `results/pipeline/<opname>/`:
  - `tracks_<camera>.csv`: één rij per box (`frame, track_id, conf, x1, y1, x2, y2`);
  - `tracks_meta.json`: fps, aantal frames, resolutie, stride en gebruikt model.
- Die CSV's worden hergebruikt, zodat de dure detectie maar één keer draait.

```bash
py scripts/track_cameras.py --recording 14_32
```

## Stap 2 – `scripts/sync_cameras.py` + `gass/sync.py`: de camera's koppelen

### 2a. De grootste box per frame
De camera's staan dicht bij de band: de fles die op dat moment geïnspecteerd wordt, is de **grootste box** in beeld. `load_biggest_boxes()` houdt per frame alleen die box over. Boxen kleiner dan 8% van het beeld (`--min-area 0.08`) zijn flessen veraf en worden genegeerd.

### 2b. De video's uitlijnen in de tijd (offset per camera)
De 3 video's starten niet op hetzelfde moment. Frame 1000 in front is dus niet hetzelfde moment als frame 1000 in back_left. Per camera zoeken we een **offset**: camera-frame = front-frame + offset.

`estimate_offset_purity()` probeert elke offset tussen −120 en +120 frames. Voor elke offset kijkt het per front-frame welke track in de andere camera dan de grootste is:
- bij de juiste offset hoort bij elke front-track (bijna) altijd dezelfde camera-track;
- bij een verkeerde offset raken die door elkaar.

De "purity" (het aandeel paren dat bij de dominante track hoort) is maximaal bij de juiste offset.

Dit heeft geen ground truth nodig, dus het werkt op elke nieuwe opname. Uitkomst: 14_55 back_left +46 en back_right +16; 14_32 back_left +20 en back_right +46. Wil je zelf offsets opgeven, gebruik dan `--offsets back_left=35 back_right=5`.

Een opmerking over 14_55: de lector-uitsnedes gebruiken +35 en +5, onze schatting geeft +46 en +16. Beide werken, omdat een fles ongeveer 100 frames in beeld is.

### 2c. De regel van de lector (`sync_strict`, standaard)
Per frame op de gemeenschappelijke tijdlijn (`build_combos()`) maken we een combinatie van de track-ID's van de grootste box in elke camera, bv. `[3, 55, 23]`.
- Zie je **6 keer na elkaar** (`--k 6`) dezelfde combinatie, dan is er een **nieuwe fles**, met het volgende ID.
- Een combinatie met een ontbrekende camera (`[3, None, 23]`) telt niet.
- Een track-ID dat al bij een vorige fles hoorde, mag geen nieuwe fles starten. Dat voorkomt dubbeltellingen.
- Zolang dezelfde combinatie terugkomt, wordt de lopende fles verlengd.

Er is ook een **robuuste variant** (`sync_robust`). Die laat één ontbrekende camera toe en voegt stukken samen die in de meerderheid van de camera's dezelfde track delen. Op onze data voegde die soms twee flessen samen, waardoor de ID's verschoven. Daarom gebruikt de pijplijn de strikte regel (`--main-mode strict`).

### 2d. Evaluatie
- **Tegen referentieframes** (`gass/evaluation.py`, `evaluate_against_reference`). Voor 14_55 weten we uit de lector-zip op welk frame welke fles in beeld is. Per fles uit de Excel kijken we:
  - is er een gevonden fles die op dat moment loopt (recall)?
  - is dat één-op-één, of zijn flessen samengevoegd of gesplitst?
  - is het gevonden ID gelijk aan het Excel-ID?
- **Aantal flessen** tegenover het aantal rijen in de Excel.

Uitvoer:
- `bottles_strict.csv` en `bottles_robust.csv`: per fles het start- en eindframe, de tijd, de track-ID's per camera en de offsets;
- `sync_report.json`: alle scores;
- `bottle_to_gt_reference.json`: koppeling fles → Excel-ID (alleen als er referentieframes zijn).

```bash
py scripts/sync_cameras.py --recording 14_32
```

## Stap 3 – `scripts/extract_bottle_crops.py`: uitsnedes per fles + kleurcheck

- Per fles en camera neemt het de boxen van haar track(s) binnen haar tijdvenster, verdeelt die in 5 tijdsstukken en kiest per stuk de beste box (grootste oppervlak × confidence).
- Die frames worden uit de video gesneden, met 3% marge, en opgeslagen als `crops/<fles-id>/<camera>_<frame>.jpg` (volle resolutie, nodig voor OCR). Het overzicht staat in `crops_index.csv`.
- Dit zijn de beelden die de classifier (WS5) en OCR (WS6) krijgen.
- **Kleurcheck** (voorstel van de lector):
  - `gass/colour.py` meet in het midden van de fles welk deel van de pixels duidelijk cyaan of groen is. Is de mediaan over de crops ≥ 25%, dan is de fles "cyaan", anders "zwart".
  - De kleurreeks van de gevonden flessen wordt uitgelijnd met de kolom `color` uit de Excel (Needleman-Wunsch, `align_sequences`).
  - Zo zie je of het aantal en de volgorde kloppen, en krijg je een koppeling fles → Excel-ID voor opnames zonder referentieframes.
  - Uitvoer: `colour_check.json`.

```bash
py scripts/extract_bottle_crops.py --recording 14_32
```

## Demovideo – `scripts/render_demo_video.py`

Toont de 3 camera's naast elkaar, met rond de geïnspecteerde fles een box en "ID n" (rood bij NOK). Onderaan staan de beslissing, de tarra en het jaar van de huidige fles naast de ground truth.

```bash
py scripts/render_demo_video.py --recording 14_32 --start-s 100 --duration-s 60
```

Uitvoer: `results/pipeline/<opname>/demo_<opname>_<start>s.mp4`. Die blijft lokaal (NDA).

## Resultaten

| Opname | Detector | Flessen / Excel | Één-op-één | ID = Excel-id | Kleur |
|---|---|---|---|---|---|
| 14_55 | v1 (webfoto's) | 24 / 112 | 19% | 0% | – |
| 14_55 | v2 (Primagaz) | 110 / 112 | 99,1% | 99,1% | 100% |
| 14_32 (niet getraind) | v2 | 173 / 174 | – | 172 van 173 | 100% |

De vergelijkingen staan in `results/pipeline/14_55_baseline_webdetector/` (oude detector) en `results/pipeline/14_55_known_offsets/` (offsets van de lector).

**Wat het verschil maakte:** de syncregel was niet het probleem. Met de oude detector werden de meeste flessen gemist. Met een goede detector en de video's eerst in de tijd uitgelijnd, werkt de eenvoudige regel van de lector bijna perfect.

## Beperkingen

- De top-camera wordt nog niet gebruikt.
- Snelheid: ongeveer 0,9× realtime voor 3 camera's op een laptop-GPU van 6 GB.
