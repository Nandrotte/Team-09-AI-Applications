# WS5 – OK/NOK per fles

**Doel:** per fles beslissen of ze **NOK** is (grote zichtbare schade, echt vuil, ander/oud type) of **OK**.

Kleine deuken, de tarra en het jaar horen hier niet bij. Tarra en jaar komen uit OCR (WS6).

Drie scripts: dataset maken, trainen en per fles beslissen.

## Stap 1 – `scripts/prepare_ok_nok_dataset.py`: de dataset

**Bronnen:**
- de zip van de lector `ai_applications_ok_nok_classification_training_set.zip`, met mappen `ok/` (1.005 beelden) en `nok/` (226 beelden);
- onze eigen crops uit de pijplijn (stap 3 van WS4) van 14_55. Het label komt uit de kolom `classification` van de Excel, via de koppeling fles → Excel-ID. Dit zijn exact de beelden die de classifier later in de pijplijn krijgt, zoals de lector vroeg ("in het formaat van je eindtoepassing").

**Belangrijke regels:**
- **Groeperen per fles** (`group_of()`). Namen als `14_55_front_id_66_5994.jpg` en `14_55_back_left_id_66_6012.jpg` zijn dezelfde fles (opname 14_55, id 66), alleen een andere camera. Alle beelden van één fles gaan samen naar train of naar val. Anders ziet het model bij het testen een fles die het al kent (leakage). Beelden zonder fles-id, zoals de foto's uit de fabriek, vormen elk een eigen groep.
- **Testopname weglaten** (`--holdout 14_32`). Alle beelden van 14_32 gaan eruit, zodat de echte test in stap 3 eerlijk is.
- **Val-split:** 15% van de flessen per klasse, gebruikt voor early stopping en om de drempel te kiezen.
- **Onevenwicht:** er zijn ongeveer 4,4 keer meer OK- dan NOK-beelden. NOK-beelden in train worden daarom 3 keer opgenomen (`--nok-oversample 3`).
- Beelden worden opgeslagen op maximaal 320 px (het model gebruikt 224 px). Dat maakt de training veel sneller.

**Uitvoer:**
- `data/processed/ok_nok/{train,val}/{ok,nok}/`;
- `dataset_report.json` met de aantallen.

```bash
py scripts/prepare_ok_nok_dataset.py --pipeline-recordings 14_55 --holdout 14_32
```

## Stap 2 – `scripts/train_ok_nok.py`: trainen en drempel kiezen

- Model: YOLO26n-cls (`models/pretrained/yolo26n-cls.pt`), 224 px.
- **Early stopping** (patience 8). De training stopte na 12 epochs en hield epoch 4. Daarna daalde de trainingsloss nog, maar steeg de val-loss: dat is overfitting.
- **Drempel:** het model geeft per beeld een NOK-kans. In plaats van 0,5 kiezen we de drempel die op de val-beelden de hoogste **F2** geeft. F2 telt recall 4 keer zwaarder dan precisie, omdat een gemiste slechte fles erger is dan een extra controle. Gekozen drempel: **0,32**.
- **Uitvoer:**
  - `models/ok_nok_best.pt`;
  - `models/ok_nok_threshold.json` (drempel + val-scores);
  - curves in `results/training/ok_nok_yolo26n_cls-2/`.

```bash
py scripts/train_ok_nok.py
```

## Stap 3 – `scripts/classify_bottles.py`: beslissing per fles

- Elke fles heeft 15 crops (5 frames × 3 camera's) en het model geeft voor elke crop een NOK-kans.
- **Score van de fles** = gemiddelde van de **3 hoogste** kansen (`--top 3`). Schade of vuil is vaak maar langs één kant zichtbaar. Een gemiddelde over alle 15 crops zou dat verdunnen: dan zou de recall zakken naar 1 op 6.
- De fles is NOK als die score ≥ de drempel.
- **Evaluatie, één keer per fles:**
  - tegen `classification`: het visuele oordeel van de lector, de juiste ground truth voor een visuele classifier;
  - tegen `Primagaz_status`: de beslissing van de operator, ter info. Die bevat ook redenen die je niet ziet, zoals een onleesbare tarra of een vervallen jaar.
- **Uitvoer:**
  - `classification_per_bottle.csv` en `classification_per_crop.csv` (lokaal);
  - `classification_report.json`.

```bash
py scripts/classify_bottles.py --recording 14_32
```

## Resultaten

| | NOK-recall | NOK-precisie | Vals alarm |
|---|---|---|---|
| val-beelden (drempel 0,32) | 0,98 | 0,96 | 0,7% |
| **14_32 per fles (niet getraind)** | **5/6** | **5/6** | **1 op 167** |
| 14_32 met gemiddelde van alle crops | 1/6 | 1/1 | 0 |

Tegenover `Primagaz_status` vindt de visuele classifier 3 van de 15 uitgeduwde flessen. Dat is te verwachten, want operators duwen ook flessen uit om redenen die niet zichtbaar zijn.

## Beperkingen

- De testopname bevat maar 6 NOK-flessen. Eén extra mis zou al 67% recall geven, dus het cijfer is nog onzeker.
- Gemiste fles: 14_32 id 46 (score 0,06). Vals alarm: id 149 (score 0,39, net boven de drempel).
