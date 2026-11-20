# Hoe het project in elkaar zit

Deze map legt uit hoe het project werkt: welk bestand wat doet, wat erin gaat en wat eruit komt, en waarom de belangrijkste keuzes zo gemaakt zijn. Begin hier en lees daarna het bestand van de workshop die je nodig hebt.

| Bestand | Onderwerp |
|---|---|
| [WS1_aanpak.md](WS1_aanpak.md) | De globale aanpak en hoe we meten |
| [WS2_detectie.md](WS2_detectie.md) | Flessen detecteren (YOLO-detector) |
| [WS4_tracking_en_sync.md](WS4_tracking_en_sync.md) | Tracken per camera en één ID per fles over de 3 camera's |
| [WS5_ok_nok.md](WS5_ok_nok.md) | OK/NOK-classificatie per fles |
| [WS6_ocr_en_score.md](WS6_ocr_en_score.md) | Tarra en jaar lezen (OCR) en de rankingscore |
| [verdediging.md](verdediging.md) | Voorbereiding verdediging: begrippen, cijfers, verwachte vragen, zwakke punten |

WS3 (CNN-notebook) was een persoonlijke opdracht en WS7 wordt door iemand anders gemaakt; die staan hier niet in.

## Het probleem in één alinea

Bij Primagaz rijden gasflessen op een transportband langs vaste camera's: **front**, **back_left**, **back_right** (en **top**, die we nog niet gebruiken). Per fles willen we:
- één ID dat in alle camera's hetzelfde is;
- een beslissing **OK/NOK** (grote schade, erg vuil, oud type);
- de **tarra** (leeggewicht in kg) en het **certificatiejaar**, gelezen van de inkjetprint op de schouder.

Per opname (bv. `14_55` = 14u55) is er een Excel met de ground truth: één rij per fles met `id`, `color`, `classification` (visueel oordeel van de lector), `Primagaz_status` (beslissing van de operator), `year`, `tarra` en `Pr_Tare_KG` (gewogen tarra).

## Mappenstructuur

```text
gass/                      gedeelde code (Python-package), gebruikt door alle scripts
  __init__.py              paden (ROOT, data/raw/primagaz, results), camera-namen, video_path()
  ground_truth.py          leest de Excel en maakt de waarden schoon ('?', '2025?', 'black ')
  reference_frames.py      haalt "fles X is in beeld op frame F" uit de bestandsnamen van de lector-zip
  sync.py                  synchronisatie van de camera's (offsets + regel van de lector)
  evaluation.py            scoort de sync tegen de ground truth (per fles, kleurvolgorde)
  colour.py                is een fles zwart of cyaan? (voor de kleurcheck)
  gt_mapping.py            welke workbook-id hoort bij welke gevonden fles
  metrics.py               alle metrics: NOK-recall/F2, tarra-fout, jaar-beslissing, rankingscore
  ocr_parse.py             regex + stemming om tarra en jaar uit OCR-tekst te halen
scripts/                   uitvoerbare stappen (elk script heeft --help)
data/raw/primagaz/         Primagaz-data (NDA, niet in Git): video's, Excel per opname, OK/NOK-zip
data/processed/            gegenereerde datasets (niet in Git)
models/                    gewichten; pretrained/ = basismodellen van Ultralytics
results/
  training/, test/         trainingscurves en testresultaten van de modellen
  pipeline/<opname>/       uitvoer van de pijplijn per opname (rapporten in JSON)
docs/                      deze uitleg
```

## De pijplijn (wat er met één opname gebeurt)

```text
3 video's ──1──► tracks_<camera>.csv ──2──► bottles_strict.csv ──3──► crops/<fles>/*.jpg
                                                                       │
                                     ┌─────────────────────────────────┼──────────────────┐
                                     4                                 5                  │
                         classification_per_bottle.csv        ocr_predictions.csv        │
                                     └──────────────► 6 ranking_report.json ◄────────────┘
                                                      7 demo-video (mp4)
```

| Stap | Script | Doet | Uitleg in |
|---|---|---|---|
| 1 | `track_cameras.py` | flessen detecteren en tracken per camera | WS4 |
| 2 | `sync_cameras.py` | camera's uitlijnen in de tijd, één ID per fles, evaluatie | WS4 |
| 3 | `extract_bottle_crops.py` | uitsnedes per fles en camera, kleurcheck | WS4 |
| 4 | `classify_bottles.py` | OK/NOK per fles, evaluatie | WS5 |
| 5 | `ocr_bottles.py` | tarra en jaar per fles, evaluatie | WS6 |
| 6 | `score_recording.py` | één rankingscore + eindbeslissing | WS6 |
| 7 | `render_demo_video.py` | video met de 3 camera's, ID's en resultaten | WS4 |

Alles in één keer:

```bash
py scripts/run_pipeline.py --recording 14_32
```

Met `--from-step` en `--to-step` draai je een deel, bv. `--from-step 4` als tracking en sync al klaar zijn.

## Van nul beginnen

1. **Installeren:**
   ```bash
   py -m pip install -r requirements.txt
   ```
   Je hebt een NVIDIA-GPU met CUDA nodig. Geeft EasyOCR een SSL-fout bij het downloaden van zijn modellen? Download dan `craft_mlt_25k.zip` en `english_g2.zip` van de EasyOCR-releases op GitHub en pak ze uit in `~/.EasyOCR/model/`.
2. **Data:** download via de Primagaz-link (feedback bij de NDA-opdracht op Toledo) de map `videos/<opname>/` en `images/classification/ai_applications_ok_nok_classification_training_set.zip`. Zet alles in `data/raw/primagaz/`, bv. `data/raw/primagaz/14_55/14_55_front_cropped.mp4`.
3. **Modellen trainen,** in deze volgorde:
   1. detector: `build_primagaz_detector_dataset.py` en dan `train_primagaz_detector.py` (WS2);
   2. stap 1–3 van de pijplijn op 14_55: `run_pipeline.py --recording 14_55 --to-step 3`;
   3. classifier: `prepare_ok_nok_dataset.py` en dan `train_ok_nok.py` (WS5).
4. **Testen op een opname die niet in de training zat:**
   ```bash
   py scripts/run_pipeline.py --recording 14_32
   ```

## Waarom 14_55 en 14_32

We trainen op 14_55 (en op de lector-afbeeldingen) en testen op **14_32**. Dat is een andere opname die geen enkel model gezien heeft. Zo meten we hoe goed het werkt op nieuwe beelden, en niet hoe goed het de trainingsdata onthouden heeft.

## Resultaten op 14_32 (173 flessen)

| Onderdeel | Resultaat |
|---|---|
| Sync | 173 van 174 flessen, kleurvolgorde 100% juist, 172 keer het juiste workbook-ID |
| OK/NOK | 5 van 6 NOK-flessen gevonden, 1 vals alarm op 167 OK-flessen |
| Tarra | 35% gelezen, daarvan 98% binnen 0,1 kg |
| Jaar | 49% gelezen, 89% exact, 96% juiste geldig/vervallen-beslissing |
| Rankingscore | 67,3 / 100 |

## NDA

De Primagaz-data valt onder een NDA. Video's, frames, crops, Excel-waarden per fles en modellen die op die data getraind zijn horen niet in een publieke repository. De `.gitignore` houdt ze buiten Git. Alleen code, curves en samengevatte rapporten (`*report*.json`) worden gecommit.
