# WS2 – Flessen detecteren

**Doel:** in elk videoframe een bounding box rond elke gasfles. Alles wat volgt (tracking, sync, classificatie, OCR) hangt hiervan af.

Er zijn twee versies. Versie 1 staat er nog om te kunnen vergelijken; de pijplijn gebruikt versie 2.

## Versie 1 – getraind op webfoto's (eerste iteratie)

| Script | Doet |
|---|---|
| `scripts/prepare_gas_dataset.py` | maakt een vaste 80/10/10-split (seed 42) van de Roboflow-set `data/raw/roboflow_gas_tank_yolo/` naar `data/processed/gasbottles/` |
| `scripts/check_gas_dataset.py` | controleert of beelden en labels kloppen |
| `scripts/train_gas_tank.py` | traint YOLO26n (20 epochs, 416 px) en kopieert het beste model naar `models/gas_tank_best.pt` |
| `scripts/predict_gas_tank.py` | draait het model op een foto of map |

Op de eigen validatieset haalt dit model mAP50 0,98. Op **echte Primagaz-beelden** haalt het maar **mAP50 0,32** (zie `results/test/old_detector_on_primagaz_test.json`). Dat komt doordat webfoto's er helemaal anders uitzien dan de camera's aan de lijn. Dit verschil heet *domain gap*. Daardoor mislukte de tracking in WS4 en WS5: flessen werden gemist, en relingen en dozen werden als fles gezien.

## Versie 2 – getraind op echte Primagaz-frames (gebruikt)

### Stap 1: dataset bouwen – `scripts/build_primagaz_detector_dataset.py`

Het probleem: er bestaan geen gelabelde boxen van de Primagaz-beelden, en alles met de hand labelen kost veel tijd.

De oplossing: de zip van de lector (`ai_applications_ok_nok_classification_training_set.zip`) bevat honderden uitsnedes uit de video's, met namen als `14_55_back_left_id_2_180.jpg` (opname, camera, fles-id, framenummer). Zo'n uitsnede is een letterlijk stuk van het frame. Als we die uitsnede in het frame terugvinden, kennen we de exacte box.

Wat het script doet:
1. **Offset per camera schatten.** De framenummers in de namen volgen de tijdlijn van de front-camera, maar de back-video's lopen verschoven. Voor 6 uitsnedes per camera zoekt het script binnen ±60 frames waar de uitsnede het best past (template matching op 20% van de resolutie) en neemt het de mediaan. Uitkomst voor 14_55: back_left **+35**, back_right **+5** frames.
2. **Elke uitsnede terugvinden** in frame `nummer + offset`: eerst grof op 25% van de resolutie, daarna exact op volle resolutie (`locate()`). Een match telt alleen bij een score ≥ 0,9. Positie plus grootte van de uitsnede geeft de box. Alle 916 uitsnedes van 14_55 werden gevonden.
3. **Buurflessen toevoegen.** Soms staat er een tweede fles half in beeld. Die zou anders als "achtergrond" geleerd worden. Het oude model voegt boxen toe met confidence ≥ 0,6, maar alleen als ze hoog zijn (≥ 60% van de beeldhoogte) en tegen de linker- of rechterrand staan (`is_neighbour_bottle()`). Zo vallen valse boxen op dozen en relingen weg.
4. **Splitsen per fles-id in tijdsblokken:** de eerste 70% van de fles-ID's is train, de volgende 15% val, de laatste 15% test. Dezelfde fles komt dus nooit in twee splits.
5. **Wegschrijven** als YOLO-dataset naar `data/processed/primagaz_detector/`. De beelden worden opgeslagen op 1280 px breed, met `data.yaml` en `build_report.json`.

```bash
py scripts/build_primagaz_detector_dataset.py --recording 14_55
```

### Stap 2: trainen – `scripts/train_primagaz_detector.py`

- Start van het v1-model (`models/gas_tank_best.pt`): dat herkent al "fles-achtige" dingen.
- Één klasse `gas-bottle`; 640 px; maximaal 40 epochs met early stopping (patience 10).
- Test daarna op de test-split (flessen die het model niet zag). Het beste model wordt `models/primagaz_detector.pt`.
- Uitvoer: curves in `results/training/primagaz_detector/` en testresultaten in `results/test/primagaz_detector/` en `.../test_metrics.json`.

```bash
py scripts/train_primagaz_detector.py
```

## Resultaat (test-split, 141 frames van flessen die niet in de training zaten)

| Model | Precisie | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| v1 webfoto's | 0,39 | 0,38 | 0,32 | 0,16 |
| v2 Primagaz | 0,99 | 0,97 | 0,99 | 0,95 |

De validatieloss daalt mee met de trainingsloss, dus er is geen overfitting.

## Goed om te weten

- Alle trainingsframes komen uit één opname (14_55). Of het model ook op andere opnames werkt, zie je in WS4: op 14_32 vindt de sync 173 van de 174 flessen.
- `track_cameras.py` neemt automatisch `models/primagaz_detector.pt` als dat bestand bestaat, anders het v1-model. Het kiest ook zelf de juiste klasse.
