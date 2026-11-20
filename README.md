# Team 09 – Gasflessen-inspectie voor Primagaz (AI Applications)

Een prototype dat alleen uit software bestaat. Het volgt elke gasfles op de transportband van Primagaz door de camera's front, back_left en back_right, en geeft per fles:
- **één oplopend ID**, hetzelfde in alle camera's;
- een **OK/NOK**-beslissing (schade, vuil, oud type);
- de **tarra** en het **certificatiejaar**, gelezen met OCR.

Elke stap wordt per fles vergeleken met de ground truth (Excel) van Primagaz.

Groep 9: Hendrik Beeken, Nand Rotté, Andres Rafael Rivas Barreto.

## Uitleg

**Lees eerst [docs/README.md](docs/README.md).** Daar staan de mappenstructuur, de dataflow en hoe je alles draait. Daarna per workshop:

| Workshop | Uitleg |
|---|---|
| WS1 – aanpak en metingen | [docs/WS1_aanpak.md](docs/WS1_aanpak.md) |
| WS2 – detectie | [docs/WS2_detectie.md](docs/WS2_detectie.md) |
| WS4 – tracking en sync over 3 camera's | [docs/WS4_tracking_en_sync.md](docs/WS4_tracking_en_sync.md) |
| WS5 – OK/NOK per fles | [docs/WS5_ok_nok.md](docs/WS5_ok_nok.md) |
| WS6 – OCR en rankingscore | [docs/WS6_ocr_en_score.md](docs/WS6_ocr_en_score.md) |
| Voorbereiding verdediging | [docs/verdediging.md](docs/verdediging.md) |

## Snel starten

```bash
py -m pip install -r requirements.txt
```

```bash
py scripts/run_pipeline.py --recording 14_32
```

De pijplijn heeft de Primagaz-data in `data/raw/primagaz/` nodig, plus de getrainde modellen. Hoe je beide klaarzet, staat in [docs/README.md](docs/README.md#van-nul-beginnen).

## NDA

De Primagaz-data valt onder een NDA. Video's, frames, crops, Excel-waarden per fles en modellen die op die data getraind zijn, horen niet in een publieke repository; de `.gitignore` houdt ze buiten Git.

De eerste detector (`scripts/prepare_gas_dataset.py`, `train_gas_tank.py`, …) is getraind op een publieke Roboflow-set in `data/raw/roboflow_gas_tank_yolo/`. Controleer de licentie in die map voordat je ze buiten het vak deelt.
