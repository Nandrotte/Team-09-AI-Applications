# Gas Bottle Detection

YOLO-prototype voor het detecteren van gasflessen in afbeeldingen. Het model lokaliseert een gasfles met een bounding box. **OK/NOK-classificatie maakt nog geen deel uit van deze versie.**

## Resultaten

Elke training en testevaluatie wordt opgeslagen in een eigen runmap onder [`results/`](results/). De README bevat bewust geen vaste evaluatiecijfers, omdat uitkomsten kunnen verschillen wanneer de dataset, configuratie of training wordt aangepast.

Bekijk de `results.csv`, confusion matrices, detectiecurves en voorbeeldafbeeldingen in de meest recente runmap. Validatie met nieuwe, realistische beelden blijft nodig voordat conclusies over productiegebruik worden getrokken.

## Projectstructuur

```text
.
├── data/
│   ├── raw/
│   │   ├── roboflow_gas_tank_yolo/  # gedeelde brondata met afbeeldingen en labels
│   │   └── primagaz_nok/            # lokale, privéfoto's; niet voor Git
│   └── processed/
│       └── gasbottles/              # lokaal gegenereerde train/val/test-splits
├── models/
│   └── yolo26n.pt                   # YOLO26n-basismodel; lokaal
├── results/
│   ├── training/                    # trainingscurves en trainingsartefacten
│   ├── test/                        # evaluatie op de testsplit
│   └── predictions/                 # resultaten op nieuwe afbeeldingen
├── scripts/
│   ├── 03_dataset_check.py
│   ├── prepare_gas_dataset.py
│   ├── train_gas_tank.py
│   └── predict_gas_tank.py
├── .gitignore
├── requirements.txt
└── README.md
```

De Roboflow-brondataset wordt gedeeld via GitHub, zodat alle teamleden met dezelfde data kunnen werken. De Primagaz-foto's, gegenereerde datasets en modelgewichten blijven lokaal en worden door `.gitignore` uitgesloten.

## Vereisten

- Python 3.14 of een compatibele Python-versie
- NVIDIA-GPU met werkende CUDA-installatie voor GPU-training
- CUDA-enabled PyTorch in de gebruikte Pythonomgeving

Installeer de projectafhankelijkheden vanuit de repository-root:

```powershell
& C:\pad\naar\python.exe -m pip install -r requirements.txt
```

Controleer CUDA:

```powershell
& C:\pad\naar\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset voorbereiden

De gedeelde Roboflow-export staat in:

```text
data/raw/roboflow_gas_tank_yolo/
```

Maak de lokale dataset aan en controleer de annotaties:

```powershell
& C:\pad\naar\python.exe scripts\prepare_gas_dataset.py
& C:\pad\naar\python.exe scripts\03_dataset_check.py
```

Het script maakt met seed `42` dezelfde verdeling:

- 80% training
- 10% validatie
- 10% test

De verwerkte dataset komt in `data/processed/gasbottles/`. Deze map is reproduceerbaar en hoeft niet naar GitHub te worden geüpload.

## Trainen en evalueren

Start een training met:

```powershell
& C:\pad\naar\python.exe scripts\train_gas_tank.py
```

Standaardinstellingen:

- basismodel: `models/yolo26n.pt`
- 20 epochs
- image size: 416
- batch size: 8
- device: `cuda`
- dataloader workers: 0, voor stabiele werking op Windows

De training en testevaluatie worden opgeslagen onder:

```text
results/training/
results/test/
```

Elke nieuwe run krijgt een eigen map. Het beste model wordt daarnaast gekopieerd naar:

```text
models/gas_tank_best.pt
```

## Voorspellen op nieuwe beelden

Gebruik het laatst getrainde model op één afbeelding:

```powershell
& C:\pad\naar\python.exe scripts\predict_gas_tank.py `
  --source "C:\pad\naar\foto.jpg" `
  --device cuda `
  --conf 0.25
```

De standaardlocatie voor het getrainde model is `models/gas_tank_best.pt`. De geannoteerde afbeeldingen worden opgeslagen onder:

```text
results/predictions/gas_tank/
```

Voor een lokale test met Primagaz-foto's:

```powershell
& C:\pad\naar\python.exe scripts\predict_gas_tank.py `
  --source "data\raw\primagaz_nok" `
  --device cuda
```

Deze foto's zijn privé en worden niet naar GitHub gestuurd. Een predictie-run is geen formele evaluatie, omdat er voor deze foto's geen ground-truth-labels beschikbaar zijn.

## Resultaten bekijken

Na een training vind je de artefacten in de runmap onder [`results/training/`](results/training/):

- `results.png` — trainingsverloop per epoch;
- `results.csv` — waarden per epoch;
- `confusion_matrix.png` — confusion matrix;
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxPR_curve.png` en `BoxF1_curve.png` — detectiecurves;
- `val_batch*_labels.jpg` en `val_batch*_pred.jpg` — echte annotaties tegenover voorspellingen.

De testartefacten staan onder [`results/test/`](results/test/). Predictie-uitvoer staat onder [`results/predictions/`](results/predictions/), waaronder de bestaande [`detections.csv`](results/predictions/detections.csv).

## Volgende stap: OK/NOK

Voor OK/NOK-classificatie zijn gelabelde voorbeelden van beide klassen nodig. De dataset moet dan minstens bevatten:

- OK-gasflessen;
- NOK-gasflessen;
- voldoende variatie in positie, belichting, achtergrond en type beschadiging.

Dat is een aparte classificatie- of detectietaak en moet niet worden afgeleid uit de huidige `gas-tank`-detectieklasse.

## Dataset en licentie

De brondata is een YOLO-export uit Roboflow Universe. Controleer de licentie en vereiste attributie in de bestanden in [`data/raw/roboflow_gas_tank_yolo/`](data/raw/roboflow_gas_tank_yolo/) voordat de dataset buiten de onderwijsopdracht wordt verspreid.
