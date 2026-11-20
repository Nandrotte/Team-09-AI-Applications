# WS1 – Aanpak en hoe we meten

Dit bestand legt de globale keuzes uit. De details van elk onderdeel staan in WS2–WS6.

## Opbouw van de oplossing

Elke stap gebruikt de uitvoer van de vorige:

1. **Detecteren:** waar staat een fles in het beeld? We gebruiken een YOLO-detector die getraind is op echte Primagaz-beelden (WS2).
2. **Tracken per camera:** dezelfde fles houdt binnen één video hetzelfde track-ID (ByteTrack, WS4).
3. **Synchroniseren:** de track-ID's van de 3 camera's worden aan elkaar gekoppeld tot één oplopend fles-ID, dat gelijk moet zijn aan de `id` in de Excel (WS4).
4. **Classificeren:** per fles OK of NOK, op basis van uitsnedes uit alle camera's (WS5).
5. **Lezen:** tarra en jaar met OCR (WS6).
6. **Beslissen:** de fles wordt uitgeduwd als ze visueel NOK is of als het gelezen jaar vervallen is (< 2025) (WS6, `score_recording.py`).

We werken met **opgenomen video's (offline)**: een prototype dat alleen uit software bestaat volstaat volgens de opdracht. Dezelfde code kan in principe op live camerabeelden draaien. Snelheid op een laptop-GPU van 6 GB:
- detectie en tracking van 3 camera's: ongeveer 0,9× realtime;
- OCR: ongeveer 5,7 s per fles.

## Gebruikte technologie

| Bestaand | Waarvoor |
|---|---|
| Ultralytics YOLO26n | detectie (WS2) en classificatie (WS5) |
| ByteTrack (in Ultralytics) | tracking per camera |
| OpenCV | video lezen, template matching, kleuren |
| EasyOCR | tekstherkenning |
| pandas + openpyxl | de Excel-bestanden lezen |

| Zelf gemaakt | Waar |
|---|---|
| Detector-dataset uit Primagaz-frames, automatisch gelabeld | `scripts/build_primagaz_detector_dataset.py` |
| Synchronisatie (tijdsoffset + regel van de lector) | `gass/sync.py` |
| Classifier-dataset zonder leakage + drempelkeuze | `scripts/prepare_ok_nok_dataset.py`, `train_ok_nok.py` |
| OCR-regex + stemming | `gass/ocr_parse.py` |
| Evaluatie per fles + rankingscore | `gass/evaluation.py`, `gass/metrics.py` |

## Hoe we meten: elke fles één keer, tegen haar eigen rij

Een score per afbeelding of per track-fragment zegt weinig: één fles levert tientallen beelden op, en een fragment kan een halve fles zijn. Daarom wordt **elke fles precies één keer** vergeleken met **haar eigen rij** in de Excel.

Om te weten welke rij bij welke gevonden fles hoort (`gass/gt_mapping.py`), zijn er twee bronnen:
- **Referentieframes** (beste bron): bestandsnamen in de lector-zip zoals `14_55_front_id_66_5994.jpg` zeggen dat fles 66 op frame 5994 in beeld is. Die info bestaat alleen voor 14_55.
- **Kleurvolgorde:** de reeks kleuren (zwart/cyaan) van de gevonden flessen wordt uitgelijnd met de kolom `color` in de Excel. Zo werkt het ook voor opnames zonder referentieframes, zoals 14_32.

**Trainen en testen zijn gescheiden per opname.** Modellen leren van 14_55 en van de afbeeldingen van de lector. Getest wordt op 14_32, waarvan niets in de training zat.

Welke metric per onderdeel, en waarom:

| Onderdeel | Metric | Waarom |
|---|---|---|
| Sync | gevonden flessen, één-op-één matches, ID = workbook-id, kleurovereenkomst | elk ander resultaat hangt af van het juiste ID |
| OK/NOK | NOK-recall, precisie, F2, vals-alarmratio | een gemiste slechte fles is erger dan een extra controle |
| Tarra | dekking, gemiddelde absolute fout (kg), % binnen 0,1 kg | de grootte van de fout telt (verkeerd vullen) |
| Jaar | dekking, exact, fout in jaren, juiste beslissing | wat telt is of de fles terecht wordt tegengehouden |

Al die metrics staan in `gass/metrics.py`. Hoe ze samen één rankingscore vormen, staat in WS6.

## Openstaande vragen

- **Voor Primagaz:** waarom duwen operators nog flessen uit, buiten schade, vuil, type, tarra en jaar? Wat is erger voor hen: een gemiste NOK of een vals alarm?
- **Voor de lector:** is de certificatieregel `jaar ≥ 2025` correct?
