# WS6 – Tarra en jaar lezen (OCR) en de rankingscore

**Doel:** per fles de **tarra** (leeggewicht in kg) en het **certificatiejaar** lezen van de inkjetprint op de schouder, bv. `10.8 KG-2034 P15Y …`. Daarna meten hoe goed dat lukt, en alle onderdelen samenvatten in één score.

## OCR – `scripts/ocr_bottles.py` + `gass/ocr_parse.py`

### Wat er per fles gebeurt
1. **Crops kiezen:** per camera de eerste 3 crops uit `crops_index.csv` (`--per-camera 3`), dus 9 per fles. Met 5 per camera werd er op 14_55 niet meer gelezen, en het duurde langer.
2. **Voorbewerken** (`preprocess()`):
   - Alleen de **schouderband** wordt gehouden: 30% tot 85% van de hoogte (`--band 0.30 0.85`). Op 14_55 gemeten staat de print tussen 48% en 72%. Dat bespaart rekentijd en geeft minder valse tekst.
   - Het beeld wordt maximaal 1600 px, grijs gemaakt, en het contrast wordt versterkt met CLAHE.
3. **EasyOCR** (Engels) zoekt tekstvakjes en leest ze, met vergroting `mag_ratio 1.5`. De optie `--rotation` probeert ook ±15° gedraaid, maar is 4 keer trager zonder duidelijke winst, dus staat ze standaard uit.
4. **Cache:** de ruwe OCR-uitvoer per crop wordt bewaard in `ocr_cache/`. Pas je de regex of de stemming aan, dan herrekent alles in een seconde zonder OCR opnieuw te draaien.
5. **Lijnen samenvoegen** (`join_lines()`): "10.8 KG-2034" komt vaak in stukken terug ("10.8", "KG", "-2034"). Vakjes op dezelfde hoogte worden van links naar rechts aan elkaar gezet. Zowel de losse stukken als de samengevoegde lijn gaan naar de parser.

### Tekst → getallen (`parse_text()` in `gass/ocr_parse.py`)
- **Normaliseren:** hoofdletters, en in "cijferachtige" woorden worden typische OCR-fouten verbeterd (O→0, I/l→1, S→5, B→8, Z→2). `K6` en `KC` worden `KG`.
- **Tarra:** een getal met een decimaalteken direct voor `KG`, tussen 8 en 22 kg (`10.8 KG`, `10,7KG`).
  - Lukt dat niet, dan zijn varianten toegelaten die we in de video's zagen: `1064KG` (punt weggevallen, dus 10,64) en `10 9KC` (spatie in plaats van punt). Die tellen als "zwak".
- **Stickerregel:** de sticker op de fles vermeldt het **netto**gewicht: 12,5 kg butaan of 10,5 kg propaan. Dat is **niet** de tarra (waarschuwing van de lector). Een zwakke lezing van precies 12,5 of 10,5 (`125KG`, `10SKG`) wordt daarom genegeerd. Op 14_32 haalde dit 5 van de 6 grote fouten weg.
- **Jaar:** het jaartal na `KG-` ("sterk"), anders een losstaand `20xx` tussen 2010 en 2049 ("zwak").

### Stemmen per fles (`vote()`)
Alle lezingen van alle crops van één fles stemmen. Elke stem weegt de OCR-zekerheid (minimaal 0,05). Daarnaast:

| Type lezing | Gewicht |
|---|---|
| tarra direct gevolgd door een jaar (het inkjetpatroon), of met het woord TARE/TARA | × 1,0 |
| gewone tarra | × 0,5 |
| zwakke tarra | × 0,25 |
| jaar na `KG` | × 1,0 |
| los jaar | × 0,5 |

De tarra wordt afgerond op 0,1 kg (de nauwkeurigheid van de Excel). Per fles wint de waarde met de meeste steun. Is er geen lezing, dan is het resultaat "niet gelezen".

De parameters `--min-tare-support` en `--min-year-support` kunnen te zwakke uitkomsten weggooien. Op 14_55 kostte dat meer juiste dan foute lezingen, dus ze staan standaard op 0.

### Metrics (`gass/metrics.py`)
Een getal is niet gewoon "juist of fout": 5,2 kg lezen voor 5,0 is veel minder erg dan 5000 kg (feedback van de lector). Tarra en jaar meten we daarom verschillend, en de **dekking** (hoeveel flessen een antwoord kregen) staat altijd naast de fout.

| | Metrics | Waarom |
|---|---|---|
| Tarra (`tare_metrics`) | dekking, gemiddelde absolute fout (kg), mediaan, % binnen 0,1 kg, aantal fouten > 1 kg | de grootte van de fout bepaalt hoeveel er verkeerd gevuld wordt |
| Jaar (`year_metrics`) | dekking, % exact, fout in jaren, % juiste beslissing (geldig als jaar ≥ 2025) | wat telt is of de fles terecht tegengehouden wordt |

Waarden die in de Excel met `?` gemarkeerd zijn, tellen niet mee. De tarra wordt ook vergeleken met `Pr_Tare_KG` (de weegschaal van Primagaz).

**Uitvoer:**
- `ocr_predictions.csv` en `ocr_raw_readings.csv` (lokaal);
- `ocr_report.json`.

```bash
py scripts/ocr_bottles.py --recording 14_32
```

### Resultaten

| Opname | Tarra gelezen | Binnen 0,1 kg | Gem. fout | Jaar gelezen | Exact | Juiste beslissing |
|---|---|---|---|---|---|---|
| 14_55 (gebruikt om af te stellen) | 41% | 91% | 0,28 kg | 61% | 88% | 100% |
| **14_32 (niet gebruikt)** | **35%** | **98%** | **0,12 kg** | **49%** | **89%** | **96%** |

Wat gelezen wordt, is bijna altijd juist. Het zwakke punt is de **dekking**: bij ongeveer de helft van de flessen is de print in geen van de 9 crops leesbaar (bewegingsonscherpte, gebogen oppervlak, print aan de andere kant). Jaarfouten zijn meestal 9↔3-verwarringen (2039 gelezen als 2033).

## Rankingscore – `scripts/score_recording.py`

Om oplossingen te rangschikken heb je één getal nodig. Dat getal combineert de vier onderdelen, telkens gemeten per fles op een opname waarop niet getraind is:

```text
score = 100 × (0,25·sync + 0,30·visueel + 0,25·tarra + 0,20·jaar)
```

| Deel | Berekening | Gewicht, waarom |
|---|---|---|
| sync | flessen die precies één keer gevonden zijn ÷ max(flessen in Excel, gevonden flessen); met referentieframes als die er zijn, anders via de kleuruitlijning (zoals bij 14_32) | 0,25: alles hangt af van het juiste ID |
| visueel | F2 van NOK tegen `classification` | 0,30: een gemiste slechte fles is een veiligheidsrisico |
| tarra | gemiddelde per fles van `max(0, 1 − fout/0,5 kg)`; niet gelezen telt als 0 | 0,25: een verkeerde tarra betekent verkeerd vullen |
| jaar | 1 voor exact, 0,5 voor een fout jaar met de juiste beslissing, 0 anders | 0,20: de beslissing telt meer dan het cijfer |

Voor tarra betekent dat bij een echte waarde van 5,0 kg: 5,0 lezen geeft 1 punt, 5,2 geeft 0,6, 5000 geeft 0.

Het script berekent ook de **eindbeslissing** zoals de installatie die zou nemen: uitduwen als de fles visueel NOK is of als het gelezen jaar < 2025 is. Die beslissing wordt vergeleken met `Primagaz_status`.

**Uitvoer:** `ranking_report.json`.

```bash
py scripts/score_recording.py --recording 14_32
```

**14_32:** sync 0,994 · visueel 0,833 · tarra 0,338 · jaar 0,451, wat een score van **67,3 / 100** geeft.

De eindbeslissing vangt 5 van de 15 flessen die de operator uitduwde (de visuele check alleen vangt er 3), met 5 extra alarmen (3,2% van de OK-flessen).

## Snelheid

OCR kost ongeveer 0,5 à 1,3 s per crop op een laptop-GPU van 6 GB, dus ongeveer 6 s per fles. Dat is te traag voor de lijn, waar elke 4 s een fles passeert. Mogelijke oplossingen: minder crops per fles, een desktop-GPU, of eerst kijken welke camera de print in beeld heeft.
