# Voorbereiding verdediging

Dit bestand is een spiekbrief voor de verdediging. Het herhaalt niet alles uit WS1–WS6, maar zet op één plaats:
- de begrippen die je moet kunnen uitleggen;
- de cijfers die je moet kennen;
- de vragen die je kan verwachten, met een kort antwoord en waar het in de code staat;
- de zwakke punten, zodat je die zelf kan benoemen voor iemand anders het doet.

## 1. Het verhaal in 30 seconden

> We volgen elke gasfles door 3 camera's en geven haar één ID dat overeenkomt met de Excel van Primagaz. Per fles beslissen we OK/NOK met een classifier en lezen we tarra en jaar met OCR. Alles meten we **per fles**, op een opname (**14_32**) waarop niets getraind is.
>
> De grootste winst kwam van een detector die getraind is op echte Primagaz-frames: mAP50 ging van 0,32 naar 0,99. De boxen daarvoor hebben we automatisch gemaakt uit de uitsnedes van de lector.
>
> Het zwakste punt is de OCR-dekking: wat we lezen is bijna altijd juist, maar we lezen maar bij ongeveer de helft van de flessen iets.

## 2. Cijfers die je moet kennen

| Onderdeel | Cijfer | Waar gemeten |
|---|---|---|
| Detector v1 (webfoto's) op Primagaz | mAP50 **0,32** | test-split 14_55 |
| Detector v2 (Primagaz) | P 0,99 · R 0,97 · mAP50 **0,99** · mAP50-95 0,95 | test-split 14_55 (141 frames, andere flessen) |
| Sync 14_55 | 110 / 112 flessen, één-op-één 99,1%, ID = Excel-id 99,1% | referentieframes |
| Sync 14_55, oude detector | 24 / 112 flessen | referentieframes |
| Sync 14_32 | 173 / 174, kleurvolgorde 100%, 172× juist ID | kleuruitlijning |
| OK/NOK val | recall 0,98 · precisie 0,96 bij drempel **0,32** | val-split |
| OK/NOK 14_32 | **5/6** NOK gevonden, 1 vals alarm op 167 | per fles |
| Tarra 14_32 | 35% gelezen, 98% daarvan binnen 0,1 kg, MAE 0,12 kg | per fles |
| Jaar 14_32 | 49% gelezen, 89% exact, 96% juiste beslissing | per fles |
| Rankingscore 14_32 | **67,3 / 100** (sync 0,99 · visueel 0,83 · tarra 0,34 · jaar 0,45) | `ranking_report.json` |
| Eindbeslissing vs operator | 5 van 15 uitgeduwde flessen gevangen, 5 extra alarmen (3,2%) | `ranking_report.json` |
| Snelheid | tracking ±0,9× realtime (3 camera's), OCR ±6 s per fles | laptop-GPU 6 GB |

## 3. Begrippen

| Begrip | Uitleg in één of twee zinnen |
|---|---|
| **Precisie / recall** | Precisie: van wat het model NOK noemt, welk deel is echt NOK. Recall: van alle echte NOK-flessen, welk deel vindt het model. |
| **F2** | `5·P·R / (4·P + R)`. Zoals F1, maar recall telt 4× zwaarder dan precisie (β = 2, β² = 4). Gekozen omdat een gemiste slechte fles erger is dan een extra controle. |
| **mAP50 / mAP50-95** | Gemiddelde precisie over alle recall-niveaus. Een box telt als juist bij IoU ≥ 0,5 (mAP50), of gemiddeld over IoU 0,5 tot 0,95 (strenger, meet hoe strak de box is). |
| **IoU** | Overlap van voorspelde en echte box: oppervlak van de doorsnede ÷ oppervlak van de unie. |
| **Domain gap** | Een model getraind op beelden van één soort (webfoto's) werkt slecht op een ander soort (camera's aan de lijn): andere belichting, hoek, achtergrond. |
| **Template matching** | Een klein beeld (de uitsnede van de lector) over een groot beeld (het videoframe) schuiven en kijken waar het het best past. Score 1 = pixelperfect. |
| **ByteTrack** | Tracker die boxen van frame tot frame koppelt op basis van positie (Kalman-filter + IoU). Gebruikt ook boxen met lage confidence, waardoor tracks minder breken. |
| **Track-ID** | Nummer dat de tracker geeft, alleen geldig binnen één video. Pas na de sync wordt het een fles-ID. |
| **Offset** | Hoeveel frames een back-camera verschoven loopt tegenover front. `camera-frame = front-frame + offset`. |
| **Purity** | Bij een kandidaat-offset: welk deel van de framekoppels hoort bij de "vaste partner"-track. Maximaal bij de juiste offset. Zie `estimate_offset_purity()` in `gass/sync.py`. |
| **Leakage** | Als het model bij het testen beelden ziet die (bijna) ook in de training zaten, bv. dezelfde fles vanuit een andere camera. De test is dan te optimistisch. |
| **Holdout** | Een opname (14_32) die we volledig weglaten uit de training, alleen om te testen. |
| **Early stopping** | Stoppen met trainen als de val-loss een aantal epochs (patience) niet meer verbetert, en het beste epoch houden. |
| **Oversampling** | Beelden van de kleine klasse (NOK) meerdere keren in de training stoppen, zodat het model ze niet negeert. |
| **Drempel** | Vanaf welke NOK-kans we "NOK" zeggen. Niet 0,5, maar 0,32: die geeft de hoogste F2 op de val-set. |
| **CLAHE** | Contrastversterking per klein vlak van het beeld, zodat donkere en lichte delen allebei leesbaar worden. |
| **Regex** | Tekstpatroon, bv. "getal met decimaal, dan KG". Gebruikt om tarra en jaar uit de ruwe OCR-tekst te halen. |
| **Needleman-Wunsch** | Algoritme om twee reeksen uit te lijnen met gaten, zoals in DNA. Wij lijnen de kleurreeks van gevonden flessen uit met de kleurkolom van de Excel. |
| **MAE** | Gemiddelde absolute fout: gemiddeld hoeveel kg de gelezen tarra ernaast zit. |
| **Dekking (coverage)** | Bij hoeveel flessen er überhaupt een antwoord is. Altijd naast de fout vermelden: "98% juist" zegt niets als je maar 1% leest. |

## 4. Verwachte vragen

### Algemeen en evaluatie

**Waarom meten jullie per fles en niet per afbeelding?**
Eén fles levert tientallen beelden op. Per beeld meten telt makkelijke flessen veel keer mee, en Primagaz neemt een beslissing per fles. Elke fles wordt één keer vergeleken met haar eigen rij in de Excel (`gass/gt_mapping.py`).

**Hoe weten jullie welke Excel-rij bij welke gevonden fles hoort?**
- Voor 14_55 via referentieframes: de bestandsnamen in de lector-zip zeggen welke fles op welk frame in beeld is (`gass/reference_frames.py`).
- Voor 14_32 via de kleurvolgorde (zwart/cyaan), uitgelijnd met Needleman-Wunsch (`align_sequences()` in `gass/evaluation.py`).

**Waarom 14_32 als test?**
Het is een volledig andere opname, waarvan niets in de training van de detector of de classifier zat. Zo meten we hoe het werkt op nieuwe beelden. 14_55 hebben we gebruikt om te trainen en af te stellen, dus scores daarop zijn te optimistisch.

**Hoe is de rankingscore opgebouwd en waarom die gewichten?**
`100 × (0,25·sync + 0,30·visueel + 0,25·tarra + 0,20·jaar)` (`gass/metrics.py`, `RANKING_WEIGHTS`).
- Visueel krijgt het meeste gewicht, omdat een gemiste schade een veiligheidsrisico is.
- Sync telt mee, omdat alles verkeerd loopt als het ID fout is.
- Jaar het minst, omdat een fout jaar met de juiste beslissing nog half telt.

De gewichten zijn **onze keuze**, niet gegeven door Primagaz. Ze zijn makkelijk aan te passen in één regel code.

**Waarom krijgt een tarra die er een beetje naast zit nog punten?**
Feedback van de lector: 5,2 kg lezen voor 5,0 is veel minder erg dan 5000. Punten = `max(0, 1 − fout / 0,5 kg)`: 5,0 geeft 1, 5,2 geeft 0,6, alles vanaf 0,5 kg ernaast geeft 0. Niet gelezen geeft ook 0.

### WS2 – Detectie

**Waarom werkte de eerste detector niet?**
Hij was getraind op webfoto's van gasflessen (Roboflow). Op de camera's van Primagaz is alles anders: hoek, belichting, achtergrond (domain gap). mAP50 viel van 0,98 op de eigen val-set naar 0,32 op Primagaz-frames.

**Hoe hebben jullie zonder handwerk boxen gemaakt?**
De uitsnedes in de lector-zip zijn letterlijke stukken van videoframes. Met template matching vinden we waar elke uitsnede in het frame zit, en dat is de box. Alle 916 uitsnedes van 14_55 werden teruggevonden, met score ≥ 0,9 (`scripts/build_primagaz_detector_dataset.py`).

**Wat als er een tweede fles half in beeld staat?**
Die heeft geen uitsnede, dus geen box, en het model zou leren dat dat achtergrond is. Daarom voegen we boxen van het oude model toe, maar alleen als ze hoog zijn (≥ 60% van de beeldhoogte) en tegen de rand staan (`is_neighbour_bottle()`).

**Is 0,99 niet verdacht hoog?**
- De test-split bevat flessen die niet in de training zaten (gesplitst per fles-id in tijdsblokken), dus het zijn niet dezelfde beelden.
- Maar wel dezelfde opname, dezelfde camera's en dezelfde belichting. De echte test is 14_32: daar vindt de sync 173 van de 174 flessen, wat toont dat de detector ook op een nieuwe opname werkt.

**Waarom YOLO26n?**
Het kleinste model van de nieuwste Ultralytics-generatie. Het moet 3 video's tegelijk aankunnen op een laptop-GPU van 6 GB. Er is maar één klasse en de flessen zijn groot in beeld, dus een groter model is niet nodig.

### WS4 – Tracking en sync

**Hoe werkt de sync, in het kort?**
1. Per camera en per frame houden we alleen de grootste box: dat is de fles die geïnspecteerd wordt.
2. We schuiven de back-camera's met hun offset op één tijdlijn.
3. Per frame maken we de combinatie van track-ID's, bv. `[3, 55, 23]`. Zie je die 6 keer na elkaar zonder ontbrekende camera, dan start een nieuwe fles. Een track-ID dat al gebruikt is, mag geen nieuwe fles starten.

Dit is de regel van de lector (`sync_strict()` in `gass/sync.py`).

**Hoe vinden jullie de offset zonder ground truth?**
We proberen elke offset tussen −120 en +120. Bij de juiste offset hoort bij elke front-track bijna altijd dezelfde back-track. Dat aandeel (purity) is maximaal bij de juiste offset (`estimate_offset_purity()`).

**Jullie offset (+46/+16) is niet die van de lector (+35/+5). Wie heeft gelijk?**
- Beide werken: een fles is ongeveer 100 frames in beeld, dus 11 frames verschil valt binnen de marge.
- Met de lector-offsets is het ID op 14_55 100% juist, met de onze 99,1% (één fles verschil).
- Het verschil is constant (11 frames bij beide camera's). We hebben gecontroleerd dat het niet aan het zoeken in de video ligt. De exacte oorzaak kennen we niet; het kan zijn dat de lector naar een ander moment in de doorgang van de fles kijkt.

Zeg dit eerlijk als het gevraagd wordt.

**Waarom de strikte en niet de robuuste variant?**
De robuuste variant laat een ontbrekende camera toe, maar voegde op onze data soms twee flessen samen, waardoor alle volgende ID's verschoven. Met een goede detector ontbreekt er zelden een camera, dus de strikte regel is beter.

**Wat zegt de kleurcheck?**
Of het aantal en de volgorde van de flessen kloppen. Elke gevonden fles krijgt zwart of cyaan (`gass/colour.py`), en die reeks wordt uitgelijnd met de Excel. Een gemiste of dubbele fles geeft een gat in de uitlijning. Op 14_32 klopte de volgorde 100%.

### WS5 – OK/NOK

**Welke data, en hoe voorkomen jullie leakage?**
- Bronnen: de lector-zip (1.005 OK, 226 NOK) plus onze eigen crops van 14_55, gelabeld via de Excel.
- Alle beelden van dezelfde fles (zelfde opname + id, alle camera's) gaan samen naar train of naar val (`group_of()`).
- 14_32 is volledig weggelaten.

**Waarom drempel 0,32 en niet 0,5?**
We kiezen de drempel die op de val-set de hoogste F2 geeft. Een lagere drempel vangt meer NOK, ten koste van wat meer valse alarmen. Dat is wat we willen.

**Waarom de 3 hoogste kansen en niet het gemiddelde?**
Schade of vuil is vaak maar vanuit één camera te zien. Een gemiddelde over 15 crops verdunt dat. Gemeten op 14_32: gemiddelde van alles vindt 1/6 NOK, top-3 vindt 5/6.

**Is 5/6 betrouwbaar?**
Nee, het zijn maar 6 flessen. Het 95%-betrouwbaarheidsinterval (Wilson) loopt van ongeveer 44% tot 97%. Meer NOK-flessen in de test zouden nodig zijn om iets zekers te zeggen.

**Waarom vinden jullie maar 3 van de 15 flessen die de operator uitduwde?**
De classifier is getraind op het visuele oordeel (`classification`), niet op de beslissing van de operator (`Primagaz_status`). Operators duwen ook flessen uit om redenen die niet zichtbaar zijn, zoals een onleesbare tarra of een vervallen jaar. Daarvoor dient WS6. Samen vangen we er 5 van de 15.

**Is er overfitting?**
Ja, na epoch 4 steeg de val-loss terwijl de trainingsloss bleef dalen. Early stopping hield epoch 4 (patience 8, gestopt na 12).

### WS6 – OCR

**Waarom is de dekking zo laag?**
Op ongeveer de helft van de flessen is de print in geen van de 9 crops leesbaar: bewegingsonscherpte, de print staat op een gebogen schouder, of hij staat aan de kant die geen camera goed ziet. Wat we wél lezen, is bijna altijd juist (98% binnen 0,1 kg).

**Hoe weet je dat een getal de tarra is en niet iets anders?**
- De regex zoekt een getal met decimaal vlak voor `KG`, tussen 8 en 22 kg.
- Lezingen in het inkjetpatroon (`10.8 KG-2034`) wegen zwaarder.
- De sticker vermeldt het nettogewicht (12,5 of 10,5 kg). Een zwakke lezing van precies die waarden negeren we (`STICKER_NET_WEIGHTS` in `gass/ocr_parse.py`).

**Waarom stemmen?**
Elke fles heeft 9 crops, die elk iets anders kunnen lezen. De waarde met de meeste (gewogen) steun wint. Zo telt één toevallige foute lezing weinig.

**Welke fouten maakt de OCR?**
Jaar: vooral 9 ↔ 3 (2039 gelezen als 2033). Tarra: op 14_32 nog één grote fout (> 1 kg).

**Is het snel genoeg?**
Nee, nog niet. Ongeveer 6 s per fles, terwijl er elke 4 s een fles passeert. Opties: minder crops, alleen de camera die de print ziet, een snellere GPU, of een kleiner OCR-model.

### NDA en GitHub

**Wat staat er op GitHub?**
Alleen code, trainingscurves en samengevatte rapporten. Video's, frames, crops, Excel-waarden per fles en de modellen die op Primagaz-data getraind zijn, worden door de `.gitignore` buitengehouden. De modellen delen we binnen het team, niet publiek.

## 5. Zwakke punten (zelf benoemen)

1. **OCR-dekking** van 35–49%: het grootste verlies in de rankingscore.
2. **Weinig NOK-flessen** in de test (6): het recall-cijfer is onzeker.
3. **Sync op 14_32 is gemeten via kleur**, niet via echte referentieframes. Kleur kan een verschuiving tussen twee flessen van dezelfde kleur missen. Dat cijfer is dus een bovengrens (zo staat het ook in `ranking_report.json`).
4. **Alles getraind op één opname** (14_55). Andere belichting of een ander tijdstip is niet getest, buiten 14_32.
5. **De top-camera** wordt niet gebruikt.
6. **Niet realtime:** de OCR is te traag.
7. **De gewichten van de rankingscore** en de regel `jaar ≥ 2025` zijn onze aannames en nog niet bevestigd door Primagaz of de lector.

## 6. Wat we als volgende stap zouden doen

- Per fles eerst bepalen welke camera de print in beeld heeft, en alleen daar OCR op draaien: sneller en hogere dekking.
- Een OCR-model fine-tunen op uitsnedes van de inkjetprint.
- De top-camera toevoegen, voor schade en vuil bovenaan.
- Meer opnames labelen voor de NOK-test.
- Aan Primagaz vragen wat erger is: een gemiste NOK of een vals alarm. Dan kunnen we de drempel en de gewichten daarop afstemmen.

## 7. Live demo

Een volledige opname door de pijplijn sturen duurt te lang voor een verdediging. Zorg dat stap 1–6 vooraf gedraaid zijn, en toon dan:

1. de demovideo, met de 3 camera's naast elkaar, de ID's en de resultaten:
   ```bash
   py scripts/render_demo_video.py --recording 14_32 --start-s 100 --duration-s 60
   ```
2. de rapporten in `results/pipeline/14_32/` (`sync_report.json`, `classification_report.json`, `ocr_report.json`, `ranking_report.json`);
3. eventueel één stap live, bv. de score herberekenen (duurt een paar seconden):
   ```bash
   py scripts/score_recording.py --recording 14_32
   ```

Zorg dat de video's en modellen lokaal staan op de laptop waarmee je de demo geeft.
