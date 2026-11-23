# Auswertung: Knapsack mit planted effects (Stand 23.11.2026)

Grundlage: [knapsack_including_planted_effects.ipynb](../../notebooks/prestudy/knapsack_including_planted_effects.ipynb) und [knapsack_including_planted_effects.csv](../../results/prestudy/knapsack_including_planted_effects.csv). Setting: 2-dimensionaler 0-1-Knapsack mit 32 Items, eigener Generator (Abschnitt 2), p = 5 Features, deg ∈ {1, 2, 4, 6}, noise ∈ {0, 0.5}, n ∈ {100, 1000}, m = 5 Refits, n_test = 1000, K = 10 Permutationen. Dazu kommt eine Zusatzprüfung der gelernten Gewichte (Abschnitt 9). Zahlen ohne weitere Angabe gelten für n = 1000 und noise = 0.5. „Wahre Kostenfunktion“ meint E[c | x] des Generators (im Code `oracle`).

Bezeichnungen: **H1** = Die Rangfolgen der Features nach ΔMSE und nach ΔRegret weichen für dasselbe Modell voneinander ab. **H2** = Das SPO+-Modell stützt sich stärker auf entscheidungskritische Features als das MSE-Modell.

## Lesehilfe

- **PFI-Wert eines Features** = um wie viel der Fehler auf der Testmenge steigt, wenn man die Spalte dieses Features zufällig mischt (Mittel über K = 10 Mischungen). Groß heißt: Ohne dieses Feature wird das Modell schlechter, also stützt es sich darauf. ≈ 0 heißt: Das Modell braucht es nicht. Leicht negative Werte sind Zufallsschwankung.
- **ΔMSE** misst das am Vorhersagefehler (Wichtigkeit für die *Vorhersage*), **ΔRegret** am normierten Regret (Wichtigkeit für die *Entscheidung*).
- **Zeile „wahr“** = dieselbe Messung an der wahren Kostenfunktion E[c | x]. Hier ist bekannt, welche Features kritisch sind, die Zeile zeigt also das richtige Ergebnis.
- **Anteile** = PFI-Wert eines Features bzw. einer Gruppe geteilt durch die Summe über alle Features (negative Werte vorher auf 0 gesetzt). Damit lassen sich Modelle mit unterschiedlichem Niveau vergleichen.
- **Gepaarte Differenz** = PFI(SPO+) − PFI(MSE), je Refit gebildet (beide Modelle auf denselben Daten), dann gemittelt. Positiv heißt: Unter SPO+ ist das Feature wichtiger. `*` heißt: Das 95-%-Intervall über die 5 Refits schließt 0 aus.

## 1 Kurzfazit

1. **Der Generator tut, was er soll.** Die drei Kerngruppen werden ähnlich oft gewählt, „zu schwer“ nie, „gratis“ immer, und die häufigste Lösung kommt höchstens in 27 % der Instanzen vor. Bei der wahren Kostenfunktion haben nur x1 und x2 einen ΔRegret.
2. **H2 bestätigt sich deutlich, viel klarer als beim Shortest Path.** Der Anteil der kritischen Features am ΔMSE liegt unter SPO+ bei 0.80–0.87, beim MSE-Modell und bei der wahren Kostenfunktion bei 0.67–0.68. Der Unterschied wächst mit deg.
3. **Der Ablenker „zu schwer“ (x3) erklärt den größten Teil.** Sein ΔMSE-Anteil fällt unter SPO+ von 0.16 auf 0.5–2 %. Die Zusatzprüfung zeigt den Grund: Die SPO+-Gewichte dieser Items bleiben exakt auf ihrer Initialisierung (Korrelation 1.00), weil diese Items nie zulässig sind und SPO+ für sie keinen Gradienten erhält.
4. **Der Ablenker „gratis“ (x4) ist der einzige Fall, in dem ein Ablenker entscheidungsrelevant wird, und SPO+ verhindert das.** Das MSE-Modell sagt bei deg 6 für 10.6 % der Gratis-Items einen negativen Wert vorher und lässt sie dann weg. x4 trägt dadurch 7.6 % seines ΔRegret. Beim SPO+-Modell sind es 1.1 % negative Vorhersagen und 0.7 % des ΔRegret.
5. **Die Unterschiede zwischen den Losses sind featurespezifisch.** Anders als beim Shortest Path ist der ΔMSE unter SPO+ bei x1 und x2 höher, bei x3 und x4 niedriger, bei x5 gleich.
6. **Die Abhängigkeit von der XAI-Methode ist schwächer als beim Shortest Path, aber vorhanden.** Eine koeffizientenbasierte Wichtigkeit gibt dem Rauschfeature x5 unter SPO+ 5–8 %, unter MSE 2–3 %. Beim Shortest Path waren es 21 % gegenüber 3 %.

## 2 Generator

Werte wie bei PyEPO, c = 5 · [(Bx/√p + 3)^deg + 1] / 3.5^deg · ε, aber ohne Aufrunden und mit bewusst gesetztem B und bewusst gesetzten Gewichten:

| Gruppe | Items | Gewichte | Wert getrieben von | Rolle |
|---|---|---|---|---|
| kritisch x1 | 8 | wie PyEPO aus {3.00, …, 7.99} | x1 | konkurriert um die Kapazität |
| kritisch x2 | 8 | dieselben wie x1 | x2 | konkurriert um die Kapazität |
| konstant | 8 | dieselben wie x1 | kein Feature | Referenz |
| zu schwer | 4 | größer als die Kapazität | x3 | nie zulässig, für jedes Modell irrelevant |
| gratis | 4 | 0 | x4 | bei positivem Wert immer gewählt |
| – | – | – | x5 | Rauschfeature, wirkt auf kein Item |

Die Kapazität beträgt ein Drittel des Gesamtgewichts der 24 Kern-Items je Dimension. Bei x = 0 sind alle drei Kerngruppen gleich wertvoll, und die Kapazität erzwingt eine Auswahl zwischen ihnen. Die Skalenbereinigung ist nur s·ĉ.

**„gratis“ ist ein bedingter Ablenker:** Unter den wahren Werten (alle > 0) ist er irrelevant, für ein Modell nur, solange es positive Werte vorhersagt.

### Generator-Check

Anteil gewählter Items je Gruppe unter den wahren Werten (2000 Instanzen):

| deg | noise | kritisch x1 | kritisch x2 | konstant | zu schwer | gratis | häufigste Lösung |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.37 | 0.37 | 0.38 | 0 | 1 | 12 % |
| 1 | 0.5 | 0.36 | 0.36 | 0.36 | 0 | 1 | 0.3 % |
| 6 | 0 | 0.39 | 0.39 | 0.27 | 0 | 1 | 27 % |
| 6 | 0.5 | 0.37 | 0.37 | 0.29 | 0 | 1 | 13 % |

- Keine Lösung dominiert. Das ist wichtig, weil sonst ein Modell die Features ignorieren und immer dieselbe Auswahl treffen könnte.
- Mit steigendem deg wird die konstante Gruppe seltener gewählt. Die konvexe Transformation macht hohe Werte bei x1/x2 so groß, dass die konstante Gruppe öfter verliert.

### PFI der wahren Kostenfunktion

| deg | Maß | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | ΔMSE | 0.21 | 0.20 | 0.10 | 0.10 | 0 |
| 1 | ΔRegret | 0.015 | 0.015 | 0 | 0 | 0 |
| 6 | ΔMSE | 2.81 | 2.90 | 1.37 | 1.41 | 0 |
| 6 | ΔRegret | 0.136 | 0.135 | 0 | 0 | 0 |

Die Grundwahrheit ist sauber umgesetzt. Die Ablenker sind für die Vorhersage etwa halb so wichtig wie x1/x2, weil sie nur 4 statt 8 Items treiben, für die Entscheidung aber gar nicht.

## 3 Modellgüte

Normierter Regret, MSE-Modell / SPO+-Modell (wahre Kostenfunktion in Klammern):

| deg | noise = 0 | noise = 0.5 | noise = 0.5, n = 100 |
|---|---|---|---|
| 1 | 0.0000 / 0.0000 (0) | 0.095 / 0.098 (0.093) | 0.101 / 0.106 |
| 2 | 0.0003 / 0.0000 (0) | 0.074 / 0.075 (0.073) | 0.077 / 0.083 |
| 4 | 0.0040 / 0.0002 (0) | 0.047 / 0.043 (0.042) | 0.049 / 0.049 |
| 6 | 0.0098 / 0.0001 (0) | 0.036 / 0.026 (0.025) | 0.038 / 0.031 |

- Ohne Rauschen ist SPO+ ab deg 2 um ein bis zwei Größenordnungen besser. Bei deg = 1 ist das MSE-Modell exakt (Kontrollfall, kein Aufrunden).
- Mit Rauschen liegt SPO+ bei deg 6 praktisch auf der Untergrenze (0.026 gegenüber 0.025), das MSE-Modell 0.011 darüber.
- Der Regret sinkt mit deg, anders als bei den PyEPO-Generatoren. Vermutlich unterscheiden sich bei hohem deg die Werte der Kerngruppen stärker, sodass das multiplikative Rauschen seltener die Auswahl kippt (nicht geprüft).
- MSE skalenbereinigt, MSE-Modell / SPO+-Modell: 2.8 / 7.8 (deg 1) bis 1.7 / 4.3 (deg 6).

## 4 PFI je Feature

### 4.1 ΔMSE

| deg | Modell | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | wahr | 0.21 | 0.20 | 0.10 | 0.10 | 0 |
| 1 | MSE | 0.21 | 0.20 | 0.10 | 0.10 | 0.000 |
| 1 | SPO+ roh | 0.18 | 0.16 | −0.001 | 0.08 | 0.001 |
| 1 | SPO+ skalenbereinigt | 0.28 | 0.25 | −0.003 | 0.12 | 0.002 |
| 6 | wahr | 2.81 | 2.90 | 1.37 | 1.41 | 0 |
| 6 | MSE | 2.38 | 2.36 | 1.08 | 1.15 | 0.001 |
| 6 | SPO+ roh | 2.77 | 2.72 | −0.006 | 0.76 | −0.002 |
| 6 | SPO+ skalenbereinigt | 2.73 | 2.69 | −0.005 | 0.75 | −0.002 |

(Beim MSE-Modell sind roh und skalenbereinigt praktisch gleich.)

- **MSE-Modell:** Bei deg 1 bildet es die Wahrheit exakt ab. Bei deg 6 unterschätzt es alle Features um 15–21 %, die Verhältnisse bleiben aber erhalten.
- **SPO+-Modell:**
  - Bei x1 und x2 liegt es bei deg 6 näher an der Wahrheit als das MSE-Modell (2.77 gegenüber 2.38, wahr 2.81).
  - Bei x3 ist der ΔMSE in allen Einstellungen 0. Das Modell hat das x3-Signal nicht gelernt (Abschnitt 9).
  - Bei x4 lernt es nur einen Teil (0.76 statt 1.41), genug, um die Gratis-Items positiv zu halten.
- **Skalenbereinigung:** Bei deg ≤ 4 sind die Vorhersagen des SPO+-Modells zu klein skaliert (s > 1). Die Bereinigung hebt seinen ΔMSE dann über den der Wahrheit (deg 1, x1: 0.28 gegenüber 0.21). Für die Anteile spielt das keine Rolle.

### 4.2 ΔRegret

| deg | Modell | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | wahr | 0.015 | 0.015 | 0 | 0 | 0 |
| 1 | MSE | 0.014 | 0.013 | 0.000 | 0.000 | 0.000 |
| 1 | SPO+ | 0.013 | 0.012 | 0.000 | 0.000 | 0.000 |
| 4 | wahr | 0.095 | 0.095 | 0 | 0 | 0 |
| 4 | MSE | 0.091 | 0.091 | 0.000 | **0.007** | 0.000 |
| 4 | SPO+ | 0.095 | 0.093 | 0.000 | 0.002 | 0.000 |
| 6 | wahr | 0.136 | 0.135 | 0 | 0 | 0 |
| 6 | MSE | 0.127 | 0.126 | 0.000 | **0.021** | 0.000 |
| 6 | SPO+ | 0.137 | 0.135 | 0.000 | 0.002 | 0.000 |

- Beide Modelle erkennen x1 und x2 als entscheidungsrelevant, SPO+ bei hohem deg etwas genauer.
- Beim MSE-Modell wird x4 ab deg 4 entscheidungsrelevant, obwohl er es bei der wahren Kostenfunktion nicht ist. Ursache sind negative Vorhersagen für Gratis-Items (Abschnitt 9): Das Modell lässt ein Item weg, das in Wahrheit kostenlos Wert bringt, und das kostet Regret. Permutiert man x4, ändern sich diese Fehlentscheidungen, daher der ΔRegret.

## 5 H2: Anteil der Wichtigkeit je Featuregruppe

Anteile am ΔMSE (skalenbereinigt), MSE-Modell / SPO+-Modell / wahre Kostenfunktion:

| deg | kritisch (x1, x2) | zu schwer (x3) | gratis (x4) |
|---|---|---|---|
| 1 | .676 / .799 / .673 | .161 / .019 / .159 | .162 / .179 / .168 |
| 2 | .677 / .822 / .672 | .158 / .011 / .160 | .165 / .165 / .167 |
| 4 | .678 / .862 / .671 | .156 / .007 / .162 | .165 / .132 / .167 |
| 6 | .680 / .873 / .672 | .155 / .005 / .162 | .165 / .122 / .166 |

Anteile am ΔRegret, MSE-Modell / SPO+-Modell (wahre Kostenfunktion: kritisch 1.00, sonst 0):

| deg | kritisch (x1, x2) | zu schwer (x3) | gratis (x4) |
|---|---|---|---|
| 1 | .988 / .985 | .003 / .004 | .006 / .003 |
| 2 | .994 / 1.00 | .001 / .000 | .004 / .000 |
| 4 | .963 / .990 | .000 / .000 | .037 / .009 |
| 6 | .924 / .992 | .000 / .000 | .076 / .007 |

- **MSE-Modell:** Beim ΔMSE gibt es die Struktur der Daten exakt wieder, unabhängig von deg. Beim ΔRegret zieht es mit steigendem deg den Ablenker x4 in die Entscheidung hinein.
- **SPO+-Modell:** Beim ΔMSE stützt es sich deutlich stärker auf die kritischen Features als die Daten selbst. Das liegt vor allem an x3 („zu schwer“), ab deg 4 auch an x4. Beim ΔRegret bleibt es in allen Einstellungen bei ≥ 98 % auf x1/x2.
- **Bei n = 100** sind die Anteile fast gleich (kritisch beim SPO+-Modell 0.80–0.87, beim MSE-Modell 0.67). Die Datenmenge spielt hier kaum eine Rolle.
- **Einordnung:** Ein Teil des H2-Effekts ist mechanisch. Für x3 kann SPO+ prinzipiell nichts lernen, und das zeigt sich bei jedem deg. Der Anteil, der mit deg wächst, stammt aus x4. Er ist der eigentlich interessante Teil, weil x4 nur bedingt irrelevant ist und SPO+ aktiv lernt, die Gratis-Items positiv zu halten.

## 6 H1: Kendall's τ

τ zwischen den Rangfolgen nach ΔMSE und ΔRegret: wahre Kostenfunktion 0.32–0.95, MSE-Modell 0.56–0.88, SPO+-Modell 0.52–0.96.

Wie beim Shortest Path mit planted effects ist τ als Maß ungeeignet. x3, x4 und x5 haben bei der wahren Kostenfunktion alle ΔRegret = 0, ihre Rangfolge untereinander ist beim Modell Zufall und drückt τ. Aussagekräftiger sind die Gruppenanteile in Abschnitt 5: ΔRegret liegt zu ≥ 92 % auf x1/x2, ΔMSE nur zu 67–87 %. H1 gilt also, wie vom Generator vorgesehen.

## 7 Gepaarte Differenz SPO+ − MSE

| deg | ΔMSE skalenbereinigt | ΔRegret |
|---|---|---|
| 1 | x1, x2 positiv * (+0.07, +0.05); x3 negativ * (−0.10); x4 +0.02 *; x5 ≈ 0 | x1, x2 leicht negativ * (−0.001) |
| 2 | x1, x2 positiv * (+0.17, +0.14); x3 negativ * (−0.28); x4, x5 ≈ 0 | ≈ 0 (einzelne Werte um −0.001 signifikant) |
| 4 | x1, x2 positiv * (+0.39, +0.37); x3 negativ * (−0.69); x4 negativ * (−0.15); x5 ≈ 0 | x1, x2 positiv * (+0.004, +0.003); x4 negativ * (−0.005) |
| 6 | x1, x2 positiv * (+0.35, +0.33); x3 negativ * (−1.09); x4 negativ * (−0.39); x5 ≈ 0 | x1, x2 positiv * (+0.009, +0.010); x4 negativ * (−0.019) |

Hier zeigen die Differenzen Verschiebungen *zwischen* Features und nicht nur ein Niveau: Unter SPO+ steigt die Wichtigkeit der kritischen Features, die der Ablenker sinkt, die des Rauschens bleibt bei 0. Das ist das Muster, das H2 erwartet.

## 8 Vergleich mit dem Shortest Path

| | Shortest Path, planted | Knapsack, planted |
|---|---|---|
| Kritischer ΔMSE-Anteil, deg 6, MSE / SPO+ / wahr | .87 / .90 / .93 (5×5) | .68 / .87 / .67 |
| Ablenker werden je entscheidungsrelevant? | nein (Grundkosten 2, nie gewählt) | ja, x4 beim MSE-Modell ab deg 4 |
| Entscheidungsneutrale Richtungen | 24 von 40 (Knotenpotenziale) | Skala und die 4 „zu schweren“ Items |
| Gepaarte ΔMSE-Differenz | Niveaueffekt, alle Features negativ | featurespezifisch |

Der Knapsack zeigt H2 klarer aus zwei Gründen. Erstens enthält er einen Ablenker, der Entscheidungen kippen kann („gratis“). Zweitens verzerrt kein großer neutraler Unterraum den ΔMSE des SPO+-Modells. Für den Shortest Path folgt daraus, Ablenker-Kanten mit niedrigeren Grundkosten zu ergänzen, die knapp verfehlen.

## 9 Zusatzprüfung: gelernte Gewichte

Setting: deg ∈ {2, 6}, noise = 0.5, n = 1000, 3 Refits, dieselben Daten, Seeds und Trainingseinstellungen wie im Hauptlauf. Ausgewertet wurden die Gewichte des linearen Modells (32 Items × 5 Features), ihre Startwerte und die Vorhersagen auf der Testmenge. Das Skript ist nicht Teil des Repos.

| | MSE, deg 2 | SPO+, deg 2 | MSE, deg 6 | SPO+, deg 6 |
|---|---|---|---|---|
| corr(Gewichte, Startwerte), Items „zu schwer“ | −0.11 | **1.00** | −0.10 | **1.00** |
| corr(Gewichte, Startwerte), kritische Items | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| mittleres \|w\| der „zu schweren“ Items auf x3 | 1.09 | 0.26 | 2.13 | 0.26 |
| mittleres \|w\| auf x5 (Ziel 0) | 0.032 | 0.080 | 0.028 | 0.106 |
| corr(Gewichte, Startwerte), Spalte x5 | −0.03 | 0.38 | −0.01 | 0.24 |
| Anteil negativer Vorhersagen, Gratis-Items | 0.1 % | 0 % | **10.6 %** | 1.1 % |
| Anteil negativer Vorhersagen, Items „zu schwer“ | 0 % | 56 % | 8 % | 56 % |

Koeffizientenbasierte Wichtigkeit, Anteil von Σ\|w_j\| über alle Items. Bei unabhängigen, standardnormalen Features ist das proportional zum mittleren \|SHAP\|:

| | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| MSE, deg 6 | .326 | .317 | .167 | .173 | .016 |
| SPO+, deg 6 | .407 | .392 | .033 | .118 | .049 |
| MSE, deg 2 | .316 | .314 | .167 | .169 | .034 |
| SPO+, deg 2 | .359 | .332 | .067 | .163 | .079 |

**Interpretation:**

- **„zu schwer“:** Die Korrelation von 1.00 heißt, dass sich diese Gewichte im Training überhaupt nicht bewegen. Kein Wert dieser Items kann eine Entscheidung ändern, daher ist der SPO+-Gradient exakt 0. Die Vorhersagen für diese Items sind reines Initialisierungsrauschen (56 % negativ). PFI zeigt das korrekt als „x3 unwichtig“, denn das Modell nutzt das x3-Signal nicht. Die Koeffizienten-Wichtigkeit zeigt x3 ebenfalls klein (3–7 %), weil die Startwerte (≈ 0.26) klein gegenüber den gelernten Gewichten auf x1/x2 sind.
- **Rauschfeature x5:** Unter SPO+ behält es 3- bis 4-mal so große Gewichte wie unter MSE, teilweise noch mit den Startwerten korreliert. Der Grund: MSE bestraft *jedes* Gewicht auf x5, weil es die Vorhersage verrauscht. SPO+ bestraft es nur, soweit es Entscheidungen kippt. Kleine zufällige Gewichte überleben daher. PFI sieht davon nichts (ΔMSE und ΔRegret ≈ 0), die Koeffizienten-Wichtigkeit schon (5–8 % gegenüber 2–3 %).
- **„gratis“:** Die Vermutung aus Abschnitt 4.2 ist bestätigt. Das fehlspezifizierte MSE-Modell passt eine Gerade an die konvexe Wertfunktion an und landet für niedrige x4 im Negativen (10.6 % bei deg 6). SPO+ lernt ein kleineres x4-Gewicht (1.47 gegenüber 2.19) und hält die Vorhersagen fast immer positiv.
- **Vergleich zum Shortest Path:** Dort hatte x5 unter SPO+ einen Koeffizienten-Anteil von 21 %, so viel wie x1 ([Auswertung_shortest_path_planted_effects.md](Auswertung_shortest_path_planted_effects.md), Abschnitt 6). Beim Knapsack ist der Effekt kleiner, weil es nur wenige neutrale Richtungen gibt. Er zeigt aber in dieselbe Richtung: Die Wahl der XAI-Methode entscheidet mit darüber, wie sauber das SPO+-Modell aussieht.

**Einschränkungen:** nur zwei Settings mit je 3 Refits. Bei Null-Initialisierung blieben die „zu schweren“ Gewichte auf 0, und die Koeffizienten-Wichtigkeit von x3 und x5 würde sinken.

## 10 Folgerungen

1. **H2 über Gruppenanteile messen**, getrennt nach Ablenker-Typ. Ablenker, die nie relevant werden können („zu schwer“), und solche, die durch Fehlvorhersagen relevant werden („gratis“), zeigen verschiedene Mechanismen.
2. **Den mechanischen Teil von H2 kennzeichnen.** Der Effekt bei x3 folgt aus dem fehlenden Gradienten und gilt unabhängig von deg. Der mit deg wachsende Teil bei x4 ist der inhaltliche Befund.
3. **Eine zweite XAI-Methode neben PFI aufnehmen** (Koeffizienten bzw. SHAP) und die Initialisierung als Robustheitsprüfung variieren (Null-Init, mehrere Init-Seeds).
4. **Für den Shortest Path Ablenker ergänzen, die knapp verfehlen**, nach dem Vorbild von „gratis“.
5. **15 Refits und Holm-Korrektur** für die Hauptstudie. Die Effekte sind hier so groß, dass sie auch dann bestehen bleiben dürften.
