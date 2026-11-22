# Auswertung: Vorstudie mit planted effects (Stand 18.11.2026)

Grundlage: [shortest_path_including_planted_effects.ipynb](../../notebooks/prestudy/shortest_path_including_planted_effects.ipynb) und [shortest_path_including_planted_effects.csv](../../results/prestudy/shortest_path_including_planted_effects.csv) (Gitter 5×5 und 10×10, deg ∈ {1, 2, 4, 6}, noise ∈ {0, 0.5}, n ∈ {100, 1000}, m = 5 Refits). Dazu kommen zwei Zusatzprüfungen außerhalb des Notebooks (Abschnitt 6). Zahlen ohne weitere Angabe gelten für n = 1000 und noise = 0.5.

## 1 Kurzfazit

1. **Der Generator tut, was er soll.** Kein Pfad ist häufiger als 37 % optimal, Ablenker-Kanten werden nie genutzt, und die wahre Kostenfunktion hat ΔRegret nur bei x1 und x2.
2. **ΔRegret hängt kaum vom Loss ab.** Bei beiden Modellen liegen ≥ 98 % der ΔRegret-Wichtigkeit auf x1 und x2, nahe am Wert der wahren Kostenfunktion.
3. **ΔMSE verschiebt sich unter SPO+ leicht weg von den Ablenkern.** Bei deg = 1 gibt es keinen Effekt, bei deg 4 und 6 sind es 3 bis 15 Prozentpunkte. Das MSE-Modell bildet die Anteile der wahren Kostenfunktion fast exakt ab. H2 bestätigt sich also nur schwach und nur bei Fehlspezifikation.
4. **Kendall's τ eignet sich hier nicht als Maß für H1.** Schon die wahre Kostenfunktion liefert τ = 0.84 (5×5) bzw. −0.12 (10×10, deg ≤ 2). τ misst damit vor allem den Generator und die zufällige Rangfolge von x3 bis x5 beim ΔRegret.
5. **Wichtigster Nebenbefund:** Die SPO+-Gewichte bleiben weitgehend auf ihrer Initialisierung, auch beim reinen Rauschfeature x5. PFI sieht das nicht. Eine koeffizienten- oder SHAP-basierte Erklärung würde x5 unter SPO+ aber etwa so wichtig einstufen wie x1 (Abschnitt 6).

## 2 Generator-Check

- **Pfadanteile:** zwischen 7 % (mitte, deg = 6) und 37 % (links-unten, deg = 6), also im Zielbereich von AP 5 (≤ ca. 70 %). Bei noise = 0.5 entstehen zusätzlich Mischpfade („andere“: bis 20 % bei deg = 1, unter 5 % ab deg = 4).
- **Treppe:** Bei noise = 0 ist sie in jedem deg zu 24,9 % optimal, weil ihre Kosten konstant sind. Sie gewinnt, wenn x1 und x2 beide eher positiv sind.
- **Wahre Kostenfunktion:** ΔRegret bei x3 bis x5 ist exakt 0, ΔMSE bei x5 ist exakt 0. Die geplante Struktur ist also korrekt umgesetzt.
- **Asymmetrie x1/x2:** x2 treibt 9 Kanten (Summe B = 12), x1 nur 8 (Summe B = 11). Im 10×10-Gitter sind es 22/30 gegenüber 20/28. Der Grund: „mitte“ teilt Kanten mit beiden Eckpfaden, aber nicht gleich viele. Daher ist ΔRegret bei x2 durchgehend größer als bei x1 (wahre Kostenfunktion, 5×5, deg = 6: 0.40 vs. 0.29). Das ist harmlos, muss aber erwähnt oder symmetrisiert werden.
- **Asymmetrie x3/x4 (5×5):** Die Bernoulli-Zuordnung ergibt 8 vs. 12 Ablenker-Kanten. Deshalb ist ΔMSE bei x4 größer als bei x3.

## 3 Modellgüte

Normierter Regret, MSE-Modell / SPO+-Modell:

| deg | 5×5 | 10×10 |
|---|---|---|
| 1 | 0.064 / 0.064 | 0.031 / 0.031 |
| 2 | 0.038 / 0.038 | 0.017 / 0.015 |
| 4 | 0.046 / 0.029 | 0.030 / 0.013 |
| 6 | 0.085 / 0.026 | 0.068 / 0.015 |

- SPO+ ist bei deg ≥ 4 deutlich besser, bei deg ≤ 2 gleichauf. Das passt zur Literatur: SPO+ lohnt sich bei Fehlspezifikation.
- Bei noise = 0 und deg = 1 ist das lineare Modell korrekt spezifiziert. Das MSE-Modell erreicht dort MSE ≈ 1e-11 und Regret 0. Dieser Fall ist trivial.
- Beim MSE ist SPO+ erwartungsgemäß viel schlechter: MSE_adj 0.5–2.4 gegenüber 0.07–1.0. Ein Grund sind die Biases. Auf den Ablenker-Kanten sagt SPO+ im Mittel 0.7 voraus, die wahren Kosten liegen bei ca. 2.8. Das ist für die Entscheidung irrelevant, solange die Kanten teuer genug bleiben.

## 4 PFI und H2: Anteil der Wichtigkeit je Featuregruppe

Anteil der Ablenker x3 und x4 am ΔMSE_adj (MSE-Modell / SPO+-Modell / wahre Kostenfunktion):

| deg | 5×5 | 10×10 |
|---|---|---|
| 1 | .358 / .373 / .357 | .565 / .513 / .566 |
| 2 | .356 / .347 / .344 | .566 / .464 / .554 |
| 4 | .260 / .200 / .206 | .447 / .298 / .373 |
| 6 | .129 / .098 / .073 | .247 / .160 / .148 |

- **MSE-Modell:** Es gibt die Wichtigkeitsstruktur der Daten treu wieder. Bei deg = 6 überschätzt es die Ablenker, weil es den konvexen Effekt von x1 und x2 linear nicht abbilden kann: ΔMSE für x1 beträgt 0.75 statt 1.98.
- **SPO+-Modell:** Es hat durchgehend einen kleineren Ablenker-Anteil (Ausnahme: 5×5, deg = 1). Bei hohem deg liegt es damit näher an der wahren Kostenfunktion als das MSE-Modell.
- **Warum geht der Ablenker-Anteil unter SPO+ nicht gegen 0?** Das ist eine Hypothese, nicht geprüft. Der SPO+-Subgradient ist 2(w* − w̃) mit w̃ = argmin (2ĉ − c)ᵀw. Eine Ablenker-Kante wird in diesem Hilfsproblem attraktiv, wenn c_e im Verhältnis zu ĉ_e hoch ist. Dann wird ĉ_e genau in den Beobachtungen mit hohem c_e nach oben gedrückt. SPO+ lernt also einen Teil des x3/x4-Signals, nämlich „halte die Kante über der Schwelle“.
- **ΔRegret:** Der Anteil von x1 und x2 liegt bei beiden Modellen bei 98–100 %. Hier gibt es nichts zu unterscheiden.

## 5 H1 und die gepaarten Differenzen

**Kendall's τ (Abschnitt 7 im Notebook):**

- ΔRegret ist bei x3, x4 und x5 im Erwartungswert 0. Bei den Modellen ist die Rangfolge dieser drei Features daher reines Rauschen. Es betrifft 3 der 10 Featurepaare, und τ sinkt allein dadurch.
- Bei 10×10 und deg ≤ 2 liefert schon die wahre Kostenfunktion τ = −0.12, weil die Ablenker dort beim ΔMSE vor x1 und x2 liegen. Gewollt ist das so. Ein niedriges τ ist hier also kein Modellbefund.
- Das SPO+-Modell hat in genau diesem Fall ein höheres τ (0.36–0.72 gegenüber −0.20–0.04). Das passt zum kleineren Ablenker-Anteil aus Abschnitt 4. Bei 5×5 liegt τ dagegen beim MSE-Modell höher.

**Gepaarte Differenz SPO+ − MSE (Abschnitt 8 im Notebook):**

- **ΔMSE_adj:** Alle Werte bei x1 bis x4 sind signifikant negativ. Das ist ein reiner Niveaueffekt: Unter SPO+ ist ΔMSE bei allen Features 3- bis 5-mal kleiner (5×5, deg = 2, x1: 0.012 gegenüber 0.041), weil SPO+ den Kostenvektor schlechter abbildet. Featurespezifisch ist daran nichts. Vergleiche über die Losses ergeben nur auf normierten Werten Sinn, also mit Anteilen wie in Abschnitt 4.
- **ΔRegret:** Bei deg ≤ 2 gibt es keinen Unterschied. Bei deg 4 und 6 ist ΔRegret für x2 unter SPO+ höher (+0.02 bis +0.05) und für x1 bei deg = 6 niedriger (−0.03). Zum Vergleich die wahre Kostenfunktion bei 5×5, deg = 6: x2 = 0.40, MSE 0.32, SPO+ 0.37. Bei x2 liegt SPO+ also näher an der wahren Struktur.

## 6 Zusatzprüfung: SPO+-Gewichte bleiben auf der Initialisierung

Setting: 5×5, deg = 2, noise = 0.5, n = 1000. Das Skript liegt nur in der Scratchpad-Ablage, nicht im Projekt.

**Korrelation zwischen gelernten und initialen Gewichten** je Kantengruppe und Feature:

| | MSE-Modell | SPO+-Modell |
|---|---|---|
| corr(w, w_init) | −0.42 bis +0.37 | **+0.56 bis +0.95** |
| mittleres \|w\| bei x5 (Ziel 0) | 0.005–0.008 | **0.17–0.24** |
| mittleres \|w\| bei x1 auf kritischen Kanten | 0.17 | 0.17 |

**Globale Koeffizienten-Wichtigkeit**, Anteil von Σ\|w_j\| über alle Kanten (3 Refits). Bei unabhängigen, standardnormalen x ist das proportional zum mittleren \|SHAP\|:

| | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| MSE | .245 | .266 | .192 | .272 | **.026** |
| SPO+ | .210 | .193 | .187 | .204 | **.206** |

**Interpretation:**

- SPO+ bekommt nur in entscheidungsrelevanten Richtungen einen Gradienten (vgl. den entscheidungsneutralen Unterraum aus der ersten Vorstudie). Alle anderen Gewichte behalten ihren Zufallswert aus der Initialisierung.
- PFI ist dagegen robust. Ein zufälliges Gewicht auf einem unabhängigen Feature ändert den erwarteten Fehler nicht, wenn man das Feature permutiert. Deshalb ist ΔMSE bei x5 ≈ 0.
- SHAP und Koeffizienten zeigen unter SPO+ dagegen eine flache Wichtigkeit: x5 ist so wichtig wie x1. Das ist ein Artefakt der Initialisierung, kein Merkmal der Daten.
- Für die Forschungsfrage ist das relevant: Die Wahl der XAI-Methode entscheidet mit darüber, ob ein Unterschied zwischen MSE und SPO+ sichtbar wird.

**Einschränkungen:**

- Geprüft wurde nur ein Setting.
- Möglich ist auch zu kurzes Training (3000 Schritte, lr = 1e-2 linear fallend). Das lässt sich mit längerem Training oder Null-Initialisierung trennen.
- Bei Null-Initialisierung bliebe x5 vermutlich bei ≈ 0. Dann sähe die Erklärung sauber aus, aber wieder nur wegen der Wahl der Initialisierung.

## 7 Empfehlungen für die nächste Runde

1. **H1-Maß ersetzen.** Statt τ ein gruppenbasiertes Maß verwenden, z. B. den Anteil kritischer Features (wie in Abschnitt 4) oder die Trefferquote „Top-2 nach ΔMSE = {x1, x2}“. Das Rauschen der Nullfeatures fließt dann nicht ein.
2. **ΔMSE zwischen den Losses nur normiert vergleichen.** Die gepaarte Differenz in Abschnitt 8 des Notebooks auf Anteile umstellen.
3. **Sensitivität gegenüber der Initialisierung prüfen.** SPO+ mit Null-Initialisierung und mit mehreren Init-Seeds bei festen Daten trainieren. Dazu SHAP bzw. Koeffizienten-Wichtigkeit als zweite XAI-Methode neben PFI aufnehmen.
4. **„Knapp verfehlende“ Ablenker ergänzen.** Mit Grundkosten 2 sind die Ablenker so teuer, dass kein Modell sie je wählt. ΔRegret ist dort deshalb für jedes Modell 0, und die Losses lassen sich nicht unterscheiden. Mit a_high ≈ 0.5–1 (oder einer zweiten Ablenker-Stufe) würden Fehlvorhersagen auf diesen Kanten Entscheidungen kippen. Erst dann kann sich zeigen, ob SPO+ sie besser beachtet.
5. **Die x1/x2-Asymmetrie** im Text erwähnen oder die Pfade symmetrisieren.
6. **noise = 0, deg = 1** als Kontrollfall kennzeichnen, weil das lineare Modell dort exakt ist.

## 8 Mögliche Frage an Zähringer (ergänzend zu Roadmap AP 9)

Die SPO+-Gewichte im entscheidungsneutralen Raum bleiben auf der Initialisierung, was SHAP- oder koeffizientenbasierte Erklärungen verzerrt. Soll das als eigener Befund in die Arbeit (XAI-Methode × Loss), oder soll es durch eine feste Konvention neutralisiert werden, etwa Null-Initialisierung, Projektion oder Ridge?
