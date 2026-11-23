# Auswertung: Knapsack mit PyEPO-Generator (Stand 23.11.2026)

Grundlage: [knapsack_no_planted_effects.ipynb](../../notebooks/prestudy/knapsack_no_planted_effects.ipynb) und [knapsack_no_planted_effects.csv](../../results/prestudy/knapsack_no_planted_effects.csv). Setting: 2-dimensionaler 0-1-Knapsack mit 32 Items, Kapazität 20 je Dimension, feste Gewichte aus {3.00, …, 7.99} (Setting von Tang & Khalil 2024), unsichere Item-Werte, Maximierung. p = 5 Features, deg ∈ {1, 2, 4, 6}, noise ∈ {0, 0.5}, n ∈ {100, 1000}, m = 5 Refits, n_test = 1000, K = 10 Permutationen. Zahlen ohne weitere Angabe gelten für n = 1000 und noise = 0.5. „Wahre Kostenfunktion“ meint E[c | x] des Generators einschließlich des Aufrundens (im Code `oracle`).

Bezeichnungen: **H1** = Die Rangfolgen der Features nach ΔMSE und nach ΔRegret weichen für dasselbe Modell voneinander ab. **H2** = Das SPO+-Modell stützt sich stärker auf entscheidungskritische Features als das MSE-Modell.

## Lesehilfe

- **PFI-Wert eines Features** = um wie viel der Fehler auf der Testmenge steigt, wenn man die Spalte dieses Features zufällig mischt (Mittel über K = 10 Mischungen). Groß heißt: Ohne dieses Feature wird das Modell schlechter, also stützt es sich darauf. ≈ 0 heißt: Das Modell braucht es nicht. Leicht negative Werte sind Zufallsschwankung.
- **ΔMSE** misst das am Vorhersagefehler (Wichtigkeit für die *Vorhersage*), **ΔRegret** am normierten Regret (Wichtigkeit für die *Entscheidung*).
- **Zeile „wahr“** = dieselbe Messung an der wahren Kostenfunktion E[c | x]. Sie zeigt, wie wichtig ein Feature tatsächlich ist, und dient als Maßstab für die Modelle.
- **Anteile** = PFI-Wert eines Features geteilt durch die Summe über alle Features (negative Werte vorher auf 0 gesetzt). Damit lassen sich Modelle mit unterschiedlichem Niveau vergleichen.
- **Gepaarte Differenz** = PFI(SPO+) − PFI(MSE), je Refit gebildet (beide Modelle auf denselben Daten), dann gemittelt. Positiv heißt: Unter SPO+ ist das Feature wichtiger. `*` heißt: Das 95-%-Intervall über die 5 Refits schließt 0 aus.

## 1 Kurzfazit

1. **Regret wie in der Literatur, mit kleinen Abständen bei niedrigem deg.** Bei deg ≤ 2 sind beide Modelle gleichauf, ab deg 4 ist SPO+ besser (deg 6: 0.148 gegenüber 0.082). Ohne Rauschen ist der Abstand bei deg 6 groß (0.080 gegenüber 0.006).
2. **Anders als beim Shortest Path ist der ΔMSE des SPO+-Modells nicht nahe 0.** Skalenbereinigt liegt er auf dem Niveau des MSE-Modells, bei deg ≤ 2 sogar darüber. Das passt zur Erklärung über den entscheidungsneutralen Unterraum: Beim Knapsack gibt es außer der Skala keine neutrale Richtung.
3. **SPO+ ebnet die ΔMSE-Anteile ein.** Alle Features erhalten 19–22 %, bei der wahren Kostenfunktion liegt die Spanne bei 17–24 %.
4. **Beim ΔRegret liegt SPO+ bei hohem deg näher an der wahren Kostenfunktion.** Das MSE-Modell unterschätzt die Entscheidungsrelevanz aller Features (deg 6, x1: 0.074 gegenüber 0.134 wahr, SPO+ 0.121).
5. **H1 gilt deutlich, schon für die wahre Kostenfunktion.** Bei deg 1 hat x1 den größten ΔMSE-Anteil, aber den kleinsten ΔRegret-Anteil. τ zwischen beiden Rangfolgen liegt bei der wahren Kostenfunktion zwischen −0.6 und 0.4.

## 2 Datenstruktur

PyEPO zieht B zufällig (Bernoulli 0.5) und rundet die Werte auf: c = ⌈5 · [(Bx/√p + 3)^deg + 1] / 3.5^deg · ε⌉. Wie viele Items jedes Feature treibt:

| | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| Items (von 32) | 20 | 14 | 17 | 15 | 17 |

Bei Kapazität 20 und Gewichten zwischen 3 und 8 passen pro Instanz etwa 4 Items. Wie beim Shortest Path mit PyEPO-Generator gibt es keine festgelegte Rolle der Features, also auch keine Grundwahrheit für H2.

Die **Skalenbereinigung** ist hier nur s·ĉ mit s > 0. Eine Verschiebung aller Werte um eine Konstante ist beim Knapsack nicht entscheidungsneutral, weil die Anzahl gewählter Items von Lösung zu Lösung schwankt.

## 3 Modellgüte

Normierter Regret, MSE-Modell / SPO+-Modell (wahre Kostenfunktion in Klammern):

| deg | noise = 0 | noise = 0.5 | noise = 0.5, n = 100 |
|---|---|---|---|
| 1 | 0.0008 / 0.0011 (0) | 0.112 / 0.114 (0.112) | 0.118 / 0.131 |
| 2 | 0.004 / 0.002 (0) | 0.117 / 0.115 (0.113) | 0.123 / 0.137 |
| 4 | 0.027 / 0.003 (0) | 0.116 / 0.095 (0.089) | 0.128 / 0.121 |
| 6 | 0.080 / 0.006 (0) | 0.148 / 0.082 (0.072) | 0.167 / 0.111 |

- Bei deg = 1 ist das MSE-Modell auch ohne Rauschen nicht exakt (Regret 0.0008), weil die wahren Werte aufgerundet sind und damit nicht linear in x.
- Bei noise = 0.5 liegt die Untergrenze (wahre Kostenfunktion) bei 0.07–0.11. Bei deg 6 kommt SPO+ bis auf 0.01 an sie heran, das MSE-Modell liegt 0.08 darüber.
- Bei n = 100 ist SPO+ bei deg ≤ 2 schlechter (0.131 gegenüber 0.118).
- Der MSE des SPO+-Modells ist deutlich höher: skalenbereinigt 6.5 gegenüber 2.9 (deg 1) und 35.5 gegenüber 20.0 (deg 6), roh bei deg 1 sogar 14.6.

## 4 PFI je Feature

### 4.1 ΔMSE

ΔMSE skalenbereinigt:

| deg | Modell | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | wahr | 0.50 | 0.35 | 0.43 | 0.38 | 0.44 |
| 1 | MSE | 0.50 | 0.35 | 0.44 | 0.38 | 0.44 |
| 1 | SPO+ | 0.57 | 0.56 | 0.58 | 0.53 | 0.63 |
| 6 | wahr | 24.0 | 17.4 | 19.7 | 20.6 | 22.7 |
| 6 | MSE | 10.3 | 8.2 | 8.7 | 8.5 | 10.7 |
| 6 | SPO+ | 9.1 | 7.7 | 7.7 | 8.4 | 9.2 |

- **MSE-Modell:** Bei deg ≤ 2 bildet es die wahre Kostenfunktion fast exakt ab. Bei deg 6 erfasst es nur etwa die Hälfte, die Anteile bleiben aber nahe an der Wahrheit (L1-Abstand 0.01–0.05).
- **SPO+-Modell:** Bei deg ≤ 2 ist der ΔMSE *höher* als bei der wahren Kostenfunktion, und alle Features liegen dicht beieinander. Das Modell reagiert also stärker auf Änderungen der Features, als es die Wahrheit tut, und unterscheidet dabei kaum zwischen ihnen. Bei deg 6 liegt es auf dem Niveau des MSE-Modells.
- **Vergleich zum Shortest Path:** Dort ist der ΔMSE des SPO+-Modells fast 0 ([Auswertung_shortest_path_no_planted_effects.md](Auswertung_shortest_path_no_planted_effects.md), Abschnitt 4.1). Hier tritt das nicht auf. Beim Knapsack bestimmt jede Richtung im Werteraum außer der Skala, welche Items gewählt werden, und SPO+ erhält in jeder Richtung einen Gradienten.

### 4.2 ΔRegret

| deg | Modell | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | wahr | 0.002 | 0.010 | 0.005 | 0.005 | 0.009 |
| 1 | MSE | 0.003 | 0.011 | 0.006 | 0.006 | 0.010 |
| 1 | SPO+ | 0.003 | 0.010 | 0.005 | 0.007 | 0.011 |
| 6 | wahr | 0.134 | 0.137 | 0.127 | 0.139 | 0.131 |
| 6 | MSE | 0.074 | 0.092 | 0.077 | 0.086 | 0.084 |
| 6 | SPO+ | 0.121 | 0.131 | 0.120 | 0.132 | 0.129 |

- Bei deg 1 sind beide Modelle nicht von der wahren Kostenfunktion zu unterscheiden.
- Ab deg 4 bleibt der ΔRegret des MSE-Modells bei allen Features deutlich unter dem der wahren Kostenfunktion, der des SPO+-Modells liegt nahe daran. Bei den Anteilen ist SPO+ ebenfalls näher an der Wahrheit (L1-Abstand bei deg 6: 0.02 gegenüber 0.05).
- **Lesart mit Vorsicht:** ΔRegret misst auch die Güte. Ein Modell mit hohem Grund-Regret hat durch die Permutation weniger zu verlieren. Der niedrigere ΔRegret des MSE-Modells hängt deshalb zum Teil an seinem schlechteren Regret.

## 5 H1: ΔMSE und ΔRegret ordnen die Features unterschiedlich

Anteile bei deg = 1 (wahre Kostenfunktion):

| | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| ΔMSE | **0.24** | 0.17 | 0.20 | 0.18 | 0.21 |
| ΔRegret | **0.07** | 0.33 | 0.15 | 0.15 | 0.30 |

- x1 ist für die Vorhersage das wichtigste Feature und für die Entscheidung das unwichtigste. x2 ist es umgekehrt.
- **Mögliche Erklärung (nicht geprüft):** x1 wirkt auf 20 der 32 Items, also auf die meisten. Steigt x1, werden viele Items gleichzeitig wertvoller. Die Werte ändern sich stark (hoher ΔMSE), ihre Reihenfolge untereinander aber wenig, und die Auswahl bleibt ähnlich (niedriger ΔRegret). Ein Feature, das wenige Items treibt, verschiebt dagegen deren Rang gegenüber allen anderen.
- Ab deg 2 gleichen sich die ΔRegret-Anteile an (alle 18–23 %), weil die Werte stärker streuen und jedes Feature Entscheidungen kippen kann.
- Kendall's τ: wahre Kostenfunktion −0.6 bis 0.4, MSE-Modell −0.6 bis 0.1, SPO+-Modell −0.2 bis 0.4. Die negativen Werte zeigen eine teils gegenläufige Rangfolge. Ohne Grundwahrheit, welche Features kritisch sind, lässt sich das aber nur beschreiben, nicht bewerten.

## 6 Gepaarte Differenz SPO+ − MSE

| | ΔMSE skalenbereinigt | ΔRegret |
|---|---|---|
| deg 1 | alle fünf positiv * (+0.07 bis +0.22) | alle ≈ 0 |
| deg 2 | alle fünf positiv * (+0.24 bis +0.58) | alle fünf positiv * (+0.004 bis +0.008) |
| deg 4 | vier von fünf positiv, uneinheitlich | alle fünf positiv * (+0.017 bis +0.021) |
| deg 6 | vier von fünf negativ * (−0.46 bis −1.50) | alle fünf positiv * (+0.039 bis +0.047) |

- Beim ΔRegret haben alle Features dieselbe Richtung und fast dieselbe Größe. Das ist ein Niveaueffekt: Das SPO+-Modell entscheidet besser und verliert durch jede Permutation mehr.
- Beim ΔMSE kippt das Vorzeichen mit deg, aber wieder für alle Features gemeinsam.
- Featurespezifische Unterschiede zwischen den Losses gibt es in diesem Lauf nicht. Bei noise = 0 ist der ΔMSE unter SPO+ schon bei deg 1 und 2 niedriger, bei n = 100 und noise = 0.5 sind die ΔMSE-Differenzen bei deg ≤ 2 nicht signifikant. Der ΔRegret ist in allen vier Einstellungen (noise × n) ab deg 2 bei allen Features unter SPO+ höher, ab deg 4 überall signifikant.

## 7 Folgerungen

1. **Dieser Lauf dient als Referenz, nicht zur Prüfung von H2.** Mit zufälligem B gibt es keine kritischen Features. H2 prüft erst der Generator mit planted effects ([Auswertung_knapsack_planted_effects.md](Auswertung_knapsack_planted_effects.md)).
2. **Der Knapsack ist das Gegenstück zum Shortest Path beim entscheidungsneutralen Raum.** Er hat außer der Skala keinen, und der ΔMSE von SPO+ bricht nicht ein. Für die Arbeit ist das ein Argument, beide Probleme zu verwenden.
3. **Losses nur über Anteile vergleichen.** Die absoluten Differenzen sind Niveaueffekte.
4. **ΔRegret hängt an der Modellgüte.** Das sollte bei der Interpretation stehen, oder ΔRegret wird relativ zum Grund-Regret berichtet.
5. **Aufrunden beachten.** Durch das Aufrunden ist das lineare Modell auch bei deg = 1 nicht exakt spezifiziert. Einen sauberen Kontrollfall wie beim Shortest Path gibt es hier nicht.
