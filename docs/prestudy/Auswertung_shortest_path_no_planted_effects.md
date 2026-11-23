# Auswertung: Shortest Path mit PyEPO-Generator (Stand 23.11.2026)

Grundlage: [shortest_path_no_planted_effects.ipynb](../../notebooks/prestudy/shortest_path_no_planted_effects.ipynb) und [shortest_path_no_planted_effects.csv](../../results/prestudy/shortest_path_no_planted_effects.csv). Setting: 5×5-Gitter (40 Kanten), p = 5 Features, deg ∈ {1, 2, 4, 6}, noise ∈ {0, 0.5}, n ∈ {100, 1000}, m = 5 Refits, n_test = 1000, K = 10 Permutationen. Zahlen ohne weitere Angabe gelten für n = 1000 und noise = 0.5. „Wahre Kostenfunktion“ meint E[c | x] des Generators (im Code `oracle`).

Bezeichnungen: **H1** = Die Rangfolgen der Features nach ΔMSE und nach ΔRegret weichen für dasselbe Modell voneinander ab. **H2** = Das SPO+-Modell stützt sich stärker auf entscheidungskritische Features als das MSE-Modell.

## Lesehilfe

- **PFI-Wert eines Features** = um wie viel der Fehler auf der Testmenge steigt, wenn man die Spalte dieses Features zufällig mischt (Mittel über K = 10 Mischungen). Groß heißt: Ohne dieses Feature wird das Modell schlechter, also stützt es sich darauf. ≈ 0 heißt: Das Modell braucht es nicht. Leicht negative Werte sind Zufallsschwankung.
- **ΔMSE** misst das am Vorhersagefehler (Wichtigkeit für die *Vorhersage*), **ΔRegret** am normierten Regret (Wichtigkeit für die *Entscheidung*).
- **Zeile „wahr“** = dieselbe Messung an der wahren Kostenfunktion E[c | x]. Sie zeigt, wie wichtig ein Feature tatsächlich ist, und dient als Maßstab für die Modelle.
- **Anteile** = PFI-Wert eines Features bzw. einer Gruppe geteilt durch die Summe über alle Features (negative Werte vorher auf 0 gesetzt). Damit lassen sich Modelle mit unterschiedlichem Niveau vergleichen.
- **Gepaarte Differenz** = PFI(SPO+) − PFI(MSE), je Refit gebildet (beide Modelle auf denselben Daten), dann gemittelt. Positiv heißt: Unter SPO+ ist das Feature wichtiger. `*` heißt: Das 95-%-Intervall über die 5 Refits schließt 0 aus.

## 1 Kurzfazit

1. **Regret wie in der Literatur.** Bei deg ≤ 2 liegen beide Modelle gleichauf, ab deg 4 ist SPO+ besser (deg 6: 0.124 gegenüber 0.069).
2. **Der ΔMSE des SPO+-Modells ist fast 0, auch nach Skalenbereinigung.** Bei deg ≤ 2 liegt er je Feature bei höchstens 0.003, beim MSE-Modell bei 0.013–0.063. Nach ΔMSE wirkt es, als nutze das SPO+-Modell die Features kaum. Die wahrscheinliche Ursache ist der entscheidungsneutrale Unterraum des Gitters (Abschnitt 4.1).
3. **Beim ΔRegret liegt SPO+ näher an der wahren Kostenfunktion.** Beide Modelle geben das Muster der wahren Kostenfunktion wieder. Bei deg 4 und 6 trifft das SPO+-Modell es aber besser, das MSE-Modell unterschätzt die Entscheidungsrelevanz aller Features.
4. **H1 gilt schon für die wahre Kostenfunktion.** x1 hat einen mittleren ΔMSE, aber den kleinsten ΔRegret, x5 den kleinsten ΔMSE, aber einen ΔRegret wie x2. Ohne bekannte Struktur lässt sich daraus aber keine Aussage über kritische Features ableiten. Dafür braucht es den Generator mit planted effects.
5. **Die gepaarten Differenzen SPO+ − MSE sind Niveaueffekte.** ΔMSE ist unter SPO+ bei allen Features kleiner, ΔRegret bei deg 6 bei allen größer. Featurespezifisch ist daran nichts.

## 2 Datenstruktur

Der PyEPO-Generator zieht B zufällig (Bernoulli 0.5). Jedes Feature wirkt deshalb auf viele Kanten:

| | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| Kanten (von 40) | 22 | 19 | 22 | 27 | 16 |

Es gibt keine Features, die nur Kosten, aber keine Entscheidung beeinflussen, und keine reinen Rauschfeatures. Jedes Feature ist mehr oder weniger entscheidungsrelevant. Welche Features „kritisch“ sind, ist daher nicht festgelegt, H2 lässt sich hier nicht prüfen.

## 3 Modellgüte

Normierter Regret, MSE-Modell / SPO+-Modell (wahre Kostenfunktion in Klammern):

| deg | noise = 0 | noise = 0.5 | noise = 0.5, n = 100 |
|---|---|---|---|
| 1 | 0.000 / 0.000 (0) | 0.154 / 0.155 (0.151) | 0.167 / 0.177 |
| 2 | 0.001 / 0.000 (0) | 0.100 / 0.100 (0.099) | 0.107 / 0.121 |
| 4 | 0.019 / 0.004 (0) | 0.078 / 0.068 (0.057) | 0.092 / 0.092 |
| 6 | 0.067 / 0.014 (0) | 0.124 / 0.069 (0.045) | 0.154 / 0.105 |

- Bei noise = 0 und deg = 1 ist das lineare Modell korrekt spezifiziert. Das MSE-Modell ist dort exakt (MSE ≈ 1e-12, Regret 0). Dieser Fall ist ein Kontrollfall.
- Bei noise = 0.5 kann kein Modell einen Regret von 0 erreichen, weil das Rauschen ε aus x nicht vorhersagbar ist. Der Regret der wahren Kostenfunktion ist die Untergrenze. Bei deg ≤ 2 liegen beide Modelle praktisch auf ihr.
- Bei n = 100 ist SPO+ bei deg ≤ 2 etwas schlechter als das MSE-Modell (0.177 gegenüber 0.167), bei deg 6 weiter klar besser.
- MSE skalenbereinigt, MSE-Modell / SPO+-Modell: 0.11 / 0.46 (deg 1) bis 1.07 / 2.01 (deg 6). Roh liegt der MSE des SPO+-Modells bei 1.1–3.3.

## 4 PFI je Feature

### 4.1 ΔMSE

ΔMSE skalenbereinigt:

| deg | Modell | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | wahr | 0.015 | 0.015 | 0.019 | 0.021 | 0.013 |
| 1 | MSE | 0.015 | 0.015 | 0.018 | 0.021 | 0.013 |
| 1 | SPO+ | −0.002 | 0.001 | 0.001 | 0.000 | 0.001 |
| 6 | wahr | 0.952 | 0.892 | 1.258 | 1.082 | 0.715 |
| 6 | MSE | 0.425 | 0.381 | 0.494 | 0.523 | 0.313 |
| 6 | SPO+ | 0.036 | 0.052 | 0.122 | 0.083 | 0.062 |

- **MSE-Modell:** Bei deg ≤ 2 bildet es die wahre Kostenfunktion fast exakt ab. Bei deg 6 halbiert sich der ΔMSE, weil das lineare Modell den konvexen Zusammenhang nicht abbilden kann. Die Anteile je Feature bleiben aber nahe an der Wahrheit (L1-Abstand der Anteilsvektoren 0.02–0.06).
- **SPO+-Modell:** Der ΔMSE ist um eine bis zwei Größenordnungen kleiner, bei x1 und deg 1 sogar leicht negativ. Die Anteile sind verzerrt: x1 erhält 4–10 % statt 18–19 %, x3 34 % statt 22–26 % (L1-Abstand 0.25–0.46).

**Erklärung über den entscheidungsneutralen Unterraum.** Im Gitter ändert jede Verschiebung v = Aᵀπ der Kantenkosten (A = Knoten-Kanten-Inzidenzmatrix, π = beliebige Knotenpotenziale) die Länge *jedes* Pfads von s nach t um dieselbe Konstante π_t − π_s. Die optimale Entscheidung bleibt also gleich. Bei 25 Knoten sind das 24 der 40 Richtungen im Kostenraum. Der SPO+-Subgradient 2(w*(c) − w*(2ĉ − c)) ist die Differenz zweier Pfadvektoren und steht senkrecht auf diesen Richtungen. Der Anteil der Vorhersage im neutralen Unterraum wird deshalb vom Loss nicht gesteuert, er hängt nur von Initialisierung und Optimierer ab.

Daraus folgt dreierlei:

- Der MSE des SPO+-Modells wird von Abweichungen in diesen 24 Richtungen dominiert.
- Der Kostenanteil der Features in diesen Richtungen wird nicht gelernt und trägt nichts zum ΔMSE bei.
- Die Skalenbereinigung a + s·ĉ korrigiert nur eine der 24 Richtungen (a·1).

Die Beobachtung passt zu dieser Erklärung. Geprüft per Projektion ist sie nicht. Im Design mit planted effects ist bestätigt, dass die SPO+-Gewichte weitgehend auf ihrer Initialisierung bleiben ([Auswertung_shortest_path_planted_effects.md](Auswertung_shortest_path_planted_effects.md), Abschnitt 6). Beim Knapsack, der außer der Skala keine neutrale Richtung hat, tritt der Effekt nicht auf ([Auswertung_knapsack_no_planted_effects.md](Auswertung_knapsack_no_planted_effects.md)).

### 4.2 ΔRegret

| deg | Modell | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|---|
| 1 | wahr | 0.009 | 0.022 | 0.028 | 0.021 | 0.022 |
| 1 | MSE | 0.009 | 0.021 | 0.026 | 0.021 | 0.021 |
| 1 | SPO+ | 0.009 | 0.020 | 0.025 | 0.022 | 0.019 |
| 6 | wahr | 0.104 | 0.177 | 0.298 | 0.184 | 0.199 |
| 6 | MSE | 0.070 | 0.136 | 0.185 | 0.156 | 0.126 |
| 6 | SPO+ | 0.090 | 0.169 | 0.266 | 0.180 | 0.170 |

- Bei deg ≤ 2 sind beide Modelle von der wahren Kostenfunktion nicht zu unterscheiden.
- Bei deg 4 und 6 liegt das SPO+-Modell bei allen Features näher an der wahren Kostenfunktion, sowohl absolut als auch bei den Anteilen (L1-Abstand 0.03 / 0.05 gegenüber 0.08 / 0.12 beim MSE-Modell).
- **Lesart mit Vorsicht:** ΔRegret misst nicht nur, wie stark ein Modell ein Feature nutzt, sondern auch, wie gut es entscheidet. Ein Modell, das ohnehin schlecht entscheidet, verliert durch die Permutation weniger. Der niedrigere ΔRegret des MSE-Modells bei deg 6 hängt also zum Teil an seinem höheren Grund-Regret. Vergleiche zwischen Modellen sind über Anteile aussagekräftiger.

## 5 H1: ΔMSE und ΔRegret ordnen die Features unterschiedlich

Anteile an der Summe der PFI-Werte bei deg = 1 (wahre Kostenfunktion):

| | x1 | x2 | x3 | x4 | x5 |
|---|---|---|---|---|---|
| ΔMSE | 0.18 | 0.19 | 0.23 | 0.25 | 0.16 |
| ΔRegret | **0.09** | 0.21 | 0.27 | 0.21 | **0.22** |

- x1 wirkt auf 22 Kanten und hat einen mittleren ΔMSE, aber mit Abstand den kleinsten ΔRegret. x5 wirkt nur auf 16 Kanten und hat den kleinsten ΔMSE, beim ΔRegret aber gleichauf mit x2 und x4. Entscheidend ist nicht, auf wie viele Kanten ein Feature wirkt, sondern auf welche.
- Kendall's τ zwischen beiden Rangfolgen: wahre Kostenfunktion 0.0–0.4, MSE-Modell 0.24–0.52, SPO+-Modell 0.20–0.80. Das SPO+-Modell hat bei hohem deg das höchste τ, was aber am verzerrten ΔMSE liegt (Abschnitt 4.1), nicht an einer besseren Übereinstimmung mit der Wahrheit.
- τ ist bei 5 Features grob (nur 11 mögliche Werte) und ohne Grundwahrheit nicht deutbar. Es zeigt nur, dass sich die Rangfolgen unterscheiden. Ob ein Modell die „richtigen“ Features für die Entscheidung nutzt, lässt sich hier nicht beurteilen.

## 6 Gepaarte Differenz SPO+ − MSE

Mittel ± 95-%-Halbbreite über 5 Refits, `*` = Intervall schließt 0 aus (ohne Mehrfachtest-Korrektur):

| | ΔMSE skalenbereinigt | ΔRegret |
|---|---|---|
| deg 1 | alle fünf negativ * (−0.011 bis −0.021) | alle ≈ 0 |
| deg 2 | alle fünf negativ * (−0.037 bis −0.062) | alle ≈ 0 |
| deg 4 | alle fünf negativ * (−0.11 bis −0.18) | x3, x5 positiv * (+0.02, +0.01) |
| deg 6 | alle fünf negativ * (−0.25 bis −0.44) | alle fünf positiv * (+0.02 bis +0.08) |

Bei allen Features gibt es dieselbe Richtung. Die Differenzen zeigen daher nur das Niveau (Abschnitt 4), keine Verschiebung zwischen Features. Bei noise = 0 und n = 100 sieht es genauso aus.

## 7 Folgerungen

1. **ΔMSE beim Shortest Path im entscheidungsrelevanten Raum messen.** Neben der rohen und der skalenbereinigten Variante sollte ΔMSE nach Projektion auf den entscheidungsrelevanten Teil berichtet werden, also mit P = I − Aᵀ(AAᵀ)⁺A auf Vorhersage und wahre Kosten. Erst dann sind die Werte von MSE- und SPO+-Modell vergleichbar. Die Lücke zwischen roher und projizierter Variante ist selbst ein Befund zu H1.
2. **Losses nur über Anteile vergleichen.** Absolute PFI-Werte sind beim ΔMSE durch das Niveau und beim ΔRegret durch die Modellgüte verzerrt.
3. **Dieser Lauf dient als Referenz, nicht zur Prüfung von H2.** Mit zufälligem B gibt es keine Grundwahrheit, welche Features kritisch sind. H2 lässt sich nur mit planted effects prüfen.
4. **noise = 0, deg = 1 als Kontrollfall kennzeichnen.** Dort ist das MSE-Modell exakt.
