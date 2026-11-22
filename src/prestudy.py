"""Vorstudie: lineares Modell mit MSE vs. SPO+ auf zwei Testproblemen.

Probleme (Parameter ``problem``):
- "shortest_path": kürzester Weg im k×k-Gitter (LP, Minimierung); size = k.
- "knapsack":      2-dimensionaler 0-1-Knapsack mit festen Gewichten und unsicheren
                   Item-Werten (binäres IP, Maximierung); size = Anzahl Items.

Je Problem zwei Generatoren (Parameter ``design``), beide von der Form
c = a + s · ((Bx/√p + 3)^deg + 1) / 3.5^deg · ε  (s = 1 beim Shortest Path, 5 beim Knapsack):
- "pyepo":   PyEPO-Standardgenerator (a = 0, B zufällig Bernoulli; Knapsack-Werte aufgerundet),
- "planted": B (beim Shortest Path auch a, beim Knapsack die Gewichte) bewusst gesetzt:
             kritische, Ablenker- und Rauschfeatures.

Je Einstellung (deg, noise, n) und Refit werden beide Modelle trainiert und per
Permutation Feature Importance (PFI) bewertet: ΔMSE (roh und skalenbereinigt) und
Δ normierter Regret je Feature. Die wahre Kostenfunktion E[c|x] dient als Referenz
(in den Ergebnissen loss = "oracle").
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache, partial
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyepo
import torch
from pyepo import EPO
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Pfade relativ zur Repo-Wurzel (src/prestudy.py liegt eine Ebene darunter), unabhängig vom Arbeitsverzeichnis
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "prestudy"

P = 5  # Anzahl Features
DEFAULT_SIZE = {"shortest_path": 5, "knapsack": 32}
SCALE = {"shortest_path": 1.0, "knapsack": 5.0}  # Faktor s wie in PyEPO
KNAPSACK_DIM = 2      # Ressourcen-Dimensionen (wie Tang & Khalil 2024)
KNAPSACK_CAP = 20     # Kapazität je Dimension beim PyEPO-Generator (wie Tang & Khalil 2024)
KNAPSACK_CAP_FRAC = 1 / 3  # planted: Kapazität = Anteil am Gesamtgewicht der Kern-Items


# --- Optimierungsmodelle -----------------------------------------------------

def _tune(om):
    """Ein Thread je Solve (bei kleinen Modellen schneller, parallele Prozesse stören sich nicht) und exakte MIP-Lösung."""
    om._model.setParam("Threads", 1)
    om._model.setParam("MIPGap", 0)
    return om


@cache
def grid_model(k):
    """Gurobi-Modell für den kürzesten Weg im k×k-Gitter (Knoten = Zeile·k + Spalte)."""
    return _tune(pyepo.model.grb.shortestPathModel((k, k)))


@cache
def opt_model(problem, design, size, seed):
    """Optimierungsmodell; beim Knapsack hängen Gewichte und Kapazität vom Generator ab."""
    if problem == "shortest_path":
        return grid_model(size)
    _, _, weights, caps = structure(problem, design, size, seed)
    return _tune(pyepo.model.grb.knapsackModel(weights, caps))


def solve(om, C):
    """Optimale Lösungen W und Zielwerte z für jede Zeile von C."""
    W = np.empty_like(C)
    for i, ci in enumerate(C):
        om.setObj(ci)
        W[i] = om.solve()[0]
    return W, (W * C).sum(1)


def sense(om):
    """+1 bei Minimierung, −1 bei Maximierung (Vorzeichen des Regrets)."""
    return 1.0 if om.modelSense == EPO.MINIMIZE else -1.0


# --- Generatoren: Struktur ---------------------------------------------------

def cost_mean(X, a, B, deg, scale=1.0):
    """E[c|x] ohne Aufrunden; das multiplikative Rauschen ε hat Erwartungswert 1."""
    return a + scale * ((X @ B.T / np.sqrt(X.shape[1]) + 3) ** deg + 1) / 3.5**deg


def planted_paths(k):
    """Vier konkurrierende Pfade als Zugfolge (R = rechts, D = unten) mit ihrem Treiber in x.

    "mitte" teilt Kanten mit "links-unten" und "treppe", hat also weniger exklusive Kanten;
    der Faktor 2 gleicht das aus (mit Faktor 1 wäre der Pfad bei noise = 0 kaum je optimal).
    """
    h = (k - 1) // 2
    return {
        "oben-rechts": ("R" * (k - 1) + "D" * (k - 1), [1, 0, 0, 0, 0]),
        "links-unten": ("D" * (k - 1) + "R" * (k - 1), [0, 1, 0, 0, 0]),
        "treppe": ("RD" * (k - 1), [0, 0, 0, 0, 0]),
        "mitte": ("D" * h + "R" * (k - 1) + "D" * (k - 1 - h), [2, 2, 0, 0, 0]),
    }


def path_vector(k, moves):
    """0/1-Vektor über die Kanten von grid_model(k) für eine Zugfolge."""
    idx = {arc: i for i, arc in enumerate(grid_model(k).arcs)}
    w, node = np.zeros(len(idx)), 0
    for mv in moves:
        nxt = node + (1 if mv == "R" else k)
        w[idx[(node, nxt)]] = 1
        node = nxt
    return w


def planted_structure(k, seed, a_high=2.0):
    """Shortest Path: Grundkosten a und Matrix B mit planted effects.

    - kritisch (x1, x2): Kanten, die genau einer der vier Pfade nutzt, tragen dessen Treiber,
      geteilte Kanten keinen. Alle Pfade haben Grundkosten 0.
    - Ablenker (x3, x4): alle übrigen Kanten; Grundkosten a_high, damit nie optimal;
      je Kante x3 und/oder x4 per Bernoulli(0.5).
    - Rauschen (x5): wirkt auf keine Kante.
    """
    paths = planted_paths(k)
    W = np.array([path_vector(k, mv) for mv, _ in paths.values()])  # Pfade × Kanten
    drivers = np.array([b for _, b in paths.values()])               # Pfade × p
    use = W.sum(0)
    B = np.where((use == 1)[:, None], W.T @ drivers, 0.0)
    distr = use == 0
    B[distr, 2:4] = np.random.RandomState(seed).binomial(1, 0.5, (distr.sum(), 2))
    return np.where(distr, a_high, 0.0), B


KNAPSACK_GROUPS = {  # Gruppe -> treibendes Feature (Spalte in B) bzw. None
    "kritisch x1": 0, "kritisch x2": 1, "konstant": None, "zu schwer (x3)": 2, "gratis (x4)": 3,
}


def knapsack_groups(m):
    """Gruppe je Item beim planted-Knapsack: je m/8 Ablenker, der Rest zu gleichen Teilen kritisch x1, x2, konstant."""
    q = m // 8
    g, rest = divmod(m - 2 * q, 3)
    if rest or q == 0:
        raise ValueError(f"m = {m}: m - 2·(m // 8) muss durch 3 teilbar und m >= 8 sein (z. B. m = 32)")
    sizes = [g, g, g, q, q]
    return np.repeat(list(KNAPSACK_GROUPS), sizes)


def knapsack_planted_structure(m, seed, cap_frac=KNAPSACK_CAP_FRAC):
    """Knapsack: Gewichte, Kapazität und Matrix B mit planted effects (alle Grundwerte a = 0).

    - kritisch (x1, x2) und konstant: drei Gruppen mit *denselben* Gewichtsvektoren (wie in PyEPO
      aus {3,00; …; 7,99} gezogen); x1 treibt die eine, x2 die andere, die dritte hat konstanten
      Wert (B-Zeile 0, Referenz wie die Treppe beim Shortest Path). Bei x = 0 sind alle drei gleich
      wertvoll; die Kapazität (cap_frac des Gesamtgewichts) erzwingt die Auswahl.
    - Ablenker x3: Items, die schwerer sind als die Kapazität, also nie zulässig.
    - Ablenker x4: Items mit Gewicht 0, also bei positivem Wert immer gewählt (alle wahren Werte sind > 0).
    - Rauschen (x5): wirkt auf kein Item.
    """
    groups = knapsack_groups(m)
    g = (groups == "kritisch x1").sum()
    w_core = np.random.RandomState(seed).choice(range(300, 800), size=(KNAPSACK_DIM, g)) / 100
    caps = np.round(cap_frac * 3 * w_core.sum(1), 2)
    weights = np.zeros((KNAPSACK_DIM, m))
    weights[:, :3 * g] = np.tile(w_core, 3)
    weights[:, groups == "zu schwer (x3)"] = (caps + 1)[:, None]
    B = np.zeros((m, P))
    for name, j in KNAPSACK_GROUPS.items():
        if j is not None:
            B[groups == name, j] = 1
    return weights, caps, B


@cache
def structure(problem, design, size, seed):
    """Grundwerte a, Matrix B (Items/Kanten × p) und beim Knapsack Gewichte (dim × m) und Kapazität."""
    if problem == "shortest_path":
        d = grid_model(size).num_cost
        if design == "pyepo":
            a, B = np.zeros(d), np.random.RandomState(seed).binomial(1, 0.5, (d, P))  # = erster Zug in genData
        else:
            a, B = planted_structure(size, seed)
        return a, B, None, None
    if design == "pyepo":
        rnd = np.random.RandomState(seed)  # gleiche Zugfolge wie pyepo.data.knapsack.genData
        weights = rnd.choice(range(300, 800), size=(KNAPSACK_DIM, size)) / 100
        B = rnd.binomial(1, 0.5, (size, P))
        caps = np.full(KNAPSACK_DIM, float(KNAPSACK_CAP))
    else:
        weights, caps, B = knapsack_planted_structure(size, seed)
    return np.zeros(size), B, weights, caps


# --- Generatoren: Daten ------------------------------------------------------

def expected_ceil(y, h):
    """E[⌈y·ε⌉] für ε ~ U(1 − h, 1 + h) und y ≥ 0 (PyEPO rundet die Knapsack-Werte auf).

    Mit U = y·ε ~ U(lo, hi) gilt E[⌈U⌉] = Σ_{j≥0} P(U > j); Summanden mit j ≤ lo sind 1,
    mit lo < j < hi linear fallend, darüber 0.
    """
    if h == 0:
        return np.ceil(y)
    lo, hi = y * (1 - h), y * (1 + h)
    j0, j1 = np.floor(lo) + 1, np.ceil(hi) - 1
    cnt = np.maximum(j1 - j0 + 1, 0)
    return j0 + (cnt * hi - cnt * (j0 + j1) / 2) / (hi - lo)


def true_cost(X, problem, design, size, deg, noise, seed):
    """Wahre Kostenfunktion E[c|x] des Generators (picklebar über functools.partial)."""
    a, B, _, _ = structure(problem, design, size, seed)
    mean = cost_mean(X, a, B, deg, SCALE[problem])
    if problem == "knapsack" and design == "pyepo":
        return expected_ceil(mean, noise)
    return mean


def make_data(problem, design, size, N, deg, noise, seed):
    """N Beobachtungen (X, C) und die wahre Kostenfunktion E[c|x] des Generators."""
    a, B, weights, _ = structure(problem, design, size, seed)
    truth = partial(true_cost, problem=problem, design=design, size=size, deg=deg, noise=noise, seed=seed)
    if design == "pyepo" and problem == "shortest_path":
        X, C = pyepo.data.shortestpath.genData(N, P, (size, size), deg, noise, seed)
    elif design == "pyepo":
        w, X, C = pyepo.data.knapsack.genData(N, P, size, KNAPSACK_DIM, deg, noise, seed)
        assert np.allclose(w, weights), "Gewichte weichen von pyepo.data.knapsack.genData ab"
    else:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, P))
        eps = rng.uniform(1 - noise, 1 + noise, (N, len(a)))
        C = (a + cost_mean(X, 0, B, deg, SCALE[problem]) * eps).astype(np.float32)
    return X, C, truth


# --- Prüfung der planted-Generatoren ----------------------------------------

def path_shares(k, deg, noise, N=2000, seed=0):
    """Shortest Path: wie oft ist welcher Pfad optimal, werden Ablenker-Kanten genutzt?"""
    _, C, _ = make_data("shortest_path", "planted", k, N, deg, noise, seed)
    W, _ = solve(grid_model(k), C)
    shares = {name: (np.abs(W - path_vector(k, mv)).max(1) < 0.5).mean()
              for name, (mv, _) in planted_paths(k).items()}
    shares["andere"] = 1 - sum(shares.values())
    shares["Ablenker genutzt"] = (W[:, planted_structure(k, seed)[0] > 0].sum(1) > 0).mean()
    return pd.Series(shares)


def item_shares(m, deg, noise, N=2000, seed=0):
    """Knapsack: Anteil gewählter Items je Gruppe und Anteil der häufigsten Lösung (wahre Werte).

    Soll: "zu schwer" = 0, "gratis" = 1; die häufigste Lösung nicht über ca. 70 % (AP 5).
    """
    _, C, _ = make_data("knapsack", "planted", m, N, deg, noise, seed)
    W, _ = solve(opt_model("knapsack", "planted", m, seed), C)
    groups = knapsack_groups(m)
    shares = {name: W[:, groups == name].mean() for name in KNAPSACK_GROUPS}
    _, counts = np.unique(W.round().astype(int), axis=0, return_counts=True)
    shares["Items je Lösung"] = W.sum(1).mean()
    shares["häufigste Lösung"] = counts.max() / N
    shares["verschiedene Lösungen"] = len(counts)
    return pd.Series(shares)


def decision_shares(problem, size, deg, noise, N=2000, seed=0):
    """Prüfung des planted-Generators für das jeweilige Problem."""
    if problem == "shortest_path":
        return path_shares(size, deg, noise, N, seed)
    return item_shares(size, deg, noise, N, seed)


# --- Training und PFI --------------------------------------------------------

def train(om, X, C, loss, steps, seed, lr=1e-2, batch_size=32):
    """Lineares Modell p -> d; beide Losses mit gleichem Optimierer, Schrittzahl und linear fallender Lernrate."""
    torch.manual_seed(seed)
    W, z = solve(om, C)
    data = TensorDataset(*(torch.tensor(a, dtype=torch.float32) for a in (X, C, W, z)))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    model = nn.Linear(X.shape[1], C.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    spo = pyepo.func.SPOPlus(om)
    epochs = int(np.ceil(steps / len(loader)))
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=epochs * len(loader))
    for _ in range(epochs):
        for xb, cb, wb, zb in loader:
            pred = model(xb)
            l = spo(pred, cb, wb, zb) if loss == "spo+" else nn.functional.mse_loss(pred, cb)
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
    return lambda X: model(torch.tensor(X, dtype=torch.float32)).detach().numpy()


def scale_fit(C_hat, C, shift=True):
    """Entscheidungsneutrale Anpassung a + s·ĉ mit s > 0, per Kleinste-Quadrate auf (ĉ, c).

    Shortest Path: Verschiebung a erlaubt (im Gitter hat jeder Pfad gleich viele Kanten).
    Knapsack: nur Skalierung (a = 0), s = Σ ĉ·c / Σ ĉ².
    """
    if shift:
        s, a = np.polyfit(C_hat.ravel(), C.ravel(), 1)
    else:
        a, s = 0.0, (C_hat * C).sum() / (C_hat**2).sum()
    return (a, s) if s > 0 else (0.0, 1.0)  # s <= 0 wäre nicht entscheidungsneutral -> roh


def scores(om, C_hat, C, z, a, s):
    """MSE roh, MSE skalenbereinigt, normierter Regret (wie pyepo.metric.regret)."""
    W, _ = solve(om, C_hat)
    regret = sense(om) * ((W * C).sum(1) - z).sum() / np.abs(z).sum()
    return ((C_hat - C) ** 2).mean(), ((a + s * C_hat - C) ** 2).mean(), regret


def pfi(om, predict, X, C, z, a, s, K, seed):
    """Mittlerer Anstieg von (MSE, MSE_adj, Regret) über K Permutationen je Feature.

    Eigener Generator je Aufruf: gleicher seed -> beide Losses sehen dieselben Permutationen.
    """
    rng = np.random.default_rng(seed)
    base = np.array(scores(om, predict(X), C, z, a, s))
    rows = []
    for j in range(X.shape[1]):
        deltas = []
        for _ in range(K):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(np.array(scores(om, predict(Xp), C, z, a, s)) - base)
        d = np.mean(deltas, axis=0)
        rows.append(dict(feature=f"x{j + 1}", mse=base[0], mse_adj=base[1], regret=base[2],
                         d_mse=d[0], d_mse_adj=d[1], d_regret=d[2]))
    return rows


def _job(spec, loss, refit, Xtr, Ctr, Xte, Cte, zte, steps, K, truth):
    """Ein Modell (oder das Orakel) trainieren und per PFI bewerten; läuft auch in einem Worker-Prozess."""
    problem, design, size, seed = spec
    om = opt_model(*spec)
    if loss == "oracle":
        return pfi(om, truth, Xte, Cte, zte, 0.0, 1.0, K, seed)
    predict = train(om, Xtr, Ctr, loss, steps, seed=refit)
    a, s = scale_fit(predict(Xtr), Ctr, shift=problem == "shortest_path")
    return pfi(om, predict, Xte, Cte, zte, a, s, K, seed=refit)


def _init_worker():
    torch.set_num_threads(1)


def run(problem="shortest_path", design="pyepo", size=None, degs=(1, 2, 4, 6), noises=(0.0, 0.5),
        ns=(100, 1000), m=5, n_test=1000, K=10, steps=3000, seed=0, n_jobs=None):
    """Ein Datenaufruf je (deg, noise): Test + m disjunkte Trainingsblöcke mit gleicher Struktur.

    Die Modelle (je n, Refit und Loss) und das Orakel laufen als unabhängige Jobs, bei
    n_jobs > 1 parallel in eigenen Prozessen; das Ergebnis hängt nicht von n_jobs ab.
    n_jobs = None: alle Kerne bis auf zwei.
    """
    size = size or DEFAULT_SIZE[problem]
    spec = (problem, design, size, seed)
    om = opt_model(*spec)
    n_jobs = n_jobs or max(1, (os.cpu_count() or 1) - 2)

    jobs, labels = [], []  # labels[i] = Kennzeichnung der Zeilen von Job i
    for deg, noise in product(degs, noises):
        X, C, truth = make_data(problem, design, size, n_test + m * max(ns), deg, noise, seed)
        Xte, Cte = X[:n_test], C[:n_test]
        _, zte = solve(om, Cte)
        setting = dict(problem=problem, size=size, deg=deg, noise=noise)
        jobs.append((spec, "oracle", -1, None, None, Xte, Cte, zte, steps, K, truth))
        labels.append(setting | dict(n=np.nan, refit=-1, loss="oracle"))
        for n, d in product(ns, range(m)):
            lo = n_test + d * max(ns)
            for loss in ("mse", "spo+"):
                jobs.append((spec, loss, d, X[lo:lo + n], C[lo:lo + n], Xte, Cte, zte, steps, K, None))
                labels.append(setting | dict(n=n, refit=d, loss=loss))

    results = [None] * len(jobs)
    per_setting = len(ns) * m * 2 + 1
    todo = {i: per_setting for i in range(0, len(jobs), per_setting)}

    def done(i):
        start = i - i % per_setting
        todo[start] -= 1
        if not todo[start]:
            lab = labels[start]
            finished = sum(v == 0 for v in todo.values())
            print(f"{problem} {design} (size {size}): deg={lab['deg']}, noise={lab['noise']} fertig "
                  f"({finished}/{len(todo)})", flush=True)

    if n_jobs == 1:
        for i, job in enumerate(jobs):
            results[i] = _job(*job)
            done(i)
    else:
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("spawn"), initializer=_init_worker) as ex:
            futures = {ex.submit(_job, *job): i for i, job in enumerate(jobs)}
            for f in as_completed(futures):
                i = futures[f]
                results[i] = f.result()
                done(i)
    return pd.DataFrame([lab | r for lab, rows in zip(labels, results) for r in rows])


# --- Auswertung --------------------------------------------------------------

COLORS = {"mse": "#2a78d6", "spo+": "#eb6834", "oracle": "#6b6a64"}
LABELS = {"mse": "MSE", "spo+": "SPO+", "oracle": "Orakel"}
MEASURES = {"d_mse": "ΔMSE roh", "d_mse_adj": "ΔMSE skalenbereinigt", "d_regret": "Δ normierter Regret"}


def ci95(s):
    """Halbe Breite des 95-%-t-Intervalls über die Refits."""
    return stats.t.ppf(0.975, len(s) - 1) * s.sem() if len(s) > 1 else 0.0


def performance(df):
    """Eine Zeile je Modell (Basiswerte stehen in jeder Feature-Zeile)."""
    keys = [c for c in ("problem", "size", "deg", "noise", "n", "refit", "loss") if c in df]
    return df.drop_duplicates(keys)


def plot_over_deg(df, metric, ylabel, title=""):
    """Kleine Vielfache: Zeilen noise, Spalten n; Linien je Loss mit 95-%-KI über Refits."""
    perf = performance(df)
    noises, ns = sorted(perf.noise.unique()), sorted(perf.n.dropna().unique())
    fig, axes = plt.subplots(len(noises), len(ns), figsize=(9, 6), sharex=True, sharey=True, squeeze=False)
    for (i, noise), (j, n) in product(enumerate(noises), enumerate(ns)):
        ax = axes[i, j]
        sub = perf[(perf.noise == noise) & ((perf.n == n) | (perf.loss == "oracle"))]
        for loss, g in sub.groupby("loss"):
            s = g.groupby("deg")[metric].agg(["mean", ci95])
            ax.errorbar(s.index, s["mean"], yerr=s["ci95"], label=LABELS[loss], color=COLORS[loss],
                        lw=2, marker="o", ms=5, capsize=3, ls="--" if loss == "oracle" else "-")
        ax.set_title(f"noise = {noise}, n = {int(n)}", fontsize=10)
        ax.set_xticks(sorted(perf.deg.unique()))
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("deg")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()


def plot_pfi(df, n=1000, noise=0.5, title=""):
    """PFI je Feature: Zeilen Maße, Spalten deg; Balken MSE/SPO+ mit 95-%-KI, Striche Orakel."""
    degs = sorted(df.deg.unique())
    fig, axes = plt.subplots(len(MEASURES), len(degs), figsize=(13, 8), sharex=True)
    sub = df[(df.noise == noise) & ((df.n == n) | (df.loss == "oracle"))]
    feats = sorted(df.feature.unique())
    xs = np.arange(len(feats))
    for (i, (m, label)), (j, deg) in product(enumerate(MEASURES.items()), enumerate(degs)):
        ax = axes[i, j]
        g = sub[sub.deg == deg].groupby(["loss", "feature"])[m].agg(["mean", ci95])
        for off, loss in [(-0.2, "mse"), (0.2, "spo+")]:
            ax.bar(xs + off, g.loc[loss, "mean"], 0.38, yerr=g.loc[loss, "ci95"], color=COLORS[loss],
                   label=LABELS[loss], capsize=2, edgecolor="white", linewidth=1)
        ax.scatter(xs, g.loc["oracle", "mean"], marker="_", s=300, lw=2.5, color=COLORS["oracle"],
                   label=LABELS["oracle"], zorder=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(xs, feats)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_title(f"deg = {deg}")
        if j == 0:
            ax.set_ylabel(label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title} PFI, n = {n}, noise = {noise}".strip())
    fig.tight_layout()


def kendall_tau(df):
    """Kendall's τ zwischen den Rangfolgen nach ΔMSE (bereinigt) und ΔRegret, je Refit, gemittelt."""
    keys = ["noise", "n", "deg", "loss"]
    tau = (df.groupby(keys + ["refit"], dropna=False)
           .apply(lambda g: stats.kendalltau(g.d_mse_adj, g.d_regret).statistic, include_groups=False)
           .groupby(keys, dropna=False).mean())
    return tau.unstack("loss").round(2)


def paired_diff(df, measures=("d_mse_adj", "d_regret")):
    """Gepaarte Differenz SPO+ − MSE je Feature: 'Mittel ± KI-Halbbreite', * = KI schließt 0 aus."""
    wide = df[df.loss != "oracle"].pivot_table(index=["noise", "n", "deg", "feature", "refit"],
                                               columns="loss", values=list(measures))
    diff = wide.xs("spo+", axis=1, level="loss") - wide.xs("mse", axis=1, level="loss")

    def fmt(s):
        m, h = s.mean(), ci95(s)
        return f"{m:+.4f} ± {h:.4f}" + (" *" if abs(m) > h else "")

    return diff.groupby(["noise", "n", "deg", "feature"]).agg(fmt)


def plot_structure(problem, size, seed=0):
    """Struktur des planted-Generators: Kantengruppen im Gitter bzw. Item-Gruppen nach Gewicht."""
    if problem == "shortest_path":
        return _plot_grid_structure(size, seed)
    return _plot_knapsack_structure(size, seed)


def _plot_grid_structure(k, seed):
    a, B = planted_structure(k, seed)
    crit, distr = (B[:, :2] != 0).any(1), a > 0
    groups = {"kritisch (x1, x2)": (crit, "#1baf7a"), "Ablenker (x3, x4)": (distr, "#eda100"),
              "Pfadkante ohne Feature\n(Treppe oder geteilt)": (~crit & ~distr, "#6b6a64")}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for label, (mask, color) in groups.items():
        for (u, v), on in zip(grid_model(k).arcs, mask):
            if on:
                ax.plot([u % k, v % k], [-(u // k), -(v // k)], color=color, lw=3, solid_capstyle="round")
        ax.plot([], [], color=color, lw=3, label=label)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"planted effects, {k}×{k} (Start oben links, Ziel unten rechts)", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)


KNAPSACK_COLORS = {"kritisch x1": "#1baf7a", "kritisch x2": "#0d7a55", "konstant": "#6b6a64",
                   "zu schwer (x3)": "#eda100", "gratis (x4)": "#b3741a"}


def _plot_knapsack_structure(m, seed):
    """Gewichte der Items in beiden Dimensionen (Ausschnitt um die Kern-Items).

    Die drei Kerngruppen haben dieselben Gewichte und sind nur zur Sichtbarkeit leicht versetzt;
    die "zu schweren" Items liegen außerhalb des Ausschnitts und stehen in der Legende.
    """
    _, _, weights, caps = structure("knapsack", "planted", m, seed)
    groups = knapsack_groups(m)
    offsets = {"kritisch x1": -0.12, "kritisch x2": 0.0, "konstant": 0.12}
    fig, ax = plt.subplots(figsize=(5, 4))
    for (name, color), marker in zip(KNAPSACK_COLORS.items(), ["o", "s", "^", "D", "v"]):
        sel = groups == name
        label = f"{name}: {sel.sum()} Items"
        if name.startswith("zu schwer"):
            label += f"\n(Gewicht {weights[0, sel][0]:g} / {weights[1, sel][0]:g}, außerhalb)"
        elif name.startswith("gratis"):
            label += " (Gewicht 0)"
        off = offsets.get(name, 0.0)
        ax.scatter(weights[0, sel] + off, weights[1, sel], color=color, marker=marker, s=45, label=label,
                   edgecolor="white", linewidth=0.5, zorder=3)
    core = weights[:, np.isin(groups, list(offsets))]
    ax.set_xlim(-0.5, core[0].max() + 1)
    ax.set_ylim(-0.5, core[1].max() + 1)
    ax.set_xlabel("Gewicht Dimension 1")
    ax.set_ylabel("Gewicht Dimension 2")
    ax.set_title(f"planted effects, {m} Items, Kapazität {caps[0]:g} / {caps[1]:g}", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=8)
    ax.grid(alpha=0.3)


def result_path(problem, design):
    """CSV der Vorstudie, z. B. results/prestudy/knapsack_including_planted_effects.csv."""
    return RESULTS / f"{problem}_{'including' if design == 'planted' else 'no'}_planted_effects.csv"


if __name__ == "__main__":
    run().to_csv(result_path("shortest_path", "pyepo"), index=False)
