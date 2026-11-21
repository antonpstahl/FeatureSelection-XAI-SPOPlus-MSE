"""Vorstudie: lineares Modell mit MSE vs. SPO+ auf Shortest-Path-Daten im k×k-Gitter.

Beide Generatoren haben die Form c = a + ((Bx/√p + 3)^deg + 1) / 3.5^deg · ε:
- "pyepo":   PyEPO-Standardgenerator (a = 0, B zufällig Bernoulli),
- "planted": a und B bewusst gesetzt (kritische, Ablenker- und Rauschfeatures).

Je Einstellung (deg, noise, n) und Refit werden beide Modelle trainiert und per
Permutation Feature Importance (PFI) bewertet: ΔMSE (roh und skalenbereinigt) und
Δ normierter Regret je Feature. Die wahre Kostenfunktion E[c|x] dient als Referenz
(in den Ergebnissen loss = "oracle").
"""

from functools import cache
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyepo
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

P = 5  # Anzahl Features


@cache
def grid_model(k):
    """Gurobi-Modell für den kürzesten Weg im k×k-Gitter (Knoten = Zeile·k + Spalte)."""
    return pyepo.model.grb.shortestPathModel((k, k))


def solve(om, C):
    """Optimale Lösungen W und Zielwerte z für jede Zeile von C."""
    W = np.empty_like(C)
    for i, ci in enumerate(C):
        om.setObj(ci)
        W[i] = om.solve()[0]
    return W, (W * C).sum(1)


# --- Generatoren -------------------------------------------------------------

def cost_mean(X, a, B, deg):
    """E[c|x]; das multiplikative Rauschen ε hat Erwartungswert 1."""
    return a + ((X @ B.T / np.sqrt(X.shape[1]) + 3) ** deg + 1) / 3.5**deg


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
    """Grundkosten a und Matrix B mit planted effects.

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


def make_data(design, k, N, deg, noise, seed):
    """N Beobachtungen (X, C) und die wahre Kostenfunktion E[c|x] des Generators."""
    d = grid_model(k).num_cost
    if design == "pyepo":
        a, B = np.zeros(d), np.random.RandomState(seed).binomial(1, 0.5, (d, P))  # = erster Zug in genData
        X, C = pyepo.data.shortestpath.genData(N, P, (k, k), deg, noise, seed)
    else:
        a, B = planted_structure(k, seed)
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(N, P))
        eps = rng.uniform(1 - noise, 1 + noise, (N, d))
        C = (a + cost_mean(X, 0, B, deg) * eps).astype(np.float32)
    return X, C, lambda X: cost_mean(X, a, B, deg)


def path_shares(k, deg, noise, N=2000, seed=0):
    """Prüfung des planted-Generators: wie oft ist welcher Pfad optimal, werden Ablenker-Kanten genutzt?"""
    _, C, _ = make_data("planted", k, N, deg, noise, seed)
    W, _ = solve(grid_model(k), C)
    shares = {name: (np.abs(W - path_vector(k, mv)).max(1) < 0.5).mean()
              for name, (mv, _) in planted_paths(k).items()}
    shares["andere"] = 1 - sum(shares.values())
    shares["Ablenker genutzt"] = (W[:, planted_structure(k, seed)[0] > 0].sum(1) > 0).mean()
    return pd.Series(shares)


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


def scale_fit(C_hat, C):
    """Entscheidungsneutrale Anpassung a + s*ĉ (im Gitter hat jeder Pfad gleich viele Kanten)."""
    s, a = np.polyfit(C_hat.ravel(), C.ravel(), 1)
    return (a, s) if s > 0 else (0.0, 1.0)  # s <= 0 wäre nicht entscheidungsneutral -> roh


def scores(om, C_hat, C, z, a, s):
    """MSE roh, MSE skalenbereinigt, normierter Regret (wie pyepo.metric.regret)."""
    W, _ = solve(om, C_hat)
    regret = ((W * C).sum(1) - z).sum() / np.abs(z).sum()
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


def run(design="pyepo", k=5, degs=(1, 2, 4, 6), noises=(0.0, 0.5), ns=(100, 1000), m=5,
        n_test=1000, K=10, steps=3000, seed=0):
    """Ein Datenaufruf je (deg, noise): Test + m disjunkte Trainingsblöcke mit gleicher Struktur (a, B)."""
    om = grid_model(k)
    rows = []
    for deg, noise in product(degs, noises):
        print(f"{design} {k}×{k}: deg={deg}, noise={noise}", flush=True)
        X, C, truth = make_data(design, k, n_test + m * max(ns), deg, noise, seed)
        Xte, Cte = X[:n_test], C[:n_test]
        _, zte = solve(om, Cte)
        setting = dict(grid=k, deg=deg, noise=noise)

        for r in pfi(om, truth, Xte, Cte, zte, 0.0, 1.0, K, seed):
            rows.append(setting | dict(n=np.nan, refit=-1, loss="oracle") | r)

        for n, d in product(ns, range(m)):
            lo = n_test + d * max(ns)
            Xtr, Ctr = X[lo:lo + n], C[lo:lo + n]
            for loss in ("mse", "spo+"):
                predict = train(om, Xtr, Ctr, loss, steps, seed=d)
                a, s = scale_fit(predict(Xtr), Ctr)
                for r in pfi(om, predict, Xte, Cte, zte, a, s, K, seed=d):
                    rows.append(setting | dict(n=n, refit=d, loss=loss) | r)
    return pd.DataFrame(rows)


# --- Auswertung --------------------------------------------------------------

COLORS = {"mse": "#2a78d6", "spo+": "#eb6834", "oracle": "#6b6a64"}
LABELS = {"mse": "MSE", "spo+": "SPO+", "oracle": "Orakel"}
MEASURES = {"d_mse": "ΔMSE roh", "d_mse_adj": "ΔMSE skalenbereinigt", "d_regret": "Δ normierter Regret"}


def ci95(s):
    """Halbe Breite des 95-%-t-Intervalls über die Refits."""
    return stats.t.ppf(0.975, len(s) - 1) * s.sem() if len(s) > 1 else 0.0


def performance(df):
    """Eine Zeile je Modell (Basiswerte stehen in jeder Feature-Zeile)."""
    return df.drop_duplicates(["deg", "noise", "n", "refit", "loss"])


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


def plot_structure(k, seed=0):
    """Gitter mit den Kantengruppen des planted-Generators."""
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


if __name__ == "__main__":
    run().to_csv("results/prestudy_no_planted_effects.csv", index=False)
