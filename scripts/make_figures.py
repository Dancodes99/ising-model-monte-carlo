"""Make the README figures and estimate T_c from the saved sweep data.

Usage:
    python scripts/make_figures.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ising import (T_C_EXACT, random_lattice, seed, run,
                   peak_position, extrapolate_tc)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"

# Colours: one fixed hue per lattice size; text stays neutral
COLOURS = {16: "#2a78d6", 32: "#eb6834", 64: "#1baf7a", 128: "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "lines.linewidth": 2,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def load(folder=DATA):
    data = {}
    for path in sorted(folder.glob("ising_L*.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        L = int(rows[0]["L"])
        data[L] = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "L"}
    return dict(sorted(data.items()))


def tc_line(ax, label=None):
    ax.axvline(T_C_EXACT, color=INK2, linestyle="--", linewidth=1.2, label=label)


def plot_observables(data):
    panels = [("m", "Magnetisation per spin  |m|"), ("e", "Energy per spin  e"),
              ("C", "Specific heat per spin  C"), ("chi", "Susceptibility per spin  χ")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, label) in zip(axes.flat, panels):
        for L, d in data.items():
            ax.errorbar(d["T"], d[key], yerr=d[key + "_err"], color=COLOURS.get(L),
                        marker="o", markersize=4, linewidth=1.5, capsize=2,
                        label=f"L = {L}")
        tc_line(ax, f"exact T$_c$ = {T_C_EXACT:.3f}" if key == "m" else None)
        ax.set_ylabel(label)
    for ax in axes[1]:
        ax.set_xlabel("Temperature  T  (units of J / k$_B$)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("2D Ising model: phase transition from Wolff cluster Monte Carlo", fontsize=14)
    fig.savefig(FIGS / "observables_vs_T.png")
    plt.close(fig)


def plot_finite_size(data):
    """Peak positions of chi and C vs 1/L, extrapolated to L -> infinity."""
    Ls = list(data)
    x = 1.0 / np.array(Ls)
    xs = np.linspace(0, x.max() * 1.08, 50)
    fits = {}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for key, name, colour in [("chi", "susceptibility \u03c7", "#2a78d6"),
                              ("C", "specific heat C", "#eb6834")]:
        peaks = [peak_position(d["T"], d[key]) for d in data.values()]
        tc, slope, tc_err = extrapolate_tc(Ls, peaks)
        fits[key] = {"peaks": peaks, "tc": tc, "tc_err": tc_err}
        ax.plot(xs, tc + slope * xs, color=colour, linewidth=1.5, alpha=0.8)
        ax.plot(x, peaks, "o", markersize=8, color=colour,
                markeredgecolor="#fcfcfb", markeredgewidth=2,
                label=f"{name} peaks \u2192 T$_c$ = {tc:.4f}")
    ax.plot(0, T_C_EXACT, "*", markersize=15, color=INK, label=f"Onsager exact T$_c$ = {T_C_EXACT:.4f}")
    for L, xp in zip(Ls, x):
        ax.annotate(f"L = {L}", (xp, fits["chi"]["peaks"][Ls.index(L)]),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=9, color=INK2)
    ax.set_xlim(-0.003, x.max() * 1.12)
    ax.set_xlabel("1 / L")
    ax.set_ylabel("Temperature of the peak  (J / k$_B$)")
    ax.set_title("Finite-size scaling: peak positions extrapolated to L \u2192 \u221e")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(FIGS / "finite_size_scaling.png")
    plt.close(fig)
    return Ls, fits


def plot_algorithm_comparison(wolff_data, L=64):
    """Metropolis vs Wolff near T_c for one lattice size (critical slowing down)."""
    metro = load(DATA / "metropolis")
    if L not in metro or L not in wolff_data:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, d, colour in [("Metropolis (50,000 sweeps)", metro[L], "#2a78d6"),
                             ("Wolff (10,000 sweeps)", wolff_data[L], "#eb6834")]:
        mask = (d["T"] >= 2.1) & (d["T"] <= 2.5)
        ax.errorbar(d["T"][mask], d["chi"][mask], yerr=d["chi_err"][mask], color=colour,
                    marker="o", markersize=5, linewidth=1.5, capsize=3, label=label)
    tc_line(ax)
    ax.set_xlabel("Temperature  T  (units of J / k$_B$)")
    ax.set_ylabel("Susceptibility per spin  \u03c7")
    ax.set_title(f"Critical slowing down, L = {L}: Metropolis vs Wolff")
    ax.legend(frameon=False)
    fig.savefig(FIGS / "metropolis_vs_wolff.png")
    plt.close(fig)


def plot_snapshots(L=128, sweeps=500):
    temps = [1.8, T_C_EXACT, 3.2]
    titles = ["T = 1.8 (ordered)", f"T = {T_C_EXACT:.2f} (critical)", "T = 3.2 (disordered)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, T, title in zip(axes, temps, titles):
        seed(7)
        s = random_lattice(L, np.random.default_rng(7))
        run(s, T, sweeps, 1, True)  # Wolff updates: equilibrates quickly at every T
        ax.imshow(s, cmap="Greys", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Spin configurations on a {L}×{L} lattice (black = up, white = down)")
    fig.savefig(FIGS / "lattice_snapshots.png")
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    data = load()
    if not data:
        raise SystemExit("No data found. Run scripts/run_sweep.py first.")
    plot_observables(data)
    Ls, fits = plot_finite_size(data)
    plot_algorithm_comparison(data)
    plot_snapshots()

    def dev(tc):
        return round(100 * abs(tc - T_C_EXACT) / T_C_EXACT, 2)

    results = {
        "T_c_exact": round(T_C_EXACT, 4),
        "lattice_sizes": Ls,
        "from_susceptibility": {
            "T_c": round(fits["chi"]["tc"], 4), "fit_error": round(fits["chi"]["tc_err"], 4),
            "deviation_percent": dev(fits["chi"]["tc"]),
            "peaks_by_L": {str(L): round(p, 4) for L, p in zip(Ls, fits["chi"]["peaks"])}},
        "from_specific_heat": {
            "T_c": round(fits["C"]["tc"], 4), "fit_error": round(fits["C"]["tc_err"], 4),
            "deviation_percent": dev(fits["C"]["tc"]),
            "peaks_by_L": {str(L): round(p, 4) for L, p in zip(Ls, fits["C"]["peaks"])}},
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
