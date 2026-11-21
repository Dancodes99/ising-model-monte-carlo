"""Run the temperature sweep for several lattice sizes and save results to CSV.

Usage:
    python scripts/run_sweep.py                              # full run, Wolff algorithm
    python scripts/run_sweep.py --algorithm metropolis       # single-spin Metropolis
    python scripts/run_sweep.py --sizes 16 --n-meas 2000     # quick test
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ising import random_lattice, seed, run, observables

ROOT = Path(__file__).resolve().parents[1]


def temperature_grid():
    """Coarse grid over 1.5-3.5, fine grid around the critical point."""
    coarse = np.round(np.arange(1.5, 3.51, 0.1), 3)
    fine = np.round(np.arange(2.10, 2.501, 0.02), 3)
    return np.unique(np.concatenate([coarse, fine]))[::-1]  # high -> low


def sweep(L, temps, n_eq, n_meas, rng_seed, wolff):
    seed(rng_seed)
    s = random_lattice(L, np.random.default_rng(rng_seed))  # hot start at the highest T
    N = L * L
    rows = []
    for T in temps:
        t0 = time.time()
        Es, Ms = run(s, T, n_eq, n_meas, wolff)  # s is reused: annealing from high to low T
        obs = observables(Es, Ms, N, T)
        rows.append({"L": L, "T": T, **obs})
        print(f"L={L:3d}  T={T:.3f}  |m|={obs['m']:.3f}  C={obs['C']:.3f}  "
              f"chi={obs['chi']:.2f}  ({time.time() - t0:.1f}s)")
    return rows[::-1]  # store low -> high


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--algorithm", choices=["wolff", "metropolis"], default="wolff")
    p.add_argument("--n-eq", type=int, default=1000, help="equilibration sweeps per T")
    p.add_argument("--n-meas", type=int, default=10000, help="measurement sweeps per T")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "data")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    temps = temperature_grid()
    for L in args.sizes:
        rows = sweep(L, temps, args.n_eq, args.n_meas, args.seed + L,
                     args.algorithm == "wolff")
        path = args.out / f"ising_L{L}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
