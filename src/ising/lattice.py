"""Lattice helpers for the 2D Ising model (J = 1, k_B = 1, no external field).

Spins live on an L x L square lattice with periodic boundary conditions,
so every spin has exactly four nearest neighbours.
"""

import numpy as np
from numba import njit


def random_lattice(L, rng=None):
    """Return an L x L lattice of random +1/-1 spins (hot start)."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L))


def ordered_lattice(L, value=1):
    """Return an L x L lattice with every spin equal to `value` (cold start)."""
    return np.full((L, L), value, dtype=np.int64)


@njit(cache=True)
def energy(s):
    """Total energy E = -sum_<ij> s_i s_j.

    Each bond is counted exactly once by only looking at the right and
    down neighbour of every site.
    """
    L = s.shape[0]
    E = 0
    for i in range(L):
        for j in range(L):
            E -= s[i, j] * (s[(i + 1) % L, j] + s[i, (j + 1) % L])
    return E


@njit(cache=True)
def magnetisation(s):
    """Total magnetisation M = sum_i s_i."""
    return s.sum()


@njit(cache=True)
def delta_energy(s, i, j):
    """Energy change if spin (i, j) were flipped: dE = 2 s_ij * (sum of 4 neighbours)."""
    L = s.shape[0]
    nb = (s[(i + 1) % L, j] + s[(i - 1) % L, j]
          + s[i, (j + 1) % L] + s[i, (j - 1) % L])
    return 2 * s[i, j] * nb
