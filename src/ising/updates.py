"""Monte Carlo updates for the 2D Ising model: Metropolis and Wolff."""

import numpy as np
from numba import njit

from .lattice import energy, magnetisation, delta_energy


@njit(cache=True)
def seed(value):
    """Seed Numba's internal random generator (separate from NumPy's)."""
    np.random.seed(value)


@njit(cache=True)
def metropolis_sweep(s, T):
    """One Monte Carlo sweep = L*L attempted single-spin flips.

    A flip is accepted if it lowers the energy, otherwise with
    probability exp(-dE / T). This satisfies detailed balance with
    respect to the Boltzmann distribution.
    """
    L = s.shape[0]
    # dE can only be -8, -4, 0, 4, 8 -> precompute the two acceptance probabilities
    p4 = np.exp(-4.0 / T)
    p8 = np.exp(-8.0 / T)
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        dE = delta_energy(s, i, j)
        if dE <= 0:
            s[i, j] = -s[i, j]
        else:
            p = p4 if dE == 4 else p8
            if np.random.random() < p:
                s[i, j] = -s[i, j]


@njit(cache=True)
def wolff_cluster(s, T):
    """Grow and flip one Wolff cluster; returns the cluster size.

    Starting from a random seed spin, aligned neighbours are added with
    probability p = 1 - exp(-2/T). Flipping the whole cluster is always
    accepted, which removes the critical slowing down that makes
    single-spin Metropolis slow near T_c.
    """
    L = s.shape[0]
    p_add = 1.0 - np.exp(-2.0 / T)
    i0 = np.random.randint(L)
    j0 = np.random.randint(L)
    spin = s[i0, j0]
    stack = np.empty((L * L, 2), dtype=np.int64)
    stack[0, 0] = i0
    stack[0, 1] = j0
    top = 1
    s[i0, j0] = -spin  # flip as we add, which also marks the site as visited
    size = 1
    while top > 0:
        top -= 1
        i = stack[top, 0]
        j = stack[top, 1]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = (i + di) % L
            nj = (j + dj) % L
            if s[ni, nj] == spin and np.random.random() < p_add:
                s[ni, nj] = -spin
                stack[top, 0] = ni
                stack[top, 1] = nj
                top += 1
                size += 1
    return size


@njit(cache=True)
def wolff_sweep(s, T, n_clusters):
    """Flip a fixed number of Wolff clusters; returns the total number of spins flipped.

    The number of clusters must NOT depend on the current state (for
    example "flip clusters until N spins have flipped"): stopping at a
    state-dependent time biases the measurements.
    """
    flipped = 0
    for _ in range(n_clusters):
        flipped += wolff_cluster(s, T)
    return flipped


@njit(cache=True)
def run(s, T, n_eq, n_meas, wolff=False):
    """Equilibrate for n_eq sweeps, then record E and |M| after each of n_meas sweeps.

    Uses single-spin Metropolis by default, or Wolff cluster updates
    with wolff=True. For Wolff, one "sweep" is a fixed number of cluster
    flips chosen during equilibration so that about L*L spins are
    flipped on average. The lattice `s` is updated in place, so it can
    be reused as the starting point for the next temperature.
    """
    N = s.shape[0] * s.shape[1]
    n_clusters = 1
    if wolff:
        total = 0
        count = 0
        for _ in range(n_eq):
            total += wolff_sweep(s, T, n_clusters)
            count += n_clusters
            # adapt during equilibration only: aim for ~N flipped spins per sweep
            n_clusters = max(1, int(np.ceil(N * count / total)))
    else:
        for _ in range(n_eq):
            metropolis_sweep(s, T)
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)
    for k in range(n_meas):
        if wolff:
            wolff_sweep(s, T, n_clusters)  # fixed number of clusters from here on
        else:
            metropolis_sweep(s, T)
        Es[k] = energy(s)
        Ms[k] = abs(magnetisation(s))
    return Es, Ms
