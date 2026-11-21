"""Turn Monte Carlo samples into physical observables with error bars."""

import numpy as np

# Onsager's exact critical temperature for the 2D square-lattice Ising model
T_C_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


def _quantities(Es, Ms, N, T):
    """Energy, |M|, specific heat and susceptibility per spin from samples."""
    e = Es.mean() / N
    m = Ms.mean() / N
    c = Es.var() / (N * T**2)
    chi = (np.mean(Ms**2) - Ms.mean() ** 2) / (N * T)
    return np.array([e, m, c, chi])


def observables(Es, Ms, N, T, n_blocks=20):
    """Means and jackknife error bars of (e, |m|, C, chi).

    Consecutive Monte Carlo samples are correlated, so the samples are
    split into `n_blocks` contiguous blocks and the jackknife is done
    over blocks, which gives honest error estimates.
    """
    Es = np.asarray(Es, dtype=float)
    Ms = np.asarray(Ms, dtype=float)
    full = _quantities(Es, Ms, N, T)

    blocks = np.array_split(np.arange(len(Es)), n_blocks)
    jack = np.empty((n_blocks, 4))
    for b, idx in enumerate(blocks):
        keep = np.ones(len(Es), dtype=bool)
        keep[idx] = False
        jack[b] = _quantities(Es[keep], Ms[keep], N, T)
    err = np.sqrt((n_blocks - 1) / n_blocks * ((jack - jack.mean(axis=0)) ** 2).sum(axis=0))

    names = ["e", "m", "C", "chi"]
    out = {name: full[k] for k, name in enumerate(names)}
    out.update({name + "_err": err[k] for k, name in enumerate(names)})
    return out


def peak_position(T, y, window=3):
    """Locate the maximum of y(T) by fitting a parabola to the points around the peak."""
    T = np.asarray(T)
    y = np.asarray(y)
    k = int(np.argmax(y))
    lo, hi = max(0, k - window), min(len(T), k + window + 1)
    a, b, _ = np.polyfit(T[lo:hi], y[lo:hi], 2)
    if a >= 0:  # not a maximum -> fall back to the grid point
        return float(T[k])
    return float(-b / (2 * a))


def extrapolate_tc(Ls, peaks):
    """Finite-size scaling with nu = 1: T_peak(L) = T_c + a / L.

    Returns (T_c estimate, slope a, standard error of T_c).
    """
    x = 1.0 / np.asarray(Ls, dtype=float)
    y = np.asarray(peaks, dtype=float)
    coeffs, cov = np.polyfit(x, y, 1, cov=True) if len(x) > 2 else (np.polyfit(x, y, 1), None)
    slope, tc = coeffs
    tc_err = float(np.sqrt(cov[1, 1])) if cov is not None else float("nan")
    return float(tc), float(slope), tc_err
