"""Unit and physics tests for the Ising simulation."""

import numpy as np
import pytest

from ising import (ordered_lattice, random_lattice, energy, magnetisation, delta_energy,
                   seed, metropolis_sweep, wolff_cluster, run, observables, peak_position,
                   extrapolate_tc, T_C_EXACT)


@pytest.mark.parametrize("L", [4, 8, 16])
def test_all_up_ground_state(L):
    s = ordered_lattice(L)
    assert energy(s) == -2 * L * L          # 2 bonds per site, each -1
    assert magnetisation(s) == L * L


@pytest.mark.parametrize("L", [4, 8, 16])
def test_checkerboard_is_highest_energy(L):
    i, j = np.indices((L, L))
    s = np.where((i + j) % 2 == 0, 1, -1).astype(np.int64)
    assert energy(s) == 2 * L * L
    assert magnetisation(s) == 0


def test_delta_energy_matches_direct_difference():
    rng = np.random.default_rng(0)
    s = random_lattice(10, rng)
    for _ in range(50):
        i, j = rng.integers(10, size=2)
        before = energy(s)
        dE = delta_energy(s, i, j)
        s[i, j] *= -1
        assert energy(s) - before == dE


def test_spins_stay_plus_minus_one():
    seed(1)
    s = random_lattice(16, np.random.default_rng(1))
    for _ in range(20):
        metropolis_sweep(s, 2.3)
    assert set(np.unique(s)) <= {-1, 1}


def test_low_temperature_stays_ordered():
    seed(2)
    s = ordered_lattice(16)
    for _ in range(200):
        metropolis_sweep(s, 0.5)
    assert abs(magnetisation(s)) / s.size > 0.99


def test_high_temperature_is_disordered():
    seed(3)
    s = ordered_lattice(32)
    _, Ms = run(s, 10.0, 200, 500)
    assert Ms.mean() / s.size < 0.1


def test_known_energy_at_low_temperature():
    # At T = 1.0 nearly all spins align: energy per spin is close to -2
    seed(4)
    s = ordered_lattice(16)
    Es, _ = run(s, 1.0, 200, 2000)
    assert Es.mean() / s.size == pytest.approx(-2.0, abs=0.02)


def test_observables_and_errors_are_finite():
    seed(5)
    s = random_lattice(16, np.random.default_rng(5))
    Es, Ms = run(s, 2.3, 200, 2000)
    obs = observables(Es, Ms, s.size, 2.3)
    for key in ["e", "m", "C", "chi"]:
        assert np.isfinite(obs[key]) and obs[key + "_err"] >= 0


def test_peak_position_of_parabola():
    T = np.linspace(2.0, 2.6, 31)
    y = -(T - 2.31) ** 2
    assert peak_position(T, y) == pytest.approx(2.31, abs=1e-6)


def test_extrapolation_recovers_intercept():
    Ls = [16, 32, 64]
    peaks = [T_C_EXACT + 0.5 / L for L in Ls]
    tc, slope, _ = extrapolate_tc(Ls, peaks)
    assert tc == pytest.approx(T_C_EXACT, abs=1e-9)
    assert slope == pytest.approx(0.5, abs=1e-9)


def test_wolff_cluster_flips_aligned_spins_only():
    # At very low T, p_add -> 1, so one cluster flips the whole ordered lattice
    seed(6)
    s = ordered_lattice(8)
    size = wolff_cluster(s, 0.01)
    assert size == 64
    assert magnetisation(s) == -64


def _exact_4x4(T):
    """Exact <e> and <|m|> per spin for a 4x4 lattice by summing all 2^16 states."""
    Es, Ms = [], []
    for bits in range(1 << 16):
        s = np.array([1 if (bits >> k) & 1 else -1 for k in range(16)],
                     dtype=np.int64).reshape(4, 4)
        Es.append(energy(s))
        Ms.append(abs(s.sum()))
    Es, Ms = np.array(Es, float), np.array(Ms, float)
    w = np.exp(-(Es - Es.min()) / T)
    w /= w.sum()
    return (w * Es).sum() / 16, (w * Ms).sum() / 16


@pytest.mark.parametrize("wolff", [False, True], ids=["metropolis", "wolff"])
def test_matches_exact_enumeration(wolff):
    """Both algorithms must reproduce the exact Boltzmann averages (4x4, T = 3)."""
    T = 3.0
    e_exact, m_exact = _exact_4x4(T)
    seed(11)
    s = random_lattice(4, np.random.default_rng(11))
    Es, Ms = run(s, T, 1000, 100000, wolff)
    assert Es.mean() / 16 == pytest.approx(e_exact, abs=0.015)
    assert Ms.mean() / 16 == pytest.approx(m_exact, abs=0.01)
