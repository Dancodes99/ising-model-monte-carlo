# Study guide: understand every part before you present it

Go through this before you put the project on your CV. In an interview, or if a professor looks at the repo, you should be able to answer every question below in your own words.

## 1. The physics

- **What is the Ising model?** Spins ±1 on a lattice; neighbours that point the same way lower the energy by J. It is the simplest model of a ferromagnet.
- **Why is there a phase transition?** Energy favours order, entropy favours disorder. Below T<sub>c</sub> energy wins (spontaneous magnetisation); above T<sub>c</sub> entropy wins.
- **Why does the 2D model matter?** It is one of the few interacting models solved exactly (Onsager, 1944), so a simulation can be checked against an exact T<sub>c</sub> = 2/ln(1+√2) ≈ 2.2692.
- **Why do C and χ peak at T<sub>c</sub>?** They are fluctuations: C ∝ variance of the energy, χ ∝ variance of the magnetisation. Fluctuations are largest at the critical point.
- **Why are the peaks at higher T for small lattices, and why do they sharpen with L?** A finite lattice cannot have an infinite correlation length, so the transition is smeared and shifted. Finite-size scaling: T<sub>peak</sub>(L) − T<sub>c</sub> ∝ L<sup>−1/ν</sup>, with ν = 1 in 2D.

## 2. The algorithms

- **Metropolis:** pick a spin, compute ΔE = 2 s<sub>i</sub> Σ neighbours, flip if ΔE ≤ 0, else flip with probability e<sup>−ΔE/T</sup>. Be ready to explain **detailed balance**: P(a)·W(a→b) = P(b)·W(b→a), with P the Boltzmann weight.
- **Why can ΔE only be −8, −4, 0, 4, 8?** Each spin has four neighbours, so Σ neighbours ∈ {−4, −2, 0, 2, 4}. That is why `metropolis_sweep` precomputes e<sup>−4/T</sup> and e<sup>−8/T</sup>.
- **Critical slowing down:** near T<sub>c</sub>, large domains form and single-spin flips change them very slowly, so successive samples are strongly correlated. See `figures/metropolis_vs_wolff.png`.
- **Wolff:** grow a cluster of aligned spins, adding each aligned neighbour with probability p = 1 − e<sup>−2J/T</sup>, then flip the whole cluster. This choice of p makes the flip always accepted while keeping detailed balance.
- **The bug the exact test caught:** an early version stopped each Wolff "sweep" once about L² spins had flipped. Because the number of clusters then depends on the current state, measurements were biased (about 20% too much order at T = 3 on a 4×4 lattice). The fix is a fixed number of clusters per sweep, chosen during equilibration. This is a good story for an interview: it shows why validation against exact results matters.

## 3. The code (read it in this order)

1. `src/ising/lattice.py`: periodic boundaries with `% L`; why `energy()` only counts the right and down neighbour (each bond once).
2. `src/ising/updates.py`: `metropolis_sweep`, `wolff_cluster` (stack-based cluster growth; flipping as you add marks sites as visited), `run`.
3. `src/ising/analysis.py`: observables, jackknife error bars over blocks, parabola fit for the peak, linear fit in 1/L.
4. `scripts/run_sweep.py`: why temperatures go from high to low (annealing).
5. `tests/test_ising.py`: especially `test_matches_exact_enumeration` (sums all 2<sup>16</sup> states of a 4×4 lattice).

## 4. Statistics

- **Equilibration:** the first sweeps depend on the starting state and are discarded.
- **Correlated samples:** naive error bars (σ/√n) are too small. Block jackknife: split the samples into 20 blocks, leave one block out at a time, and use the spread of the results.
- **Why the fit error is smaller than the real deviation:** systematic effects (corrections to 1/L scaling, the 0.02 temperature grid) are not in the statistical error. Saying this honestly is a strength.

## 5. Questions to practise

1. Derive ΔE for flipping one spin.
2. Show that Metropolis satisfies detailed balance.
3. Why is Numba used, and what would the code cost in pure Python?
4. Why does χ use ⟨|M|⟩ rather than ⟨M⟩ on a finite lattice?
5. What would you do next? (Critical exponents from peak heights: χ<sub>max</sub> ∝ L<sup>γ/ν</sup> with γ/ν = 7/4; |m|(T<sub>c</sub>) ∝ L<sup>−β/ν</sup> with β/ν = 1/8.)

## 6. Practise by changing things

The best way to own this code is to modify it yourself:

- Re-run with a different seed and check the results stay within the error bars.
- Add L = 256 for the Wolff run and see whether the extrapolation improves.
- Implement the critical-exponent fit from question 5 as your own extension.
