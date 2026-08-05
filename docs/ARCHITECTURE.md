# ARCHITECTURE — `topological-vqpu`

**GPU-accelerated tensor-network and BdG simulation of an engineered
topological-superconductor lattice Hamiltonian, with a companion ballistic
phonon-transport solver for the thermal substrate.**

Revision: drafted against *Solid-State 0.3 K Cuprate–TI Topological Processor,
Rev. 2.1* (6 July 2026), hereafter **Rev 2.1**, which is the source of truth for
every physical parameter in §2.

---

## 0. Scope — what this repository is, and is not

This is a **classical simulator**. It runs float64 linear algebra on CPUs and
GPUs. It is not a quantum processor, virtual or otherwise, and it does not
"emulate" one in any sense that would survive a technical reviewer.

Three statements bound every claim made by this codebase:

1. **We simulate a Hamiltonian, we do not measure a device.** Every number this
   repo produces is conditional on the model in §2 and on the pairing gap
   Δ = 2 meV, which Rev 2.1 §9.2 correctly identifies as the single unverified
   load-bearing assumption of the whole architecture.
2. **Matrix product states work precisely because the states involved are not
   very entangled.** Anything this repo can simulate is, by construction, in the
   regime where quantum advantage does not exist. Classical simulation maps
   where quantum hardware *is not* needed. See `docs/tensor_networks.md` §7.
3. **Topological protection reduces the physical error rate. It does not
   abolish error correction.** Rev 2.1 §7.2 states this correctly and the
   resource estimator in `tvqpu.fte` enforces it.

The repository is open-source and public. Nothing in it should assert a
capability that a program manager could disprove in thirty seconds.

---

**Implementation status.** Modules 1, 3 and the memory budgeter are
implemented and validated. Modules 2, 4 and 5 are specified here but not yet
written — see the status table in [README.md](README.md). Where this document
describes a module in the present tense, read it as the contract that module
must satisfy, not as a claim that it exists.

---

## 1. Module map

```
                    ┌──────────────────────────────────────┐
                    │  Module 1  tvqpu.lattice             │
   geometry ───────►│  geometry → Hamiltonian              │
   (Rev 2.1 §2)     │    ├─ .to_bdg_dense()  → 4N×4N BdG   │───► exact diag
                    │    └─ .to_mpo()        → MPO         │───► Module 2
                    └──────────────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │  Module 2  tvqpu.dmrg                │
                    │  DMRG / TEBD over MPS (quimb)        │───► Module 3
                    │  ONLY for interacting extensions     │
                    └────────────────┬─────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │  Module 3  tvqpu.kernels             │
                    │  Triton fused two-site update        │
                    │  + telemetry counters                │
                    └──────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  Module 4  tvqpu.substrate   — SEPARATE PHYSICS, NO TENSOR   │
   │  NETWORKS.  Ballistic phonon focusing (Christoffel/BTE).     │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  (planned) tvqpu.fte  — surface-code resource estimator      │
   │            NOT IMPLEMENTED, not present in this repository    │
   └──────────────────────────────────────────────────────────────┘
```

**Division of labour between Modules 1 and 2 (resolves flag #1 from onboarding):**

| Question | Method | Why |
|---|---|---|
| Δ_top, ξ, δE, phase boundary, disorder ensembles | **BdG exact diagonalization** (Module 1) | The model is quadratic. Dense `eigh` on 4N×4N is exact, costs seconds, and is what produced Rev 2.1 Figs. 1–2 and the 1.05 meV gap. |
| Effect of Coulomb repulsion in the channel | **DMRG over an MPO** (Module 2) | Interactions make the model non-quadratic. BdG is mean-field by construction and cannot answer this. Bond dimension is a real cost here. |

**Δ_top = 1.05 meV is a BdG result, not a DMRG result.** This attribution has
been made incorrectly more than once in the project's history; the docstring of
`tvqpu.lattice.MajoranaChannel` restates it, and `tests/test_lattice.py` pins
the value.

---

## 2. The physical node (Rev 2.1, verbatim parameter set)

Gate-defined quasi-1D channel on a Bi₂Se₃ surface, proximity-coupled to a
45°-twisted BSCCO bilayer with emergent chiral *d + id′* pairing. Braiding via
T-junction networks of gate-defined channels; **no vortex is created or moved**
(Rev 2.1 §3.2 analyses and rejects Fu–Kane vortex-core operation on mini-gap
grounds: δ ≃ Δ_top²/E_F = 11–22 µeV, only 0.4–0.9 k_BT at 0.3 K).

### 2.1 Tight-binding BdG Hamiltonian

N sites, lattice constant a = 10 nm, 4-component Nambu ⊗ spin basis
Ψ_j = (c_j↑, c_j↓, c†_j↑, c†_j↓)ᵀ:

```
H_on  = (2t − µ)(τ_z ⊗ σ_0) + Δ(τ_x ⊗ σ_0) + V_z(τ_0 ⊗ σ_z)
H_hop = −t(τ_z ⊗ σ_0) − iα(τ_z ⊗ σ_y)
```

Bulk: `H(k) = [2t(1−cos ka) − µ](τ_z⊗σ_0) + 2α sin(ka)(τ_z⊗σ_y) + Δ(τ_x⊗σ_0) + V_z(τ_0⊗σ_z)`

**The `+2t` onsite offset is load-bearing.** It puts µ = 0 at the band bottom.
Rev 2.0 omitted it, placing the Fermi level mid-band, so the simulated wire
never entered the topological phase and the true critical field was
√((2t)² + Δ²) ≈ 20 meV rather than 2 meV (Rev 2.1 Appendix B.1). Module 1
carries a regression test for exactly this.

Particle–hole symmetry `P H(k) P⁻¹ = −H(−k)` with `P = (τ_y ⊗ σ_y)K`, `P² = +1`
⇒ Altland–Zirnbauer class **D**, ℤ₂ Pfaffian invariant, `Q = −1` for
`V_z > √(µ² + Δ²)` (Rev 2.1 Appendix A).

### 2.2 Parameter set — `tvqpu.lattice.REV21`

| Symbol | Value | Note |
|---|---|---|
| t | 20 meV | ℏ²/2m\*a², m\* ≈ 0.019 mₑ |
| α | 10 meV | effective spin–orbit scale |
| Δ | 2.0 meV | **design assumption, never measured** |
| V_z | 3.0 meV | operating point; B∥ = 4.15 T at g ≈ 25 |
| µ | 0 | band bottom; E_F ≲ 8 meV from Dirac point |
| a | 10 nm | |
| T | 0.3 K | k_BT = 25.9 µeV, Δ_top/k_BT = 40.6 |

### 2.3 Targets the simulator must reproduce

These are the acceptance gates for Module 1. They come from Rev 2.1 §2 and
Table 2, and an independent reimplementation reproduced them earlier in the
project's history (values in parentheses).

| Target | Rev 2.1 | This repo | Gate |
|---|---|---|---|
| V_z,crit at µ = 0 | 2.0 meV | **2.000** | exact, `√(µ²+Δ²)` |
| Δ_top at V_z = 3 meV | 1.05 meV | **1.000** | closed form, see below |
| ξ | 21 sites ≈ 210 nm | **19.1 sites (191 nm)** | within 15% |
| δE at N = 200 (L = 2 µm) | 57 neV | **57.0 neV** | within 10% |
| δE at L ≥ 6.3 µm = 30ξ | ≲ 0.1 feV | extrapolates | order of magnitude |

**Δ_top has a closed form at the µ = 0 sweet spot.** The k = 0 bulk
eigenvalues are `±√(µ²+Δ²) + V_z·s` for spin sector `s = ±1`, so the lowest
positive excitation is exactly `|V_z − √(µ²+Δ²)|`, and a Brillouin-zone scan
confirms k = 0 is the global minimum at the operating point. At V_z = 3 meV,
µ = 0, Δ = 2 meV that is **1.000 meV exactly**. Rev 2.1's 1.05 meV is read
off the finite N = 80 chain of Fig. 1, where level discretization lifts it
~5%. Both are "the gap"; only one is exact, and a reviewer who rederives it
will get 1.000. Quoting the closed form is stronger than quoting the numeric,
and it makes the V_z-dependence of every downstream budget explicit.

**Figure-1 caveat worth keeping in the caption.** Rev 2.1 Fig. 1 uses N = 80
(L = 0.8 µm), where the end-mode splitting is ~92 µeV — invisible on a ±8 meV
axis but not zero. The figure is illustrative, not at the production operating
point (L ≥ 6.3 µm). A reviewer who replots it will notice; better to state it.

### 2.4 What DMRG adds, and in what units

The interacting extension is the spinless effective model (Zeeman-polarised
band), Jordan–Wigner mapped:

```
H = −t Σ (c†_j c_{j+1} + h.c.) + Δ Σ (c_j c_{j+1} + h.c.)
    − µ Σ (n_j − ½) + V Σ n_j n_{j+1}
```

Under JW with `n_j − ½ → Z_j/2`:

```
H = −(µ/2) Σ Z_j − ((t−Δ)/2) Σ X_j X_{j+1} − ((t+Δ)/2) Σ Y_j Y_{j+1}
    + (V/4) Σ Z_j Z_{j+1} + (V/4) Σ (Z_j + Z_{j+1}) + const
```

At t = Δ this is the transverse-field Ising chain with order parameter
⟨Y_i Y_j⟩ and exact critical point µ_c = 2t. Established results, reproduced
earlier in the project:

- DMRG vs exact diagonalization at L = 12: agreement to **2×10⁻¹¹**, including
  with V ≠ 0.
- V = 0 boundary reproduces µ_c = 2t; mid-chain entanglement entropy peaks at
  µ ≈ 1.9 (finite-size signature of the critical point).
- Repulsive V **widens** the topological µ-window: µ_c ≈ 2.0 (V=0) → ≈ 3.0
  (V=+1), i.e. roughly 50%.

**Two hard constraints on how this may be quoted.**

1. **V and µ here are dimensionless, in units of t. There are no meV in this
   calculation.** Converting V to a screened Coulomb strength in gated Bi₂Se₃
   requires a screening calculation that has not been done, so we cannot say
   where on the V-axis the real device sits.
2. **The "50% wider window" is a property of the effective model, not of the
   device.** The defensible sentence is: *going beyond mean-field BdG does not
   threaten the phase; in the repulsive regime the µ-window is more forgiving
   than the BdG phase diagram implies.* That answers a reviewer asking "your
   phase diagram is mean-field — what do interactions do?" without overclaiming.

This is also not novel. It is characterised in the literature
([arXiv:1511.02817](https://arxiv.org/abs/1511.02817),
[arXiv:2402.12897](https://arxiv.org/abs/2402.12897)), and Stoudenmire, Alicea,
Starykh & Fisher, *PRB* **84**, 014503 (2011). We reproduce it as a toolchain
check and cite them.

### 2.4b The Go/No-Go thresholds, derived rather than asserted

The Stage A slide states a kill-floor and a decision band on the *pairing* gap
Δ, which is the quantity Milestone 1 measures:

> Hard kill-floor: Δ ≥ 0.6 meV — below it, T_max < 300 mK.
> Go/No-Go: ≥ 1.4 meV = GO · 0.6–1.4 meV = redesign (degraded) · < 0.6 = No-Go.

But every protection budget depends on **Δ_top**, not Δ, and the map Δ → Δ_top
is exactly what Module 1 computes. Running it (`--gap`, sweeping Δ) gives a
result that does not have the shape the band assumes.

**At fixed V_z = 3 meV**, Δ_top is a *tent function* of Δ, because two minima
compete:

| branch | location | value | trend in Δ |
|---|---|---|---|
| Zeeman-limited | k = 0 | `V_z − √(µ²+Δ²)` | **falling** |
| pairing-limited | ka ≈ 0.941 | ≈ 0.983 Δ | **rising** |

They cross at Δ ≈ 1.51 meV. Converged values (k-grid to 2×10⁵ points):

| Δ (meV) | 0.6 | 1.0 | 1.4 | **1.51** | 1.8 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|---|
| Δ_top (meV) | 0.590 | 0.983 | 1.377 | **≈1.49** | 1.200 | 1.000 | 0.500 |
| T_max = Δ_top/20k_B (K) | 0.342 | 0.571 | 0.799 | **0.86** | 0.696 | 0.580 | 0.290 |

Three consequences:

1. **Δ = 2 meV is on the falling side of the curve.** The design point is not
   the optimum. At V_z = 3 meV the operational gap is maximised at
   Δ ≈ 1.5 meV, where Δ_top ≈ 1.49 meV — **~50% larger** than the 1.00 meV the
   design assumption delivers.
2. **The 1.4 meV "degraded redesign" threshold is inverted in that region.**
   A measurement returning Δ = 1.4 meV would give Δ_top = 1.38 meV, *better*
   than the design point, not worse. As written, the band would trigger a
   redesign on the strongest plausible outcome.
3. **The 0.6 meV kill-floor checks out — at fixed V_z.** T_max crosses 300 mK
   at Δ_top = 0.517 meV, i.e. Δ ≈ 0.53 meV on the pairing-limited branch. The
   stated 0.6 meV is correct and slightly conservative.

**The caveat that decides all of this: what is held fixed as Δ falls?**
If instead V_z tracks Δ by the design rule `V_z = 1.5 Δ` — which is what
reproduces Rev 2.1's "roughly half the bare pairing scale" — then
Δ_top = 0.5 Δ *exactly*, monotonic, and:

| Δ (meV) | 0.6 | 1.0 | 1.4 | 2.0 |
|---|---|---|---|---|
| Δ_top = 0.5Δ | 0.300 | 0.500 | 0.700 | 1.000 |
| T_max (K) | 0.174 | 0.290 | 0.406 | 0.580 |

Under that rule the kill-floor moves to **Δ ≈ 1.03 meV**, not 0.6 meV — the
band tightens by 1.7×, and Δ = 0.6 meV would force operation at 174 mK, which
breaks the dry-ADR premise the whole architecture rests on.

Rev 2.1 §3.2 chose V_z = 3 meV "comfortably inside the topological phase while
limiting Zeeman pair-breaking of the parent condensate." If the measured Δ
comes back small, the parent condensate is weaker and a 4.15 T in-plane field
is *more* likely to pair-break it — which argues for the V_z = 1.5Δ rule and
the tighter floor. **This is a device-physics question the simulation cannot
settle, and the Go/No-Go criteria should state which rule they assume.** As
written they silently assume the fixed-V_z case, which is the permissive one.

Reproduce with:

```bash
python -m tvqpu.lattice --gap --delta 1.5 --v-z 3.0
```

### 2.5 Open item: the σ ≤ 0.2% tolerance is still asserted

Rev 2.1 §8.1 sets σ_max = 0.2% as a 15× margin below an observed degradation
onset at σ ≈ 3%. The disorder sweep run earlier in the project (L = 60, 8
realizations, W ≤ 4) **never reached a threshold** — the order parameter had
not collapsed at W = 4, twice the clean phase boundary. So the number is
inherited from the twist-metrology requirement, not derived from a computed
tolerance curve.

Deriving it needs: W extended to ~10, ≥ 50 realizations (8 gives the standard
deviations themselves ~25% uncertainty), and finite-size scaling across
L = 40/60/80/120. `scripts/sigma_tolerance_sweep.py` runs exactly that,
resumably; `tvqpu.dmrg` is the driver.

**The unit conversion is the part that decides whether the answer means
anything.** A threshold quoted as W/t does *not* transfer to the device.
Rev 2.1 runs at gap/t = 1.05/20 = 0.05; the Jordan-Wigner toy chain runs at
t = Δ, where that ratio is ~1. A tolerance expressed as a fraction of t is
therefore off by a factor of ~20 between the two models — exactly the kind of
slip that turns a simulation result into a wrong spec.

The transferable quantity is **δµ_rms / gap**, and for a uniform box of width
W, δµ_rms = W/√12. `tvqpu.dmrg.tolerance_in_gap_units` does the conversion.
Rev 2.1's own numbers in that form:

| | σ | δµ_rms | δµ_rms / Δ_top |
|---|---|---|---|
| production spec | 0.2% | 40 µeV | **0.038** |
| observed degradation onset | ~3% | 600 µeV | **0.571** |

So the question the sweep answers is whether the computed threshold sits above
0.571 (Rev 2.1's onset is consistent), and by how much the 0.038 spec is
therefore margined.

### 2.5a Result — and why it stops short of a single number

The full run completed: L = 40/60/80/120, W ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6,
7, 8, 10}, 50 realizations per point, 2204 DMRG solves. Order parameter at the
95% collapse criterion (Rev 2.1's onset is *degradation beginning*, not a full
collapse, so this is the criterion comparable to their 0.571 — see the table
in §2.5, and never compare their onset to a 50% threshold):

| L | 40 | 60 | 80 | 120 |
|---|---|---|---|---|
| W_c @ 95% | 2.701 | 2.720 | 3.424 | 3.022 |
| ratio δµ_rms/Δ_top | 0.780 | 0.785 | 0.988 | 0.872 |

**This is not monotonic in 1/L, at any of the four collapse criteria (95, 90,
75, 50%).** L = 80 sits *above* both neighbours in every row — a bulge, not a
step in a trend. A linear fit through four points where one bulges is fitting
noise, not physics, and reporting its intercept as "the" answer would be
exactly the kind of curve-fit-through-scatter this project's validation
discipline exists to catch.

**Where the noise comes from, quantified:** at 50 realizations per point, sem
on ⟨|YY|⟩ is ~0.03–0.04 near the transition. That propagates directly into
several-percent jitter in the extracted W_c — comparable to, or larger than,
the finite-size trend itself. `scripts/sigma_tolerance_sweep.py` now defaults
to 200 realizations (sem ~1/√n, so roughly half) for exactly this reason.

**A second, independent limitation sits on top of it.** Bond dimension
saturated in 16/2204 points, concentrated exactly where it would corrupt the
answer most:

| L | W=6 | W=7 | W=8 | W=10 |
|---|---|---|---|---|
| 40 | 0% | 0% | 0% | 0% |
| 60 | 0% | 0% | 0% | 2% |
| 80 | 0% | 0% | 2% | 4% |
| 120 | 0% | 2% | 4% | **18%** |

At L = 120, W = 10, nearly one in five realizations has unknown discarded
weight. Those high-W points anchor the 50%/75% thresholds directly, which is
the second reason those two criteria are not trustworthy here — independent of
the sampling-noise problem above.

**Best estimate, honestly bounded rather than extrapolated to a point:**

> δµ_rms / Δ_top ≈ **0.9 ± 0.15** (95% criterion, χ = 96, n = 50)

This sits comfortably above Rev 2.1's asserted onset (0.571) at every length
and every criterion measured — **the many-body treatment does not threaten
σ ≤ 0.2%.** That conclusion is solid. The exact multiplicative margin on the
spec is not, and should not be quoted more precisely than the ± above.

### 2.5b χ = 256 confirmation — a real finding, not the one it looks like at first

`runs/sigma_chi256` re-ran W ∈ {6, 7, 8, 10} at χ = 256, n = 20, using the
**same disorder seeds** as the χ = 96 baseline's first 20 realizations, so
every point has an exact paired comparison at fixed disorder.

**Naively re-running `--report` on this ledger looks alarming**: it prints
95%/90% ratios of 0.31–0.50, *below* Rev 2.1's 0.571 onset. **This is a
methodology artifact, not a physics result, and must be discarded.** The
`--w-min 6` scope means the ledger contains only W = 0 (clean) and
W ≥ 6 — no points in the W ≈ 1–4 range where the 95%/90% crossing actually
sits. `disorder_threshold` at those criteria is therefore interpolating
*linearly between exactly two points 6 apart in W*, not reading a resolved
curve. Confirmed by hand: the printed L = 40 value (1.465) reproduces exactly
from `6 × (0.9306 − 0.884)/(0.9306 − 0.7401)` — a straight line between the
clean point and W = 6, nothing else. **Only the 50%/75% rows are inside this
study's sampled range and are meaningful; 95%/90% from this ledger should
never be quoted.**

**The 50%/75% ensemble means agree closely with χ = 96** (differences of
0.001–0.02 in ⟨|YY|⟩, small against sem ≈ 0.03–0.06) — reassuring: the 16/2204
saturated points did **not** measurably bias the aggregate threshold that
matters for those criteria.

**But the per-realization paired comparison exposes a different, more
serious problem.** At fixed (L, W, seed), χ = 96 and χ = 256 disagree by up
to **0.38** in |⟨YY⟩| — and the worst disagreements are *not* the points
where χ = 96 saturated its ceiling. Examples: L = 120, W = 6, seed = 15 —
χ = 96 reached bond dim **8** (nowhere near its 96 cap) and got 0.036;
χ = 256 got 0.417. L = 80, W = 8, seed = 12 — χ = 96 reached 32, got 0.201;
χ = 256 reached 145, got 0.014.

That pattern — large disagreement uncorrelated with whether the bond-dimension
*ceiling* bound — is the signature of **DMRG converging to different
near-degenerate local optima** for the same disordered Hamiltonian, depending
on the bond-dimension growth schedule, not of insufficient χ per se. This is
expected and documented behaviour for two-site DMRG on strongly disordered 1D
chains near a transition (rare-region / Griffiths physics produces
near-degenerate low-lying states), and it means **realization-to-realization
scatter in this regime partly reflects optimization landscape multistability,
not only true sample-to-sample physical variation.**

**Consequence: this is a more important caveat than truncation, and it is not
fixed by more realizations or larger χ alone.**

### 2.5c Confirmed by energy, and fixed

The diagnosis above was checked the only way that settles it — **by comparing
energies**, since DMRG is variational and the lower energy is the better
answer regardless of which χ produced it:

| point | χ=96 energy | χ=256 energy | lower | χ=96 OP | χ=256 OP |
|---|---|---|---|---|---|
| L=80, W=8, s=12 | **−117.43145** | −117.42991 | χ=96 | 0.201 | 0.014 |
| L=60, W=10, s=5 | −95.88125 | **−95.88171** | χ=256 | 0.371 | 0.259 |
| L=120, W=8, s=6 | −173.94799 | **−173.95203** | χ=256 | 0.098 | 0.341 |
| L=120, W=6, s=15 | −149.79176 | **−149.88386** | χ=256 | 0.036 | 0.417 |
| L=120, W=10, s=7 | **−187.45682** | −187.45647 | χ=96 | 0.280 | 0.218 |

**χ=96 wins twice, χ=256 wins three times.** Truncation error would make the
larger χ systematically lower; a coin-flip split is the signature of *basin
trapping*. At L=80/W=8/s=12 the larger bond dimension was decisively **worse**
— it found a higher-energy state, so its 0.014 order parameter was simply the
wrong answer.

**Multi-restart independently confirms it.** Four restarts at χ=96 on that
same point give energies −117.43145053 / −117.43144834 / −117.43109838 /
−117.43056932 with order parameters 0.2006 / 0.2014 / 0.1236 / 0.0703. The
lowest energy carries OP ≈ 0.2006, matching the χ=96 single shot and
confirming χ=256's 0.014 as a trapped solution. The spread — **0.13 in the
order parameter at fixed disorder and fixed χ** — reproduces the effect
entirely within one bond dimension, which is what attributes it to the
optimizer landscape rather than to truncation.

**Fix implemented.** `tvqpu.dmrg.ground_state` now takes `n_restarts` and
`restart_seed`, running from several seeded random initial states and
returning the lowest-energy result. `DMRGResult` gained `energy_spread`,
`order_parameter_spread`, and `multistable`, and `trustworthy` now requires
*not multistable* in addition to converged and non-saturated.

Two properties worth knowing:

- **`n_restarts=1` cannot detect multistability** and reports
  `multistable=False` by construction — absence of evidence, not evidence of
  absence. It also deliberately keeps the original unseeded quimb code path so
  existing ledgers stay comparable, at the cost of being reproducible only to
  solver tolerance (~10⁻¹³) rather than bit-for-bit.
- **A clean gapped chain shows zero spread** (< 10⁻⁸), so the machinery
  exposes multistability rather than manufacturing it. Pinned in
  `tests/test_dmrg_restarts.py`.

**Standing guidance:** deep in the collapsing regime (W ≳ 6) use
`n_restarts ≥ 4` and check `energy_spread` before trusting a number. The
existing `sigma_v0` and `sigma_chi256` ledgers were produced with a single
restart and retain this caveat; their aggregate 50%/75% thresholds agreed
between the two χ values, so the *ensemble* conclusion stands, but individual
realizations in that regime should not be quoted.

**Interaction convention (fixed during driver bring-up).** `InteractingChain`
now defaults to the particle-hole symmetric form `V Σ(n_i−½)(n_{i+1}−½)`,
which maps to a pure (V/4)ZZ term. The plain `V Σ n_i n_{i+1}` form also
generates a field shift +V/2 in the interior, so at V = µ the transverse field
cancels *exactly*, leaving a ferromagnetic Ising point whose symmetry-broken
doublet splits by only ~3×10⁻⁴ at L = 12 — DMRG converged it five orders of
magnitude worse than every other point (1.1×10⁻⁶ against 2×10⁻¹¹). That is a
real near-degeneracy rather than a solver bug, but there is no reason to walk
into it, and the PH-symmetric form is also the convention of the literature
being reproduced (Stoudenmire et al., PRB **84**, 014503), so µ_c comparisons
transfer. Both forms remain available and both are asserted MPO-equals-dense.

With that convention the driver reproduces all five reference energies from an
**independently written** quimb `SpinHam1D` construction (earlier session) to
~10⁻¹⁰ — a genuine cross-implementation check, since Module 1 builds its MPO
from a finite-state machine instead.

---

## 3. Module 1 — `tvqpu.lattice`

**Contract: geometry in, Hamiltonian out. No solving.**

```python
from tvqpu.lattice import MajoranaChannel, REV21

ch = MajoranaChannel(n_sites=200, params=REV21)
H  = ch.to_bdg_dense()          # (4N, 4N) float64/complex128, exact
mpo = ch.interacting().to_mpo() # site-dependent MPO for DMRG
```

Three geometry sources, one Hamiltonian interface:

| Class | Origin | Emits |
|---|---|---|
| `MajoranaChannel` | Rev 2.1 §2.1 | dense BdG + spin-chain MPO |
| `HoneycombSuperlattice` | ported from `115 sim/generate_labels.py` | tight-binding Bloch H, band statistics |
| `LatticeGeometry` | generic sites + bonds | either |

The honeycomb path is a direct port of the validated superlattice solver from
the metamaterial GNN pipeline (`generate_labels.py`), preserving its four
known-physics gates: 6t total bandwidth, gapless Dirac point, exact exchange
splitting, superlattice band narrowing. Its **honesty contract carries over
unchanged**: it computes band-structure *ingredients* of a single-orbital
nearest-neighbour model. It does not compute T_c, pairing, or many-body physics.

### 3.1 MPO construction

Finite-state-machine MPO, bond dimension **D = 5** for the model of §2.4:

```
        ┌ I     0     0     0     0 ┐
        │ X     0     0     0     0 │
W[j] =  │ Y     0     0     0     0 │      v_L = e₀ᵀ,  v_R = e₄
        │ Z     0     0     0     0 │
        └ h_j·Z Jx·X  Jy·Y  Jz·Z  I ┘
```

Per-site coefficients (`h_j`, and optionally per-bond `J`) make disorder
realizations free: no rebuild, just a new coefficient vector. This is what the
σ-tolerance sweep of §2.5 needs to run 50 realizations cheaply.

The builder emits both a pure-NumPy MPO (always available) and a `quimb`
`MatrixProductOperator` when quimb is installed. A dense reconstruction path
exists solely so `tests/test_lattice.py` can assert the MPO equals an
independently-built dense Hamiltonian at L = 8 — **validate against a known
answer before trusting a novel system** is the house rule, and uncheckable
numerics on a novel Hamiltonian is how people convince themselves of artifacts.

### 3.2 Two API traps that silently produce wrong physics

Both were hit in this project and neither raised an error:

- **`quimb`'s `.correlation()` returns the *connected* correlator.** In a
  symmetry-broken DMRG ground state that is ≈ 0 even deep in the ordered phase
  — it looks exactly like the topological phase has vanished everywhere. The
  order parameter needs the raw correlator, so ⟨Y_i⟩⟨Y_j⟩ must be added back.
- **`.magnetization()` returns ⟨S^y⟩ = ⟨Y⟩/2, not ⟨Y⟩.** Factor 2 per site,
  hence 4× in the correlator.

`tvqpu.dmrg.raw_correlator()` wraps both and is verified against direct
contraction at L = 20 (0.93052 vs 0.93038).

---

## 4. Module 3 — `tvqpu.kernels`, Triton fused two-site update

### 4.1 Where the memory pressure actually is

Not in generic matrix multiplication. The DMRG inner loop is a Lanczos/Davidson
eigensolve on the two-site effective Hamiltonian, and the hot operation is
applying `H_eff` to the two-site wavefunction θ, once per Krylov vector:

```
θ'[a,s₁,s₂,b] = Σ  L[a,m,a'] · W₁[m,n,s₁,s₁'] · W₂[n,p,s₂,s₂'] · R[b,p,b'] · θ[a',s₁',s₂',b']
               a'm n p b' s₁' s₂'
```

with χ = bond dimension, d = physical dimension (2 here), D = MPO bond
dimension (5 here). Contracted naively in four steps, the intermediates are:

| step | intermediate | size (complex128) at χ=4096, d=2, D=5 |
|---|---|---|
| θ·L | `[m,a,s₁,s₂,b]` | χ²d²D · 16 B = **5.4 GB** |
| ·W₁ | `[n,a,s₁,s₂,b]` | χ²d²D · 16 B = 5.4 GB |
| ·W₂ | `[p,a,s₁,s₂,b]` | χ²d²D · 16 B = 5.4 GB |
| ·R | `[a,s₁,s₂,b]` | χ²d² · 16 B = 1.1 GB |

Every one of those is written to and re-read from HBM, and the Lanczos loop
repeats it 10–40 times per site per sweep. That round-tripping — not the FLOPs
— is what caps χ on a 32 GB card. Fusing the chain so the D-indexed
intermediates never leave SRAM removes three full HBM round-trips per Krylov
step.

### 4.2 Kernel design

`tvqpu/kernels/fused_twosite.py` implements
`fused_heff_apply(L, W1, W2, R, theta)`:

- **Tiling.** Output tile `(BLOCK_A × BLOCK_B)` over the two virtual indices,
  with `(s₁,s₂)` unrolled (d² = 4 for spin-½). The MPO indices `m,n,p` are
  reduction axes held entirely in registers — D = 5 is tiny, which is exactly
  why this fuses well.
- **Residency.** `W₁` and `W₂` together are `D²d²·16 B = 1.6 kB`. They are
  loaded once per program into SRAM and reused across the whole tile. The
  D-indexed intermediates are register-resident and never materialized.
- **Precision.** float64 / complex128 throughout, non-negotiable. Lanczos loses
  orthogonality in float32 and — worse — the truncation-error diagnostic stops
  being meaningful, which is the property that makes DMRG trustworthy at all.
  There is no low-precision path and no tensor-core path for this kernel.
- **Complex arithmetic.** Triton has no native complex dtype. Real and
  imaginary parts are carried as separate tensors with an explicit
  4-multiply/2-add complex FMA in-kernel. A real-only fast path is provided and
  is the common case: the JW-mapped Hamiltonian of §2.4 is real, and real
  float64 halves memory versus complex128 at zero accuracy cost.

### 4.3 Correctness and portability, stated up front

- **Every kernel has a `torch.einsum` reference implementation**, and
  `tests/test_fused_twosite.py` asserts agreement to 1e-12 relative. The
  reference path is the default; the Triton path is opt-in via
  `TVQPU_BACKEND=triton`. A fused kernel that is fast and wrong is worse than
  no kernel.
- **The kernel does not run on the development laptop.** That machine is 8
  cores / 31.6 GB / integrated graphics, no CUDA device. Triton also has no
  first-class Windows support. Module 3 is developed against the reference
  path locally and executed on the 4× RTX 5090 vast.ai box. Imports are
  guarded so the package works with neither Triton nor CUDA present.

### 4.4 Memory budget — the area-law bound, made explicit

From `docs/tensor_networks.md`:

```
MPS state memory ≈ 32 · n · χ²  bytes   (complex128)
DMRG sweep cost  ≈ n · χ³
Max entanglement entropy across any cut: S = log₂ χ
```

| n sites | max χ at ~8 GB state budget |
|---|---|
| 100 | ~1500 |
| 500 | ~700 |
| 1000 | ~500 |

Upper bounds — real DMRG also stores environment tensors. `tvqpu.budget`
computes these and **refuses to start a run whose projected peak exceeds the
declared VRAM budget**, rather than discovering it via an OOM 40 minutes in.

The bound is legitimate here because 1D gapped ground states obey an area law:
entanglement entropy saturates at a constant set by the correlation length,
*independent of system size*, so required χ does not grow with n and cost is
strictly O(n·χ³). This holds for the gapped topological phase of §2. It does
**not** hold at the critical point (µ → µ_c), where S(ℓ) ≈ (c/6)·log ℓ and χ
must grow polynomially. Runs near the transition must not silently inherit the
gapped budget, and `tvqpu.budget` flags them.

**Truncation error is the run's own self-diagnostic.** DMRG reports discarded
weight every sweep: ~10⁻¹² means trust the result; climbing toward 10⁻³ means
the run has left the valid regime. The method does not fail silently, and the
harness logs the diagnostic to the ledger every sweep.

---

## 5. Module 4 — `tvqpu.substrate` (ballistic phonon transport)

**No tensor networks here.** Phonon transport in BAs is harmonic modes plus
perturbative three-phonon scattering — Boltzmann transport and caustic
geometry. There is no many-body entangled state for a PEPS to represent. See
§8 on nomenclature.

### 5.1 Two distinct "metamaterials" that must never be conflated

The project contains two objects both called metamaterials. They are separated
by eleven orders of magnitude in frequency and share no physics:

| | **Vibration mount** (Rev 2.1 §6) | **BAs phonon focusing** (Li et al. 2026) |
|---|---|---|
| Frequency | 50–150 Hz | ~1–20 THz |
| Physics | locally resonant elastic metamaterial | ballistic lattice phonons, caustics |
| Structure | 112×75×7 mm, triangular lattice a = 9 mm, gyroid cells, 2–4 g tungsten proof masses on Si flexures, rainbow-graded | zincblende BAs crystal, no engineered structure |
| Purpose | isolate the 0.3 K stage from Stirling piston vibration; 20–30 dB insertion loss | guide heat along ray-like paths |
| Model | continuum elastodynamics / Bloch band structure | Christoffel equation + BTE on the fc2 group-velocity field |
| Temperature | 4 K / 0.3 K | **room temperature** |

Rev 2.1 explicitly notes that flexural wavelengths at 50–150 Hz are 0.6–1.0 m,
far larger than the mount, so **Bragg and valley-Hall mechanisms are inoperative
in that band** — the mount works by local resonance and rainbow grading alone.
Both submodules exist; they are separate namespaces and separate solvers.

### 5.1a The measured effect, from HU2026

Li, Wu, Qin, Su, Nguyen & Hu, *Phonon focusing at room temperature*, Nature
Physics (2026), [doi:10.1038/s41567-026-03335-y](https://doi.org/10.1038/s41567-026-03335-y)
— hereafter **HU2026**. Tip-enhanced Raman thermometry of a self-assembled
molecular monolayer, imaging the steady-state temperature field around a
nanoscale STM hot spot on BAs single crystals.

**The formalism, which is what Module 4 implements.** Mode heat flux
`q_ks = ħ ω_ks v_ks δn_ks` (eq. 1). Real-space directional flux integrates the
Brillouin zone against a delta function on the *group-velocity* direction:

```
q(θ,φ,r) = (2π)⁻³ Σ_s ∫ |q_ks| δ(Ω_θφ − Ω_ks) dk = Σ_s ∫ q_s(θ,φ,r,ω) dω    (2)
q_s(θ,φ,r,ω) = ħ ω v_s(θ,φ,ω) δn_s(θ,φ,r,ω) DOS_s(θ,φ,ω)                     (3)
δn_s(θ,φ,r,ω) ≈ δn_s(θ,φ,0,ω) · exp(−|r| / L_s(θ,φ,ω))                       (4)
```

**That delta function is the Jacobian.** `δ(Ω_θφ − Ω_ks)` pushes forward the
measure from k-space onto the group-velocity sphere; its density is exactly the
inverse Jacobian of the q̂ → v̂ map, and the caustics are where that Jacobian
vanishes. HU2026 calls the result the **directional density of states**
`DOS_s(θ,φ,ω)` — modes per unit frequency per steradian. So the refactor
specified in §5.2 is not an invention; it is eq. (2) done correctly, and the
paper supplies published values to check it against.

**Calibration targets — these are hard numbers, not qualitative.**

| Quantity | HU2026 | Where |
|---|---|---|
| Ballistic conductance G₀, BAs (111) | **8.68 W m⁻¹ rad⁻¹ K⁻¹** | Fig. 3d caption |
| Peak G on the principal direction, 300 K | ~40 W m⁻¹ rad⁻¹ K⁻¹ | Fig. 3d |
| **Peak G/G₀** | **≈ 4.6** | Fig. 3d |
| Angular width of the peak | narrow within ±15° | Fig. 3d φ-axis |
| Directional-DOS anisotropy | ~1.4–1.5 peak vs ~0.5 min | Fig. 4a–c insets |
| Iso-frequency surface shown | LA branch at 7.8 THz, \|k\| ≈ 0.8–1.1 Å⁻¹ | Fig. 3a |
| Propagation length at 300 K | ~µm up to 8 THz | Fig. 3c |

**G/G₀ ≈ 4.6 is the number that condemns the old code.** The previous
histogram-binning implementation reported peak enhancements of 23–44 — roughly
an order of magnitude too large, on top of the degenerate-branch artifact.
`G_OVER_G0_PEAK_300K` in `tvqpu.substrate.orientation` records the published
value so the refactor has a target rather than a free parameter.

Note that G/G₀ > 1 is real and is the paper's point: focusing *redistributes*
conductance, so the peak exceeds the ballistic radiation limit even though the
angular average cannot.

**Measured ray lengths — and they get longer as it gets colder.**

| T | focused ray length |
|---|---|
| 80 K | ~1.2 µm (plus six ⟨211⟩ minor rays) |
| 150 K | ~350 nm (minor rays weakening) |
| 300 K | ~250 nm (minor rays gone) |
| > 400 K | pattern merges into a hexagon, trending diffusive |

### 5.1b Surface orientation sets the symmetry — implemented and validated

HU2026 predicts *and measures* three different fold-symmetries on the same
material (Fig. 4a–c theory, Fig. 4d–f experiment). `tvqpu.substrate.orientation`
reproduces all three from pure crystallography — no DFT, no force constants:

```
surface (111): 6-fold   principal [1-10] [-110] [10-1] [-101] [01-1] [0-11]
                        minor     <211> family, low-temperature only
surface (100): 8-fold   principal <001> (4) + <011> (4)
surface (110): 4-fold   principal <001> (2) + <1-10> (2), INEQUIVALENT pairs
```

The rule is one line: a direction `[uvw]` contributes iff `hu + kv + lw = 0`,
i.e. it lies *in* the surface plane. Members of ⟨100⟩ and ⟨110⟩ that pass give
the principal rays; ⟨211⟩ gives the minor ones.

**A correction that matters for the drain geometry.** The six (111) principal
directions all sum to zero. The superficially similar set **[110], [101],
[011] sums to +2 and does not lie in the (111) plane at all** — those point
into the bulk. A drain specified on the mixed set would aim part of its
routing into the substrate rather than along the surface.
`tests/test_orientation.py` pins this explicitly.

### 5.2 The `focusing.py` refactor (flag #2)

The existing phonon-focusing calculation is **marked do-not-reuse** and its
peak enhancement factor must not be quoted. The defect: all three acoustic
branches peak in the identical angular bin with identical `A_max`, which is
physically impossible — different branches have different caustic structures.
The likely cause is degenerate-mode handling in the group-velocity routine at
|q| → 10⁻⁴ with non-analytic correction active. The percentile statistics
(p99 ≈ 23–44, median 1.2–1.65) are probably sound; the peak value is not.

**The fix is a real Jacobian, not histogram binning.** Phonon focusing is the
statement that the map from wavevector direction q̂ to group-velocity direction
v̂ = ∇_q ω / |∇_q ω| is not area-preserving. The enhancement factor is the
inverse Jacobian determinant of that map:

```
A(v̂) = Σ        |det ∂v̂/∂q̂|⁻¹
     q̂ ∈ preimage(v̂)
```

Caustics are the loci where `det ∂v̂/∂q̂ → 0`, i.e. where `A → ∞`; the physical
divergence is cut off by finite phonon wavelength and lifetime. Requirements
for the refactored solver:

1. **Analytic derivatives where possible.** ∂v̂/∂q̂ involves second derivatives
   of ω(q) — the Hessian of the dynamical matrix eigenvalue. Finite-differencing
   v̂ on a q-grid is what produced the current artifact.
2. **Degenerate branches handled explicitly.** The two TA branches are
   degenerate along high-symmetry directions. Eigenvector tracking (overlap
   continuation between adjacent q-points) is mandatory; naive eigenvalue
   sorting swaps branches and smears the caustics into a common bin.
3. **Multivalued preimages summed, not binned.** A given v̂ can receive flux
   from several q̂. The solver must find all preimages, not histogram whichever
   one it visited.
4. **Validation before use.** The acceptance ladder is now fully specified by
   HU2026, in increasing difficulty:
   a. **Symmetry** — reproduce 6/8/4-fold on (111)/(100)/(110).
      *Already done* by `tvqpu.substrate.orientation`, no DFT required.
   b. **Angular profile** — a narrow peak within ±15° of each principal
      direction (Fig. 3d).
   c. **Magnitude** — peak `G/G₀ ≈ 4.6` against `G₀ = 8.68 W m⁻¹ rad⁻¹ K⁻¹`.
      This is the gate the old code fails by ~10×.
   d. **Spectral** — propagation length staying of order µm up to 8 THz at
      300 K (Fig. 3c), which is the actual reason the effect survives to room
      temperature.
   Silicon remains a useful warm-up (Wolfe's phonon imaging is textbook), but
   BAs now has better-specified published targets than Si does.

**Provenance of the input data, stated plainly.** The fc2/fc3 data is a
published VASP/PBE `phono3py_params` dataset for *bulk zincblende BAs* from the
NIMS MDR repository (512-atom harmonic supercell, third-order displacement
forces, Born charges). It is someone else's calculation on one of the
most-studied thermal materials of the last decade. Our contribution is the
solver, not the material.

### 5.2b Where the BAs substrate can and cannot be claimed

The proposal is to place BAs (111) under the logic stack as a thermal drain.
Two facts from HU2026 bound what that can be claimed to do, and both are
enforced in code (`focusing_regime_note`, `dominant_phonon_frequency_thz`).

**1. Frequency.** HU2026 is explicit that room-temperature focusing is carried
by **THz thermal phonons**, and that its survival to 300 K is due to
propagation lengths persisting to ~µm *up to 8 THz*. The paper contrasts this
with sub-GHz acoustic waves, which it treats as a different regime entirely.
At the logic layer's 0.3 K the dominant phonon scale is `k_BT/h ≈ 6.3 GHz` —
**three orders of magnitude below the measured band**, and the THz modes that
carry the effect are essentially unoccupied. Transport there is
boundary-limited (Casimir, κ ∝ T³), not Umklapp-limited, so neither the
1300 W/m·K figure nor the ray patterns transfer.

**2. Length.** The measured rays are 250 nm at 300 K and 1.2 µm at 80 K.
Rays do lengthen as temperature falls, which is the favourable direction — but
the parts that need thermal management are the 5 × 5 × 0.5 mm die, the
90 × 60 × 4 mm stage plate and the 220 × 160 × 10 mm baseplate. Focusing acts
**three to four orders of magnitude below the length scale of the hardware.**
It is a die-attach and interface-layer phenomenon, not a plate-scale one.

**Where it is on-label.** The 4 K and warmer stages sit inside the band HU2026
actually measured, and the same group has published BAs specifically as a
device cooling substrate and thermal interface material (Kang et al.,
*Nat. Electron.* **4**, 416 (2021); Cui et al., *Nat. Commun.* **12**, 1284
(2021)). A BAs interface layer at the 4 K baseplate or on the room-temperature
side of the wiring loom is a defensible, citable application of this result.
A "room-temperature phonon-focusing thermal shield for the 0.3 K logic layer"
is not, and Rev 2.1 does not need one — its 0.3 K load budget already closes
with 10× margin (≲4.9 µW against 50 µW ADR capacity, Table 1).

**On "replacing the (111)-oriented silicon substrate of Rev 2.1":** Rev 2.1
specifies no silicon substrate and no crystal orientation anywhere. The only
silicon it names is the **flexures** of the vibration mount (§6): mm-scale
mechanical springs carrying 2–4 g tungsten proof masses in a 9 mm-pitch
lattice, tuned for 50–150 Hz. Substituting BAs there would be a structural
change to a resonator, and phonon focusing is irrelevant at 100 Hz, where
flexural wavelengths are 0.6–1.0 m. The two components are not
interchangeable and should not be conflated.

### 5.2c Gate results for the implemented solver

`tvqpu.substrate.directional_dos` implements eq. (2) as specified. Status
against the four ranked gates, with the elastic (Christoffel) provider:

| Gate | Target | Result | |
|---|---|---|---|
| **calibration** | isotropic ⇒ A ≡ 1 | **1.000000** everywhere, covering degree 1.00000000 | ✅ |
| **measure** | ⟨G/G₀⟩ over v̂ sphere = 1 | **1.024** at 400 probes (BAs, folded map) | ✅ |
| **(a) symmetry** | (111) 6 principal + 6 minor | **17.30** (⟨110⟩) and **1.16** (⟨211⟩), ratio 14.9 | ✅ |
| | (110) 4-fold | **4-fold** | ✅ |
| | (100) 8-fold | **1** — plane uniformly bright, no contrast | ❌ |
| **(b) angular** | narrow peak, ±15° | rays over ~20% of azimuth | ✅ |
| **(c) magnitude** | peak G/G₀ ≈ 4.6 | **17.30** — overshoots by 3.8× | ❌ |
| **(d) spectral** | L ~ µm to 8 THz | not attempted — needs the fc3 scattering matrix | — |

**On the (111) ray count.** The elastic model gives **twelve** rays above the
ballistic limit, at 30° spacing — converged from 72 to 720 azimuthal samples.
That is not a discrepancy with HU2026, it is the low-temperature limit of it:
six principal ⟨110⟩ rays at G/G₀ = 17.30 *plus* six minor ⟨211⟩ rays at 1.16,
which is exactly the principal/minor structure the paper reports, with the
minors ~15× weaker. HU2026 sees both at 80 K (Fig. 2b), the minors weakening
at 150 K and gone by 300 K. A ballistic model has no scattering, so it is the
80 K pattern that it reproduces. Any threshold in (1.2, 17) selects the six
principal rays that survive to room temperature.

A single "fold number" is therefore not well defined without stating the
temperature regime, and `fold_symmetry` takes an explicit `threshold` rather
than pretending otherwise.

Each star is internally uniform to three decimals — all six ⟨110⟩ give 17.30,
all six ⟨211⟩ give 1.16. No cubic symmetry is built into the solver, so that
equality is earned and is an independent correctness check.

**The solver is validated; the elastic model is the limitation *at 300 K*.**
The isotropic case has an exact known answer and the solver reproduces it to
eight digits. The two gates that miss HU2026's numbers both miss for the same
knowable reason:

- **(c) overshoots because the elastic ballistic limit has no scattering.**
  Caustics are sharper than in a real 300 K crystal whose propagation length
  is ~250 nm. HU2026's 4.6 comes from the full scattering matrix, explicitly
  *not* the single-mode relaxation-time approximation.
- **(a) on (100) fails because the elastic limit is k → 0.** HU2026's pattern
  is read off the iso-frequency surface at **7.8 THz**, where the dispersion
  has flattened substantially — our own fc2 data gives \|v_g\| along [100]
  falling from 7209 m/s at \|q\| = 0.05 Å⁻¹ to 3193 m/s at 0.90 Å⁻¹. A
  long-wavelength model cannot know about that surface.

### 5.2d Regime correction: at 4 K the elastic model is not an approximation

The gates above compare against HU2026's **300 K** measurement. The BAs
interface in this architecture sits at the **4 K** stage (§5.2b), and that
changes which model is correct rather than merely which is convenient.

| T | dominant phonon k_BT/h | regime |
|---|---|---|
| 300 K | 6.25 THz | near HU2026's 7.8 THz surface; dispersion strongly non-linear |
| 80 K | 1.67 THz | HU2026's low-T measurement; minor rays visible |
| **4 K** | **0.083 THz** | **linear dispersion; elastic limit exact** |
| 0.3 K | 0.006 THz | linear, but see §5.2b — this stage is not the BAs use case |

BAs acoustic branches already sit at 0.42 THz at \|q\| = 0.05 Å⁻¹, so 83 GHz
is far inside the linear regime. The Christoffel description is **exact**
there, not approximate. This is also why the classic phonon-imaging
experiments HU2026 cites as refs 1–3 (Taylor/Maris/Elbaum 1969;
Northrop & Wolfe 1979; Hensel & Dynes 1979) were done at 1.8–3.6 K with
elastic-continuum theory: the ballistic elastic limit *is* the cryogenic
regime.

**Three-phonon (fc3) scattering is genuinely frozen out at 4 K.** Umklapp is
exponentially suppressed, so omitting fc3 is correct here — not a shortcut.

**But "fc3 frozen" does not mean "no scattering."** At 4 K the mean free path
is set by:
- **boundary (Casimir) scattering** — geometric, and the dominant term;
- **isotope disorder** — natural boron is ~20% ¹⁰B, scattering as ω⁴, weak at
  83 GHz but not zero.

These set the propagation length L, and L is what cuts off the caustic
divergence. `caustic_cutoff_from_geometry()` turns (T, L) into an
order-of-magnitude ceiling via the diffraction angle λ/L — at 4 K the thermal
wavelength is ~64 nm, so a 1 µm propagation length supports enhancements only
up to ~16. **The default cutoff of 50 is not physically justified for a thin
interface layer**, and any 4 K magnitude quoted from this solver must state
the assumed L.

### 5.2e Why the 4 K gates cannot simply be "12-fold and 17.30"

Retargeting gate (a) to **12-fold is legitimate**: it is HU2026's own 80 K
observation (six ⟨110⟩ principal plus six ⟨211⟩ minor, Fig. 2b), it is an
independent published result, and the solver reproduces it with the correct
15× intensity ordering. That is a real gate and it passes.

Retargeting gate (c) to **17.30 is not legitimate**, because 17.30 is this
solver's own output. A number cannot validate the code that produced it. Doing
so converts gate (c) from *validation* into a *regression test* — still useful,
still worth pinning, but it must be labelled as such or the gate table becomes
self-certifying.

Two ways to restore a real magnitude gate in the ballistic regime, neither of
which needs fc3:

1. **Validate against the phonon-imaging literature.** Northrop & Wolfe
   published ballistic caustic patterns and enhancement factors for Si and Ge
   at liquid-helium temperature — the same elastic-ballistic regime, with
   independent numbers. Si is the natural target because its elastic constants
   are unambiguous, unlike the contested BAs C₄₄.
2. **Constrain by geometry.** Require the reported peak to be consistent with
   `caustic_cutoff_from_geometry(4 K, L)` for the actual interface thickness.
   That is a falsifiable consistency condition rather than a free parameter.

Until one of those is in place, the 4 K magnitude is a **prediction of this
model, not a validated result**, and should be reported that way.

### 5.2f Formal acceptance gates, 4 K ballistic regime

These supersede the 300 K gates of §5.2c for the BAs interface at the 4 K
stage. (Note on numbering: phonon focusing is **Module 4**; Module 3 is the
Triton kernel layer. The gates below belong to Module 4.)

| # | Gate | Target | Source | Status |
|---|---|---|---|---|
| **4K-0** | isotropic medium ⇒ A ≡ 1 | 1.000000 | exact, closed form | ✅ |
| **4K-1** | measure conservation | ⟨G/G₀⟩ = 1 | exact, pushforward | ✅ 1.024 |
| **4K-2** | (111) ray structure | 6 principal + 6 minor, principal ≫ minor | HU2026 Fig. 2b, **80 K** | ✅ 17.30 / 1.16, ratio 14.9 |
| **4K-3** | (110) symmetry | 4-fold | HU2026 Fig. 4c/4f | ✅ |
| **4K-4** | **magnitude ceiling** | peak ≤ `caustic_cutoff_from_geometry(4 K, L)` | geometry + diffraction | ⚠ see below |
| **4K-5** | **Si/Ge ballistic caustics** | published patterns and enhancements | Northrop & Wolfe, *PRL* **43**, 1424 (1979); Hensel & Dynes, *PRL* **43**, 1033 (1979); Wolfe, *Imaging Phonons* | ☐ not yet run |
| **4K-R** | 17.30 along ⟨110⟩ | regression pin only | this solver's own output | — |

**4K-4 is a falsifiable ceiling, not a target.** For a 1 µm BAs die-attach
layer at 4 K, λ_th ≈ 64 nm gives a per-caustic ceiling of **≈ 15.6**. The
solver's unconstrained ⟨110⟩ output is 17.30, which exceeds it, so the ceiling
binds and the run must be re-done with `caustic_cutoff=15.6`.

**The constrained result is G/G₀ ≈ 6.0, not 15.6.** The ceiling clips each
*preimage's* divergence; the reported G/G₀ is the normalized *sum* over
preimages, so it lands well below the ceiling. **6.0 is the number to quote
for a 1 µm layer at 4 K** — not 15.6, and not 17.30.

Worth noting where that lands: 6.0 against HU2026's measured 4.6 at 300 K. The
two are within ~30% and for a coherent reason — both are finite-propagation-
length results, one limited by 250 nm of anharmonic scattering at 300 K, the
other by a 1 µm layer thickness at 4 K. That is a consistency observation, not
a validation; 4K-5 remains the real magnitude gate.

The ceiling scales with layer thickness (10 µm → ~156, which clears the
caustic entirely and returns the unconstrained 17.30), so **the quoted
magnitude is a function of the interface design and must always be stated with
its assumed L.**

**4K-R is explicitly not a gate.** It exists so that future refactors are
caught if they move the number, and it is marked so nobody mistakes a
regression pin for validation. 4K-5 is the real magnitude validation and is
outstanding.

That (111) and (110) *do* come out right in the elastic limit is the
substantive physics result here: the ⟨110⟩ focusing that produces the six-fold
(111) pattern is already present in the acoustic branches. `enhancement_along`
gives G/G₀ = 17.30 along ⟨110⟩, 6.05 along ⟨100⟩ and 1.59 along ⟨111⟩.

**Estimator bug found and fixed, recorded because the class of error is
generic.** The first implementation returned Σ(A·w)/Σ(w) over the preimages —
a *mean* — where the pushforward density is a *sum*. On an unfolded map the
two coincide, so the isotropic calibration passed and hid it completely. It
only surfaced on the real dispersion at 7.8 THz, where the covering degree is
**6.03** and every reported value came out ~6× too small (all below 1). The
fix normalizes by the isotropic ballistic reference instead of by the number
of preimages found, and `verify_measure_conservation` now tests the invariant
that catches it: the solid-angle average of G/G₀ over the destination sphere
must be 1, which is only true for a sum. **An exact calibration on a
degenerate case is not sufficient validation** — the invariant has to be
checked where the map actually folds.

**Closing gates (a·100), (c) and (d) requires `PhonopyFC2Field`**, which is
implemented and loads the real 512-atom BAs fc2, plus a Boltzmann scattering
weight that is not yet written. That is the next piece of work, not a claim.

### 5.3 What is already validated in this module

| Result | Value | Status |
|---|---|---|
| BAs κ(300 K), 3-phonon RTA + isotope | 1244 W/m·K at 23³ mesh (1272 at 15³, 1255 at 19³) | vs ~1000–1300 experimental (*Science* 2018) |
| Acoustic sum rule | exact to 10⁻⁷ THz | gate |
| Acoustic–optic gap | 8.90 THz | from mass ratio 6.93 ✓ |
| LA[100] / TA[100] | 7222 / 5311 m/s | −3.2% vs C₁₁; TA within 0.5% of the ultrasonic C₄₄ |
| 103-compound fc3 reference corpus | κ spanning 0.3 → 1272.7 W/m·K, 0 gate failures | held-out MLIP test set |

**Caveat that must accompany the κ number.** 1244 vs ~1300 lands close partly
through error cancellation: omitting four-phonon scattering *overestimates* κ
(large for BAs specifically, whose acoustic–optic gap suppresses three-phonon
channels), while RTA instead of the full iterative LBTE *underestimates* it
(BAs is normal-scattering-dominated, RTA's worst case). Two approximations
pulling opposite ways. This validates that the pipeline is wired correctly. It
is not evidence that either approximation is individually accurate.

**The TA velocity "agreement" is with a contested measurement.** Brillouin
scattering gives C₄₄ = 173 ± 6 GPa; picosecond ultrasonics gives 149 GPa. The
two experiments disagree by 16%. Our 5311 m/s sits within 0.5% of the
ultrasonic value and 7.7% off the Brillouin value. Report it as such.

---

## 6. (Planned) `tvqpu.fte` — fault-tolerance resource estimator

> **NOT IMPLEMENTED.** This section specifies a module that does not
> exist in this repository. It is retained as a specification, not a
> description of shipped code.

Standard surface-code scaling, `d ≈ 2·log(1/p_L)/log(p_th/p)` rounded up to
odd, `n_phys ≈ 2d²`, threshold `p_th = 10⁻²`.

**The physical error rate p is an input parameter swept over 10⁻³–10⁻⁶, not a
derived quantity.** For a Majorana device p is set by non-equilibrium
quasiparticle poisoning, quasiparticle bursts, and control error — none of
which follow the thermal Arrhenius form. The thermal suppression factor
`exp(−Δ_top/k_BT) = 2.3×10⁻¹⁸` is reported as a **lower bound only**.

Result, and it is not the flattering one:

| p | d | phys/logical | vs 99.9% SC baseline |
|---|---|---|---|
| 10⁻³ | 31 | 1,922 | 1.0× |
| 10⁻⁴ | 15 | 450 | 4.3× |
| 10⁻⁵ | 11 | 242 | 7.9× |
| 10⁻⁶ | 9 | 162 | 11.9× |

**A 1000× better error rate buys ~12× fewer qubits**, because d ∝ 1/log(p_th/p)
— overhead scales with the *logarithm* of the error rate, and below ~3×10⁻⁶ the
distance stops decreasing at all. This does not show the architecture "needs
vastly fewer physical qubits than Google or IBM." It shows a 4–12× advantage,
conditional on an unmeasured p.

**The primary claim is SWaP-C**: 0.3 K without a dilution refrigerator, ~30×
cryogenic-plant reduction (Rev 2.1 §10). That is larger than the qubit-count
effect and does not depend on an unmeasured error rate. The overhead table is
the compounding secondary benefit. Counts exclude magic-state distillation,
routing, and control overhead; distillation commonly dominates algorithm-level
totals.

---

## 7. Telemetry (flag #5)

An honest performance-counter layer for the Triton kernels: per-kernel wall
time, achieved bandwidth, occupancy, HBM traffic, peak resident bytes,
truncation error and χ per sweep, exported as Prometheus text and JSONL.

**Explicitly out of scope:** the MCDM virtual-driver bridge that would make
tensor contractions register as NPU activity in Windows Task Manager. It
reports untrue information about which device executed the work, and MCDM
drivers require attestation signing to load at all. Dropped by agreement.

---

## 8. Nomenclature warning

"Tensor" means three unrelated things in this repository. They share a word and
nothing else, and conflating them is how a project drifts into confident
nonsense:

1. **Tensor networks / MPS / MPO** — Modules 1–3. Many-body quantum states.
2. **Lattice-dynamics force-constant tensors** (fc2, fc3) — Module 4. Phonons.
   No connection to (1).
3. **Tensor-decomposed neural-network weights** — not used here.

Likewise "metamaterial" means two unrelated things; see §5.1.

---

## 9. Validation ledger

Nothing enters a proposal, paper, or README from the right-hand column.

**Validated — reproducible, with a known answer to check against:**

- BdG Oreg–Lutchyn/Rev 2.1 model: Δ_top = 1.029 meV (vs 1.05 claimed),
  V_z,crit = 2.000 meV, splitting extrapolating to ~0.1 feV at 6.3 µm.
- DMRG vs exact diagonalization, L = 12: agreement to 2×10⁻¹¹ with V ≠ 0.
- V = 0 phase boundary reproduces the analytic µ_c = 2t.
- BAs κ = 1244–1272 W/m·K vs ~1300 experimental.
- 103-compound DFT fc3 reference corpus, 0 gate failures, κ ordering correct
  across 5034× (boron/carbon compounds at the ceiling, silver halides and lead
  chalcogenides at the floor).
- MACE-MP-0 gives κ 3.6× low on BAs **while passing every stability gate** —
  standard diagnostics do not catch anharmonic failure. Force RMSE was 1.5%.

**Asserted, unverified, or broken — do not build on without doing the work:**

- **Δ = 2 meV proximity-induced gap.** Never measured. Rev 2.1 §9.2 is right to
  make tunneling spectroscopy of the induced gap Milestone 1, before any qubit
  is fabricated. Every protection budget scales exponentially with it.
- **σ ≤ 0.2% electrostatic tolerance** — inherited from twist metrology, not
  derived; see §2.5.
- **Peak phonon-focusing enhancement** — current code produces an artifact; see
  §5.2.
- **The vibration mount's 20–30 dB insertion loss** — a design prediction, not
  a simulated or measured result. No unit-cell band-structure calculation for
  the gyroid/tungsten lattice exists in this repository yet.
- **No BAs thermal-shield structure has been designed.** The Li et al. result
  is room-temperature phonon focusing in bulk BAs. Applying it as a substrate
  under a 0.3 K stage is an idea, not a modelled component, and the temperature
  mismatch is the first thing to work out: BAs's record κ is a room-temperature
  Umklapp-limited property, whereas transport at sub-kelvin is boundary-limited
  (Casimir regime, κ ∝ T³). Room-temperature focusing does not automatically
  transfer to 0.3 K, and the honest first calculation is whether the caustic
  structure survives the change in the dominant scattering mechanism.

---

## 10. Evaluated and rejected pathways

Recorded so they are not re-proposed. Both papers below are good work; neither
does what an integration into this architecture would need it to do. The
rejection is on mechanism, not on quality.

### 10.1 Implosion carving (ImpCarv) as a manufacturing pathway

Yang, Yang, Nambara et al., *Isotropic shrinkage of patterned vacancies
enables three-dimensional nanoprecise metastructures for visible light
applications*, Nature Photonics **20**, 653 (June 2026),
doi:10.1038/s41566-026-01896-1.

**What it actually is.** Two-photon activation of rhodamine B generates
reactive oxygen species that cleave a **poly(acrylate-co-acrylamide) hydrogel**
backbone at photo-targeted points (ROS diffusion range ~100 nm). Cation
exchange (Na⁺ → Mg²⁺ → Ca²⁺) plus supercritical CO₂ drying shrinks the gel
isotropically by **13.18 ± 0.28×** (HSF) or 5.03 ± 0.06× (MSF). The product is
a **transparent organic polymer** of refractive index ~1.5 containing
vacancies, Δn ≈ 0.5. Demonstrated application: a diffractive optical network
doing digit classification at 532 nm.

**Why it cannot do the three proposed jobs:**

1. **BAs die-attach interface.** The six-fold symmetry is not something a
   template imposes — it is a property of the BAs crystal lattice and its
   (111) surface, arising from reciprocal-space anisotropy of the phonon
   dispersion (§5.1b). You obtain it by cutting the crystal on (111), which
   `tvqpu.substrate.orientation` already computes. Meanwhile ImpCarv cannot
   make BAs at all: BAs is grown by chemical vapour transport at **1058–1083 K**
   (HU2026 Methods), which incinerates a polyacrylate aerogel, and ROS cleavage
   acts on hydrogel backbones, not on covalent III–V bonds. The requirement
   here is wafer orientation and polishing — ordinary crystallography, not
   nanofabrication.
2. **Metamaterial mount gyroid voids.** Scale and material both fail. The
   mount is 112 × 75 × 7 mm on a **9 mm** lattice pitch carrying **2–4 g
   tungsten proof masses**; ImpCarv's demonstrated build volume is sub-
   millimetre, and a polymer aerogel cannot carry a 4 g mass. The resonator
   frequency is √(k/m), so the flexure material *is* the design — substituting
   an aerogel for silicon changes the thing being designed. And nothing about
   a 9 mm-pitch structure needs tens-of-nanometres resolution: mm-scale
   gyroid/TPMS cells are routine for conventional metal AM. This solves a
   problem the mount does not have, with a material that cannot do it.
3. **45.0° ± 0.1° twist alignment scaffold.** One point in its favour:
   isotropic shrinkage preserves angles exactly, so a scaffold could in
   principle define an angle well. But the tolerance is on the **lattice**
   angle, not on a mechanical fiducial — a scaffold does not know where the
   crystal axes are, which is exactly why the state of the art is tear-and-
   stack (the two flakes come from the *same* crystal, guaranteeing registry)
   with in-situ electron-diffraction feedback. Worse, Rev 2.1 §8.2 requires an
   adsorbate-free interface because the c-axis coherence length is shorter
   than one unit cell and "leaves no tolerance for interfacial residue." A
   polymer scaffold at the junction introduces precisely the contaminant the
   process forbids.

**Verdict: no role in this architecture.** No honest substitute application was
found; inventing one would be worse than saying so.

### 10.2 Floquet rotational super-radiance as drive and readout

Nasari, Moussa, Kasahara, Thielens & Alù, *Observation of Floquet rotational
super-radiance*, Nature **655**, 608 (16 July 2026),
doi:10.1038/s41586-026-10725-y.

**What it actually is.** A ring of **N = 3** parallel RLC tank resonators in
delta topology, modulated by **varactor diodes**, ~2 cm across, carrier
**100 MHz**, at room temperature. Travelling-wave modulation
ω_i(t) = ω₀[1 + δ_m cos(Ω_m t + (i−1)2π/N)] synthesizes rotation; above a
threshold rate the rotational Doppler shift goes negative, angular-momentum
bandgaps open, and parametric coupling between negative- and positive-frequency
components amplifies a selected OAM order. **Maximum raw gain 7.8 dB.**

**Why it cannot do the two proposed jobs:**

1. **Synthetic Majorana braiding.** Category error. Braiding MZMs is not
   rotation of anything physical — it is adiabatic exchange of two zero modes
   in real space, implemented (Rev 2.1 §3.2, after Karzig et al.) by keyboard-
   gate control of the topological/trivial boundary in a T-junction. The thing
   that moves is a **domain wall**, moved by DC gate voltages. Rev 2.1 already
   states "no vortex is created or moved during operation," so "without
   mechanical gate switching" describes a problem that does not exist: gate
   switching is already electrical. The paper's "effective motion" is a
   rotating *modulation pattern* in classical resonators emulating a moving
   medium for EM waves; it does not translate a topological domain wall in a
   proximitized channel.
   Two further hard conflicts: **braiding must be adiabatic** — Rev 2.1
   specifies t_b = 1 µs ≫ ħ/Δ_top = 0.63 ps, so *ultrafast* drive is precisely
   what destroys the topological protection; and a ring of 100 MHz-modulated
   coils around the channel generates AC magnetic fields, directly against the
   B⊥ ≤ 2 mT / 0.028° alignment budget (§3.2) that excludes pancake vortices.
2. **Super-radiant parity readout.** The mechanism's **gain increases with
   parasitic loss** — the paper states that higher resonator losses give
   larger effective amplification, because this is the time-reversed regime in
   which loss becomes gain. A qubit preamplifier must be **quantum-limited**,
   adding noise near ħω/2; an amplifier whose gain requires dissipation adds
   noise in proportion to it. 7.8 dB of loss-enabled gain is not a qubit
   readout chain. "Across the temperature gradient" is also not a thing an
   amplifier does — it sits at a stage. And it cannot be built by ImpCarv,
   which produces polymer, not metal.

**The one salvageable idea, marked speculative.** The paper's genuine
contribution is **magnetic-free non-reciprocity** — replacing gyromagnetic
isolators and circulators with spatio-temporal modulation. Ferrite circulators
are bulky, lossy, and magnetic, and a magnet-free circulator is a real and
active line of work for superconducting-qubit readout (Alù's group among
others). For an architecture whose entire field budget is 2 mT and 0.028°, a
magnet-free circulator in the readout chain is worth tracking as a **future
component option**. That is a different claim from rotational super-radiance,
and it is not part of the current design.

---

## 11. References

- Rev 2.1, *Solid-State 0.3 K Cuprate–TI Topological Processor*, 6 July 2026.
- M. Li, H. Wu, Z. Qin, C. Su, H. D. Nguyen, Y. Hu, "Phonon focusing at room
  temperature," *Nature Physics* (23 July 2026),
  [doi:10.1038/s41567-026-03335-y](https://doi.org/10.1038/s41567-026-03335-y).
- A. Y. Kitaev, *Phys.-Usp.* **44**, 131 (2001).
- Y. Oreg, G. Refael, F. von Oppen, *PRL* **105**, 177002 (2010);
  R. M. Lutchyn, J. D. Sau, S. Das Sarma, *PRL* **105**, 077001 (2010).
- O. Can et al., "High-temperature topological superconductivity in twisted
  double-layer copper oxides," *Nat. Phys.* **17**, 519 (2021).
- S. Y. F. Zhao et al., *Science* **382**, 1422 (2023).
- T. Karzig et al., *PRB* **95**, 235305 (2017).
- E. M. Stoudenmire, J. Alicea, O. A. Starykh, M. P. A. Fisher, *PRB* **84**,
  014503 (2011).
- L. Lindsay, D. A. Broido, T. L. Reinecke, *PRL* **111**, 025901 (2013);
  BAs κ measurements, *Science* **361** (2018).
- J. P. Wolfe, *Imaging Phonons* (Cambridge, 1998) — phonon focusing and
  caustics.
