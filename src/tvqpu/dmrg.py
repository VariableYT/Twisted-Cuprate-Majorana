"""
dmrg.py -- Module 2.  DMRG over the Module 1 MPO.

SCOPE.  DMRG is used here ONLY for the interacting extension, where the model
stops being quadratic and bond dimension becomes a real cost.  Everything
quadratic -- Delta_top, xi, the edge splitting, the BdG phase boundary -- is
exact diagonalization in ``tvqpu.lattice`` and must not be recomputed here.
See ARCHITECTURE.md section 1.

UNITS.  This module is DIMENSIONLESS, in units of t.  There are no meV past
the ``MajoranaChannel.interacting()`` handoff.  A disorder strength computed
here transfers to the device only as a RATIO TO THE GAP -- see
``tolerance_in_gap_units`` for why, and for the arithmetic that makes the
comparison to Rev 2.1's sigma spec meaningful rather than misleading.

TWO QUIMB TRAPS, both of which silently produce wrong physics rather than
errors, and both of which this module wraps:

  * ``.correlation()`` returns the CONNECTED correlator.  In a symmetry-broken
    DMRG ground state that is ~0 even deep in the ordered phase -- it looks
    exactly like the topological phase has vanished everywhere.  The order
    parameter needs the raw correlator, so <Y_i><Y_j> must be added back.
  * ``.magnetization()`` returns <S^y> = <Y>/2, not <Y>.  Factor 2 per site,
    hence 4x in the correlator.

The reconstruction was verified against direct contraction at L=20 in an
earlier session (0.93052 vs 0.93038) and is re-verified by ``validate()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from tvqpu.lattice import InteractingChain

__all__ = [
    "DMRGResult", "ground_state", "raw_correlator", "order_parameter",
    "clean_gap", "disorder_ensemble", "disorder_threshold",
    "tolerance_in_gap_units", "validate",
]

PAULI_Y = np.array([[0, -1j], [1j, 0]])


def _require_quimb():
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Module 2 needs quimb; `pip install quimb`") from exc
    return qtn


# --------------------------------------------------------------------------
# Ground state
# --------------------------------------------------------------------------
@dataclass
class DMRGResult:
    energy: float
    order_parameter: float
    entropy: float
    cutoff: float          # SVD cutoff requested (an upper bound per truncation)
    bond_dim: int          # bond dimension actually reached
    bond_dim_max: int      # bond dimension allowed
    energy_delta: float    # |E_last - E_previous| across the final sweep
    n_sweeps: int
    converged: bool

    # ---- multi-restart diagnostics (see ``ground_state``) -----------------
    n_restarts: int = 1
    #: Energies from every restart, ascending.  Length == n_restarts.
    restart_energies: tuple[float, ...] = ()
    #: Order parameters from every restart, in the SAME order as
    #: ``restart_energies`` (i.e. index 0 is the lowest-energy solution).
    restart_order_parameters: tuple[float, ...] = ()

    @property
    def energy_spread(self) -> float:
        """E_max - E_min across restarts.  Zero for a single restart.

        THIS IS THE MULTISTABILITY DIAGNOSTIC.  A strictly positive spread
        means different random initial states relaxed to genuinely different
        variational minima, so the single-shot answer was landing on whichever
        basin the initialization happened to fall into.
        """
        if len(self.restart_energies) < 2:
            return 0.0
        return float(max(self.restart_energies) - min(self.restart_energies))

    @property
    def order_parameter_spread(self) -> float:
        """Range of the order parameter across restarts.

        The number that actually bit us: ARCHITECTURE.md section 2.5b records
        paired chi=96 vs chi=256 runs disagreeing by up to 0.38 in |<YY>| at
        *fixed disorder*, at bond dimensions nowhere near either ceiling.  A
        large spread here on a single chi reproduces that effect directly and
        attributes it correctly -- to the optimizer landscape, not truncation.
        """
        if len(self.restart_order_parameters) < 2:
            return 0.0
        vals = [abs(v) for v in self.restart_order_parameters]
        return float(max(vals) - min(vals))

    @property
    def multistable(self, energy_tol: float = 1e-9) -> bool:
        """True if restarts found distinct minima rather than the same one.

        Uses the energy, not the order parameter: two restarts reaching the
        same energy but different order parameters would indicate a genuine
        degeneracy (e.g. the two symmetry-broken ferromagnetic states), which
        is physics rather than an optimization failure.
        """
        return self.energy_spread > energy_tol

    @property
    def bond_saturated(self) -> bool:
        """True if the run hit its bond-dimension ceiling.

        THIS IS THE REAL SELF-DIAGNOSTIC AVAILABLE HERE.  quimb's DMRG2 does
        not expose the discarded weight, so ``cutoff`` is only the requested
        bound, not a measurement.  What can be measured is which constraint
        bound: if the bond dimension stayed BELOW the ceiling, the SVD cutoff
        was binding and the truncation really is at or under ``cutoff``.  If
        it saturated, the ceiling was binding, the true discarded weight is
        unknown and possibly large, and the run may have left the
        low-entanglement regime where MPS is valid.

        Near a critical point this will saturate -- S(l) ~ (c/6) log l means
        the required chi grows with length -- which is exactly when the result
        should not be trusted at face value.
        """
        return self.bond_dim >= self.bond_dim_max

    @property
    def trustworthy(self) -> bool:
        """Converged, not truncation-limited, and not multistable.

        All three matter.  The third was added after the chi=256 study
        (ARCHITECTURE 2.5b) showed that the first two can both pass while the
        answer is still landing in an arbitrary local minimum.  A run with
        n_restarts=1 cannot detect multistability at all, so it reports
        ``multistable = False`` by construction -- absence of evidence, not
        evidence of absence.
        """
        return (self.converged and not self.bond_saturated
                and not self.multistable)


def raw_correlator(psi, i: int, j: int) -> float:
    """<Y_i Y_j>, RAW rather than connected.  See the module docstring."""
    conn = psi.correlation(PAULI_Y, i, j)
    m_i = 2.0 * psi.magnetization(i, direction="Y")
    m_j = 2.0 * psi.magnetization(j, direction="Y")
    return float(np.real(conn + m_i * m_j))


def order_parameter(psi, n_sites: int) -> float:
    """Long-distance <Y_i Y_j> at quarter and three-quarter points.

    Nonzero exactly in the topological phase.  Measured away from the edges so
    the Majorana end modes do not contaminate the bulk order parameter.
    """
    return raw_correlator(psi, n_sites // 4, 3 * n_sites // 4)


def _single_run(mpo, chain: InteractingChain, chi: int, tol: float,
                max_sweeps: int, cutoff: float, init_seed: int | None):
    """One DMRG solve from one initial state.  Returns (result, psi)."""
    qtn = _require_quimb()

    p0 = None
    if init_seed is not None:
        import quimb
        # Seed quimb's generators so the random initial MPS is reproducible.
        # Without this the restarts are irreproducible, which would make a
        # multistability finding impossible to re-examine later.
        quimb.seed_rand(init_seed)
        p0 = qtn.MPS_rand_state(chain.n_sites, bond_dim=min(chi, 16),
                                dtype="complex128")

    schedule = [b for b in (16, 32, 64, 128, 256) if b < chi] + [chi]
    dmrg = qtn.DMRG2(mpo, bond_dims=schedule, cutoffs=cutoff, p0=p0)
    converged = dmrg.solve(tol=tol, verbosity=0, max_sweeps=max_sweeps)
    psi = dmrg.state
    energies = list(dmrg.energies or [])
    delta = (abs(float(np.real(energies[-1] - energies[-2])))
             if len(energies) >= 2 else float("nan"))
    res = DMRGResult(
        energy=float(np.real(dmrg.energy)),
        order_parameter=order_parameter(psi, chain.n_sites),
        entropy=float(np.real(psi.entropy(chain.n_sites // 2))),
        cutoff=float(cutoff),
        bond_dim=int(max(psi.bond_sizes())),
        bond_dim_max=int(chi),
        energy_delta=delta,
        n_sweeps=len(energies),
        converged=bool(converged),
    )
    return res, psi


def ground_state(chain: InteractingChain, chi: int = 80,
                 tol: float = 1e-8, max_sweeps: int = 25,
                 cutoff: float = 1e-11, n_restarts: int = 1,
                 restart_seed: int = 0) -> DMRGResult:
    """Two-site DMRG ground state of ``chain``, via the Module 1 MPO.

    Uses the real (Y -> iY) MPO storage by default -- half the memory, and the
    dense reconstruction is asserted equal to the complex form in
    ``tvqpu.lattice.validate()``.

    MULTI-RESTART (``n_restarts > 1``)
    ----------------------------------
    Runs the solve ``n_restarts`` times from *different random initial states*
    and returns the LOWEST-ENERGY result.  This is standard practice for
    disordered systems and it is here because of a measured failure, not on
    principle.

    ARCHITECTURE.md section 2.5b records paired chi=96 vs chi=256 runs on
    *identical* disorder realizations disagreeing by up to 0.38 in the order
    parameter -- at bond dimensions of 8 and 32, nowhere near either ceiling.
    Truncation could not explain that.  The cause is that strongly disordered
    1D chains near the transition have near-degenerate low-lying states
    (rare-region / Griffiths physics), so a variational method lands in
    whichever basin its initialization and bond-dimension growth schedule
    steer it toward.  Restarting from several initial states and keeping the
    best energy is the standard fix.

    IMPORTANT -- this changes what "converged" means.  A single restart cannot
    detect multistability, so ``DMRGResult.multistable`` is False by
    construction at ``n_restarts=1``.  That is absence of evidence.  Deep in
    the collapsing regime (W >~ 6 for the sigma sweep) use several restarts
    and check ``energy_spread`` / ``order_parameter_spread`` before trusting a
    number.

    ``restart_seed`` makes the restarts reproducible: restart k is
    initialized from ``restart_seed * 1000 + k``, so two calls with the same
    arguments give bit-identical answers.  That is what makes a multistability
    finding re-examinable rather than anecdotal.

    NOTE the asymmetry: ``n_restarts=1`` is deliberately NOT seeded, because it
    reproduces the original code path exactly (quimb's own unseeded default
    initial state) so existing ledgers stay comparable.  The cost is that
    single-restart calls are only reproducible to solver tolerance (~1e-13 in
    the energy), not bit-for-bit.  If you need bit-reproducibility, use
    ``n_restarts>=1`` *with* an explicit ``restart_seed`` -- which takes the
    seeded path even for a single restart is not the case here, so use
    ``n_restarts=2`` as the smallest reproducible setting.

    WHICH ANSWER IS RIGHT is decided by energy, not by bond dimension.
    Measured on L=80, W=8, seed=12 (ARCHITECTURE 2.5b): chi=96 found
    E=-117.43145 with order parameter 0.201, while chi=256 found the *higher*
    E=-117.42991 with 0.014.  The larger chi was not more accurate -- it was
    trapped in a worse basin.  Four restarts at chi=96 confirm the lower
    energy and its 0.2006 order parameter.  Across five checked disagreements
    chi=96 won twice and chi=256 won three times, which is the signature of
    basin trapping rather than truncation (truncation would favour chi=256
    systematically).

    Cost is linear in ``n_restarts``.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    mpo = chain.to_mpo(real=False).to_quimb()

    if n_restarts == 1:
        # Preserve the original code path bit-for-bit: quimb's own default
        # initial state, no seeding. Existing ledgers stay reproducible.
        res, _ = _single_run(mpo, chain, chi, tol, max_sweeps, cutoff, None)
        return replace(res, n_restarts=1,
                       restart_energies=(res.energy,),
                       restart_order_parameters=(res.order_parameter,))

    runs = [_single_run(mpo, chain, chi, tol, max_sweeps, cutoff,
                        restart_seed * 1000 + k)[0]
            for k in range(n_restarts)]
    order = sorted(range(len(runs)), key=lambda i: runs[i].energy)
    best = runs[order[0]]
    return replace(
        best,
        n_restarts=n_restarts,
        restart_energies=tuple(runs[i].energy for i in order),
        restart_order_parameters=tuple(runs[i].order_parameter for i in order),
        # A restart set is only "converged" if every member converged --
        # one failed restart means the minimum may not have been found.
        converged=all(r.converged for r in runs),
    )


# --------------------------------------------------------------------------
# The clean gap -- the denominator that makes disorder transferable
# --------------------------------------------------------------------------
def clean_gap(chain: InteractingChain) -> float:
    """Bulk excitation gap of the CLEAN chain, in units of t.

    At t = Delta and V = 0 the Jordan-Wigner image is the transverse-field
    Ising model H = -J sum Y Y - h sum Z with J = t and h = mu/2, whose bulk
    gap is 2|J - h| = |2t - mu|.  That is the closed form used here.

    For V != 0 there is no closed form and this returns the V = 0 value as a
    reference scale, which is what the tolerance ratio needs -- the point is a
    consistent denominator, not a precise many-body gap.
    """
    if abs(chain.t - chain.delta) > 1e-12:
        raise NotImplementedError(
            "clean_gap has a closed form only in the t = Delta (TFIM) limit; "
            "for Delta/t != 1 compute E1 - E0 numerically instead")
    return abs(2.0 * chain.t - chain.mu)


def tolerance_in_gap_units(w_threshold: float, gap: float) -> float:
    """Convert a box-disorder threshold W_c to the dimensionless ratio
    delta-mu_rms / gap, which is the ONLY form that transfers to the device.

    For a uniform box of width W, delta-mu_rms = W / sqrt(12).

    WHY THE RATIO AND NOT W/t.  Rev 2.1's sigma is delta-mu_rms as a fraction
    of t, and its t is 20 meV against a 1.05 meV gap -- a gap-to-hopping ratio
    of 0.05.  This toy chain runs at t = Delta, where that ratio is ~1.  A
    tolerance quoted as a fraction of t is therefore off by a factor of ~20
    between the two models, which is exactly the kind of unit slip that turns
    a simulation result into a wrong spec.  The ratio to the GAP is
    model-independent to leading order and is what should be compared.

    Reference points on the Rev 2.1 side:
        spec  sigma = 0.2%  -> delta-mu_rms = 40 ueV  -> ratio 0.038
        onset sigma ~ 3%    -> delta-mu_rms = 600 ueV -> ratio 0.571
    """
    if gap <= 0:
        raise ValueError("gap must be positive")
    return (w_threshold / math.sqrt(12.0)) / gap


# --------------------------------------------------------------------------
# Disorder ensembles
# --------------------------------------------------------------------------
@dataclass
class EnsembleResult:
    w: float
    mean: float
    std: float
    sem: float
    n_realizations: int
    n_trustworthy: int

    def __str__(self) -> str:
        return (f"W={self.w:5.2f}  <YY> = {self.mean:.4f} +/- {self.sem:.4f} "
                f"(sd {self.std:.4f}, n={self.n_realizations}, "
                f"ok={self.n_trustworthy})")


def disorder_ensemble(chain: InteractingChain, w: float, n_realizations: int,
                      chi: int = 80, seed0: int = 0, **kw) -> EnsembleResult:
    """Order parameter over ``n_realizations`` disorder realizations.

    Eight realizations -- what an earlier sweep used -- leaves the standard
    deviation itself uncertain at the ~25% level, which is why the variance
    claims from that run were called suggestive rather than established.
    Fifty is the minimum for a quotable spread.
    """
    vals, ok = [], 0
    for k in range(n_realizations):
        realization = (chain if w == 0.0
                       else chain.with_disorder(w=w, seed=seed0 + k))
        r = ground_state(realization, chi=chi, **kw)
        vals.append(abs(r.order_parameter))
        ok += int(r.trustworthy)
    a = np.asarray(vals)
    return EnsembleResult(w=w, mean=float(a.mean()), std=float(a.std(ddof=1)),
                          sem=float(a.std(ddof=1) / math.sqrt(len(a))),
                          n_realizations=len(a), n_trustworthy=ok)


def disorder_threshold(ensembles: list[EnsembleResult],
                       fraction: float = 0.5) -> float | None:
    """Disorder strength at which the order parameter falls to ``fraction`` of
    its clean value, by linear interpolation.

    Returns None if the sweep never reached the threshold -- which is what
    happened to the earlier W <= 4 sweep, and why that run could not convert
    the asserted sigma spec into a derived one.  A None here is a real result
    and must not be reported as a large tolerance.
    """
    if not ensembles:
        return None
    ordered = sorted(ensembles, key=lambda e: e.w)
    clean = ordered[0].mean
    target = fraction * clean
    for a, b in zip(ordered, ordered[1:]):
        if a.mean >= target > b.mean:
            span = a.mean - b.mean
            if span <= 0:
                return b.w
            return a.w + (b.w - a.w) * (a.mean - target) / span
    return None


# --------------------------------------------------------------------------
# Validation gate
# --------------------------------------------------------------------------
def validate(verbose: bool = True) -> bool:
    """Known-answer checks.  Must pass before any sweep is trusted."""
    ok = True

    def check(label: str, good: bool, detail: str) -> bool:
        if verbose:
            print(f"[{'PASS' if good else 'FAIL'}] {label}: {detail}")
        return good

    # 1. DMRG vs exact diagonalization, including with interactions on.
    for mu, v in ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (1.0, 1.0), (1.0, -1.0)):
        chain = InteractingChain(n_sites=12, t=1.0, delta=1.0, mu=mu, v_int=v)
        e_dmrg = ground_state(chain, chi=64).energy
        e_exact = float(np.linalg.eigvalsh(chain.to_dense())[0].real)
        rel = abs(e_dmrg - e_exact) / abs(e_exact)
        ok &= check(f"DMRG vs ED (mu={mu}, V={v})", rel < 1e-9,
                    f"{e_dmrg:.11f} vs {e_exact:.11f}, rel {rel:.1e}")

    # 2. The V = 0 phase boundary must reproduce the analytic mu_c = 2t.
    ordered = ground_state(InteractingChain(n_sites=48, mu=1.0), chi=64)
    trivial = ground_state(InteractingChain(n_sites=48, mu=3.0), chi=64)
    ok &= check("V=0 phase boundary at mu_c = 2t",
                ordered.order_parameter > 0.5 and abs(trivial.order_parameter) < 1e-2,
                f"<YY> = {ordered.order_parameter:.4f} at mu=1, "
                f"{trivial.order_parameter:.2e} at mu=3")

    # 3. The connected-correlator trap: the raw correlator must NOT be ~0 in
    #    the ordered phase.  If this fires, .correlation() is being used bare.
    ok &= check("raw correlator reconstruction", ordered.order_parameter > 0.5,
                "nonzero deep in the ordered phase (connected would be ~0)")

    # 4. Disorder is reproducible.
    a = disorder_ensemble(InteractingChain(n_sites=32, mu=1.0), w=1.0,
                          n_realizations=3, chi=48, seed0=7)
    b = disorder_ensemble(InteractingChain(n_sites=32, mu=1.0), w=1.0,
                          n_realizations=3, chi=48, seed0=7)
    ok &= check("disorder reproducibility", abs(a.mean - b.mean) < 1e-9,
                f"{a.mean:.10f} vs {b.mean:.10f}")

    # 5. The unit conversion, against the Rev 2.1 reference points.
    ratio = tolerance_in_gap_units(w_threshold=math.sqrt(12.0) * 0.571, gap=1.0)
    ok &= check("gap-units conversion", abs(ratio - 0.571) < 1e-9,
                f"W/sqrt(12)/gap round-trips ({ratio:.3f})")

    print("\nVALIDATION", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)
