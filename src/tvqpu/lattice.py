"""
lattice.py — Module 1. Geometry in, Hamiltonian out.

Builds the lattice Hamiltonians of ARCHITECTURE.md §2-3 and emits them in the
two forms the rest of the stack consumes:

  * ``.to_bdg_dense()``  -> (4N, 4N) Bogoliubov-de Gennes matrix for exact
    diagonalization.  The model is quadratic, so this is EXACT, not an
    approximation, and it costs seconds.
  * ``.to_mpo()``        -> finite-state-machine matrix product operator for
    DMRG.  Used ONLY for the interacting extension, where the model stops
    being quadratic and bond dimension becomes a real cost.

This module SOLVES NOTHING beyond what a validation gate needs.  Ground states,
sweeps, and disorder ensembles live in ``tvqpu.dmrg`` and ``tvqpu.campaign``.

PROVENANCE CONTRACT
  * Parameters in ``REV21`` are transcribed from *Solid-State 0.3 K Cuprate-TI
    Topological Processor, Rev. 2.1* (6 July 2026), Table 2.  Do not edit them
    to make a test pass; edit the test, and say why.
  * The operational topological gap Delta_top = 1.05 meV is a BdG
    exact-diagonalization result.  It is NOT a DMRG result.  It has been
    misattributed to DMRG more than once in this project's history.  DMRG's
    contribution is the *interacting* extension of section 2.4, whose V and mu
    are DIMENSIONLESS, in units of t, and contain no meV.
  * The pairing gap Delta = 2 meV has never been measured.  Every protection
    budget downstream scales exponentially with it.  Rev 2.1 section 9.2 is
    right to make tunneling spectroscopy of the induced gap Milestone 1.
  * The honeycomb superlattice solver is ported from the metamaterial GNN
    pipeline (``115 sim/generate_labels.py``) and inherits its honesty
    contract unchanged: it computes BAND-STRUCTURE INGREDIENTS (flatness,
    gaps) of a single-orbital nearest-neighbour model.  It does not compute
    Tc, pairing, or many-body physics.

  * ``--validate`` must pass before any batch is trusted.  It checks the
    solvers against exact known physics (analytic V_z,crit; the +2t
    band-bottom convention; MPO vs an independently built dense Hamiltonian;
    graphene bandwidth 6t and the Dirac point).

Usage:
  python -m tvqpu.lattice --validate
  python -m tvqpu.lattice --gap --n-sites 400
"""

from __future__ import annotations

import os

# Pin BLAS to 1 thread BEFORE numpy import.  Parallelism in this project comes
# from running many shard processes; N <= 800 eigensolves gain nothing from
# threads, and unpinned shards have previously produced 640 threads on a
# 32-core box (load 553, ~50x slowdown).  Env exports from a parent shell
# proved unreliable on the cluster image, so it is done here.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import sys
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "ModelParams", "REV21", "REV20_BUGGY",
    "MajoranaChannel", "InteractingChain", "MPO",
    "HoneycombSuperlattice",
    "PAULI_I", "PAULI_X", "PAULI_Y", "PAULI_Z",
    "validate",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

K_B_MEV_PER_K = 0.08617333262  # meV/K


def _kron(tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """tau (x) sigma in the Nambu (x) spin basis (c_up, c_dn, c^dag_up, c^dag_dn)."""
    return np.kron(tau, sigma)


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelParams:
    """Tight-binding BdG parameters.  All energies in meV, lengths in nm."""

    t: float = 20.0           # hopping, hbar^2 / 2 m* a^2  (m* ~ 0.019 m_e)
    alpha: float = 10.0       # effective spin-orbit scale
    delta: float = 2.0        # proximity-induced pairing -- DESIGN ASSUMPTION
    v_z: float = 3.0          # Zeeman energy; B_par = 4.15 T at g ~ 25
    mu: float = 0.0           # chemical potential, measured FROM THE BAND BOTTOM
    a_nm: float = 10.0        # lattice constant
    temperature_k: float = 0.3

    # The +2t onsite offset is what makes "mu measured from the band bottom"
    # true.  Rev 2.0 omitted it, which put the Fermi level mid-band: the wire
    # never entered the topological phase and the true critical field was
    # sqrt((2t)^2 + Delta^2) ~ 20 meV rather than 2 meV.  Rev 2.1 Appendix B.1.
    # Kept switchable ONLY so the regression test can demonstrate the bug.
    band_bottom_offset: bool = True

    @property
    def v_z_crit(self) -> float:
        """Analytic bulk gap-closing field, sqrt(mu^2 + Delta^2) (Rev 2.1 eq. after 4)."""
        if not self.band_bottom_offset:
            # The Rev 2.0 convention: mu = 0 sits mid-band, so the offset the
            # topological criterion sees is 2t rather than mu.
            return math.hypot(2.0 * self.t + self.mu, self.delta)
        return math.hypot(self.mu, self.delta)

    @property
    def k_b_t(self) -> float:
        """k_B T in meV."""
        return K_B_MEV_PER_K * self.temperature_k

    def thermal_suppression(self, gap_mev: float) -> float:
        """exp(-Delta/k_B T).  LOWER BOUND on the error rate only -- real
        Majorana devices are limited by non-equilibrium quasiparticle
        poisoning, bursts, and control error, none of which follow this form.
        See ARCHITECTURE.md section 6."""
        return math.exp(-gap_mev / self.k_b_t)


#: Rev 2.1 Table 2, the unified specification.  Source of truth.
REV21 = ModelParams()

#: The Rev 2.0 convention, retained solely as a regression fixture.
REV20_BUGGY = replace(REV21, band_bottom_offset=False)


# --------------------------------------------------------------------------
# Module 1a -- the BdG channel
# --------------------------------------------------------------------------
@dataclass
class MajoranaChannel:
    """Gate-defined quasi-1D channel on a Bi2Se3 surface, proximitized by a
    45-degree-twisted BSCCO bilayer.  ARCHITECTURE.md section 2.1.

    N sites in the 4-component Nambu (x) spin basis:

        H_on  = (2t - mu)(tau_z (x) sigma_0) + Delta(tau_x (x) sigma_0)
                                             + V_z(tau_0 (x) sigma_z)
        H_hop = -t(tau_z (x) sigma_0) - i alpha(tau_z (x) sigma_y)

    Particle-hole symmetry P H(k) P^-1 = -H(-k), P = (tau_y (x) sigma_y) K,
    P^2 = +1  =>  Altland-Zirnbauer class D, Z2 Pfaffian invariant.
    """

    n_sites: int = 200
    params: ModelParams = REV21

    # Optional per-site disorder, in meV, added to mu_j and t_j.  Length
    # n_sites and n_sites-1 respectively.  ``None`` means pristine.
    delta_mu: np.ndarray | None = None
    delta_t: np.ndarray | None = None

    # ---------------- geometry ----------------
    @property
    def length_nm(self) -> float:
        return self.n_sites * self.params.a_nm

    @property
    def length_um(self) -> float:
        return self.length_nm / 1000.0

    def with_disorder(self, sigma: float, seed: int) -> "MajoranaChannel":
        """Gaussian site-to-site disorder of relative variance ``sigma`` on
        mu_j and t_j, as in Rev 2.1 section 8.1 (delta ~ N(0, (sigma t)^2))."""
        rng = np.random.default_rng(seed)
        scale = sigma * self.params.t
        return replace(
            self,
            delta_mu=rng.normal(0.0, scale, self.n_sites),
            delta_t=rng.normal(0.0, scale, max(self.n_sites - 1, 0)),
        )

    # ---------------- Hamiltonian emission ----------------
    def onsite_block(self, j: int = 0) -> np.ndarray:
        p = self.params
        mu_j = p.mu + (0.0 if self.delta_mu is None else float(self.delta_mu[j]))
        offset = 2.0 * p.t if p.band_bottom_offset else 0.0
        return (
            (offset - mu_j) * _kron(PAULI_Z, PAULI_I)
            + p.delta * _kron(PAULI_X, PAULI_I)
            + p.v_z * _kron(PAULI_I, PAULI_Z)
        )

    def hopping_block(self, j: int = 0) -> np.ndarray:
        p = self.params
        t_j = p.t + (0.0 if self.delta_t is None else float(self.delta_t[j]))
        return -t_j * _kron(PAULI_Z, PAULI_I) - 1j * p.alpha * _kron(PAULI_Z, PAULI_Y)

    def to_bdg_dense(self) -> np.ndarray:
        """The (4N, 4N) real-space BdG matrix.  Hermitian by construction."""
        n = self.n_sites
        h = np.zeros((4 * n, 4 * n), dtype=complex)
        for j in range(n):
            h[4 * j:4 * j + 4, 4 * j:4 * j + 4] = self.onsite_block(j)
        for j in range(n - 1):
            blk = self.hopping_block(j)
            h[4 * j:4 * j + 4, 4 * (j + 1):4 * (j + 1) + 4] = blk
            h[4 * (j + 1):4 * (j + 1) + 4, 4 * j:4 * j + 4] = blk.conj().T
        # Guard: a non-Hermitian BdG matrix is a construction bug, not physics.
        asym = np.abs(h - h.conj().T).max()
        if asym > 1e-10:
            raise RuntimeError(f"BdG matrix not Hermitian (max asym {asym:.3e})")
        return h

    def bulk_hamiltonian(self, k: float) -> np.ndarray:
        """H(k) of Rev 2.1 eq. (4).  k in rad/nm."""
        p = self.params
        ka = k * p.a_nm
        # The +2t convention, written explicitly: with the offset, mu = 0 sits
        # at the band bottom (f1(k=0) = -mu).  Without it, mu = 0 sits
        # mid-band (f1(k=0) = -2t - mu) -- the Rev 2.0 bug.
        f1 = (2.0 * p.t * (1.0 - math.cos(ka)) if p.band_bottom_offset else
              -2.0 * p.t * math.cos(ka)) - p.mu
        f2 = 2.0 * p.alpha * math.sin(ka)
        return (
            f1 * _kron(PAULI_Z, PAULI_I)
            + f2 * _kron(PAULI_Z, PAULI_Y)
            + p.delta * _kron(PAULI_X, PAULI_I)
            + p.v_z * _kron(PAULI_I, PAULI_Z)
        )

    # ---------------- the minimum solving a gate needs ----------------
    def spectrum(self) -> np.ndarray:
        """Sorted BdG eigenvalues in meV."""
        return np.linalg.eigvalsh(self.to_bdg_dense())

    def eigensystem(self) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.to_bdg_dense())

    def bulk_gap(self, nk: int = 2001) -> float:
        """Minimum positive excitation energy of H(k) over the Brillouin zone.

        This is Delta_top: the energy scale that actually suppresses poisoning
        and bounds adiabaticity.  Rev 2.1 quotes 1.05 meV at V_z = 3 meV.
        Computed from the BULK Hamiltonian, so it is free of the end-mode
        contribution that would otherwise dominate a finite chain.
        """
        kmax = math.pi / self.params.a_nm
        ks = np.linspace(-kmax, kmax, nk)
        best = math.inf
        for k in ks:
            ev = np.linalg.eigvalsh(self.bulk_hamiltonian(float(k)))
            pos = ev[ev > 0]
            if pos.size:
                best = min(best, float(pos.min()))
        return best

    def edge_splitting(self) -> float:
        """Majorana hybridization splitting delta-E in meV: the lowest positive
        eigenvalue of the finite chain.  Rev 2.1 reports 57 neV at N = 200."""
        ev = self.spectrum()
        pos = ev[ev > 0]
        return float(pos.min())

    def majorana_density(self) -> np.ndarray:
        """Site-resolved probability density of the lowest-energy BdG pair,
        summed over the two members and the 4 Nambu-spin components.
        Normalized to sum 1.  Reproduces Rev 2.1 Fig. 2."""
        ev, vecs = self.eigensystem()
        order = np.argsort(np.abs(ev))[:2]  # the +/- E pair closest to zero
        dens = np.zeros(self.n_sites)
        for idx in order:
            v = vecs[:, idx].reshape(self.n_sites, 4)
            dens += np.sum(np.abs(v) ** 2, axis=1)
        return dens / dens.sum()

    def localization_length(self, fit_frac: float = 0.25) -> float:
        """Fitted Majorana localization length xi, IN SITES.

        Exponential fit of log(density) over the outer ``fit_frac`` of the
        chain from the left edge, skipping site 0 (lattice-scale structure) and
        stopping before the density flattens into the mid-chain floor, which
        would bias the slope toward zero.  Rev 2.1 reports xi ~ 21 sites
        (~210 nm).
        """
        dens = self.majorana_density()
        n_fit = max(int(self.n_sites * fit_frac), 5)
        y = dens[1:n_fit]
        floor = dens[self.n_sites // 2 - 2:self.n_sites // 2 + 2].mean()
        keep = y > max(floor * 10.0, 1e-300)
        if keep.sum() < 3:
            raise RuntimeError(
                "not enough dynamic range to fit xi -- chain too short, or the "
                "wire is in the trivial phase (check V_z > V_z,crit)")
        x = np.arange(1, n_fit)[keep]
        slope = np.polyfit(x, np.log(y[keep]), 1)[0]
        if slope >= 0:
            raise RuntimeError("Majorana density is not decaying from the edge")
        # density ~ exp(-2 x / xi):  the mode amplitude decays with xi, the
        # probability density with xi/2.
        return float(-2.0 / slope)

    def edge_weight(self, n_edge: int = 25) -> float:
        """eta_edge: fraction of the Majorana density in the outer ``n_edge``
        sites at each end.  Rev 2.1 Fig. 5 disorder metric."""
        dens = self.majorana_density()
        return float(dens[:n_edge].sum() + dens[-n_edge:].sum())

    # ---------------- handoff to the interacting model ----------------
    def interacting(self, v_int: float = 0.0, t_over_delta: float | None = None
                    ) -> "InteractingChain":
        """Project onto the spinless (Zeeman-polarised) effective model and
        hand off to DMRG.

        WARNING -- UNITS.  The returned chain is DIMENSIONLESS, in units of t.
        There are no meV past this point.  Converting ``v_int`` to a screened
        Coulomb strength in gated Bi2Se3 needs a screening calculation that has
        not been done, so we cannot say where on the V axis the real device
        sits.  See ARCHITECTURE.md section 2.4.
        """
        ratio = (self.params.t / self.params.delta if t_over_delta is None
                 else t_over_delta)
        return InteractingChain(
            n_sites=self.n_sites, t=1.0, delta=1.0 / ratio,
            mu=self.params.mu / self.params.t * 1.0, v_int=v_int)


# --------------------------------------------------------------------------
# Module 1b -- MPO container and the interacting chain
# --------------------------------------------------------------------------
@dataclass
class MPO:
    """Finite-state-machine matrix product operator.

    ``tensors[j]`` has shape (D_left, D_right, d, d).  Boundary vectors
    ``v_left`` (D0,) and ``v_right`` (D_{L},) close the chain, so

        H = sum_{a...}  v_L[a0] W[0][a0,a1] W[1][a1,a2] ... W[L-1][a_{L-1},aL] v_R[aL]
    """

    tensors: list[np.ndarray]
    v_left: np.ndarray
    v_right: np.ndarray

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def bond_dim(self) -> int:
        return max(t.shape[1] for t in self.tensors)

    @property
    def phys_dim(self) -> int:
        return self.tensors[0].shape[2]

    def to_dense(self) -> np.ndarray:
        """Reconstruct the full 2^L x 2^L operator.

        For VALIDATION ONLY -- exponential in L.  Refuses past L = 14 (256 MB
        at complex128) rather than quietly exhausting memory.
        """
        n = self.n_sites
        if n > 14:
            raise ValueError(
                f"to_dense() is exponential; refusing n_sites={n} (> 14). "
                "This path exists to check the MPO against an independently "
                "built dense Hamiltonian, not to run physics.")
        d = self.phys_dim
        dtype = np.result_type(*(t.dtype for t in self.tensors),
                               self.v_left.dtype, self.v_right.dtype)
        # acc[a, i, j]: the partial operator, with the bond index a still open.
        acc = self.v_left.astype(dtype).reshape(-1, 1, 1)
        for j in range(n):
            w = self.tensors[j]
            # acc[a,i,j] , w[a,b,s,s'] -> new[b, (i s), (j s')]
            new = np.einsum("aij,absc->bisjc", acc, w)
            dim = acc.shape[1] * d
            acc = new.reshape(w.shape[1], dim, dim)
        return np.einsum("aij,a->ij", acc, self.v_right.astype(dtype))

    def to_quimb(self):
        """Return a ``quimb.tensor.MatrixProductOperator``.  Requires quimb."""
        try:
            from quimb.tensor import MatrixProductOperator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "quimb is required for to_quimb(); `pip install quimb`") from exc
        arrays = []
        n = self.n_sites
        for j, w in enumerate(self.tensors):
            # quimb wants (left, right, phys_out, phys_in), with the boundary
            # tensors already contracted against the boundary vectors.
            a = w
            if j == 0:
                a = np.einsum("a,absc->bsc", self.v_left.astype(complex), a)
            if j == n - 1:
                a = (np.einsum("bsc,b->sc", a, self.v_right.astype(complex))
                     if j == 0 else
                     np.einsum("absc,b->asc", a, self.v_right.astype(complex)))
            arrays.append(a)
        return MatrixProductOperator(arrays)


#: Y~ = iY, real and antisymmetric.  Y_j Y_k = -Y~_j Y~_k, so a Hamiltonian
#: whose only imaginary operators appear in YY pairs can be expressed with
#: entirely REAL MPO tensors.  That halves memory versus complex128 at zero
#: accuracy cost, and it is what lets the Triton kernel take its real float64
#: path in the common case rather than paying 4x for complex arithmetic.
PAULI_Y_REAL = np.array([[0.0, 1.0], [-1.0, 0.0]])


def build_fsm_mpo(n_sites: int,
                  hz: Sequence[float] | np.ndarray,
                  jx: Sequence[float] | np.ndarray,
                  jy: Sequence[float] | np.ndarray,
                  jz: Sequence[float] | np.ndarray,
                  real: bool = False) -> MPO:
    """Bond-dimension-5 MPO for

        H = sum_j hz[j] Z_j
          + sum_j ( jx[j] X_j X_{j+1} + jy[j] Y_j Y_{j+1} + jz[j] Z_j Z_{j+1} )

    ``hz`` has length n_sites; ``jx/jy/jz`` have length n_sites - 1 (bond j
    connects sites j and j+1).  Scalars are broadcast.

    ``real=True`` emits float64 tensors by substituting Y -> Y~ = iY and
    flipping the sign of the YY coupling.  The represented operator is
    identical; only the storage changes.  Requires hz/jx/jy/jz to be real.

    Per-site / per-bond coefficients make disorder realizations free: no
    rebuild, just a new coefficient vector.  That is what the sigma-tolerance
    sweep of ARCHITECTURE.md section 2.5 needs to run 50 realizations cheaply.

    Layout (row = incoming state, column = outgoing state):

        0 : nothing emitted yet
        1 : an X is waiting for its partner on the next site
        2 : a Y is waiting
        3 : a Z is waiting
        4 : the term is complete
    """
    nb = max(n_sites - 1, 0)
    dtype = np.float64 if real else complex
    if real:
        for name, arr in (("hz", hz), ("jx", jx), ("jy", jy), ("jz", jz)):
            if np.iscomplexobj(np.asarray(arr)) and np.any(np.imag(np.asarray(arr))):
                raise ValueError(f"real=True but {name} has an imaginary part")
    hz = np.broadcast_to(np.asarray(hz, dtype=dtype), (n_sites,))
    jx = np.broadcast_to(np.asarray(jx, dtype=dtype), (nb,))
    jy = np.broadcast_to(np.asarray(jy, dtype=dtype), (nb,))
    jz = np.broadcast_to(np.asarray(jz, dtype=dtype), (nb,))

    # Y -> Y~ = iY with jy -> -jy leaves Y_j Y_{j+1} unchanged (see above).
    y_op = PAULI_Y_REAL if real else PAULI_Y
    y_sign = -1.0 if real else 1.0
    eye = np.eye(2, dtype=dtype)
    x_op = PAULI_X.real if real else PAULI_X
    z_op = PAULI_Z.real if real else PAULI_Z

    d_bond = 5
    tensors = []
    for j in range(n_sites):
        w = np.zeros((d_bond, d_bond, 2, 2), dtype=dtype)
        w[0, 0] = eye
        w[4, 4] = eye
        w[0, 4] = hz[j] * z_op
        if j < nb:  # this site opens the bond to j+1
            w[0, 1] = x_op
            w[0, 2] = y_op
            w[0, 3] = z_op
        if j > 0:   # this site closes the bond from j-1, carrying its coupling
            w[1, 4] = jx[j - 1] * x_op
            w[2, 4] = y_sign * jy[j - 1] * y_op
            w[3, 4] = jz[j - 1] * z_op
        tensors.append(w)

    v_left = np.zeros(d_bond, dtype=dtype); v_left[0] = 1.0
    v_right = np.zeros(d_bond, dtype=dtype); v_right[4] = 1.0
    return MPO(tensors, v_left, v_right)


@dataclass
class InteractingChain:
    """Spinless interacting Kitaev chain, Jordan-Wigner mapped.

    DIMENSIONLESS -- energies are in units of t.  See the warning in
    ``MajoranaChannel.interacting``.

        H = -t sum (c^dag_j c_{j+1} + h.c.) + Delta sum (c_j c_{j+1} + h.c.)
            - mu sum (n_j - 1/2) + V sum n_j n_{j+1}

    Under JW with n_j - 1/2 -> Z_j / 2:

        H = -(mu/2) sum Z_j
            - ((t-Delta)/2) sum X_j X_{j+1}
            - ((t+Delta)/2) sum Y_j Y_{j+1}
            + (V/4) sum Z_j Z_{j+1}
            + (V/4) sum (Z_j + Z_{j+1})           [+ an irrelevant constant]

    At t = Delta this is the transverse-field Ising chain with order parameter
    <Y_i Y_j> and exact critical point mu_c = 2t -- the known answer this
    module validates against before anything novel is trusted.
    """

    n_sites: int = 60
    t: float = 1.0
    delta: float = 1.0
    mu: float = 0.0
    v_int: float = 0.0
    mu_disorder: np.ndarray | None = None  # per-site additive shift on mu

    #: Interaction convention.  True (default) uses the particle-hole
    #: symmetric form V sum (n_i - 1/2)(n_{i+1} - 1/2), which maps to a pure
    #: (V/4) ZZ term.  False uses the plain V sum n_i n_{i+1}, which also
    #: generates a field shift (V/4) sum (Z_i + Z_{i+1}).
    #:
    #: PH-symmetric is the default for three reasons: it is the convention of
    #: the literature this reproduces (Stoudenmire, Alicea, Starykh & Fisher,
    #: PRB 84, 014503), so mu_c comparisons transfer; and the plain form has a
    #: pathological point where the field shift exactly cancels the transverse
    #: field (V = mu, interior hz = 0), leaving a ferromagnetic Ising point
    #: whose two symmetry-broken ground states are degenerate to ~1e-4 at
    #: L = 12 -- which DMRG converges an order of magnitude worse than
    #: everywhere else.  That is a real near-degeneracy, not a solver bug, but
    #: there is no reason to run into it on purpose.
    ph_symmetric: bool = True

    def with_disorder(self, w: float, seed: int) -> "InteractingChain":
        """Uniform box disorder mu_j -> mu + U(-w/2, w/2), in units of t."""
        rng = np.random.default_rng(seed)
        return replace(self, mu_disorder=rng.uniform(-w / 2, w / 2, self.n_sites))

    def couplings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n, nb = self.n_sites, max(self.n_sites - 1, 0)
        mu_j = np.full(n, float(self.mu))
        if self.mu_disorder is not None:
            mu_j = mu_j + np.asarray(self.mu_disorder, dtype=float)

        # The plain V n_i n_{i+1} form also generates a field shift; the
        # particle-hole symmetric form does not.  See ``ph_symmetric``.
        if self.ph_symmetric:
            shift = np.zeros(n)
        else:
            n_bonds_at = np.full(n, 2.0)
            if n >= 1:
                n_bonds_at[0] = 1.0
                n_bonds_at[-1] = 1.0
            if n == 1:
                n_bonds_at[0] = 0.0
            shift = (self.v_int / 4.0) * n_bonds_at

        hz = -mu_j / 2.0 + shift
        jx = np.full(nb, -(self.t - self.delta) / 2.0)
        jy = np.full(nb, -(self.t + self.delta) / 2.0)
        jz = np.full(nb, self.v_int / 4.0)
        return hz, jx, jy, jz

    def to_mpo(self, real: bool = True) -> MPO:
        """MPO for this chain.  ``real=True`` (the default) emits float64
        tensors via the Y -> iY substitution -- half the memory, and it is the
        path the Triton kernel of Module 3 is fastest on.  The represented
        operator is identical either way, which ``validate()`` asserts."""
        return build_fsm_mpo(self.n_sites, *self.couplings(), real=real)

    def to_dense(self) -> np.ndarray:
        """Independently built dense Hamiltonian, by explicit Kronecker
        products.  Deliberately does NOT go through the MPO -- this is the
        reference the MPO is checked against."""
        n = self.n_sites
        if n > 14:
            raise ValueError(f"dense reference is exponential; refusing n={n}")
        hz, jx, jy, jz = self.couplings()
        dim = 2 ** n

        # Sparse Kronecker products: the operators are at most 2-local, so
        # every term has exactly 2^n nonzeros before summation.  Building
        # these densely costs 35 x dim^2 complex allocations at L = 12, which
        # is minutes; sparse makes it milliseconds.
        try:
            import scipy.sparse as sp
        except ImportError:  # pragma: no cover - numpy fallback
            sp = None

        if sp is None:
            eye, kron, zeros = np.eye(2, dtype=complex), np.kron, np.zeros
            h = zeros((dim, dim), dtype=complex)

            def build(mats):
                out = mats[0]
                for m in mats[1:]:
                    out = kron(out, m)
                return out
        else:
            eye = sp.identity(2, dtype=complex, format="csr")
            h = sp.csr_matrix((dim, dim), dtype=complex)

            def build(mats):
                out = mats[0]
                for m in mats[1:]:
                    out = sp.kron(out, m, format="csr")
                return out

        def wrap(op):
            return sp.csr_matrix(op) if sp is not None else op

        x, y, z = wrap(PAULI_X), wrap(PAULI_Y), wrap(PAULI_Z)

        for j in range(n):
            mats = [eye] * n
            mats[j] = z
            h = h + hz[j] * build(mats)
        for j in range(n - 1):
            for coeff, op in ((jx[j], x), (jy[j], y), (jz[j], z)):
                if abs(coeff) < 1e-15:
                    continue
                mats = [eye] * n
                mats[j] = mats[j + 1] = op
                h = h + coeff * build(mats)
        return np.asarray(h.todense()) if sp is not None else h

    @property
    def mu_c_free(self) -> float:
        """Analytic V = 0 phase boundary, mu_c = 2t."""
        return 2.0 * self.t


# --------------------------------------------------------------------------
# Module 1c -- honeycomb superlattice (ported from 115 sim/generate_labels.py)
# --------------------------------------------------------------------------
T_HOP_GRAPHENE = 2.7  # eV, nearest-neighbour hopping
A0_GRAPHENE = 2.46    # Angstrom
LAYOUTS = ("triangular", "square", "honeycomb", "kagome", "quasiperiodic")


@dataclass
class HoneycombSuperlattice:
    """Static tight-binding solver for a 2D honeycomb lattice under a passive
    structural-metamaterial configuration: a patterned superlattice potential
    plus embedded localized moments coupling through a collinear exchange term.

    HONESTY CONTRACT (carried over verbatim from generate_labels.py)
      * Computes BAND-STRUCTURE INGREDIENTS (flatness, gaps) of a
        single-orbital nearest-neighbour model.  Does NOT compute Tc, pairing,
        or many-body physics.  Labels mean "this passive layout reshapes the
        bands this way" -- nothing more.
      * Collinear approximation: only m_z enters the Hamiltonian.
      * Single-orbital NN graphene is a model system, not any specific real
        compound.  Transferring claims to real materials requires DFT.
    """

    n_cells: int = 6
    layout: str = "triangular"
    void_frac: float = 0.3
    well_depth_ev: float = 1.5
    moment_mu_b: float = 0.0
    j_ex_mev: float = 0.0
    seed: int = 0

    # ---------------- geometry ----------------
    def build_supercell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_cells
        a1 = np.array([A0_GRAPHENE, 0.0])
        a2 = np.array([A0_GRAPHENE / 2, A0_GRAPHENE * math.sqrt(3) / 2])
        basis = [np.zeros(2), (a1 + a2) / 3]  # A and B sublattice
        pos = [i * a1 + j * a2 + b
               for i in range(n) for j in range(n) for b in basis]
        pos = np.array(pos)
        lat = np.array([n * a1, n * a2])
        return pos @ np.linalg.inv(lat), pos, lat

    @staticmethod
    def neighbor_table(frac: np.ndarray, lat: np.ndarray, rcut: float = 1.6):
        """Nearest-neighbour pairs including periodic images."""
        cart = frac @ lat
        pairs = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                shift = di * lat[0] + dj * lat[1]
                d = cart[:, None, :] - (cart[None, :, :] + shift)
                dist = np.linalg.norm(d, axis=-1)
                ii, jj = np.where((dist > 1e-3) & (dist < rcut))
                pairs.extend((int(a), int(b), np.array([di, dj]))
                             for a, b in zip(ii, jj))
        return pairs

    def pattern_centers(self, rng) -> np.ndarray:
        """Well / moment centres in FRACTIONAL supercell coordinates."""
        table = {
            "triangular": np.array([[0.0, 0.0]]),
            "square": np.array([[0.0, 0.0], [0.5, 0.5]]),
            "honeycomb": np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]]),
            "kagome": np.array([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]),
        }
        if self.layout in table:
            return table[self.layout]
        if self.layout == "quasiperiodic":
            # frozen random motif -- a periodic supercell cannot host true
            # quasiperiodic order; this is an approximant.
            return rng.random((4, 2))
        raise ValueError(f"unknown layout {self.layout!r}")

    def site_masks(self, frac, lat, rng) -> np.ndarray:
        centers = self.pattern_centers(rng)
        cell_area = abs(lat[0][0] * lat[1][1] - lat[0][1] * lat[1][0])
        r_well = math.sqrt(max(self.void_frac, 1e-4) * cell_area
                           / (len(centers) * math.pi))
        cart = frac @ lat
        mask = np.zeros(len(frac), dtype=bool)
        for c in centers:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cc = (c + [di, dj]) @ lat
                    mask |= np.linalg.norm(cart - cc, axis=1) < r_well
        return mask

    # ---------------- Hamiltonian ----------------
    @staticmethod
    def bloch_bands(frac, lat, pairs, onsite_up, onsite_dn, kgrid: int = 6):
        """Diagonalize H(k) for both (decoupled, collinear) spin sectors over a
        kgrid x kgrid grid of the supercell BZ."""
        n = len(frac)
        b = 2 * math.pi * np.linalg.inv(lat).T
        ks = [b[0] * u + b[1] * v
              for u in np.linspace(0, 1, kgrid, endpoint=False)
              for v in np.linspace(0, 1, kgrid, endpoint=False)]
        evs = []
        for k in ks:
            h = np.zeros((n, n), dtype=complex)
            for i, j, img in pairs:
                r = img[0] * lat[0] + img[1] * lat[1]
                h[i, j] += -T_HOP_GRAPHENE * np.exp(1j * k @ r)
            e_up = np.linalg.eigvalsh(h + np.diag(onsite_up))
            e_dn = np.linalg.eigvalsh(h + np.diag(onsite_dn))
            evs.append(np.sort(np.concatenate([e_up, e_dn])))
        return np.array(evs)

    def solve(self, kgrid: int = 6) -> dict:
        rng = np.random.default_rng(self.seed)
        frac, _cart, lat = self.build_supercell()
        pairs = self.neighbor_table(frac, lat)
        wells = self.site_masks(frac, lat, rng)

        onsite = np.where(wells, self.well_depth_ev, 0.0)
        ex = (self.j_ex_mev / 1000.0) * self.moment_mu_b / 2.0
        onsite_up = onsite - np.where(wells, ex, 0.0)
        onsite_dn = onsite + np.where(wells, ex, 0.0)

        evs = self.bloch_bands(frac, lat, pairs, onsite_up, onsite_dn, kgrid)
        flat, gap_mev, w_mev = extract_band_labels(evs)
        return {"flatness": flat, "gap_mev": gap_mev, "width_mev": w_mev,
                "n_sites": len(frac), "wells": wells, "frac": frac, "lat": lat}


def extract_band_labels(evs: np.ndarray) -> tuple[float, float, float]:
    """Band nearest charge neutrality (E=0): width W, and the spectral gap.

    The gap is the GLOBAL spectral gap at E = 0 over the whole k-grid.  A
    per-band 'isolation gap' is ill-defined here: dense supercell spectra have
    hundreds of energy-overlapping sorted bands, which made that label
    identically zero.  (Lesson recorded in the original pipeline.)
    """
    band_centers = evs.mean(axis=0)
    idx = int(np.argmin(np.abs(band_centers)))
    band = evs[:, idx]
    w = float(band.max() - band.min())
    below, above = evs[evs < 0.0], evs[evs >= 0.0]
    gap = (max(0.0, float(above.min() - below.max()))
           if below.size and above.size else 0.0)
    flatness = -math.log10(max(w, 1e-6) / 1.0)  # W in eV
    return flatness, gap * 1000.0, w * 1000.0


# --------------------------------------------------------------------------
# Validation gate -- known physics must reproduce before anything is trusted
# --------------------------------------------------------------------------
def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def validate(verbose: bool = True) -> bool:
    """Run every known-answer check.  Returns True iff all pass."""
    ok = True

    # --- 1. Analytic bulk critical field, V_z,crit = sqrt(mu^2 + Delta^2) ----
    for mu in (0.0, 1.0, 2.5):
        p = replace(REV21, mu=mu)
        expect = math.hypot(mu, p.delta)
        # numerically: the k=0 gap closes when V_z = expect
        ch_lo = MajoranaChannel(n_sites=1, params=replace(p, v_z=expect - 0.02))
        ch_hi = MajoranaChannel(n_sites=1, params=replace(p, v_z=expect + 0.02))
        g_at = MajoranaChannel(n_sites=1, params=replace(p, v_z=expect))
        e0 = np.abs(np.linalg.eigvalsh(g_at.bulk_hamiltonian(0.0))).min()
        e_lo = np.abs(np.linalg.eigvalsh(ch_lo.bulk_hamiltonian(0.0))).min()
        e_hi = np.abs(np.linalg.eigvalsh(ch_hi.bulk_hamiltonian(0.0))).min()
        good = e0 < 1e-10 and e_lo > 1e-3 and e_hi > 1e-3
        ok &= _check(f"V_z,crit(mu={mu})", good,
                     f"analytic {expect:.4f} meV, |E|(k=0) = {e0:.2e} at the "
                     f"boundary, {e_lo:.4f}/{e_hi:.4f} on either side")

    # --- 2. The +2t band-bottom convention (Rev 2.1 Appendix B.1) -----------
    good_conv = abs(REV21.v_z_crit - 2.0) < 1e-9
    bad_conv = REV20_BUGGY.v_z_crit
    ok &= _check("band-bottom offset", good_conv and bad_conv > 19.0,
                 f"Rev 2.1 V_z,crit = {REV21.v_z_crit:.4f} meV; without the "
                 f"+2t offset it would be {bad_conv:.2f} meV (the Rev 2.0 bug)")

    # --- 3. Particle-hole symmetry, class D ---------------------------------
    p_op = _kron(PAULI_Y, PAULI_Y)
    ch = MajoranaChannel(n_sites=1, params=REV21)
    resid = 0.0
    for k in np.linspace(-math.pi / REV21.a_nm, math.pi / REV21.a_nm, 41):
        h_k = ch.bulk_hamiltonian(float(k))
        h_mk = ch.bulk_hamiltonian(float(-k))
        resid = max(resid, np.abs(p_op @ h_mk.conj() @ p_op.conj().T + h_k).max())
    ok &= _check("particle-hole symmetry", resid < 1e-10,
                 f"max |P H*(-k) P^dag + H(k)| = {resid:.2e} (class D)")

    # --- 4. Finite-chain spectrum is symmetric about zero -------------------
    ch = MajoranaChannel(n_sites=60, params=REV21)
    ev = ch.spectrum()
    sym = np.abs(ev + ev[::-1]).max()
    ok &= _check("BdG spectrum +/-E symmetry", sym < 1e-9,
                 f"max |E_i + E_{{-i}}| = {sym:.2e} meV")

    # --- 5. MPO == independently built dense Hamiltonian --------------------
    # Both storage forms must reconstruct the SAME operator: the real (Y->iY)
    # form is an optimization, not a different model.
    for (mu, v) in ((0.0, 0.0), (1.3, 0.0), (1.0, 1.0), (0.7, -0.8)):
        chain = InteractingChain(n_sites=8, t=1.0, delta=1.0, mu=mu, v_int=v)
        ref = chain.to_dense()
        for tag, use_real in (("complex", False), ("real Y->iY", True)):
            err = np.abs(chain.to_mpo(real=use_real).to_dense() - ref).max()
            ok &= _check(f"MPO vs dense (mu={mu}, V={v}, {tag})", err < 1e-12,
                         f"max abs difference = {err:.2e}")

    ok &= _check("real MPO is float64",
                 InteractingChain(n_sites=8).to_mpo(real=True).tensors[0].dtype
                 == np.float64, "dtype float64 (half the memory of complex128)")

    # MPO must also survive per-site disorder (the sweep depends on it).
    chain = InteractingChain(n_sites=8, t=1.0, delta=1.0, mu=1.0,
                             v_int=0.5).with_disorder(w=3.0, seed=11)
    err = np.abs(chain.to_mpo().to_dense() - chain.to_dense()).max()
    ok &= _check("MPO vs dense (disordered)", err < 1e-12,
                 f"max abs difference = {err:.2e}")

    ok &= _check("MPO bond dimension", InteractingChain(n_sites=8).to_mpo().bond_dim == 5,
                 f"D = {InteractingChain(n_sites=8).to_mpo().bond_dim} (expected 5)")

    # --- 6. TFIM limit: exact ground-state energy at the critical point -----
    # At t = Delta, V = 0, mu = 2t the chain is the critical TFIM.  Check the
    # dense ground state against the free-fermion closed form for OBC.
    n = 10
    chain = InteractingChain(n_sites=n, t=1.0, delta=1.0, mu=2.0, v_int=0.0)
    e_dense = float(np.linalg.eigvalsh(chain.to_dense()).min())
    # H = -J sum Y Y - h sum Z with J = 1, h = mu/2 = 1; exact OBC spectrum
    # via the standard Jordan-Wigner free-fermion solution.
    e_exact = _tfim_obc_ground_energy(n, j=1.0, h=1.0)
    rel = abs(e_dense - e_exact) / abs(e_exact)
    ok &= _check("TFIM ground energy (exact)", rel < 1e-10,
                 f"dense {e_dense:.12f} vs exact {e_exact:.12f} (rel {rel:.2e})")

    # --- 7. Honeycomb solver: the four original gates -----------------------
    lattice = HoneycombSuperlattice(n_cells=6)
    frac, _, lat = lattice.build_supercell()
    pairs = lattice.neighbor_table(frac, lat)
    zeros = np.zeros(len(frac))
    evs = lattice.bloch_bands(frac, lat, pairs, zeros, zeros, kgrid=12)
    total_w = float(evs.max() - evs.min())
    good = abs(total_w - 6 * T_HOP_GRAPHENE) < 0.15 * T_HOP_GRAPHENE
    ok &= _check("graphene bandwidth", good,
                 f"{total_w:.3f} eV (exact 6t = {6*T_HOP_GRAPHENE:.2f})")

    min_abs = float(np.min(np.abs(evs)))
    ok &= _check("Dirac point gapless", min_abs < 0.3,
                 f"min|E| = {min_abs*1000:.1f} meV")

    j_mev, m = 200.0, 1.0
    ex = (j_mev / 1000.0) * m / 2.0
    evs_x = lattice.bloch_bands(frac, lat, pairs, zeros - ex, zeros + ex, kgrid=6)
    shift = float(evs.min() - evs_x.min())
    split_ok = abs(abs(shift) - ex) < 1e-6
    ok &= _check("exchange splitting", split_ok,
                 f"edge shift {abs(shift)*1000:.2f} meV (expected {ex*1000:.2f})")

    r = HoneycombSuperlattice(n_cells=8, layout="triangular", void_frac=0.3,
                              well_depth_ev=1.5).solve()
    plain = HoneycombSuperlattice(n_cells=8, void_frac=1e-4,
                                  well_depth_ev=0.0).solve()
    narrow = r["width_mev"] < plain["width_mev"]
    ok &= _check("superlattice band narrowing", narrow,
                 f"W {plain['width_mev']:.0f} -> {r['width_mev']:.0f} meV")

    print("\nVALIDATION", "PASSED" if ok else "FAILED")
    return ok


def _tfim_obc_ground_energy(n: int, j: float, h: float) -> float:
    """Exact ground-state energy of H = -J sum_{i<n-1} Y_i Y_{i+1} - h sum_i Z_i
    with open boundaries, via the free-fermion (BdG) solution.

    Independent of the MPO path -- this is a genuine external check.
    """
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    for i in range(n):
        a[i, i] = -2.0 * h
    for i in range(n - 1):
        a[i, i + 1] += -j
        a[i + 1, i] += -j
        b[i, i + 1] += -j
        b[i + 1, i] += +j
    # Standard Lieb-Schultz-Mattis: energies are the singular values of A + B.
    eps = np.linalg.svd(a + b, compute_uv=False)
    return float(-0.5 * eps.sum())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validate", action="store_true",
                   help="run known-physics checks and exit")
    p.add_argument("--gap", action="store_true",
                   help="report Delta_top, xi, and the edge splitting")
    p.add_argument("--sweep-delta", action="store_true",
                   help="Delta -> Delta_top -> T_max table (the Go/No-Go map)")
    p.add_argument("--n-sites", type=int, default=200)
    p.add_argument("--v-z", type=float, default=REV21.v_z)
    p.add_argument("--mu", type=float, default=REV21.mu)
    p.add_argument("--delta", type=float, default=REV21.delta,
                   help="pairing gap in meV (the Milestone 1 quantity)")
    p.add_argument("--vz-rule", choices=("fixed", "scaled"), default="fixed",
                   help="'fixed' holds V_z at --v-z; 'scaled' uses V_z = 1.5*Delta")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.sweep_delta:
        print(f"V_z rule: {args.vz_rule}"
              + (f" (V_z = {args.v_z} meV)" if args.vz_rule == "fixed"
                 else " (V_z = 1.5 x Delta)"))
        print(f"{'Delta':>7} {'V_z':>6} {'Delta_top':>10} {'T_max':>8}  gate")
        for d in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0):
            v_z = args.v_z if args.vz_rule == "fixed" else 1.5 * d
            params = replace(REV21, delta=d, v_z=v_z, mu=args.mu)
            gap = MajoranaChannel(n_sites=1, params=params).bulk_gap(nk=4001)
            t_max = gap / (20 * K_B_MEV_PER_K)
            print(f"{d:7.2f} {v_z:6.2f} {gap:10.4f} {t_max:7.3f} K  "
                  f"{'OK' if t_max >= 0.3 else 'BELOW 300 mK -- kill floor'}")
        return 0

    if args.validate:
        return 0 if validate() else 1

    if args.gap:
        params = replace(REV21, v_z=args.v_z, mu=args.mu, delta=args.delta)
        ch = MajoranaChannel(n_sites=args.n_sites, params=params)
        gap = ch.bulk_gap()
        print(f"N = {args.n_sites}  (L = {ch.length_um:.2f} um), "
              f"V_z = {params.v_z} meV, mu = {params.mu} meV")
        print(f"  V_z,crit (analytic) = {params.v_z_crit:.4f} meV")
        print(f"  Delta_top (bulk)    = {gap:.4f} meV   "
              f"[Rev 2.1 quotes 1.05 meV at V_z = 3]")
        print(f"  Delta_top / k_B T   = {gap / params.k_b_t:.1f}   "
              f"(T = {params.temperature_k} K)")
        print(f"  exp(-Delta_top/kT)  = {params.thermal_suppression(gap):.2e}"
              "   <- LOWER BOUND on the error rate only")
        if params.v_z > params.v_z_crit:
            print(f"  xi                  = {ch.localization_length():.1f} sites "
                  f"({ch.localization_length() * params.a_nm:.0f} nm)")
        split = ch.edge_splitting()
        print(f"  edge splitting      = {split * 1e6:.1f} neV")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
