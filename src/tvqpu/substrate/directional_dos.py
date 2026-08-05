"""
directional_dos.py -- Module 4b.  Directional density of states, phonon
focusing factor, and directional thermal conductance.

This replaces the histogram-binning implementation that ARCHITECTURE.md
section 5.2 marks do-not-reuse.  The three defects of that code and their
fixes here:

  1. It finite-differenced a BINNED velocity field.  Here the group-velocity
     field is analytic (Hellmann-Feynman derivative of the Christoffel
     eigenvalue), and only the direction map is differenced -- a far
     better-conditioned object.
  2. It sorted branches by eigenvalue, which swaps the two TA branches
     wherever they cross and smears every caustic into a common bin.  Here
     branches are tracked by EIGENVECTOR CONTINUATION (maximum overlap), which
     is the actual fix.
  3. It binned the map n_hat -> v_hat, so near a caustic the answer was set by
     bin width rather than by physics.  Here the enhancement is the inverse
     Jacobian determinant, and multivalued preimages are FOUND AND SUMMED.

THE PHYSICS (Li et al., Nature Physics 2026, doi:10.1038/s41567-026-03335-y)

    q(theta,phi,r) = (2pi)^-3 sum_s int |q_ks| delta(Omega_thetaphi - Omega_ks) dk   (2)

The delta function pushes the k-space measure forward onto the group-velocity
sphere.  Its density is exactly the inverse Jacobian of the map
q_hat -> v_hat = grad_q omega / |grad_q omega|:

    A(v_hat) = sum over preimages of  |det( d v_hat / d q_hat )|^-1

Caustics are the loci where that determinant vanishes.  The divergence is cut
off physically by finite phonon wavelength and lifetime; numerically it is
clipped, and the clip is reported rather than hidden.

Because the pushforward preserves total measure, the solid-angle average of A
over the v_hat sphere is exactly 1.  So in the ballistic limit A IS the
enhancement relative to the isotropic radiation limit -- which is what makes
the G/G0 gate directly checkable without any absolute normalization.

VELOCITY-FIELD PROVIDERS
    CubicElasticField -- Christoffel equation for a cubic crystal.  Analytic,
        exact, no data files.  This is the validation harness: for an isotropic
        medium it must return A == 1 everywhere, and its sound velocities along
        <100>/<110>/<111> have closed forms.  It captures the caustic STRUCTURE
        (which is an acoustic-branch property) but not THz magnitudes.
    PhonopyFC2Field -- real dispersion from second-order force constants.
        Needed for the quantitative gates (c) and (d), because the 4.6 figure
        comes from the full dispersion near 7.8 THz, not the elastic limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "verify_measure_conservation", "count_rays", "angular_profile",
    "caustic_cutoff_from_geometry", "dominant_phonon_wavelength_nm",
    "VelocityField", "CubicElasticField", "PhonopyFC2Field",
    "BAS_ELASTIC", "SI_ELASTIC",
    "FocusingMap", "focusing_map", "enhancement_along",
    "directional_conductance", "fold_symmetry",
]

# --------------------------------------------------------------------------
# Elastic constants
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CubicElastic:
    """Cubic elastic constants in GPa and density in kg/m^3."""

    c11: float
    c12: float
    c44: float
    density: float
    name: str = ""

    @property
    def zener(self) -> float:
        """Zener anisotropy 2*C44/(C11-C12).  Exactly 1 => elastically
        isotropic => no focusing.  This is the knob the isotropy test turns."""
        return 2.0 * self.c44 / (self.c11 - self.c12)

    def isotropic_like(self) -> "CubicElastic":
        """Same C11, C12, but C44 set to make the medium exactly isotropic."""
        return CubicElastic(self.c11, self.c12, 0.5 * (self.c11 - self.c12),
                            self.density, (self.name + " (isotropic)").strip())


#: Boron arsenide.  C11 and C12 from Brillouin scattering
#: (Phys. Rev. Materials 5, 033606).  C44 IS CONTESTED: Brillouin gives
#: 173 +/- 6 GPa, picosecond ultrasonics gives 149 GPa -- the two experiments
#: disagree by 16%.  Our phono3py-derived TA[100] velocity (5311 m/s) sits
#: within 0.5% of the ultrasonic value and 7.7% off the Brillouin one, so the
#: ultrasonic C44 is used as the default here.  Change it deliberately, not
#: to make a test pass, and report which was used.
BAS_ELASTIC = CubicElastic(c11=291.0, c12=83.0, c44=149.0, density=5220.0,
                           name="BAs (zincblende)")

#: Silicon, the textbook phonon-focusing reference (Wolfe, *Imaging Phonons*).
SI_ELASTIC = CubicElastic(c11=165.6, c12=63.9, c44=79.5, density=2329.0,
                          name="Si")


# --------------------------------------------------------------------------
# Velocity-field interface
# --------------------------------------------------------------------------
class VelocityField(Protocol):
    """Supplies phonon group velocities as a function of propagation direction.

    ``evaluate(n)`` takes unit directions of shape (N, 3) and returns

        phase_speed : (N, S)     branch phase speeds, m/s
        v_group     : (N, S, 3)  branch group velocities, m/s
        pol         : (N, S, 3)  polarization vectors, for branch continuation
    """

    n_branches: int

    def evaluate(self, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


@dataclass
class CubicElasticField:
    """Christoffel equation for a cubic crystal.

    Gamma_ik(n) = C_ijkl n_j n_l,   Gamma u = rho v_p^2 u,   omega = v_p(n) |q|

    Group velocity follows exactly, with no finite differencing of omega:

        v_g = v_p n_hat + (I - n_hat n_hat^T) grad_n v_p
        d v_p / d n_m = (1 / (2 rho v_p)) * u . (d Gamma / d n_m) . u

    the second line by Hellmann-Feynman.  The transverse projector is there
    because only the tangential part of grad_n v_p contributes -- the radial
    part is already carried by the v_p n_hat term.
    """

    elastic: CubicElastic
    n_branches: int = 3

    def _gamma(self, n: np.ndarray) -> np.ndarray:
        """(N, 3, 3) Christoffel matrices, in Pa."""
        c11 = self.elastic.c11 * 1e9
        c12 = self.elastic.c12 * 1e9
        c44 = self.elastic.c44 * 1e9
        n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]
        g = np.zeros((len(n), 3, 3))
        g[:, 0, 0] = c11 * n1**2 + c44 * (n2**2 + n3**2)
        g[:, 1, 1] = c11 * n2**2 + c44 * (n1**2 + n3**2)
        g[:, 2, 2] = c11 * n3**2 + c44 * (n1**2 + n2**2)
        off = c12 + c44
        g[:, 0, 1] = g[:, 1, 0] = off * n1 * n2
        g[:, 0, 2] = g[:, 2, 0] = off * n1 * n3
        g[:, 1, 2] = g[:, 2, 1] = off * n2 * n3
        return g

    def _dgamma(self, n: np.ndarray) -> np.ndarray:
        """(N, 3, 3, 3) derivative d Gamma_ik / d n_m, index order (N, i, k, m)."""
        c11 = self.elastic.c11 * 1e9
        c12 = self.elastic.c12 * 1e9
        c44 = self.elastic.c44 * 1e9
        off = c12 + c44
        n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]
        z = np.zeros(len(n))
        d = np.zeros((len(n), 3, 3, 3))
        # diagonal entries
        d[:, 0, 0, :] = np.stack([2 * c11 * n1, 2 * c44 * n2, 2 * c44 * n3], -1)
        d[:, 1, 1, :] = np.stack([2 * c44 * n1, 2 * c11 * n2, 2 * c44 * n3], -1)
        d[:, 2, 2, :] = np.stack([2 * c44 * n1, 2 * c44 * n2, 2 * c11 * n3], -1)
        # off-diagonal entries
        d[:, 0, 1, :] = d[:, 1, 0, :] = np.stack([off * n2, off * n1, z], -1)
        d[:, 0, 2, :] = d[:, 2, 0, :] = np.stack([off * n3, z, off * n1], -1)
        d[:, 1, 2, :] = d[:, 2, 1, :] = np.stack([z, off * n3, off * n2], -1)
        return d

    def evaluate(self, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = np.asarray(n, dtype=float)
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        rho = self.elastic.density

        lam, vecs = np.linalg.eigh(self._gamma(n))       # (N,3), (N,3,3)
        lam = np.clip(lam, 1e-12, None)
        v_p = np.sqrt(lam / rho)                          # (N, S)
        pol = np.transpose(vecs, (0, 2, 1))               # (N, S, 3)

        dg = self._dgamma(n)                              # (N,i,k,m)
        # dlam/dn_m = u_i u_k dGamma_ikm
        dlam = np.einsum("nsi,nikm,nsk->nsm", pol, dg, pol)      # (N,S,3)
        dvp = dlam / (2.0 * rho * v_p[..., None])                # (N,S,3)

        # transverse projection: (I - n n^T) grad_n v_p
        radial = np.einsum("nsm,nm->ns", dvp, n)[..., None] * n[:, None, :]
        v_g = v_p[..., None] * n[:, None, :] + (dvp - radial)
        return v_p, v_g, pol


@dataclass
class PhonopyFC2Field:
    """Group velocities from real second-order force constants, at a fixed
    phonon frequency shell.

    This is the provider the quantitative gates need.  Li et al. show the
    iso-frequency surface at 7.8 THz for the longitudinal branch, and their
    G/G0 = 4.6 comes from the full dispersion, not the elastic limit -- the
    elastic model has no way to know about 7.8 THz.

    Requires phonopy and a ``phonopy_params``/``phono3py_params`` YAML.  The
    BAs dataset (512-atom supercell, NIMS MDR) is the intended input.

    PERFORMANCE TRAP.  The loaded phonon object is cached per INSTANCE, and
    building it re-reads the 512-atom dataset and rebuilds fc2 -- tens of
    seconds.  To scan the frequency shell, mutate ``q_invang`` on ONE instance:

        f = PhonopyFC2Field(path, q_invang=0.5)
        f.evaluate(n)          # pays the load once
        f.q_invang = 0.85      # free
        f.evaluate(n)

    Constructing a new instance per q value turns a 24-step bisection into
    half an hour of pure I/O.
    """

    yaml_path: str
    q_invang: float = 0.30      # |q| in 1/Angstrom -- selects the frequency shell
    n_branches: int = 3
    _phonon: object = None
    _rec: np.ndarray = None

    def _load(self):
        if self._phonon is None:
            import phono3py
            from phonopy import Phonopy
            # The harmonic force constants live on the 4x4x4 (512-atom) PHONON
            # supercell, not the smaller fc3 cell.  Rebuilding a Phonopy object
            # on the phonon supercell is the load path that works; calling
            # phonopy.load() directly on this YAML leaves the dynamical matrix
            # unbuilt.
            p3 = phono3py.load(self.yaml_path, produce_fc=True, log_level=0)
            ph = Phonopy(p3.unitcell,
                         supercell_matrix=p3.phonon_supercell_matrix,
                         primitive_matrix=p3.primitive_matrix)
            ph.force_constants = p3.fc2
            ph.nac_params = p3.nac_params
            self._phonon = ph
            self._rec = np.linalg.inv(ph.primitive.cell).T * 2 * math.pi
        return self._phonon

    def frequency_shell_thz(self) -> np.ndarray:
        """Acoustic-branch frequencies at ``q_invang`` along [100], for
        checking which part of the spectrum is being sampled.  Li et al. show
        the iso-frequency surface at 7.8 THz."""
        _, _, _ = self.evaluate(np.array([[1.0, 0.0, 0.0]]))
        return self._last_freqs[0]

    def evaluate(self, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ph = self._load()
        n = np.asarray(n, dtype=float)
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        # cartesian |q| in 1/Ang -> reduced coordinates of the primitive cell
        q_red = np.linalg.solve(self._rec.T, (n * self.q_invang).T).T

        ph.run_qpoints(q_red, with_group_velocities=True,
                       with_eigenvectors=True)
        freqs = ph.qpoints.frequencies                       # (N, 3n_at), THz
        v_all = ph.qpoints.group_velocities * 100.0          # THz*Ang -> m/s
        eigvecs = ph.qpoints.eigenvectors                    # (N, 3n_at, 3n_at)

        s = self.n_branches
        order = np.argsort(freqs, axis=1)[:, :s]
        f = np.take_along_axis(freqs, order, axis=1)                   # (N,S)
        self._last_freqs = f
        v_g = np.stack([np.take_along_axis(v_all[..., k], order, axis=1)
                        for k in range(3)], axis=-1)                   # (N,S,3)
        # phase speed omega/|q|, with omega in rad/s and |q| in 1/m
        v_p = (f * 1e12 * 2 * math.pi) / (self.q_invang * 1e10)

        # polarizations: the first atom's displacement block is enough to
        # track branches through degeneracies.
        pol = np.zeros((len(n), s, 3))
        for i in range(len(n)):
            for j in range(s):
                vec = np.real(eigvecs[i][:3, order[i, j]])
                nrm = np.linalg.norm(vec)
                pol[i, j] = vec / nrm if nrm > 1e-12 else np.array([1.0, 0, 0])
        return v_p, v_g, pol


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangent frame at each unit vector, chosen continuously
    enough for a local Jacobian (the Jacobian determinant's magnitude does not
    depend on the choice)."""
    n = np.atleast_2d(n)
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(n), 1))
    bad = np.abs(n[:, 2]) > 0.9
    helper[bad] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2, axis=-1, keepdims=True)
    return e1, e2


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Near-uniform points on the unit sphere.  Equal-area to within O(1/N),
    which is what makes the solid-angle average of A converge to 1."""
    i = np.arange(n_points) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_points)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=-1)


def _track_branch(field: VelocityField, n: np.ndarray, ref_pol: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the field at ``n`` and select, for each point, the branch whose
    polarization best matches ``ref_pol``.

    THIS IS THE DEGENERATE-BRANCH FIX.  Sorting by eigenvalue swaps the two TA
    branches wherever they cross, which is precisely along the high-symmetry
    directions where the caustics live -- the old code's three branches all
    peaked in one bin because of exactly this.
    """
    v_p, v_g, pol = field.evaluate(n)
    overlap = np.abs(np.einsum("nsi,ni->ns", pol, ref_pol))
    pick = np.argmax(overlap, axis=1)
    rows = np.arange(len(n))
    return v_g[rows, pick], pol[rows, pick], v_p[rows, pick]


def _branch_jacobian(field: VelocityField, n: np.ndarray, ref_pol: np.ndarray,
                     step: float = 1e-4):
    """Tangent-frame Jacobian of the direction map n_hat -> v_hat, for one
    tracked branch, vectorized over ``n``.

    Returns ``(v_hat, jac, det, pol, v_p, e1, e2, f1, f2)`` where ``jac`` has
    shape (N, 2, 2) expressed in the tangent frames (e1, e2) at n_hat and
    (f1, f2) at v_hat.

    Only the DIRECTION map is differenced.  The velocity field itself is
    analytic, so this is a smooth map away from caustics -- unlike the old
    code, which differenced a binned field.
    """
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    v_g, pol, v_p = _track_branch(field, n, ref_pol)
    v_hat = v_g / np.linalg.norm(v_g, axis=-1, keepdims=True)
    e1, e2 = _tangent_basis(n)
    f1, f2 = _tangent_basis(v_hat)

    cols = []
    for e in (e1, e2):
        plus = n + step * e
        plus /= np.linalg.norm(plus, axis=-1, keepdims=True)
        minus = n - step * e
        minus /= np.linalg.norm(minus, axis=-1, keepdims=True)
        vp_g, _, _ = _track_branch(field, plus, pol)
        vm_g, _, _ = _track_branch(field, minus, pol)
        vp = vp_g / np.linalg.norm(vp_g, axis=-1, keepdims=True)
        vm = vm_g / np.linalg.norm(vm_g, axis=-1, keepdims=True)
        cols.append((vp - vm) / (2.0 * step))

    jac = np.empty((len(n), 2, 2))
    jac[:, 0, 0] = np.einsum("ni,ni->n", cols[0], f1)
    jac[:, 1, 0] = np.einsum("ni,ni->n", cols[0], f2)
    jac[:, 0, 1] = np.einsum("ni,ni->n", cols[1], f1)
    jac[:, 1, 1] = np.einsum("ni,ni->n", cols[1], f2)
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    return v_hat, jac, det, pol, v_p, e1, e2, f1, f2


# --------------------------------------------------------------------------
# The focusing factor
# --------------------------------------------------------------------------
@dataclass
class FocusingMap:
    """Result of pushing an n_hat sampling forward onto the v_hat sphere."""

    n_hat: np.ndarray        # (N, S, 3) sampled propagation directions
    v_hat: np.ndarray        # (N, S, 3) group-velocity directions
    enhancement: np.ndarray  # (N, S)    A = 1/|det J|, clipped
    det_j: np.ndarray        # (N, S)    signed Jacobian determinant
    phase_speed: np.ndarray  # (N, S)
    pol: np.ndarray          # (N, S, 3) polarizations, for branch tracking
    clipped: int             # how many samples hit the caustic cutoff

    @property
    def covering_degree(self) -> float:
        """Mean |det J| over the n_hat sampling.

        This is the average number of times the map wraps the v_hat sphere,
        and it is the sampling-independent diagnostic for the map itself.  It
        is EXACTLY 1 for a fold-free (isotropic) medium and > 1 once caustics
        appear, because folded sheets cover parts of the v_hat sphere more
        than once.

        Note this is deliberately NOT a mean of the enhancement A = 1/|det J|.
        The solid-angle average of A over the n_hat sphere is not 1 -- by
        Jensen's inequality it is >= 1 -- and mistaking one for the other is
        how a focusing calculation ends up reporting an enhancement of 3.6
        where it should report 1.
        """
        return float(np.mean(np.abs(self.det_j)))


def focusing_map(field: VelocityField, n_points: int = 20000,
                 step: float = 1e-4, caustic_cutoff: float = 50.0,
                 ) -> FocusingMap:
    """Compute the phonon-focusing enhancement over the whole sphere.

    ``step`` is the angular half-step used for the central difference of the
    DIRECTION MAP.  Note what is and is not differenced: the group-velocity
    field itself is analytic (for the elastic provider) or comes from
    phonopy's analytic derivative; only the unit-vector map n_hat -> v_hat is
    differenced, and that map is smooth away from caustics.

    ``caustic_cutoff`` clips A where det J -> 0.  The physical cutoff is finite
    phonon wavelength and lifetime; this one is numerical, and the number of
    clipped samples is reported so the clip can never silently set the answer.
    """
    n0 = fibonacci_sphere(n_points)
    v_p0, v_g0, pol0 = field.evaluate(n0)
    n_br = v_g0.shape[1]

    enh = np.zeros((n_points, n_br))
    detj = np.zeros((n_points, n_br))
    vhat = np.zeros((n_points, n_br, 3))
    pols = np.zeros((n_points, n_br, 3))
    vps = np.zeros((n_points, n_br))
    nhat = np.repeat(n0[:, None, :], n_br, axis=1)

    for s in range(n_br):
        v_hat, _jac, det, pol, v_p, *_ = _branch_jacobian(
            field, n0, pol0[:, s, :], step=step)
        vhat[:, s, :] = v_hat
        pols[:, s, :] = pol
        vps[:, s] = v_p
        detj[:, s] = det
        with np.errstate(divide="ignore"):
            enh[:, s] = np.minimum(1.0 / np.maximum(np.abs(det), 1e-300),
                                   caustic_cutoff)

    return FocusingMap(n_hat=nhat, v_hat=vhat, enhancement=enh, det_j=detj,
                       phase_speed=vps, pol=pols,
                       clipped=int(np.sum(enh >= caustic_cutoff)))


def enhancement_along(field: VelocityField, fmap: FocusingMap, direction,
                      seed_cone_deg: float = 6.0, tol: float = 1e-9,
                      max_iter: int = 40, caustic_cutoff: float = 50.0,
                      dedupe_deg: float = 0.5, branch_weights: bool = True,
                      step: float = 1e-4) -> float:
    """Enhancement A(v_hat) along one group-velocity direction, by finding and
    summing the PREIMAGES.

        A(v_hat) = sum over {n_hat : v_hat(n_hat) = v_hat_0} of 1/|det J(n_hat)|

    This is the estimator ARCHITECTURE.md section 5.2 item 3 requires, and the
    reason it is not a histogram: near a caustic, binning gives an answer set
    by the bin width, whereas this gives the analytic density with the
    divergence explicitly clipped.

    Seeds come from ``fmap`` (samples whose v_hat lies within
    ``seed_cone_deg``), are refined by Newton iteration on the 2x2 tangent
    Jacobian, then deduplicated by angular proximity in n_hat so that a basin
    contributing several seeds is counted once.

    With ``branch_weights`` the ballistic Debye weight 1/v_p^2 is applied and
    the result is G/G0; without it, the bare geometric focusing factor.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    cos_seed = math.cos(math.radians(seed_cone_deg))

    total = 0.0
    n_br = fmap.v_hat.shape[1]
    # Ballistic reference: the isotropic radiation limit G0 is the
    # solid-angle average of the summed branch weights.  Dividing by this --
    # rather than by the number of preimages actually found -- is what makes
    # the result a DENSITY (a sum over sheets) instead of a MEAN over them.
    #
    # The two coincide when the map is unfolded, which is why the isotropic
    # calibration cannot catch the difference.  On a folded map they differ by
    # the covering degree, so a strongly folded iso-frequency surface reports
    # everything below 1 if this is got wrong.
    if branch_weights:
        reference = float(np.mean(np.sum(1.0 / fmap.phase_speed**2, axis=1)))
    else:
        reference = float(n_br)
    for s in range(n_br):
        vh = fmap.v_hat[:, s, :]
        sel = np.nonzero(vh @ d >= cos_seed)[0]
        if sel.size == 0:
            continue
        n = fmap.n_hat[sel, s, :].copy()
        ref = fmap.pol[sel, s, :].copy()

        for _ in range(max_iter):
            v_hat, jac, det, ref, _vp, e1, e2, f1, f2 = _branch_jacobian(
                field, n, ref, step=step)
            resid = np.stack([np.einsum("ni,ni->n", d - v_hat, f1),
                              np.einsum("ni,ni->n", d - v_hat, f2)], axis=-1)
            if np.max(np.abs(resid)) < tol:
                break
            # damped Newton; near a caustic det -> 0 so the step is limited
            safe = np.where(np.abs(det) > 1e-12, det, 1e-12)
            inv = np.empty_like(jac)
            inv[:, 0, 0] = jac[:, 1, 1] / safe
            inv[:, 1, 1] = jac[:, 0, 0] / safe
            inv[:, 0, 1] = -jac[:, 0, 1] / safe
            inv[:, 1, 0] = -jac[:, 1, 0] / safe
            delta = np.einsum("nab,nb->na", inv, resid)
            mag = np.linalg.norm(delta, axis=-1, keepdims=True)
            delta = np.where(mag > 0.1, delta * 0.1 / np.maximum(mag, 1e-30),
                             delta)
            n = n + delta[:, :1] * e1 + delta[:, 1:] * e2
            n /= np.linalg.norm(n, axis=-1, keepdims=True)

        v_hat, jac, det, ref, v_p, *_ = _branch_jacobian(field, n, ref, step=step)
        ok = (v_hat @ d) > math.cos(math.radians(0.05))
        if not ok.any():
            continue
        n_ok, det_ok, vp_ok = n[ok], det[ok], v_p[ok]

        # dedupe: one contribution per distinct preimage basin
        keep: list[int] = []
        cos_dedupe = math.cos(math.radians(dedupe_deg))
        for i in range(len(n_ok)):
            if all(float(n_ok[i] @ n_ok[j]) < cos_dedupe for j in keep):
                keep.append(i)
        a = np.minimum(1.0 / np.maximum(np.abs(det_ok[keep]), 1e-12),
                       caustic_cutoff)
        w = 1.0 / vp_ok[keep] ** 2 if branch_weights else np.ones_like(a)
        total += float(np.sum(a * w))

    return total / reference


def dominant_phonon_wavelength_nm(temperature_k: float,
                                  sound_speed: float = 5300.0) -> float:
    """Thermal phonon wavelength v / (k_B T / h), in nm.

    At 4 K in BAs this is ~64 nm; at 300 K it is ~0.85 nm.  The long
    wavelength at cryogenic temperatures is why the elastic (linear-dispersion)
    description is exact there and why the classic phonon-imaging experiments
    were done at liquid-helium temperatures.
    """
    k_b, h = 1.380649e-23, 6.62607015e-34
    return sound_speed / (k_b * float(temperature_k) / h) * 1e9


def caustic_cutoff_from_geometry(temperature_k: float,
                                 propagation_length_nm: float,
                                 sound_speed: float = 5300.0) -> float:
    """Physical ceiling on the focusing enhancement, from finite wavelength
    and finite propagation length.

    A fold caustic diverges in ray optics.  Two things cut it off: diffraction,
    which blurs the pattern over an angle ~lambda/L, and scattering, which
    limits L.  Taking the angular resolution delta ~ lambda/L, an
    enhancement cannot exceed roughly 1/delta.

    THIS IS AN ORDER-OF-MAGNITUDE CEILING, NOT A CALIBRATED MODEL.  It exists
    so that ``caustic_cutoff`` can be set from physics instead of from the
    arbitrary default of 50, and so that a reported peak can be checked
    against whether the geometry could support it at all.

    Note what sets L at cryogenic temperature: NOT three-phonon scattering,
    which is frozen out, but boundary (Casimir) scattering and isotope
    disorder.  Natural boron is ~20% B-10, and isotope scattering goes as
    omega^4 -- weak at 83 GHz but not absent.  "fc3 is frozen out" does not
    mean "there is no scattering."
    """
    lam = dominant_phonon_wavelength_nm(temperature_k, sound_speed)
    if propagation_length_nm <= 0:
        raise ValueError("propagation length must be positive")
    return max(float(propagation_length_nm) / lam, 1.0)


def verify_measure_conservation(field: VelocityField, fmap: FocusingMap,
                                n_probe: int = 120, **kw) -> float:
    """Solid-angle average of G/G0 over the v_hat sphere.  Must be 1.

    THIS IS THE INVARIANT THAT CATCHES SUM-VS-MEAN ERRORS.  The pushforward
    conserves measure, so however the sheets fold, the average enhancement
    over the destination sphere is exactly 1.  A sampling that is too coarse
    to find every preimage reports LESS than 1; an estimator that averages
    over preimages instead of summing them also reports less than 1, by
    roughly the covering degree.

    The isotropic calibration cannot detect either failure, because an
    unfolded map has exactly one preimage per branch.  Run this on an
    ANISOTROPIC field, where the covering degree exceeds 1, before trusting
    any absolute magnitude.
    """
    probes = fibonacci_sphere(n_probe)
    vals = [enhancement_along(field, fmap, d, **kw) for d in probes]
    return float(np.mean(vals))


def in_plane_frame(plane) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis spanning the surface ``plane``."""
    nrm = np.asarray(plane, dtype=float)
    nrm = nrm / np.linalg.norm(nrm)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ nrm)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - float(ref @ nrm) * nrm
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(nrm, e1)


def angular_profile(field: VelocityField, fmap: FocusingMap, plane,
                    n_angles: int = 180, **kw) -> tuple[np.ndarray, np.ndarray]:
    """G/G0 as a function of azimuth within the surface ``plane``.

    This is the object gates (a) and (b) are both read from: peak COUNT gives
    the fold symmetry, peak WIDTH gives the angular profile, peak HEIGHT gives
    the magnitude.
    """
    e1, e2 = in_plane_frame(plane)
    angles = np.linspace(0.0, 2 * math.pi, n_angles, endpoint=False)
    prof = np.array([
        enhancement_along(field, fmap, math.cos(t) * e1 + math.sin(t) * e2, **kw)
        for t in angles])
    return angles, prof


def count_rays(profile: np.ndarray, threshold: float = 1.0) -> int:
    """Number of contiguous arcs (circularly) where the profile exceeds
    ``threshold``.

    A focusing ray is a contiguous BRIGHT ARC, not a local maximum.  Counting
    local maxima overcounts badly in the elastic limit, because a fold caustic
    produces a cusp pair -- two divergences flanking a shallow dip -- inside
    what is physically one ray.  Measured patterns do not show that
    substructure because finite propagation length smooths it.

    The default threshold is 1.0, which is not a tuning knob: G/G0 = 1 is the
    ballistic isotropic limit, so 'above threshold' means 'focused rather than
    defocused'.
    """
    above = profile > threshold
    if above.all():
        return 1
    if not above.any():
        return 0
    # rotate so index 0 starts a run, then count rising edges
    start = int(np.argmax(~above))
    rolled = np.roll(above, -start)
    return int(np.sum(rolled[1:] & ~rolled[:-1]) + (1 if rolled[0] else 0))


def fold_symmetry(field: VelocityField, fmap: FocusingMap, plane,
                  n_angles: int = 180, threshold: float = 1.0,
                  **kw) -> tuple[int, np.ndarray, np.ndarray]:
    """Count focusing rays lying IN the surface ``plane``.

    Li et al. Fig. 4 target: (111) -> 6, (100) -> 8, (110) -> 4.

    Returns ``(fold, angles, profile)``.
    """
    angles, prof = angular_profile(field, fmap, plane, n_angles=n_angles, **kw)
    return count_rays(prof, threshold), angles, prof


def directional_conductance(field: VelocityField, direction,
                            n_points: int = 20000, **kw) -> float:
    """G(v_hat)/G0 along one direction, with the ballistic Debye branch weight.

    In that limit the modes per solid angle per unit frequency go as
    omega^2 / v_p^3 and the flux carries one more factor of v_g ~ v_p, so the
    branch weight is 1/v_p^2.  Slow branches dominate -- which is why the
    transverse branches, not the longitudinal one, set the focusing pattern in
    most cubic crystals.
    """
    fmap = focusing_map(field, n_points=n_points)
    return enhancement_along(field, fmap, direction, branch_weights=True, **kw)
