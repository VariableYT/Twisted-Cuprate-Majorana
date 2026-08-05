"""Module 2 tests -- DMRG over the Module 1 MPO.

The reference energies pinned here were produced by an INDEPENDENTLY WRITTEN
quimb ``SpinHam1D`` construction in an earlier session.  This module builds its
MPO from a finite-state machine in ``tvqpu.lattice`` instead, so agreement to
1e-10 is a genuine cross-check of two separate implementations rather than a
regression test against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("quimb", reason="Module 2 needs quimb")

from tvqpu.dmrg import (  # noqa: E402
    clean_gap,
    disorder_ensemble,
    disorder_threshold,
    ground_state,
    tolerance_in_gap_units,
    validate,
)
from tvqpu.dmrg import EnsembleResult  # noqa: E402
from tvqpu.lattice import InteractingChain  # noqa: E402


# --------------------------------------------------------------------------
# Cross-checks against the independently written earlier implementation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mu,v,ref", [
    (0.0, 0.0, -11.000000000),
    (1.0, 0.0, -11.892044873),
    (3.0, 0.0, -19.879107043),
    (1.0, 1.0, -11.828653692),
    (1.0, -1.0, -12.694597707),
])
def test_dense_energies_match_the_independent_construction(mu, v, ref):
    chain = InteractingChain(n_sites=12, t=1.0, delta=1.0, mu=mu, v_int=v)
    e0 = float(np.linalg.eigvalsh(chain.to_dense())[0].real)
    assert e0 == pytest.approx(ref, abs=1e-8)


def test_dmrg_reproduces_exact_diagonalization():
    """Machine precision, including with interactions on."""
    for mu, v in ((1.0, 0.0), (1.0, 1.0), (1.0, -1.0)):
        chain = InteractingChain(n_sites=12, t=1.0, delta=1.0, mu=mu, v_int=v)
        e_dmrg = ground_state(chain, chi=64).energy
        e_exact = float(np.linalg.eigvalsh(chain.to_dense())[0].real)
        assert abs(e_dmrg - e_exact) / abs(e_exact) < 1e-9


# --------------------------------------------------------------------------
# The interaction convention
# --------------------------------------------------------------------------
def test_ph_symmetric_is_the_default_and_has_no_field_shift():
    c = InteractingChain(n_sites=8, mu=1.0, v_int=1.0)
    assert c.ph_symmetric
    hz, _, _, jz = c.couplings()
    assert np.allclose(hz, -0.5)              # -mu/2, no V contribution
    assert np.allclose(jz, 0.25)              # V/4


def test_plain_convention_has_a_pathological_cancellation_at_V_equals_mu():
    """The reason PH-symmetric is the default.

    With the plain V*n_i*n_{i+1} form the induced field shift is +V/2 in the
    interior, so at V = mu the transverse field vanishes exactly and the chain
    becomes a pure ferromagnetic Ising point.  Its two symmetry-broken ground
    states are then split only by an exponentially small finite-size gap,
    which DMRG converges roughly five orders of magnitude worse.
    """
    plain = InteractingChain(n_sites=12, mu=1.0, v_int=1.0, ph_symmetric=False)
    hz, _, _, _ = plain.couplings()
    assert hz[6] == pytest.approx(0.0, abs=1e-12), "interior field should cancel"

    ev = np.sort(np.linalg.eigvalsh(plain.to_dense()).real)
    assert ev[1] - ev[0] < 1e-3, "expected a near-degenerate doublet"


def test_both_conventions_still_agree_mpo_with_dense():
    for ph in (True, False):
        c = InteractingChain(n_sites=8, mu=0.7, v_int=1.3, ph_symmetric=ph)
        assert np.allclose(c.to_mpo().to_dense(), c.to_dense(), atol=1e-12)


# --------------------------------------------------------------------------
# The quimb bridge and the two API traps
# --------------------------------------------------------------------------
def test_module1_mpo_bridges_to_quimb_exactly():
    c = InteractingChain(n_sites=10, t=1.0, delta=1.0, mu=1.3, v_int=0.5)
    ref = float(np.linalg.eigvalsh(c.to_dense())[0].real)
    for real in (False, True):
        got = float(np.linalg.eigvalsh(c.to_mpo(real=real).to_quimb().to_dense())[0].real)
        assert got == pytest.approx(ref, abs=1e-10)


def test_raw_correlator_is_not_the_connected_one():
    """The trap: quimb's .correlation() is connected, and in a symmetry-broken
    ground state that is ~0 even deep in the ordered phase.  If this ever
    fails, somebody has used .correlation() bare and the topological phase
    will look like it vanished everywhere."""
    r = ground_state(InteractingChain(n_sites=48, mu=1.0), chi=64)
    assert r.order_parameter > 0.5, r.order_parameter


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
def test_v0_phase_boundary_is_mu_c_equals_2t():
    ordered = ground_state(InteractingChain(n_sites=48, mu=1.0), chi=64)
    trivial = ground_state(InteractingChain(n_sites=48, mu=3.0), chi=64)
    assert ordered.order_parameter > 0.5
    assert abs(trivial.order_parameter) < 1e-2


def test_clean_gap_closed_form():
    assert clean_gap(InteractingChain(mu=1.0)) == pytest.approx(1.0)
    assert clean_gap(InteractingChain(mu=0.0)) == pytest.approx(2.0)
    with pytest.raises(NotImplementedError, match="TFIM"):
        clean_gap(InteractingChain(t=1.0, delta=0.1))


def test_bond_saturation_is_reported():
    """The only self-diagnostic quimb actually exposes here."""
    tight = ground_state(InteractingChain(n_sites=32, mu=1.0), chi=4)
    assert tight.bond_saturated and not tight.trustworthy
    loose = ground_state(InteractingChain(n_sites=32, mu=1.0), chi=64)
    assert not loose.bond_saturated and loose.trustworthy


def test_disorder_is_reproducible():
    kw = dict(w=1.0, n_realizations=3, chi=48, seed0=7)
    a = disorder_ensemble(InteractingChain(n_sites=32, mu=1.0), **kw)
    b = disorder_ensemble(InteractingChain(n_sites=32, mu=1.0), **kw)
    assert a.mean == pytest.approx(b.mean, abs=1e-9)


# --------------------------------------------------------------------------
# Units -- the part that decides whether the result means anything
# --------------------------------------------------------------------------
def test_gap_units_conversion_and_the_rev21_reference_points():
    """delta-mu_rms/gap is the only form that transfers to the device.

    Rev 2.1: spec sigma = 0.2% -> 40 ueV / 1.05 meV = 0.038;
             onset sigma ~ 3%  -> 600 ueV / 1.05 meV = 0.571.
    """
    assert tolerance_in_gap_units(np.sqrt(12.0), 1.0) == pytest.approx(1.0)
    assert tolerance_in_gap_units(np.sqrt(12.0) * 0.571, 1.0) == pytest.approx(0.571)
    # the Rev 2.1 numbers themselves
    assert 0.002 * 20.0 / 1.05 == pytest.approx(0.038, abs=0.001)
    assert 0.03 * 20.0 / 1.05 == pytest.approx(0.571, abs=0.001)
    with pytest.raises(ValueError, match="gap must be positive"):
        tolerance_in_gap_units(1.0, 0.0)


def test_threshold_returns_none_when_the_sweep_never_collapses():
    """A None is a real result -- it is what the earlier W <= 4 sweep produced
    -- and must never be reported as a large tolerance."""
    ens = [EnsembleResult(w=w, mean=0.9, std=0.0, sem=0.0,
                          n_realizations=8, n_trustworthy=8)
           for w in (0.0, 1.0, 2.0, 4.0)]
    assert disorder_threshold(ens) is None


def test_threshold_interpolates():
    ens = [EnsembleResult(w=w, mean=m, std=0.0, sem=0.0, n_realizations=8,
                          n_trustworthy=8)
           for w, m in ((0.0, 1.0), (2.0, 0.75), (4.0, 0.25))]
    wc = disorder_threshold(ens, fraction=0.5)
    assert wc == pytest.approx(3.0, abs=1e-9)


# --------------------------------------------------------------------------
@pytest.mark.slow
def test_full_validation_gate():
    assert validate(verbose=False)
