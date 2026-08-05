"""Module 1 acceptance tests.

The numbers pinned here come from Rev 2.1 Table 2 and section 2.  If one of
these starts failing, the correct response is to find out which side moved --
not to widen the tolerance.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tvqpu.lattice import (
    K_B_MEV_PER_K,
    REV21,
    REV20_BUGGY,
    HoneycombSuperlattice,
    InteractingChain,
    MajoranaChannel,
    PAULI_I,
    PAULI_Y,
    _tfim_obc_ground_energy,
    validate,
)


# --------------------------------------------------------------------------
# The full gate, as one test
# --------------------------------------------------------------------------
def test_validate_gate_passes():
    assert validate(), "known-physics validation gate failed"


# --------------------------------------------------------------------------
# Rev 2.1 section 2 targets
# --------------------------------------------------------------------------
def test_vz_crit_is_analytic():
    """V_z,crit = sqrt(mu^2 + Delta^2); 2.000 meV at the mu = 0 sweet spot."""
    assert REV21.v_z_crit == pytest.approx(2.0, abs=1e-12)
    for mu in (0.0, 0.5, 1.0, 3.0):
        p = replace(REV21, mu=mu)
        assert p.v_z_crit == pytest.approx(math.hypot(mu, p.delta), rel=1e-12)


def test_topological_gap_closed_form():
    """At mu = 0 the k = 0 gap is |V_z - Delta| exactly, and it is the global
    minimum over the Brillouin zone.

    Rev 2.1 quotes Delta_top = 1.05 meV at V_z = 3 meV.  The exact bulk value
    is 1.000 meV = 3 - 2; the 1.05 in the paper is read off a finite N = 80
    chain, where level discretization lifts it slightly.  Both are 'the gap',
    but only one of them is exact, and a reviewer who rederives it will get
    this one.
    """
    ch = MajoranaChannel(n_sites=1, params=REV21)
    gap = ch.bulk_gap(nk=4001)
    assert gap == pytest.approx(REV21.v_z - REV21.delta, abs=1e-6)
    assert gap == pytest.approx(1.05, rel=0.06)  # within 6% of the Rev 2.1 value


def test_trivial_phase_below_critical_field():
    """Below V_z,crit there is no zero mode: the finite-chain gap stays open."""
    below = MajoranaChannel(n_sites=120,
                            params=replace(REV21, v_z=1.5))
    above = MajoranaChannel(n_sites=120,
                            params=replace(REV21, v_z=3.0))
    assert below.edge_splitting() > 100 * above.edge_splitting()


def test_localization_length_matches_rev21():
    """xi ~ 21 sites (~210 nm).  Fit is slope-sensitive; 15% is the gate."""
    ch = MajoranaChannel(n_sites=200, params=REV21)
    xi = ch.localization_length()
    assert xi == pytest.approx(21.0, rel=0.15), f"xi = {xi:.2f} sites"


def test_edge_splitting_matches_rev21():
    """delta-E = 57 neV at N = 200 (L = 2 um).  Rev 2.1 section 2.3."""
    ch = MajoranaChannel(n_sites=200, params=REV21)
    split_nev = ch.edge_splitting() * 1e6
    assert split_nev == pytest.approx(57.0, rel=0.10), f"{split_nev:.1f} neV"


def test_splitting_decays_exponentially_with_length():
    """delta-E ~ Delta_top exp(-L/xi): the production spec L >= 30 xi is what
    buys the ~0.1 feV number in Rev 2.1 eq. (7)."""
    splits = [MajoranaChannel(n_sites=n, params=REV21).edge_splitting()
              for n in (100, 150, 200)]
    assert splits[0] > splits[1] > splits[2]
    # log-linear in N
    slope = np.polyfit([100, 150, 200], np.log(splits), 1)[0]
    xi_from_splitting = -1.0 / slope
    assert xi_from_splitting == pytest.approx(21.0, rel=0.25)


def test_delta_top_is_a_tent_function_of_the_pairing_gap():
    """At fixed V_z, Delta_top peaks at intermediate Delta -- two competing
    minima, one at k = 0 (Zeeman-limited, falling in Delta) and one at
    ka ~ 0.94 (pairing-limited, rising in Delta).

    Consequence for the Stage A Go/No-Go band: Delta = 2 meV sits on the
    FALLING side, so a measurement returning ~1.4-1.5 meV would give a LARGER
    operational gap than the design assumption, not a degraded one.  See
    ARCHITECTURE.md section 2.4b.
    """
    def gap(d):
        return MajoranaChannel(
            n_sites=1, params=replace(REV21, delta=d)).bulk_gap(nk=4001)

    assert gap(1.4) == pytest.approx(1.3765, abs=2e-3)
    assert gap(2.0) == pytest.approx(1.0000, abs=1e-4)
    assert gap(1.5) > gap(2.0), "Delta = 2 meV is not the optimum at V_z = 3"
    assert gap(1.5) > gap(1.0), "and the peak is interior, not at the edge"
    # the k=0 branch is exact where it wins (Delta above the crossover)
    for d in (1.8, 2.0, 2.5):
        assert gap(d) == pytest.approx(REV21.v_z - d, abs=1e-4)


def test_kill_floor_depends_on_which_quantity_is_held_fixed():
    """T_max = Delta_top / 20 k_B crosses 300 mK at a different Delta under
    the two V_z conventions.  The slide's 0.6 meV floor is the fixed-V_z
    answer; under V_z = 1.5*Delta the floor is ~1.03 meV."""
    def t_max(delta, v_z):
        g = MajoranaChannel(
            n_sites=1, params=replace(REV21, delta=delta, v_z=v_z)).bulk_gap(nk=4001)
        return g / (20 * K_B_MEV_PER_K)

    # fixed V_z = 3 meV: 0.6 meV clears 300 mK (floor is ~0.53)
    assert t_max(0.6, 3.0) == pytest.approx(0.342, abs=0.01)
    assert t_max(0.6, 3.0) > 0.3
    # scaled V_z = 1.5 Delta: 0.6 meV does NOT clear it
    assert t_max(0.6, 0.9) == pytest.approx(0.174, abs=0.01)
    assert t_max(0.6, 0.9) < 0.3
    # and the scaled rule reproduces Rev 2.1's "roughly half the pairing scale"
    for d in (0.6, 1.0, 1.4, 2.0):
        g = MajoranaChannel(
            n_sites=1, params=replace(REV21, delta=d, v_z=1.5 * d)).bulk_gap(nk=4001)
        assert g == pytest.approx(0.5 * d, abs=1e-4)


def test_thermal_suppression_at_operating_point():
    """Delta_top / k_B T = 40.6 and exp(-.) = 2.3e-18 at 1.05 meV, 0.3 K.

    Reported as a LOWER BOUND on the error rate only -- real devices are
    limited by non-equilibrium poisoning, bursts, and control error.
    """
    assert 1.05 / REV21.k_b_t == pytest.approx(40.6, rel=0.01)
    assert REV21.thermal_suppression(1.05) == pytest.approx(2.3e-18, rel=0.05)


# --------------------------------------------------------------------------
# The Rev 2.0 regression -- the bug that invalidated an entire revision
# --------------------------------------------------------------------------
def test_band_bottom_offset_regression():
    """Without the +2t offset, mu = 0 sits mid-band and the wire never enters
    the topological phase (Rev 2.1 Appendix B.1).

    Note: Rev 2.1's appendix says the true critical field 'was
    sqrt((2t)^2 + Delta^2) ~ 20 meV'.  With t = 20 meV from Table 2 that
    formula evaluates to 40.05 meV, not 20.  The formula and the number in the
    appendix disagree; the physics conclusion (a trivial-phase wire) is
    unaffected either way.  Pinned here so the discrepancy is not rediscovered.
    """
    assert REV20_BUGGY.v_z_crit == pytest.approx(
        math.hypot(2 * REV21.t, REV21.delta), rel=1e-12)
    assert REV20_BUGGY.v_z_crit > 19.0

    # At the Rev 2.1 operating point the buggy convention gives a TRIVIAL wire:
    # V_z = 3 meV sits far below its critical field, so there is no zero mode
    # and the lowest excitation is of order the bulk gap rather than
    # exponentially small.
    buggy = MajoranaChannel(n_sites=120, params=REV20_BUGGY)
    good = MajoranaChannel(n_sites=120, params=REV21)
    assert REV20_BUGGY.v_z < REV20_BUGGY.v_z_crit, "buggy wire should be trivial"
    assert buggy.edge_splitting() > 0.5, "trivial wire must have an open gap"
    assert good.edge_splitting() < 0.01, "topological wire must have a near-zero mode"


# --------------------------------------------------------------------------
# Symmetry
# --------------------------------------------------------------------------
def test_particle_hole_symmetry_class_d():
    p_op = np.kron(PAULI_Y, PAULI_Y)
    ch = MajoranaChannel(n_sites=1, params=REV21)
    for k in np.linspace(-math.pi / REV21.a_nm, math.pi / REV21.a_nm, 25):
        h_k = ch.bulk_hamiltonian(float(k))
        h_mk = ch.bulk_hamiltonian(float(-k))
        assert np.allclose(p_op @ h_mk.conj() @ p_op.conj().T, -h_k, atol=1e-12)


def test_bdg_matrix_is_hermitian_and_spectrum_symmetric():
    ch = MajoranaChannel(n_sites=40, params=REV21)
    h = ch.to_bdg_dense()
    assert np.allclose(h, h.conj().T, atol=1e-12)
    ev = ch.spectrum()
    assert np.allclose(ev, -ev[::-1], atol=1e-9)


# --------------------------------------------------------------------------
# Disorder
# --------------------------------------------------------------------------
def test_disorder_is_reproducible_and_degrades_edge_weight():
    a = MajoranaChannel(n_sites=120, params=REV21).with_disorder(0.02, seed=3)
    b = MajoranaChannel(n_sites=120, params=REV21).with_disorder(0.02, seed=3)
    assert np.array_equal(a.delta_mu, b.delta_mu)

    pristine = MajoranaChannel(n_sites=120, params=REV21).edge_weight()
    strong = np.mean([
        MajoranaChannel(n_sites=120, params=REV21)
        .with_disorder(0.10, seed=s).edge_weight() for s in range(5)])
    assert strong < pristine


# --------------------------------------------------------------------------
# MPO
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mu,v", [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0),
                                  (1.5, -0.5), (0.3, 2.0)])
@pytest.mark.parametrize("real", [False, True])
def test_mpo_reconstructs_dense_hamiltonian(mu, v, real):
    chain = InteractingChain(n_sites=8, t=1.0, delta=1.0, mu=mu, v_int=v)
    got = chain.to_mpo(real=real).to_dense()
    assert np.allclose(got, chain.to_dense(), atol=1e-12)


def test_real_mpo_is_float64_and_halves_memory():
    c = InteractingChain(n_sites=8)
    real_mpo = c.to_mpo(real=True)
    cplx_mpo = c.to_mpo(real=False)
    assert real_mpo.tensors[0].dtype == np.float64
    assert cplx_mpo.tensors[0].dtype == np.complex128
    real_bytes = sum(t.nbytes for t in real_mpo.tensors)
    cplx_bytes = sum(t.nbytes for t in cplx_mpo.tensors)
    assert real_bytes * 2 == cplx_bytes


def test_mpo_bond_dimension_is_five():
    assert InteractingChain(n_sites=12).to_mpo().bond_dim == 5


def test_mpo_handles_per_site_disorder():
    chain = InteractingChain(n_sites=8, mu=1.0, v_int=0.5).with_disorder(4.0, 7)
    assert np.allclose(chain.to_mpo().to_dense(), chain.to_dense(), atol=1e-12)


def test_to_dense_refuses_to_blow_up():
    with pytest.raises(ValueError, match="exponential"):
        InteractingChain(n_sites=20).to_mpo().to_dense()


# --------------------------------------------------------------------------
# The known answer everything else is checked against
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n,h", [(8, 0.5), (10, 1.0), (10, 1.7), (12, 1.0)])
def test_tfim_ground_energy_against_free_fermions(n, h):
    """At t = Delta, V = 0, mu = 2h the chain is the transverse-field Ising
    model, whose OBC ground energy has a closed form.  This is the external
    check: the free-fermion solution never touches the MPO code path."""
    chain = InteractingChain(n_sites=n, t=1.0, delta=1.0, mu=2 * h, v_int=0.0)
    e_dense = float(np.linalg.eigvalsh(chain.to_dense()).min())
    assert e_dense == pytest.approx(_tfim_obc_ground_energy(n, 1.0, h),
                                    rel=1e-11)


def test_free_fermion_phase_boundary_is_two_t():
    assert InteractingChain(n_sites=10, t=1.0).mu_c_free == 2.0


def test_interacting_handoff_is_dimensionless():
    """MajoranaChannel.interacting() drops meV.  Guard against anyone quoting
    a DMRG V or mu in meV -- see ARCHITECTURE.md section 2.4."""
    ch = MajoranaChannel(n_sites=60, params=REV21)
    chain = ch.interacting(v_int=1.0)
    assert chain.t == 1.0
    assert chain.delta == pytest.approx(REV21.delta / REV21.t)


# --------------------------------------------------------------------------
# Honeycomb superlattice -- the four gates inherited from generate_labels.py
# --------------------------------------------------------------------------
def test_graphene_total_bandwidth_is_6t():
    lat = HoneycombSuperlattice(n_cells=6)
    frac, _, cell = lat.build_supercell()
    pairs = lat.neighbor_table(frac, cell)
    z = np.zeros(len(frac))
    evs = lat.bloch_bands(frac, cell, pairs, z, z, kgrid=12)
    assert float(evs.max() - evs.min()) == pytest.approx(6 * 2.7, rel=0.03)


def test_graphene_is_gapless_at_dirac_point():
    lat = HoneycombSuperlattice(n_cells=6)
    frac, _, cell = lat.build_supercell()
    pairs = lat.neighbor_table(frac, cell)
    z = np.zeros(len(frac))
    evs = lat.bloch_bands(frac, cell, pairs, z, z, kgrid=12)
    assert float(np.min(np.abs(evs))) < 0.3


def test_superlattice_narrows_the_band():
    patterned = HoneycombSuperlattice(n_cells=8, void_frac=0.3,
                                      well_depth_ev=1.5).solve()
    plain = HoneycombSuperlattice(n_cells=8, void_frac=1e-4,
                                  well_depth_ev=0.0).solve()
    assert patterned["width_mev"] < plain["width_mev"]


@pytest.mark.parametrize("layout",
                         ["triangular", "square", "honeycomb", "kagome",
                          "quasiperiodic"])
def test_all_layouts_solve(layout):
    r = HoneycombSuperlattice(n_cells=5, layout=layout, seed=1).solve(kgrid=4)
    assert r["n_sites"] == 2 * 25
    assert math.isfinite(r["flatness"])
    assert r["gap_mev"] >= 0.0
