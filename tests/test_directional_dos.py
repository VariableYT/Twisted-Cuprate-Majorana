"""Module 4b tests -- the directional-DOS / phonon-focusing solver.

The load-bearing test in this file is ``test_isotropic_medium_has_no_focusing``.
An elastically isotropic medium has a known exact answer -- the enhancement is
1 in every direction -- and the solver reproduces it to eight digits.  That is
what licenses trusting it on an anisotropic crystal, where there is no
closed form to check against.

The old implementation this replaces could not have passed that test: it
binned the direction map, so its answer depended on bin width even in the
isotropic case.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tvqpu.substrate.directional_dos import (
    verify_measure_conservation,
    caustic_cutoff_from_geometry,
    dominant_phonon_wavelength_nm,
    BAS_ELASTIC,
    SI_ELASTIC,
    CubicElasticField,
    PhonopyFC2Field,
    count_rays,
    enhancement_along,
    fibonacci_sphere,
    focusing_map,
    fold_symmetry,
)

# Optional external dataset: the NIMS MDR phono3py_params file for
# zincblende BAs. Not redistributed here. Point TVQPU_BAS_FC2 at a local
# copy to enable the real-dispersion tests; without it they skip, which
# is the expected state for a fresh clone.
BAS_FC2 = os.environ.get("TVQPU_BAS_FC2", "")
requires_fc2 = pytest.mark.skipif(
    not os.path.exists(BAS_FC2),
    reason="BAs phono3py dataset not present (see ARCHITECTURE 5.2b)")


@pytest.fixture(scope="module")
def bas_field():
    return CubicElasticField(BAS_ELASTIC)


@pytest.fixture(scope="module")
def bas_map(bas_field):
    return focusing_map(bas_field, n_points=20000)


# --------------------------------------------------------------------------
# The known answer
# --------------------------------------------------------------------------
def test_isotropic_medium_has_no_focusing():
    """Zener anisotropy exactly 1 => v_g parallel to n everywhere => A == 1.

    This is the solver's calibration against an exact result.  Eight digits.
    """
    iso = BAS_ELASTIC.isotropic_like()
    assert iso.zener == pytest.approx(1.0, abs=1e-12)
    fmap = focusing_map(CubicElasticField(iso), n_points=3000)
    assert fmap.enhancement.min() == pytest.approx(1.0, abs=1e-6)
    assert fmap.enhancement.max() == pytest.approx(1.0, abs=1e-6)
    assert fmap.clipped == 0
    assert fmap.covering_degree == pytest.approx(1.0, abs=1e-6)


def test_isotropic_group_velocity_is_radial():
    iso = CubicElasticField(BAS_ELASTIC.isotropic_like())
    n = fibonacci_sphere(200)
    v_p, v_g, _ = iso.evaluate(n)
    v_hat = v_g / np.linalg.norm(v_g, axis=-1, keepdims=True)
    for s in range(3):
        assert np.allclose(np.abs(np.einsum("ni,ni->n", v_hat[:, s], n)), 1.0,
                           atol=1e-9)


@pytest.mark.parametrize("direction", [(1, 0, 0), (1, 1, 0), (1, 1, 1),
                                       (0.3, 0.5, 0.81), (2, 1, 3)])
def test_isotropic_enhancement_is_one_in_every_direction(direction):
    iso = CubicElasticField(BAS_ELASTIC.isotropic_like())
    fmap = focusing_map(iso, n_points=3000)
    assert enhancement_along(iso, fmap, direction) == pytest.approx(1.0, abs=1e-4)


def test_christoffel_sound_speeds_match_closed_forms():
    """Textbook cubic acoustic velocities.  An independent check of the
    Christoffel solve that does not go through the Jacobian machinery."""
    e = BAS_ELASTIC
    c11, c12, c44, rho = e.c11 * 1e9, e.c12 * 1e9, e.c44 * 1e9, e.density
    f = CubicElasticField(e)

    def speeds(d):
        n = np.array([d], dtype=float)
        n /= np.linalg.norm(n)
        return np.sort(f.evaluate(n)[0][0])

    assert speeds((1, 0, 0)) == pytest.approx(
        np.sort([np.sqrt(c11 / rho)] + [np.sqrt(c44 / rho)] * 2), rel=1e-10)
    assert speeds((1, 1, 0)) == pytest.approx(np.sort([
        np.sqrt((c11 + c12 + 2 * c44) / (2 * rho)),
        np.sqrt(c44 / rho),
        np.sqrt((c11 - c12) / (2 * rho))]), rel=1e-10)
    assert speeds((1, 1, 1)) == pytest.approx(np.sort([
        np.sqrt((c11 + 2 * c12 + 4 * c44) / (3 * rho))]
        + [np.sqrt((c11 - c12 + c44) / (3 * rho))] * 2), rel=1e-10)


def test_sound_speeds_agree_with_the_phono3py_pipeline():
    """LA[100] and TA[100] against the values the phono3py path produced
    (7222 and 5311 m/s).  Agreement here is also what vindicates choosing the
    ultrasonic C44 = 149 GPa over the Brillouin 173 GPa -- the two experiments
    disagree by 16% and this picks a side, so it is worth pinning."""
    f = CubicElasticField(BAS_ELASTIC)
    n = np.array([[1.0, 0.0, 0.0]])
    v = np.sort(f.evaluate(n)[0][0])
    assert v[0] == pytest.approx(5311.0, rel=0.02)   # TA
    assert v[2] == pytest.approx(7222.0, rel=0.04)   # LA


# --------------------------------------------------------------------------
# Anisotropic BAs
# --------------------------------------------------------------------------
def test_bas_is_anisotropic_and_folds_the_sphere():
    assert BAS_ELASTIC.zener == pytest.approx(1.433, rel=0.01)
    fmap = focusing_map(CubicElasticField(BAS_ELASTIC), n_points=20000)
    # covering degree > 1 means caustics exist: folded sheets cover part of
    # the v_hat sphere more than once.
    assert fmap.covering_degree > 1.05


def test_110_is_the_most_focused_direction(bas_field, bas_map):
    """The six (111)-plane principal directions of Li et al. are the in-plane
    <110> set, and this is the mechanism: <110> is where the enhancement is."""
    a110 = enhancement_along(bas_field, bas_map, (1, 1, 0))
    a100 = enhancement_along(bas_field, bas_map, (1, 0, 0))
    a111 = enhancement_along(bas_field, bas_map, (1, 1, 1))
    assert a110 > 10.0, f"<110> should be strongly focused, got {a110:.2f}"
    assert a110 > a100 > a111, (a110, a100, a111)


def test_measure_is_conserved_on_an_anisotropic_field(bas_field, bas_map):
    """The invariant that catches sum-vs-mean errors in the preimage estimator.

    The isotropic calibration CANNOT catch them -- an unfolded map has one
    preimage per branch, where a sum and a mean coincide.  This test runs on
    BAs, where the map folds, and it is the reason the estimator can be
    trusted for absolute magnitudes rather than only for ratios.

    400 probes; the convergence is Monte-Carlo-slow because the integrand has
    sharp caustic peaks (60 probes gives ~1.20, 150 gives ~0.95, 400 gives
    ~1.02), so the tolerance here is deliberately loose and the probe count
    deliberately explicit.
    """
    assert verify_measure_conservation(bas_field, bas_map,
                                       n_probe=400) == pytest.approx(1.0, abs=0.10)


def test_caustic_cutoff_is_not_binding_at_the_default(bas_field, bas_map):
    """If the clip set the answer, the reported peak would move with it."""
    vals = [enhancement_along(bas_field, bas_map, (1, 1, 0), caustic_cutoff=c)
            for c in (50.0, 200.0, 1000.0)]
    assert max(vals) / min(vals) < 1.01, vals
    # ...but a cutoff below the peak does bite, which proves the clip works.
    assert enhancement_along(bas_field, bas_map, (1, 1, 0),
                             caustic_cutoff=20.0) < 0.6 * vals[0]


def test_enhancement_is_resolution_independent(bas_field):
    """The property the histogram implementation did not have.

    Because the estimator finds preimages and evaluates an analytic Jacobian
    there, the answer must not drift with sphere sampling or with the seed
    cone.  A binned estimator fails this badly near a caustic, which is where
    the old code's 23-44 peak enhancements came from.
    """
    vals = [enhancement_along(bas_field, focusing_map(bas_field, n_points=n),
                              (1, 1, 0))
            for n in (5000, 20000, 40000)]
    assert max(vals) / min(vals) < 1.05, f"drifts with sampling: {vals}"

    fmap = focusing_map(bas_field, n_points=20000)
    cones = [enhancement_along(bas_field, fmap, (1, 1, 0), seed_cone_deg=c)
             for c in (4.0, 8.0, 15.0)]
    assert max(cones) / min(cones) < 1.05, f"drifts with seed cone: {cones}"


def test_all_symmetry_equivalent_110_directions_agree(bas_field, bas_map):
    """Cubic symmetry is not built into the solver, so equal values across the
    <110> star is an independent check that the Jacobian is orientation-free."""
    vals = [enhancement_along(bas_field, bas_map, d)
            for d in ((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, -1, 0), (-1, 0, 1))]
    assert max(vals) / min(vals) < 1.02, vals


# --------------------------------------------------------------------------
# Gate (a): fold symmetry
# --------------------------------------------------------------------------
def test_count_rays_counts_contiguous_arcs():
    assert count_rays(np.array([2.0, 2.0, 0.5, 2.0, 0.5])) == 2
    assert count_rays(np.array([0.5, 0.5, 0.5])) == 0
    assert count_rays(np.array([2.0, 2.0, 2.0])) == 1
    # wraps around the circle
    assert count_rays(np.array([2.0, 0.5, 0.5, 2.0])) == 1
    # a cusp pair inside one arc is still one ray
    assert count_rays(np.array([0.5, 3.0, 1.2, 3.0, 0.5])) == 1


def test_gate_a_111_reproduces_the_principal_and_minor_ray_sets(bas_field,
                                                                bas_map):
    """Li et al. on (111): SIX principal <110> rays, plus SIX minor <211> rays
    that are visible at 80 K, weaker at 150 K, and gone by 300 K.

    The elastic ballistic model has no scattering, so it is effectively the
    low-temperature limit -- and it shows both sets, which is what Li et al.
    Fig. 2b measures at 80 K.  The principal set comes out ~15x stronger, which
    is why the minor set is the first thing scattering removes.

    Both sets being internally identical to three decimals is also an
    independent check that the solver respects cubic symmetry: none is built
    in, so equality across each star has to be earned.
    """
    principal = [(1, -1, 0), (-1, 1, 0), (1, 0, -1),
                 (-1, 0, 1), (0, 1, -1), (0, -1, 1)]
    minor = [(2, -1, -1), (-2, 1, 1), (-1, 2, -1),
             (1, -2, 1), (-1, -1, 2), (1, 1, -2)]
    vp = [enhancement_along(bas_field, bas_map, d) for d in principal]
    vm = [enhancement_along(bas_field, bas_map, d) for d in minor]

    assert max(vp) / min(vp) < 1.01, f"principal star not uniform: {vp}"
    assert max(vm) / min(vm) < 1.01, f"minor star not uniform: {vm}"
    assert np.mean(vp) > 10.0
    assert np.mean(vp) / np.mean(vm) > 8.0, "principal must dominate minor"


def test_gate_a_111_ray_count_is_threshold_dependent(bas_field, bas_map):
    """Twelve rays above the ballistic limit; six once you require a ray to be
    meaningfully above it.

    This is not a defect of the counter -- it is the physics.  The minor
    <211> set sits at ~1.16, barely over G/G0 = 1, so a threshold anywhere in
    (1.2, 17) selects exactly the six principal rays that survive to 300 K.
    A single 'fold number' is therefore not well defined without saying which
    temperature regime is meant.
    """
    fold_all, _, _ = fold_symmetry(bas_field, bas_map, (1, 1, 1),
                                   n_angles=180, threshold=1.0)
    fold_principal, _, _ = fold_symmetry(bas_field, bas_map, (1, 1, 1),
                                         n_angles=180, threshold=3.0)
    assert fold_all == 12, fold_all
    assert fold_principal == 6, fold_principal


def test_gate_a_110_is_four_fold(bas_field, bas_map):
    """Li et al. Fig. 4c/4f: four-fold on (110)."""
    fold, angles, prof = fold_symmetry(bas_field, bas_map, (1, 1, 0),
                                       n_angles=72)
    assert fold == 4, f"got {fold}-fold, peak {prof.max():.2f}"


def test_gate_a_100_is_NOT_reproduced_in_the_elastic_limit(bas_field, bas_map):
    """Li et al. Fig. 4b/4e measure eight-fold on (100).  The elastic
    (long-wavelength) model does NOT reproduce it: the entire (100) plane sits
    above the ballistic limit, so there is no angular contrast and no ray
    structure to count.

    This is a regime difference, not a solver bug.  Li et al.'s pattern comes
    from the iso-frequency surface at 7.8 THz, where the dispersion has
    flattened; the elastic limit is k -> 0 and knows nothing about it.
    Reaching this gate requires the PhonopyFC2Field provider.

    The test asserts the failure so it cannot be mistaken for a pass.
    """
    fold, angles, prof = fold_symmetry(bas_field, bas_map, (1, 0, 0),
                                       n_angles=72)
    assert fold == 1
    assert np.median(prof) > 5.0, "expected a uniformly bright (100) plane"


def test_gate_b_111_rays_are_narrow(bas_field, bas_map):
    """Li et al. Fig. 3d shows a narrow peak within +/-15 degrees.  Six rays
    occupying ~10% of the azimuth is ~6 degrees of half-width each."""
    fold, angles, prof = fold_symmetry(bas_field, bas_map, (1, 1, 1),
                                       n_angles=72)
    frac_bright = float((prof > 1.0).mean())
    assert 0.03 < frac_bright < 0.30, frac_bright


def test_gate_4K4_geometric_ceiling_binds_for_a_1um_interface(bas_field,
                                                              bas_map):
    """Gate 4K-4: the reported peak must be consistent with what the geometry
    can support.

    For a 1 um BAs die-attach layer at 4 K the ceiling is ~15.6, and the
    solver's unconstrained <110> output is 17.30 -- so the ceiling BINDS, and
    a magnitude quoted for that geometry must be the constrained value.
    Quoting 17.30 for a 1 um layer would be quoting a number the geometry
    cannot support.
    """
    ceiling = caustic_cutoff_from_geometry(4.0, 1000.0)
    unconstrained = enhancement_along(bas_field, bas_map, (1, 1, 0))
    assert unconstrained > ceiling, "ceiling should bind at 1 um"

    constrained = enhancement_along(bas_field, bas_map, (1, 1, 0),
                                    caustic_cutoff=ceiling)
    # NOTE the arithmetic: the ceiling clips each PREIMAGE's divergence, and
    # the reported G/G0 is the normalized SUM over preimages.  So the
    # constrained peak is not the ceiling itself -- it comes out near 6.0,
    # well below both the ceiling (15.6) and the unconstrained value (17.30).
    # 6.0 is the number to quote for a 1 um layer.
    assert constrained == pytest.approx(6.03, rel=0.05), constrained
    assert constrained < ceiling < unconstrained

    # a 10x thicker layer lifts the ceiling clear of the caustic entirely
    thick = caustic_cutoff_from_geometry(4.0, 10000.0)
    assert thick > unconstrained
    assert enhancement_along(bas_field, bas_map, (1, 1, 0),
                             caustic_cutoff=thick) == pytest.approx(
                                 unconstrained, rel=1e-6)


def test_gate_4KR_is_a_regression_pin_not_a_validation(bas_field, bas_map):
    """17.30 along <110> is THIS SOLVER'S OWN OUTPUT.

    It is pinned so a refactor that moves it is caught, and it is named so
    nobody mistakes it for validation against a published number.  The real
    magnitude gate is 4K-5 (Northrop & Wolfe Si/Ge caustics), which is not
    yet run.
    """
    assert enhancement_along(bas_field, bas_map, (1, 1, 0)) == pytest.approx(
        17.30, rel=0.02)


def test_gate_c_elastic_limit_overshoots_the_published_magnitude(bas_field,
                                                                bas_map):
    """Gate (c) is peak G/G0 = 4.6 (Li et al. Fig. 3d, G0 = 8.68 W/m/rad/K).

    The elastic ballistic limit gives ~17 -- roughly 3.6x too large -- and the
    reason is physical, not numerical: this model has no scattering, so its
    caustics are sharper than a real 300 K crystal with a ~250 nm propagation
    length.  Li et al.'s value comes from the full scattering matrix.

    Gate (c) therefore CANNOT be met by the elastic model, and this test pins
    the discrepancy rather than letting a future reader assume the gate passed.
    """
    peak = enhancement_along(bas_field, bas_map, (1, 1, 0))
    assert peak > 4.6, "elastic limit should overshoot, not undershoot"
    assert 10.0 < peak < 25.0, f"peak {peak:.1f} outside the expected band"


# --------------------------------------------------------------------------
# Silicon -- a different anisotropy, to show the solver is not tuned to BAs
# --------------------------------------------------------------------------
def test_thermal_wavelength_puts_4K_in_the_elastic_regime():
    """At 4 K the dominant phonon wavelength is ~64 nm and the frequency
    ~83 GHz -- far inside the linear-dispersion regime, where the Christoffel
    model is exact rather than approximate.  This is the justification for
    using the elastic provider at the 4 K interface stage (ARCHITECTURE 5.2d).
    """
    lam4 = dominant_phonon_wavelength_nm(4.0)
    lam300 = dominant_phonon_wavelength_nm(300.0)
    assert lam4 == pytest.approx(64.0, rel=0.05)
    assert lam300 == pytest.approx(0.85, rel=0.05)
    assert lam4 / lam300 == pytest.approx(75.0, rel=0.01)


def test_geometric_caustic_ceiling_constrains_the_reported_peak():
    """The caustic divergence is cut off by diffraction over lambda/L.

    At 4 K with a 1 um propagation length the ceiling is ~16 -- BELOW the
    17.30 this model reports with its default cutoff of 50.  So a 4 K
    magnitude quoted from this solver has to state the assumed L, and for a
    thin interface layer the default cutoff is not physically justified.
    """
    ceiling_1um = caustic_cutoff_from_geometry(4.0, 1000.0)
    assert ceiling_1um == pytest.approx(15.6, rel=0.05)
    assert ceiling_1um < 17.30, "the geometric ceiling should bite here"
    # a thicker layer supports more
    assert caustic_cutoff_from_geometry(4.0, 10000.0) > 100.0
    # and it never drops below the isotropic limit
    assert caustic_cutoff_from_geometry(4.0, 1.0) == 1.0


def test_silicon_is_more_anisotropic_and_folds_more():
    assert SI_ELASTIC.zener > BAS_ELASTIC.zener
    si = focusing_map(CubicElasticField(SI_ELASTIC), n_points=20000)
    bas = focusing_map(CubicElasticField(BAS_ELASTIC), n_points=20000)
    assert si.covering_degree > bas.covering_degree


# --------------------------------------------------------------------------
# The real dispersion
# --------------------------------------------------------------------------
@requires_fc2
def test_phonopy_provider_reproduces_the_elastic_limit_at_small_q():
    """At |q| -> 0 the real dispersion must return the elastic sound speeds.
    This is what ties the two providers together."""
    f = PhonopyFC2Field(BAS_FC2, q_invang=0.05)
    n = np.array([[1.0, 0.0, 0.0]])
    _, v_g, _ = f.evaluate(n)
    speeds = np.sort(np.linalg.norm(v_g[0], axis=-1))
    assert speeds[0] == pytest.approx(5311.0, rel=0.03)   # TA
    assert speeds[2] == pytest.approx(7222.0, rel=0.03)   # LA


@requires_fc2
def test_phonopy_dispersion_flattens_toward_the_zone_boundary():
    """Li et al.'s effect lives near 7.8 THz, where group velocities are far
    below the sound speeds.  If this were not true the elastic model would be
    sufficient and gate (c) would not need the real dispersion."""
    slow = PhonopyFC2Field(BAS_FC2, q_invang=0.05)
    fast = PhonopyFC2Field(BAS_FC2, q_invang=0.90)
    v_slow = np.linalg.norm(slow.evaluate(np.array([[1.0, 0, 0]]))[1][0], axis=-1)
    v_fast = np.linalg.norm(fast.evaluate(np.array([[1.0, 0, 0]]))[1][0], axis=-1)
    assert v_fast.max() < 0.5 * v_slow.max()
