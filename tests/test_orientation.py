"""Module 4a tests -- validated against Li et al., Nature Physics (2026),
doi:10.1038/s41567-026-03335-y (HU2026).

Every symmetry claim in this file has a published, measured counterpart:
HU2026 Fig. 4d/e/f are experimental images showing six-, eight- and four-fold
patterns on (111), (100) and (110) respectively.
"""

from __future__ import annotations

import pytest

from tvqpu.substrate.orientation import (
    HU2026_MINOR_111,
    HU2026_PRINCIPAL_100,
    HU2026_PRINCIPAL_110,
    HU2026_PRINCIPAL_111,
    dominant_phonon_frequency_thz,
    family,
    focusing_regime_note,
    in_plane_directions,
    predict_focusing,
)


# --------------------------------------------------------------------------
# Cubic families
# --------------------------------------------------------------------------
@pytest.mark.parametrize("idx,n", [((1, 0, 0), 6), ((1, 1, 0), 12),
                                   ((1, 1, 1), 8), ((2, 1, 1), 24)])
def test_family_sizes(idx, n):
    assert len(family(idx)) == n


# --------------------------------------------------------------------------
# The three published surfaces
# --------------------------------------------------------------------------
@pytest.mark.parametrize("plane,fold", [((1, 1, 1), 6), ((1, 0, 0), 8),
                                        ((1, 1, 0), 4)])
def test_fold_symmetry_matches_hu2026(plane, fold):
    """HU2026 Fig. 4: six-, eight- and four-fold, predicted AND measured."""
    assert predict_focusing(plane).fold == fold


def test_111_principal_directions_are_exactly_the_published_set():
    got = set(predict_focusing((1, 1, 1)).principal)
    assert got == set(HU2026_PRINCIPAL_111)


def test_100_principal_directions_are_exactly_the_published_set():
    got = set(predict_focusing((1, 0, 0)).principal)
    assert got == set(HU2026_PRINCIPAL_100)


def test_110_principal_directions_are_exactly_the_published_set():
    got = set(predict_focusing((1, 1, 0)).principal)
    assert got == set(HU2026_PRINCIPAL_110)


def test_111_minor_directions_are_the_published_211_set():
    """Visible at 80 K, weaker at 150 K, gone by 300 K (HU2026 Fig. 2b)."""
    got = set(predict_focusing((1, 1, 1)).minor)
    assert got == set(HU2026_MINOR_111)


# --------------------------------------------------------------------------
# The in-plane test -- the error this module exists to catch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [(1, 1, 0), (1, 0, 1), (0, 1, 1)])
def test_positive_110_directions_are_not_in_the_111_plane(bad):
    """[110], [101] and [011] all sum to +2 and point OUT of the (111) plane.

    They are easy to list alongside the real principal directions by eye, and
    a 'thermal drain' built on them would aim partly into the substrate rather
    than along the surface.  HU2026's actual (111) set is the six directions
    that sum to zero.
    """
    assert sum(bad) != 0
    assert bad not in predict_focusing((1, 1, 1)).principal


def test_every_principal_direction_lies_in_its_plane():
    for plane in ((1, 1, 1), (1, 0, 0), (1, 1, 0), (2, 1, 1), (3, 1, 1)):
        for d in predict_focusing(plane).principal:
            assert sum(a * b for a, b in zip(plane, d)) == 0


def test_in_plane_directions_rejects_out_of_plane_families():
    # <111> has no member lying in the (111) plane.
    assert in_plane_directions((1, 1, 1), ((1, 1, 1),)) == []


# --------------------------------------------------------------------------
# Ray geometry
# --------------------------------------------------------------------------
def test_111_rays_are_evenly_spaced_at_60_degrees():
    angles = predict_focusing((1, 1, 1)).angles_deg()
    assert len(angles) == 6
    gaps = [round(b - a, 6) for a, b in zip(angles, angles[1:])]
    assert all(g == pytest.approx(60.0, abs=1e-6) for g in gaps)


def test_100_rays_are_evenly_spaced_at_45_degrees():
    angles = predict_focusing((1, 0, 0)).angles_deg()
    assert len(angles) == 8
    gaps = [round(b - a, 6) for a, b in zip(angles, angles[1:])]
    assert all(g == pytest.approx(45.0, abs=1e-6) for g in gaps)


def test_110_rays_are_orthogonal_but_inequivalent_pairs():
    """HU2026: the (110) four-fold arises from INEQUIVALENT horizontal and
    vertical directions -- <001> and <1-10> are not symmetry-related here."""
    pred = predict_focusing((1, 1, 0))
    angles = pred.angles_deg()
    gaps = [round(b - a, 6) for a, b in zip(angles, angles[1:])]
    assert all(g == pytest.approx(90.0, abs=1e-6) for g in gaps)
    kinds = {tuple(sorted(abs(x) for x in d)) for d in pred.principal}
    assert kinds == {(0, 0, 1), (0, 1, 1)}, "two inequivalent direction types"


def test_plane_index_is_reduced():
    assert predict_focusing((2, 2, 2)).plane == (1, 1, 1)
    assert predict_focusing((2, 2, 2)).fold == 6


def test_zero_plane_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        predict_focusing((0, 0, 0))


# --------------------------------------------------------------------------
# The temperature-regime guard
# --------------------------------------------------------------------------
def test_room_temperature_is_terahertz():
    f = dominant_phonon_frequency_thz(300.0)
    assert f == pytest.approx(6.25, rel=0.02)  # ~6.25 THz, inside HU2026's band


def test_operating_point_is_three_orders_below_the_measured_band():
    """At 0.3 K the dominant phonons are ~6 GHz -- the THz modes that carry
    the measured focusing are unoccupied.  This is why the BAs substrate
    cannot be claimed as a 0.3 K thermal shield on HU2026's evidence."""
    f = dominant_phonon_frequency_thz(0.3)
    assert f * 1000 == pytest.approx(6.25, rel=0.02)  # GHz
    assert dominant_phonon_frequency_thz(300.0) / f == pytest.approx(1000.0,
                                                                    rel=1e-6)


@pytest.mark.parametrize("t,expect", [(0.3, "OUTSIDE"), (4.0, "measured"),
                                      (300.0, "measured"), (600.0, "above")])
def test_regime_note_flags_the_right_band(t, expect):
    assert expect in focusing_regime_note(t)
