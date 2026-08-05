"""
orientation.py -- Module 4a.  Crystallographic prediction of phonon-focusing
symmetry on a cubic surface.

NO TENSOR NETWORKS HERE.  Phonon transport is Boltzmann transport and caustic
geometry; the word "tensor" in this module means a force-constant tensor and
has nothing to do with matrix product states.  See ARCHITECTURE.md section 8.

WHAT THIS MODULE DOES
---------------------
Given a surface plane (hkl) of a cubic crystal, enumerate the high-symmetry
directions that lie IN that plane and predict the fold-symmetry of the
phonon-focusing pattern.  This is pure crystallography -- it needs no DFT data
and no force constants -- and it reproduces every symmetry result in Li et al.,
"Phonon focusing at room temperature", Nature Physics (2026),
doi:10.1038/s41567-026-03335-y, hereafter HU2026:

    (111) -> six-fold      (Fig. 4a, measured Fig. 4d)
    (100) -> eight-fold    (Fig. 4b, measured Fig. 4e)
    (110) -> four-fold     (Fig. 4c, measured Fig. 4f)

WHY THE IN-PLANE TEST IS THE WHOLE POINT
----------------------------------------
A direction [uvw] lies in the plane (hkl) iff h*u + k*v + l*w = 0.  This is the
check that separates a direction which routes heat ALONG the surface from one
that dumps it into the bulk, and it is easy to get wrong by eye.

Concretely, for the (111) surface HU2026 gives the six principal directions as

    [1-10], [-110], [10-1], [-101], [01-1], [0-11]

every one of which sums to zero.  The superficially similar set

    [110], [101], [011]

sums to +2 and lies OUT of the (111) plane entirely -- those directions point
into the substrate.  Mixing the two sets produces a "thermal drain" that is
partly aimed at the thing it is supposed to be draining away from.
``in_plane_directions`` refuses to return them.

WHAT THIS MODULE DOES NOT DO
----------------------------
It predicts WHERE the focusing rays point, not HOW STRONG they are.  The
enhancement is set by the directional density of states (HU2026 eqs. 2-3),
which needs the first-principles group-velocity field.  That is
``tvqpu.substrate.focusing`` -- see ARCHITECTURE.md section 5.2 for the
Jacobian formulation and the reasons the previous histogram-binning
implementation is marked do-not-reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations, product

import numpy as np

__all__ = [
    "Direction", "FocusingPrediction",
    "family", "in_plane_directions", "predict_focusing",
    "HU2026_PRINCIPAL_111", "HU2026_MINOR_111", "HU2026_PRINCIPAL_100",
    "HU2026_PRINCIPAL_110", "G0_BAS_111", "G_OVER_G0_PEAK_300K",
    "RAY_LENGTH_NM",
]

Direction = tuple[int, int, int]

# --------------------------------------------------------------------------
# Published values from HU2026, for validation and for calibrating the
# eventual DOS solver.  Do not edit to make anything pass.
# --------------------------------------------------------------------------

#: Ballistic thermal conductance of BAs on (111), Fig. 3d caption.
#: W m^-1 rad^-1 K^-1.  G0 is the radiation limit -- normally the upper bound
#: on conductance in a solid.
G0_BAS_111 = 8.68

#: Peak G/G0 on the (111) principal direction at 300 K, read from Fig. 3d
#: (G peaks near 40 W m^-1 rad^-1 K^-1 against G0 = 8.68).  Focusing
#: redistributes conductance so the peak EXCEEDS the ballistic limit.
#: This is the number the refactored Jacobian solver must reproduce; the old
#: histogram code produced peak enhancements of 23-44, which is wrong by
#: roughly an order of magnitude.
G_OVER_G0_PEAK_300K = 4.6

#: Measured focused-ray length on BAs (111) vs temperature, HU2026 p.3.
#: Note the trend: rays get LONGER as temperature falls.
RAY_LENGTH_NM: dict[int, float] = {80: 1200.0, 150: 350.0, 300: 250.0}

#: HU2026, (111) surface: six principal directions.  All sum to zero.
HU2026_PRINCIPAL_111: tuple[Direction, ...] = (
    (1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1),
)

#: HU2026, (111) surface: six MINOR directions, visible at 80 K, gone by 300 K.
HU2026_MINOR_111: tuple[Direction, ...] = (
    (2, -1, -1), (-2, 1, 1), (-1, 2, -1), (1, -2, 1), (-1, -1, 2), (1, 1, -2),
)

#: HU2026, (100) surface: eight principal directions.
HU2026_PRINCIPAL_100: tuple[Direction, ...] = (
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0),
)

#: HU2026, (110) surface: four principal directions.  The paper notes the
#: horizontal and vertical pairs are INEQUIVALENT, so this is four-fold in
#: count but not square-symmetric in magnitude.
HU2026_PRINCIPAL_110: tuple[Direction, ...] = (
    (0, 0, 1), (0, 0, -1), (-1, 1, 0), (1, -1, 0),
)


# --------------------------------------------------------------------------
# Crystallography
# --------------------------------------------------------------------------
def family(indices: Direction) -> list[Direction]:
    """All symmetry-equivalent directions of the cubic family <indices>.

    Generated by permutation of the index magnitudes and all sign
    combinations, deduplicated.  For <100> this gives 6 members, <110> gives
    12, <111> gives 8, <211> gives 24.
    """
    mags = tuple(abs(int(i)) for i in indices)
    out: set[Direction] = set()
    for perm in set(permutations(mags)):
        for signs in product((1, -1), repeat=3):
            v = tuple(s * m for s, m in zip(signs, perm))
            if any(v):
                out.add(v)  # type: ignore[arg-type]
    return sorted(out)


def _lies_in_plane(direction: Direction, plane: Direction) -> bool:
    """h*u + k*v + l*w == 0.  Exact integer arithmetic, no tolerance."""
    return sum(int(a) * int(b) for a, b in zip(plane, direction)) == 0


def in_plane_directions(plane: Direction,
                        families: tuple[Direction, ...] = ((1, 0, 0), (1, 1, 0))
                        ) -> list[Direction]:
    """Members of the given cubic families that lie in the surface ``plane``.

    The default families <100> and <110> are the ones HU2026 identifies as
    carrying the PRINCIPAL focusing peaks.  Pass ``((2, 1, 1),)`` for the
    minor directions that appear only at low temperature.
    """
    seen: set[Direction] = set()
    out: list[Direction] = []
    for fam in families:
        for d in family(fam):
            if _lies_in_plane(d, plane) and d not in seen:
                seen.add(d)
                out.append(d)
    return sorted(out)


def _reduce(v: Direction) -> Direction:
    g = math.gcd(math.gcd(abs(v[0]), abs(v[1])), abs(v[2])) or 1
    return (v[0] // g, v[1] // g, v[2] // g)


@dataclass(frozen=True)
class FocusingPrediction:
    plane: Direction
    principal: tuple[Direction, ...]
    minor: tuple[Direction, ...]

    @property
    def fold(self) -> int:
        """Number of principal focusing rays in the surface plane."""
        return len(self.principal)

    def angles_deg(self, reference: Direction | None = None) -> list[float]:
        """In-plane azimuthal angle of each principal ray, in degrees.

        Angles are measured from ``reference`` (default: the first principal
        direction) about the plane normal, so they can be compared directly
        against a measured image.
        """
        n = np.array(self.plane, dtype=float)
        n /= np.linalg.norm(n)
        ref = np.array(reference if reference is not None else self.principal[0],
                       dtype=float)
        ref = ref - np.dot(ref, n) * n
        ref /= np.linalg.norm(ref)
        perp = np.cross(n, ref)
        out = []
        for d in self.principal:
            v = np.array(d, dtype=float)
            v = v - np.dot(v, n) * n
            out.append(math.degrees(math.atan2(float(np.dot(v, perp)),
                                               float(np.dot(v, ref)))) % 360.0)
        return sorted(out)

    def describe(self) -> str:
        def fmt(v: Direction) -> str:
            return "[" + "".join(f"{x}" if x >= 0 else f"-{-x}" for x in v) + "]"
        lines = [
            f"surface ({''.join(str(x) for x in self.plane)}): "
            f"{self.fold}-fold phonon focusing",
            "  principal: " + " ".join(fmt(d) for d in self.principal),
        ]
        if self.minor:
            lines.append("  minor    : " + " ".join(fmt(d) for d in self.minor)
                         + "   (low-temperature only; invisible by 300 K)")
        return "\n".join(lines)


def predict_focusing(plane: Direction) -> FocusingPrediction:
    """Predict the phonon-focusing symmetry of a cubic crystal surface.

    Reproduces HU2026 Fig. 4: (111) -> 6-fold, (100) -> 8-fold, (110) -> 4-fold.

    The prediction is the count and orientation of rays only.  Ray INTENSITY
    requires the directional density of states from first principles; see the
    module docstring.
    """
    plane = _reduce(tuple(int(x) for x in plane))  # type: ignore[arg-type]
    if not any(plane):
        raise ValueError("plane normal must be non-zero")
    principal = in_plane_directions(plane, ((1, 0, 0), (1, 1, 0)))
    minor = in_plane_directions(plane, ((2, 1, 1),))
    return FocusingPrediction(plane=plane,
                              principal=tuple(principal),
                              minor=tuple(minor))


# --------------------------------------------------------------------------
# Thermal-regime guard
# --------------------------------------------------------------------------
def dominant_phonon_frequency_thz(temperature_k: float) -> float:
    """Rough thermal phonon frequency scale, k_B T / h, in THz.

    The point of this helper is the sanity check it enables.  HU2026 is
    explicit that room-temperature focusing in BAs is carried by THz thermal
    phonons whose propagation length stays of order micrometres up to ~8 THz.
    At 0.3 K the scale is ~6 GHz -- three orders of magnitude below the band
    where the measured effect lives, and the THz modes that carry it are
    essentially unoccupied.

    A 'BAs phonon-focusing thermal shield' placed under a 0.3 K stage is
    therefore NOT an application of the measured effect, and this function
    exists so that claim cannot be made accidentally.  See ARCHITECTURE.md
    section 5.3.
    """
    k_b = 1.380649e-23
    h = 6.62607015e-34
    return k_b * float(temperature_k) / h / 1e12


def focusing_regime_note(temperature_k: float) -> str:
    f = dominant_phonon_frequency_thz(temperature_k)
    if temperature_k < 4.0:
        return (f"T = {temperature_k} K -> dominant phonons ~{f * 1000:.1f} GHz. "
                "OUTSIDE the 1-8 THz band in which HU2026 measured focusing "
                "(80-400 K). Transport here is boundary-limited (Casimir, "
                "kappa ~ T^3), not Umklapp-limited. Do not cite the 1300 W/m/K "
                "or the ray-like patterns for this stage.")
    if temperature_k <= 400.0:
        return (f"T = {temperature_k} K -> dominant phonons ~{f:.1f} THz. "
                "Within the range HU2026 measured directly.")
    return (f"T = {temperature_k} K -> above the 400 K point where HU2026 "
            "observed the pattern merging into a hexagon and trending toward "
            "diffusive.")
