"""Module 4 -- thermal substrate.

Ballistic phonon transport: Boltzmann transport and caustic geometry.  No
tensor networks; see ARCHITECTURE.md section 8 on the nomenclature collision.

Implemented:
  * ``orientation``      -- crystallographic focusing symmetry (no DFT needed)
  * ``directional_dos``  -- the Jacobian solver: focusing factor, directional
    DOS, and G/G0.  Gate status in ARCHITECTURE.md section 5.2c.
"""

from tvqpu.substrate.directional_dos import (  # noqa: F401
    BAS_ELASTIC,
    SI_ELASTIC,
    CubicElasticField,
    PhonopyFC2Field,
    directional_conductance,
    enhancement_along,
    focusing_map,
    fold_symmetry,
)
from tvqpu.substrate.orientation import (  # noqa: F401
    FocusingPrediction,
    dominant_phonon_frequency_thz,
    family,
    focusing_regime_note,
    in_plane_directions,
    predict_focusing,
)

__all__ = [
    "predict_focusing", "FocusingPrediction", "in_plane_directions", "family",
    "dominant_phonon_frequency_thz", "focusing_regime_note",
    "focusing_map", "enhancement_along", "fold_symmetry",
    "directional_conductance", "CubicElasticField", "PhonopyFC2Field",
    "BAS_ELASTIC", "SI_ELASTIC",
]
