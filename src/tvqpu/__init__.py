"""topological-vqpu -- GPU-accelerated tensor-network and BdG simulation of an
engineered topological-superconductor lattice Hamiltonian.

This is a classical simulator.  It runs float64 linear algebra on CPUs and
GPUs.  It is not a quantum processor, and it does not emulate one.  See
ARCHITECTURE.md section 0 for the three statements that bound every claim this
package makes.

Submodules are imported lazily (PEP 562).  That keeps ``import tvqpu`` cheap,
avoids pulling torch/quimb in for callers who only need the lattice builder,
and stops ``python -m tvqpu.lattice`` from double-importing its own module.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_LAZY = {
    "REV21": "tvqpu.lattice",
    "ModelParams": "tvqpu.lattice",
    "MajoranaChannel": "tvqpu.lattice",
    "InteractingChain": "tvqpu.lattice",
    "MPO": "tvqpu.lattice",
    "HoneycombSuperlattice": "tvqpu.lattice",
    "plan": "tvqpu.budget",
    "Budget": "tvqpu.budget",
}

__all__ = [*_LAZY, "__version__"]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from tvqpu.budget import Budget, plan
    from tvqpu.lattice import (
        MPO,
        REV21,
        HoneycombSuperlattice,
        InteractingChain,
        MajoranaChannel,
        ModelParams,
    )
