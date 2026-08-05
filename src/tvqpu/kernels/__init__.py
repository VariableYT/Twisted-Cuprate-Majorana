"""Module 3 -- GPU kernels.

Every kernel here is guarded: the package imports and the reference path works
with neither Triton nor CUDA present.  See ``fused_twosite`` for why, and for
where the memory pressure in the DMRG inner loop actually is.
"""

from tvqpu.kernels.fused_twosite import (  # noqa: F401
    active_backend,
    have_triton,
    heff_apply,
    heff_apply_reference,
    heff_matvec_operator,
    peak_bytes_fused,
    peak_bytes_naive,
)

__all__ = [
    "heff_apply", "heff_apply_reference", "heff_matvec_operator",
    "have_triton", "active_backend", "peak_bytes_naive", "peak_bytes_fused",
]
