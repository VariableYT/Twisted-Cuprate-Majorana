"""
budget.py -- area-law memory bounds, enforced before a run starts rather than
discovered via an OOM forty minutes in.

The bound is legitimate for the gapped topological phase of ARCHITECTURE.md
section 2 because 1D gapped ground states obey an area law: entanglement
entropy saturates at a constant set by the correlation length, INDEPENDENT of
system size.  Required chi therefore does not grow with n, and cost is strictly
O(n chi^3).

It is NOT legitimate near the transition.  At criticality S(l) ~ (c/6) log l
for open boundaries, so chi must grow polynomially with length and accuracy
genuinely degrades as the chain lengthens.  ``plan()`` flags runs whose
parameters sit near a known phase boundary so they cannot silently inherit the
gapped budget.

    MPS state memory ~ 32 n chi^2 bytes  (complex128)
    DMRG sweep cost  ~ n chi^3
    Max entanglement entropy across any cut:  S = log2(chi)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Budget", "plan", "max_chi_for_bytes", "max_chi_for_plan",
           "bytes_per_chi_squared", "state_bytes", "max_entanglement_entropy"]

_GIB = 1024 ** 3


def state_bytes(n_sites: int, chi: int, phys_dim: int = 2,
                dtype=np.complex128) -> int:
    """MPS state storage.  n tensors of shape (chi, d, chi)."""
    return n_sites * chi * phys_dim * chi * int(np.dtype(dtype).itemsize)


def max_chi_for_bytes(n_sites: int, budget_bytes: int, phys_dim: int = 2,
                      dtype=np.complex128) -> int:
    """Largest chi whose MPS STATE ALONE fits in ``budget_bytes``.

    This is the number in the planning table of ``docs/tensor_networks.md``,
    and it is an UPPER BOUND that real runs never reach: environments dominate
    (see ``bytes_per_chi_squared``).  Use :func:`max_chi_for_plan` to size an
    actual run.
    """
    per = n_sites * phys_dim * int(np.dtype(dtype).itemsize)
    if per <= 0:
        return 0
    return max(int(math.isqrt(max(budget_bytes // per, 0))), 0)


def bytes_per_chi_squared(n_sites: int, phys_dim: int = 2,
                          dtype=np.complex128, n_krylov: int = 40,
                          d_mpo: int = 5) -> int:
    """Every term in the peak scales as chi^2, so the whole budget is one
    coefficient:

        peak = chi^2 * ( n d      [state]
                       + 2 n D    [two environment stacks]
                       + k d^2 )  [Krylov space]  * itemsize

    The environment term is usually the largest by a wide margin -- 2D = 10
    against d = 2 -- which is why chi budgeted from the state alone is
    optimistic by roughly a factor of sqrt(6) in practice.
    """
    it = int(np.dtype(dtype).itemsize)
    return it * (n_sites * phys_dim
                 + 2 * n_sites * d_mpo
                 + n_krylov * phys_dim * phys_dim)


def max_chi_for_plan(n_sites: int, limit_gib: float, phys_dim: int = 2,
                     dtype=np.complex128, n_krylov: int = 40,
                     d_mpo: int = 5) -> int:
    """Largest chi whose FULL projected peak fits in ``limit_gib``."""
    per = bytes_per_chi_squared(n_sites, phys_dim, dtype, n_krylov, d_mpo)
    if per <= 0:
        return 0
    return max(int(math.isqrt(int(limit_gib * _GIB) // per)), 0)


def max_entanglement_entropy(chi: int) -> float:
    """S = log2(chi), in bits.  The honest exchange rate against a dense
    state-vector simulation, which supports S = N/2 for N qubits: chi = 1000
    gives S ~ 10 bits, the entanglement capacity of a 20-qubit dense sim."""
    return math.log2(max(chi, 1))


@dataclass
class Budget:
    n_sites: int
    chi: int
    phys_dim: int
    dtype: type
    state_bytes: int
    env_bytes: int
    krylov_bytes: int
    peak_bytes: int
    limit_bytes: int
    sweep_flops: float
    max_entropy_bits: float
    critical_warning: str | None = None

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.limit_bytes

    def report(self) -> str:
        def gb(x: int) -> str:
            return f"{x / _GIB:7.2f} GiB"
        lines = [
            f"n = {self.n_sites} sites, chi = {self.chi}, d = {self.phys_dim}, "
            f"{np.dtype(self.dtype).name}",
            f"  MPS state       {gb(self.state_bytes)}",
            f"  environments    {gb(self.env_bytes)}",
            f"  Krylov vectors  {gb(self.krylov_bytes)}",
            f"  PEAK            {gb(self.peak_bytes)}   "
            f"(limit {gb(self.limit_bytes)})  ->  "
            f"{'FITS' if self.fits else 'DOES NOT FIT'}",
            f"  sweep cost      {self.sweep_flops:.3e} FLOP  (~n chi^3)",
            f"  max entanglement across a cut: S = {self.max_entropy_bits:.1f} bits "
            f"(= a {2 * self.max_entropy_bits:.0f}-qubit dense simulation)",
        ]
        if self.critical_warning:
            lines.append(f"  WARNING: {self.critical_warning}")
        return "\n".join(lines)


def plan(n_sites: int, chi: int, limit_gib: float = 8.0, phys_dim: int = 2,
         dtype=np.complex128, n_krylov: int = 40, d_mpo: int = 5,
         near_critical: bool = False, strict: bool = True) -> Budget:
    """Project peak memory for a DMRG run and refuse it if it will not fit.

    ``near_critical`` marks a run whose parameters sit close to a phase
    boundary (e.g. mu -> mu_c = 2t for the chain of ARCHITECTURE.md 2.4).
    Such runs must not inherit the gapped area-law budget; the required chi
    grows with length there.
    """
    it = int(np.dtype(dtype).itemsize)
    s_bytes = state_bytes(n_sites, chi, phys_dim, dtype)
    # Two environment stacks (left and right), one tensor per site,
    # shape (chi, D, chi).
    e_bytes = 2 * n_sites * chi * d_mpo * chi * it
    # Krylov space for the two-site eigensolve: n_krylov vectors of theta.
    k_bytes = n_krylov * chi * phys_dim * phys_dim * chi * it
    peak = s_bytes + e_bytes + k_bytes
    limit = int(limit_gib * _GIB)

    warn = None
    if near_critical:
        warn = ("run flagged near-critical: entanglement grows as "
                "S(l) ~ (c/6) log l, so this chi does NOT transfer across "
                "system sizes. Converge chi explicitly and watch the "
                "truncation error.")

    b = Budget(n_sites=n_sites, chi=chi, phys_dim=phys_dim, dtype=dtype,
               state_bytes=s_bytes, env_bytes=e_bytes, krylov_bytes=k_bytes,
               peak_bytes=peak, limit_bytes=limit,
               sweep_flops=float(n_sites) * float(chi) ** 3,
               max_entropy_bits=max_entanglement_entropy(chi),
               critical_warning=warn)

    if strict and not b.fits:
        fits_at = max_chi_for_plan(n_sites, limit_gib, phys_dim, dtype,
                                   n_krylov, d_mpo)
        raise MemoryError(
            f"projected peak {peak / _GIB:.2f} GiB exceeds the "
            f"{limit_gib:.2f} GiB budget.\n" + b.report() +
            f"\n  suggestion: chi <= {fits_at} fits this budget "
            f"(S = {max_entanglement_entropy(fits_at):.1f} bits). "
            "Pass strict=False to plan anyway.")
    return b
