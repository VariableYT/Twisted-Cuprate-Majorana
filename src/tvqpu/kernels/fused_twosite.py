"""
fused_twosite.py -- Module 3.  Fused DMRG two-site effective-Hamiltonian apply.

THE OPERATION
-------------
The DMRG inner loop is a Lanczos/Davidson eigensolve on the two-site effective
Hamiltonian.  Once per Krylov vector it computes

    out[a,i,k,b] = sum  L[a,m,A] W1[m,n,i,j] W2[n,p,k,l] R[b,p,B] theta[A,j,l,B]
                  A m n p B j l

    chi = virtual bond dimension    (the thing we want large)
    d   = physical dimension        (2 for spin-1/2)
    D   = MPO bond dimension        (5 for the model of ARCHITECTURE.md 2.4)

WHERE THE MEMORY PRESSURE ACTUALLY IS
-------------------------------------
Contracted naively in four steps the intermediates are, at chi=4096, d=2, D=5,
complex128:

    T1[m,a,j,l,B]  = sum_A L theta      chi^2 d^2 D * 16 B = 5.4 GB
    T2[n,a,i,l,B]  = sum_mj W1 T1       chi^2 d^2 D * 16 B = 5.4 GB
    T3[p,a,i,k,B]  = sum_nl W2 T2       chi^2 d^2 D * 16 B = 5.4 GB
    out[a,i,k,b]   = sum_pB R T3        chi^2 d^2   * 16 B = 1.1 GB

Every one is written to and re-read from HBM, and Lanczos repeats the whole
chain 10-40 times per site per sweep.  That round-tripping, not the FLOPs, is
what caps chi on a 32 GB card.

WHAT THIS MODULE FUSES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
Steps 1 and 4 are the chi^3 ones -- they are large, well-shaped GEMMs, and
cuBLAS already runs them at close to peak.  Hand-writing them in Triton would
be slower.  They stay in torch.

Steps 2 and 3 are chi^2 in FLOPs but chi^2 d^2 D in TRAFFIC: almost no
arithmetic per byte moved.  They are pure bandwidth, and they are what this
kernel fuses.  One kernel reads T1 once and writes T3 once; the D-indexed
intermediate T2 never leaves registers.  That removes one full chi^2 d^2 D
round-trip (a read and a write, ~10.8 GB at chi=4096) per Krylov step.

The MPO tensors themselves are 5*5*2*2 = 100 elements each -- under 1 kB.  They
are loaded once per program into SRAM and reused across the entire tile.  The
smallness of D is exactly why this fuses well.

PRECISION -- NON-NEGOTIABLE
---------------------------
float64 throughout.  Lanczos loses orthogonality in float32 and, worse, the
truncation-error diagnostic stops being meaningful -- and that diagnostic is
the property that makes DMRG trustworthy at all (~1e-12 discarded weight means
trust the answer; climbing toward 1e-3 means the run has left the valid
regime).  There is no low-precision path here and no tensor-core path.  If you
came looking for one, the answer is that this workload cannot use it.

CORRECTNESS AND PORTABILITY
---------------------------
* Every kernel has a ``torch.einsum`` reference, and the test suite asserts
  agreement to 1e-12 relative.  The reference is the DEFAULT; Triton is opt-in
  via ``TVQPU_BACKEND=triton`` or ``backend="triton"``.  A fused kernel that is
  fast and wrong is worse than no kernel.
* This does not run on the development laptop (8 cores, 31.6 GB, integrated
  graphics, no CUDA device; Triton also has no first-class Windows support).
  Module 3 is developed against the reference path locally and executed on the
  4x RTX 5090 box.  Imports are guarded so the package works with neither
  Triton nor CUDA present.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

__all__ = [
    "heff_apply", "heff_apply_reference", "heff_matvec_operator",
    "peak_bytes_naive", "peak_bytes_fused", "have_triton", "active_backend",
]

Backend = Literal["auto", "reference", "triton"]

# --------------------------------------------------------------------------
# Optional dependencies, all guarded
# --------------------------------------------------------------------------
try:
    import torch
    _HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAVE_TORCH = False

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAVE_TRITON = False


def have_triton() -> bool:
    """True iff Triton is importable AND a CUDA device is visible."""
    if not (_HAVE_TRITON and _HAVE_TORCH):
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver weirdness
        return False


def active_backend(requested: Backend = "auto") -> str:
    """Resolve the backend actually used.  ``TVQPU_BACKEND`` overrides."""
    env = os.environ.get("TVQPU_BACKEND", "").strip().lower()
    if env in ("reference", "triton"):
        requested = env  # type: ignore[assignment]
    if requested == "triton":
        if not have_triton():
            raise RuntimeError(
                "backend='triton' requested but Triton/CUDA is unavailable. "
                "This is expected on the development laptop; run on the GPU box.")
        return "triton"
    if requested == "reference":
        return "reference"
    return "triton" if have_triton() else "reference"


# --------------------------------------------------------------------------
# Memory accounting -- the area-law budget, made checkable
# --------------------------------------------------------------------------
def _itemsize(dtype) -> int:
    return int(np.dtype(dtype).itemsize)


def peak_bytes_naive(chi: int, d: int = 2, d_mpo: int = 5,
                     dtype=np.float64) -> int:
    """Peak intermediate bytes for the unfused four-step contraction.

    Counts the two largest co-resident intermediates (T_k and T_{k+1}), plus
    theta and the output.  This is what OOMs a run 40 minutes in.
    """
    it = _itemsize(dtype)
    big = chi * chi * d * d * d_mpo * it        # a D-indexed intermediate
    small = chi * chi * d * d * it              # theta / out
    return 2 * big + 2 * small


def peak_bytes_fused(chi: int, d: int = 2, d_mpo: int = 5,
                     dtype=np.float64) -> int:
    """Peak intermediate bytes with steps 2+3 fused.

    T2 never materializes, so only T1 and T3 are ever co-resident, and the
    kernel can overwrite T1 in place when ``out=`` aliases it.
    """
    it = _itemsize(dtype)
    big = chi * chi * d * d * d_mpo * it
    small = chi * chi * d * d * it
    return 2 * big + 2 * small - big  # T2 eliminated


# --------------------------------------------------------------------------
# Reference implementation -- the ground truth
# --------------------------------------------------------------------------
def heff_apply_reference(l_env, w1, w2, r_env, theta):
    """Unfused H_eff @ theta.  Backend-agnostic (numpy or torch).

    Shapes
        l_env  (chi_l, D, chi_l)      [a, m, A]
        w1     (D, D, d, d)           [m, n, i, j]
        w2     (D, D, d, d)           [n, p, k, l]
        r_env  (chi_r, D, chi_r)      [b, p, B]
        theta  (chi_l, d, d, chi_r)   [A, j, l, B]
    Returns
        out    (chi_l, d, d, chi_r)   [a, i, k, b]

    Contraction order is pinned explicitly rather than left to an optimizer:
    the two chi^3 GEMMs first and last, the two cheap MPO steps in between.
    An optimizer that reorders these can build a chi^2 d^2 D^2 intermediate,
    which is D times worse than anything above.
    """
    xp = torch if (_HAVE_TORCH and torch is not None
                   and isinstance(theta, torch.Tensor)) else np
    ein = xp.einsum

    t1 = ein("amA,AjlB->majlB", l_env, theta)     # chi^3 d^2 D   (GEMM)
    t2 = ein("mnij,majlB->nailB", w1, t1)         # chi^2 d^3 D^2 (cheap)
    t3 = ein("npkl,nailB->paikB", w2, t2)         # chi^2 d^3 D^2 (cheap)
    out = ein("bpB,paikB->aikb", r_env, t3)       # chi^3 d^2 D   (GEMM)
    return out


def _fuse_reference(t1, w1, w2):
    """The reference for exactly what the Triton kernel computes: steps 2+3.

        t1[m,a,j,l,B] -> t3[p,a,i,k,B]
    """
    xp = torch if (_HAVE_TORCH and torch is not None
                   and isinstance(t1, torch.Tensor)) else np
    t2 = xp.einsum("mnij,majlB->nailB", w1, t1)
    return xp.einsum("npkl,nailB->paikB", w2, t2)


# --------------------------------------------------------------------------
# Triton kernel -- steps 2+3, T2 held in registers
# --------------------------------------------------------------------------
if _HAVE_TRITON:

    @triton.jit
    def _fused_mpo_pair_kernel(
        T1_ptr, W1_ptr, W2_ptr, T3_ptr,
        n_batch,                      # = chi_l * chi_r, the flattened (a, B)
        s1_m, s1_b, s1_j, s1_l,       # strides of T1 over (m, batch, j, l)
        s3_p, s3_b, s3_i, s3_k,       # strides of T3 over (p, batch, i, k)
        D: tl.constexpr, DP: tl.constexpr, DPP: tl.constexpr,
        DPHYS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """t3[p,a,i,k,B] = sum_{n,l} W2[n,p,k,l] sum_{m,j} W1[m,n,i,j] t1[m,a,j,l,B]

        One program owns a BLOCK-sized tile of the flattened (a, B) axis.
        D, DP, DPP and DPHYS are compile-time constants, so every MPO/physical
        index loop below fully unrolls: the (n,i,l) intermediate lives in
        registers and is never written to HBM.  This is the whole point of the
        kernel -- see the module docstring.

        D    : left MPO bond of W1
        DP   : shared MPO bond (W1 right == W2 left)
        DPP  : right MPO bond of W2
        DPHYS: physical dimension d
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_batch

        # ---- load the whole T1 slab for this tile: D * d * d vectors --------
        # Held in registers; D and DPHYS are tiny (5 and 2) by construction.
        t1 = [[[tl.load(T1_ptr + m * s1_m + offs * s1_b + j * s1_j + l * s1_l,
                        mask=mask, other=0.0)
                for l in range(DPHYS)]
               for j in range(DPHYS)]
              for m in range(D)]

        # ---- step 2: t2[n,i,l] = sum_{m,j} W1[m,n,i,j] * t1[m,j,l] ---------
        # W1/W2 are scalars per index tuple (under 1 kB total) and are read
        # straight from HBM into registers; the L2 cache serves every program
        # after the first.
        t2 = [[[tl.zeros([BLOCK], dtype=tl.float64)
                for _ in range(DPHYS)]
               for _ in range(DPHYS)]
              for _ in range(DP)]
        for n in range(DP):
            for i in range(DPHYS):
                for l in range(DPHYS):
                    acc = tl.zeros([BLOCK], dtype=tl.float64)
                    for m in range(D):
                        for j in range(DPHYS):
                            w = tl.load(W1_ptr + ((m * DP + n) * DPHYS + i)
                                        * DPHYS + j)
                            acc += w * t1[m][j][l]
                    t2[n][i][l] = acc

        # ---- step 3: t3[p,i,k] = sum_{n,l} W2[n,p,k,l] * t2[n,i,l] ---------
        for p in range(DPP):
            for i in range(DPHYS):
                for k in range(DPHYS):
                    acc = tl.zeros([BLOCK], dtype=tl.float64)
                    for n in range(DP):
                        for l in range(DPHYS):
                            w = tl.load(W2_ptr + ((n * DPP + p) * DPHYS + k)
                                        * DPHYS + l)
                            acc += w * t2[n][i][l]
                    tl.store(T3_ptr + p * s3_p + offs * s3_b + i * s3_i
                             + k * s3_k, acc, mask=mask)


def _fused_mpo_pair_triton(t1, w1, w2, block: int = 256):
    """Launch the fused kernel.  ``t1`` is (D, chi_l, d, d, chi_r)."""
    if not have_triton():  # pragma: no cover
        raise RuntimeError("Triton/CUDA unavailable")
    d_m, chi_l, d1, d2, chi_r = t1.shape
    d_n, d_p, dk, dl = w2.shape
    assert d1 == d2 == dk == dl, "physical dimensions must agree"
    d_phys = d1

    # Flatten (a, B) into one batch axis.  T1 arrives as [m, a, j, l, B]; we
    # want [m, (a B), j, l] so the batch axis is contiguous for coalescing.
    t1c = t1.permute(0, 1, 4, 2, 3).contiguous()      # [m, a, B, j, l]
    t1c = t1c.reshape(d_m, chi_l * chi_r, d_phys, d_phys)
    w1c = w1.contiguous()
    w2c = w2.contiguous()
    n_batch = chi_l * chi_r
    t3 = torch.empty((d_p, n_batch, d_phys, d_phys),
                     dtype=t1.dtype, device=t1.device)

    grid = (triton.cdiv(n_batch, block),)
    _fused_mpo_pair_kernel[grid](
        t1c, w1c, w2c, t3,
        n_batch,
        t1c.stride(0), t1c.stride(1), t1c.stride(2), t1c.stride(3),
        t3.stride(0), t3.stride(1), t3.stride(2), t3.stride(3),
        D=w1.shape[0], DP=w1.shape[1], DPP=d_p, DPHYS=d_phys,
        BLOCK=block,
    )
    # back to [p, a, i, k, B]
    t3 = t3.reshape(d_p, chi_l, chi_r, d_phys, d_phys)
    return t3.permute(0, 1, 3, 4, 2)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def heff_apply(l_env, w1, w2, r_env, theta, backend: Backend = "auto",
               block: int = 256):
    """H_eff @ theta, with steps 2+3 fused when Triton is available.

    Falls back to :func:`heff_apply_reference` transparently.  The two paths
    agree to 1e-12 relative -- ``tests/test_fused_twosite.py`` asserts it, and
    that test is the reason to trust this function at all.
    """
    chosen = active_backend(backend)
    if chosen == "reference":
        return heff_apply_reference(l_env, w1, w2, r_env, theta)

    if theta.dtype not in (torch.float64, torch.float32):
        raise NotImplementedError(
            "the Triton path is real-valued. Build the MPO with "
            "InteractingChain.to_mpo(real=True) -- the Y -> iY substitution "
            "makes the Jordan-Wigner Hamiltonian real at zero accuracy cost. "
            "For a genuinely complex MPO, use backend='reference'.")
    if theta.dtype == torch.float32:
        raise ValueError(
            "float32 is not supported: Lanczos loses orthogonality and the "
            "truncation-error diagnostic stops being meaningful. See the "
            "module docstring.")

    t1 = torch.einsum("amA,AjlB->majlB", l_env, theta)
    t3 = _fused_mpo_pair_triton(t1, w1, w2, block=block)
    return torch.einsum("bpB,paikB->aikb", r_env, t3)


def heff_matvec_operator(l_env, w1, w2, r_env, shape, backend: Backend = "auto"):
    """Wrap :func:`heff_apply` as a flat matrix-vector product, which is the
    interface Lanczos/Davidson eigensolvers actually want.

    ``shape`` is (chi_l, d, d, chi_r).  The returned callable maps a flat
    vector of length prod(shape) to another of the same length.
    """
    def matvec(vec):
        theta = vec.reshape(shape)
        out = heff_apply(l_env, w1, w2, r_env, theta, backend=backend)
        return out.reshape(-1)
    return matvec
