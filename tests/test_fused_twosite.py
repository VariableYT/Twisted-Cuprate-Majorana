"""Module 3 tests.

The reference path is exercised everywhere.  The Triton path is exercised only
when a CUDA device is present -- which it is not on the development laptop, so
those tests skip locally and run on the GPU box.  A skipped test is not a
passing test, and the CI matrix must include a CUDA runner before the Triton
path is considered verified.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvqpu.kernels.fused_twosite import (
    _fuse_reference,
    active_backend,
    have_triton,
    heff_apply,
    heff_apply_reference,
    heff_matvec_operator,
    peak_bytes_fused,
    peak_bytes_naive,
)
from tvqpu.lattice import InteractingChain

torch = pytest.importorskip("torch", reason="Module 3 needs torch")

requires_gpu = pytest.mark.skipif(
    not have_triton(), reason="no Triton/CUDA device (expected on the laptop)")


# --------------------------------------------------------------------------
# Fixtures: a two-site environment with the real MPO of ARCHITECTURE.md 2.4
# --------------------------------------------------------------------------
def make_case(chi_l=6, chi_r=7, d=2, d_mpo=5, seed=0, dtype=torch.float64,
              device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64).to(
            device=device, dtype=dtype)

    chain = InteractingChain(n_sites=8, t=1.0, delta=0.7, mu=1.1, v_int=0.4)
    mpo = chain.to_mpo(real=(dtype in (torch.float64, torch.float32)))
    w1 = torch.as_tensor(np.ascontiguousarray(mpo.tensors[3])).to(
        device=device, dtype=dtype)
    w2 = torch.as_tensor(np.ascontiguousarray(mpo.tensors[4])).to(
        device=device, dtype=dtype)

    l_env = rnd(chi_l, d_mpo, chi_l)
    r_env = rnd(chi_r, d_mpo, chi_r)
    theta = rnd(chi_l, d, d, chi_r)
    return l_env, w1, w2, r_env, theta


# --------------------------------------------------------------------------
# The reference is itself checked against a single fully-naive einsum
# --------------------------------------------------------------------------
def test_reference_matches_single_shot_einsum():
    """The pinned four-step order must equal the one-shot contraction.  If the
    step ordering is ever 'optimized', this catches an index error."""
    l_env, w1, w2, r_env, theta = make_case()
    got = heff_apply_reference(l_env, w1, w2, r_env, theta)
    want = torch.einsum("amA,mnij,npkl,bpB,AjlB->aikb",
                        l_env, w1, w2, r_env, theta)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_reference_works_on_numpy_too():
    l_env, w1, w2, r_env, theta = make_case()
    got_np = heff_apply_reference(*(x.numpy() for x in
                                    (l_env, w1, w2, r_env, theta)))
    got_t = heff_apply_reference(l_env, w1, w2, r_env, theta)
    assert np.allclose(got_np, got_t.numpy(), rtol=1e-12, atol=1e-12)


def _dense_heff(l_env, w1, w2, r_env, shape, dtype=torch.float64):
    """Materialize H_eff column by column, for small shapes only."""
    dim = int(np.prod(shape))
    mat = torch.zeros(dim, dim, dtype=dtype)
    matvec = heff_matvec_operator(l_env, w1, w2, r_env, shape,
                                  backend="reference")
    for c in range(dim):
        e = torch.zeros(dim, dtype=dtype)
        e[c] = 1.0
        mat[:, c] = matvec(e)
    return mat


def test_heff_reproduces_the_exact_two_site_hamiltonian():
    """The known-answer check that ties Module 3 back to Module 1.

    With chi_l = chi_r = 1 and the MPO boundary vectors as environments, the
    two-site effective Hamiltonian IS the exact Hamiltonian of a 2-site chain,
    which ``InteractingChain.to_dense()`` builds independently by Kronecker
    products.  If the kernel's index conventions are wrong, this fails.
    """
    chain = InteractingChain(n_sites=2, t=1.0, delta=0.7, mu=1.1, v_int=0.4)
    mpo = chain.to_mpo(real=True)
    d_mpo = mpo.tensors[0].shape[0]

    l_env = torch.as_tensor(mpo.v_left, dtype=torch.float64).reshape(1, d_mpo, 1)
    r_env = torch.as_tensor(mpo.v_right, dtype=torch.float64).reshape(1, d_mpo, 1)
    w1 = torch.as_tensor(np.ascontiguousarray(mpo.tensors[0]), dtype=torch.float64)
    w2 = torch.as_tensor(np.ascontiguousarray(mpo.tensors[1]), dtype=torch.float64)

    got = _dense_heff(l_env, w1, w2, r_env, (1, 2, 2, 1))
    want = torch.as_tensor(chain.to_dense().real, dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-12)
    # ...and it is symmetric, as a physical Hamiltonian must be.
    assert torch.allclose(got, got.T, atol=1e-12)


def test_heff_is_hermitian_with_hermitian_environments():
    """H_eff is Hermitian when the MPO operators are Hermitian and the
    environments are.

    GOTCHA, recorded so nobody 'fixes' the kernel over it: this does NOT hold
    term-by-term for the real (Y -> iY) MPO with *arbitrary* environments.
    Y~ = iY is antisymmetric, so a dangling Y~ contracted against a random
    symmetric environment gives an antisymmetric contribution.  In a real DMRG
    sweep the environments are built from the same MPO and those terms cancel
    -- which is what the two-site test above demonstrates.  Random tensors are
    not a physical environment.
    """
    chi_l, chi_r, d = 4, 5, 2
    chain = InteractingChain(n_sites=8, t=1.0, delta=0.7, mu=1.1, v_int=0.4)
    mpo = chain.to_mpo(real=False)  # Hermitian Y, not the real Y~ form
    w1 = torch.as_tensor(np.ascontiguousarray(mpo.tensors[3]))
    w2 = torch.as_tensor(np.ascontiguousarray(mpo.tensors[4]))

    g = torch.Generator().manual_seed(5)
    l_env = torch.randn(chi_l, 5, chi_l, generator=g, dtype=torch.float64)
    r_env = torch.randn(chi_r, 5, chi_r, generator=g, dtype=torch.float64)
    l_env = (0.5 * (l_env + l_env.transpose(0, 2))).to(torch.complex128)
    r_env = (0.5 * (r_env + r_env.transpose(0, 2))).to(torch.complex128)

    mat = _dense_heff(l_env, w1, w2, r_env, (chi_l, d, d, chi_r),
                      dtype=torch.complex128)
    assert torch.allclose(mat, mat.conj().T, atol=1e-11)


def test_matvec_operator_shape_roundtrip():
    chi_l, chi_r, d = 3, 4, 2
    l_env, w1, w2, r_env, theta = make_case(chi_l, chi_r, d, seed=9)
    matvec = heff_matvec_operator(l_env, w1, w2, r_env,
                                  (chi_l, d, d, chi_r), backend="reference")
    out = matvec(theta.reshape(-1))
    assert out.shape == (chi_l * d * d * chi_r,)
    assert torch.allclose(
        out.reshape(chi_l, d, d, chi_r),
        heff_apply_reference(l_env, w1, w2, r_env, theta), atol=1e-12)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
def test_auto_backend_falls_back_without_cuda():
    assert active_backend("auto") in ("reference", "triton")
    assert active_backend("reference") == "reference"


def test_explicit_triton_request_fails_loudly_without_cuda():
    if have_triton():
        pytest.skip("CUDA is present; nothing to assert")
    with pytest.raises(RuntimeError, match="Triton/CUDA is unavailable"):
        active_backend("triton")


def test_env_var_overrides_backend(monkeypatch):
    monkeypatch.setenv("TVQPU_BACKEND", "reference")
    assert active_backend("auto") == "reference"


def test_heff_apply_uses_reference_when_no_gpu():
    if have_triton():
        pytest.skip("CUDA is present")
    l_env, w1, w2, r_env, theta = make_case()
    assert torch.allclose(heff_apply(l_env, w1, w2, r_env, theta),
                          heff_apply_reference(l_env, w1, w2, r_env, theta),
                          atol=1e-12)


# --------------------------------------------------------------------------
# Memory accounting
# --------------------------------------------------------------------------
def test_fusion_removes_one_full_intermediate():
    chi, d, d_mpo = 4096, 2, 5
    naive = peak_bytes_naive(chi, d, d_mpo, np.complex128)
    fused = peak_bytes_fused(chi, d, d_mpo, np.complex128)
    one_big = chi * chi * d * d * d_mpo * 16
    assert naive - fused == one_big
    assert one_big / 1024 ** 3 == pytest.approx(5.0, rel=0.15)  # ~5.4 GB


def test_peak_bytes_scale_as_chi_squared():
    a = peak_bytes_naive(1000)
    b = peak_bytes_naive(2000)
    assert b == pytest.approx(4 * a, rel=1e-9)


# --------------------------------------------------------------------------
# Precision guards
# --------------------------------------------------------------------------
@requires_gpu
def test_float32_is_refused():
    l_env, w1, w2, r_env, theta = make_case(dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="float32 is not supported"):
        heff_apply(l_env, w1, w2, r_env, theta, backend="triton")


@requires_gpu
def test_complex_mpo_is_refused_with_a_useful_message():
    l_env, w1, w2, r_env, theta = make_case(dtype=torch.complex128,
                                            device="cuda")
    with pytest.raises(NotImplementedError, match="real=True"):
        heff_apply(l_env, w1, w2, r_env, theta, backend="triton")


# --------------------------------------------------------------------------
# The tests that actually verify the kernel -- GPU only
# --------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("chi_l,chi_r", [(1, 1), (4, 4), (16, 9), (64, 64),
                                         (127, 33)])
def test_triton_matches_reference(chi_l, chi_r):
    args = make_case(chi_l, chi_r, seed=chi_l + chi_r, device="cuda")
    got = heff_apply(*args, backend="triton")
    want = heff_apply_reference(*args)
    rel = (got - want).abs().max() / want.abs().max()
    assert rel < 1e-12, f"relative error {rel:.3e}"


@requires_gpu
def test_fused_step_matches_its_own_reference():
    """Isolate the kernel: check steps 2+3 alone, not the whole chain."""
    from tvqpu.kernels.fused_twosite import _fused_mpo_pair_triton
    l_env, w1, w2, r_env, theta = make_case(32, 24, device="cuda")
    t1 = torch.einsum("amA,AjlB->majlB", l_env, theta)
    got = _fused_mpo_pair_triton(t1, w1, w2)
    want = _fuse_reference(t1, w1, w2)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-12)


@requires_gpu
def test_triton_handles_non_multiple_of_block():
    """Tail masking: n_batch not divisible by BLOCK must not read past the end."""
    args = make_case(13, 11, seed=2, device="cuda")
    got = heff_apply(*args, backend="triton", block=64)
    assert torch.allclose(got, heff_apply_reference(*args), atol=1e-12)
