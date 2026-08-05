"""Multi-restart DMRG tests.

The restart machinery exists because of a measured failure, not on principle:
paired chi=96 vs chi=256 runs on identical disorder realizations disagreed by
up to 0.38 in the order parameter at bond dimensions nowhere near either
ceiling (ARCHITECTURE.md 2.5b).  These tests pin the behaviour that fixes it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("quimb", reason="Module 2 needs quimb")

from tvqpu.dmrg import ground_state  # noqa: E402
from tvqpu.lattice import InteractingChain  # noqa: E402


def test_single_restart_preserves_previous_behaviour():
    """n_restarts=1 must take the original unseeded code path, so ledgers
    produced before the restart feature existed remain comparable."""
    r = ground_state(InteractingChain(n_sites=24, mu=1.0), chi=48)
    assert r.n_restarts == 1
    assert r.restart_energies == (r.energy,)
    assert r.energy_spread == 0.0
    # A single restart CANNOT detect multistability -- absence of evidence.
    assert r.multistable is False


def test_multi_restart_is_bit_reproducible():
    """Without this, a multistability finding is anecdote rather than data."""
    kw = dict(chi=48, n_restarts=3, restart_seed=11)
    a = ground_state(InteractingChain(n_sites=24, mu=1.0), **kw)
    b = ground_state(InteractingChain(n_sites=24, mu=1.0), **kw)
    assert a.restart_energies == b.restart_energies
    assert a.restart_order_parameters == b.restart_order_parameters


def test_returns_the_lowest_energy_restart():
    """Energy is the arbiter -- that is the whole point.  The returned
    order_parameter must belong to the minimum-energy solution, not to an
    average or to whichever ran last."""
    r = ground_state(InteractingChain(n_sites=24, mu=1.0), chi=48,
                     n_restarts=3, restart_seed=5)
    assert r.energy == min(r.restart_energies)
    assert r.energy == pytest.approx(r.restart_energies[0], abs=1e-14)
    assert r.order_parameter == r.restart_order_parameters[0]
    assert len(r.restart_energies) == 3 == len(r.restart_order_parameters)


def test_restart_energies_are_sorted_ascending():
    r = ground_state(InteractingChain(n_sites=24, mu=1.0), chi=48,
                     n_restarts=4, restart_seed=3)
    assert list(r.restart_energies) == sorted(r.restart_energies)


def test_clean_chain_is_not_multistable():
    """A clean gapped chain has one well-separated ground state; restarts must
    all find it.  If this fails, the restart machinery is injecting noise
    rather than exposing it."""
    r = ground_state(InteractingChain(n_sites=32, mu=1.0), chi=64,
                     n_restarts=3, restart_seed=1)
    assert r.energy_spread < 1e-8, r.restart_energies
    assert not r.multistable
    assert r.order_parameter_spread < 1e-6


@pytest.mark.slow
def test_strong_disorder_is_detectably_multistable():
    """The regime the fix was built for.

    At L=80, W=8, seed=12 the chi=96 and chi=256 single-shot runs disagreed by
    0.19 in the order parameter.  Restarts reproduce that spread WITHIN a
    single chi, which is what attributes it to the optimizer landscape rather
    than to truncation.
    """
    chain = InteractingChain(n_sites=80, mu=1.0).with_disorder(w=8.0, seed=12)
    r = ground_state(chain, chi=96, n_restarts=4, restart_seed=1)
    assert r.multistable, r.restart_energies
    assert r.order_parameter_spread > 0.05, r.restart_order_parameters
    # ...and trustworthy must reflect it
    assert not r.trustworthy


def test_invalid_restart_count_rejected():
    with pytest.raises(ValueError, match="n_restarts must be >= 1"):
        ground_state(InteractingChain(n_sites=8), n_restarts=0)


def test_trustworthy_requires_all_three_conditions():
    """converged AND not truncation-limited AND not multistable."""
    r = ground_state(InteractingChain(n_sites=24, mu=1.0), chi=48,
                     n_restarts=2, restart_seed=1)
    assert r.trustworthy == (r.converged and not r.bond_saturated
                             and not r.multistable)
