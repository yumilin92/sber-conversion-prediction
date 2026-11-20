import numpy as np
import pandas as pd

from src.monitor import bin_edges, psi, shares


def test_psi_is_zero_for_the_same_distribution():
    values = pd.Series(np.random.default_rng(0).normal(size=5000))
    edges = bin_edges(values)
    assert psi(shares(values, edges), shares(values, edges)) == 0


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=5000))
    edges = bin_edges(ref)
    small = psi(shares(ref, edges), shares(pd.Series(rng.normal(0.1, size=5000)), edges))
    large = psi(shares(ref, edges), shares(pd.Series(rng.normal(1.0, size=5000)), edges))
    assert small < 0.1 < 0.25 < large


def test_missing_values_have_their_own_bin():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0] * 100)
    edges = bin_edges(ref)
    with_missing = pd.Series([1.0, 2.0, None, None] * 100)
    assert shares(with_missing, edges)[-1] == 0.5
