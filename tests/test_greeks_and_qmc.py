"""Greeks and variance-reduction checks against bumps and closed forms."""
import numpy as np

from src import greeks
from src.cva import cs01, cva, cva_weights
from src.swap import Swap, discount_factors, expected_discounted_value, par_rate, swap_mtm
from src.variance_reduction import cva_by_method, pca_factor
from src.vasicek import Vasicek

MODEL = Vasicek(a=0.10, b=0.03, sigma=0.01, r0=0.025)
TIMES = np.linspace(0.0, 5.0, 5 * 52 + 1)
SWAP = Swap(1e7, par_rate(MODEL, 5.0), 5.0)
W = cva_weights(TIMES, 0.015, 0.4)
Z = np.random.default_rng(11).standard_normal((4_000, len(TIMES) - 1))


def test_pathwise_delta_matches_crn_bump():
    pw = greeks.delta_pathwise(MODEL, SWAP, TIMES, W, Z)
    fd = greeks.delta_bump_crn(MODEL, SWAP, TIMES, W, Z, h=1e-6)
    assert abs(pw.value - fd.value) < 1e-3 * abs(pw.value)


def test_likelihood_ratio_is_unbiased():
    pw = greeks.delta_pathwise(MODEL, SWAP, TIMES, W, Z)
    lr = greeks.delta_likelihood_ratio(MODEL, SWAP, TIMES, W, Z)
    assert abs(lr.value - pw.value) < 4 * lr.std_error
    assert lr.std_error > 5 * pw.std_error  # ... but much noisier


def test_analytic_cs01_matches_bump():
    ee = np.linspace(0, 1e5, len(TIMES))
    bump = (cva(TIMES, ee, 0.0151, 0.4) - cva(TIMES, ee, 0.0149, 0.4)) / 2
    assert abs(cs01(TIMES, ee, 0.015, 0.4) - bump) < 1e-6 * abs(bump)


def test_control_variate_mean_matches_simulation():
    z = np.concatenate([Z, -Z])
    rates = MODEL.simulate_from_normals(TIMES, z)
    x = discount_factors(TIMES, rates) * swap_mtm(SWAP, MODEL, TIMES, rates)
    mu = expected_discounted_value(SWAP, MODEL, TIMES)
    assert np.abs(x.mean(axis=0) - mu).max() < 2e-4 * SWAP.notional


def test_pca_factor_reproduces_covariance():
    f = pca_factor(MODEL, TIMES)
    t = TIMES[1:]
    i, j = 50, 200  # s = t_i <= t_j
    expected = (MODEL.sigma**2 / (2 * MODEL.a) * np.exp(-MODEL.a * (t[j] - t[i]))
                * (1 - np.exp(-2 * MODEL.a * t[i])))
    assert np.isclose((f @ f.T)[i, j], expected, rtol=1e-8)


def test_sobol_pca_agrees_with_plain_mc():
    qmc = cva_by_method("Sobol + PCA", MODEL, SWAP, TIMES, W, 4096, seed=0)
    mc = np.mean([cva_by_method("Plain MC", MODEL, SWAP, TIMES, W, 4096, seed=s) for s in range(4)])
    assert abs(qmc - mc) < 0.03 * qmc
