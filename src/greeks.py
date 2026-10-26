"""Interest-rate sensitivities of the CVA: four Monte Carlo estimators compared.

The CVA is a sum over paths, CVA = mean(c), with per-path contribution
    c = sum_k w_k * D(t_k) * max(V(t_k), 0)
(w = cva_weights). We differentiate with respect to the short rate r0.

  1. bump_independent : bump & revalue with independent random numbers
  2. bump_crn         : bump & revalue with common random numbers (same z)
  3. pathwise         : exact derivative of each path, tangent mode (what AAD does)
  4. likelihood_ratio : differentiate the density instead of the payoff

Each returns (estimate, standard error) per unit of r0; multiply by 1e-4 for a
"per basis point" figure.
"""
from dataclasses import dataclass, replace

import numpy as np

from .swap import Swap, discount_factors, discount_factors_delta, swap_mtm
from .vasicek import Vasicek


@dataclass
class Estimate:
    value: float
    std_error: float


def _per_path_cva(model: Vasicek, swap: Swap, times, z, w):
    rates = model.simulate_from_normals(times, z)
    values = swap_mtm(swap, model, times, rates)
    disc = discount_factors(times, rates)
    return (disc * np.maximum(values, 0.0)) @ w


def _stats(x: np.ndarray) -> Estimate:
    return Estimate(float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))))


def delta_bump_independent(model, swap, times, w, n_paths, h=1e-4, seed=0) -> Estimate:
    rng = np.random.default_rng(seed)
    z_up = rng.standard_normal((n_paths, len(times) - 1))
    z_dn = rng.standard_normal((n_paths, len(times) - 1))
    c_up = _per_path_cva(replace(model, r0=model.r0 + h), swap, times, z_up, w)
    c_dn = _per_path_cva(replace(model, r0=model.r0 - h), swap, times, z_dn, w)
    value = (c_up.mean() - c_dn.mean()) / (2 * h)
    se = np.sqrt(c_up.var(ddof=1) / n_paths + c_dn.var(ddof=1) / n_paths) / (2 * h)
    return Estimate(float(value), float(se))


def delta_bump_crn(model, swap, times, w, z, h=1e-4) -> Estimate:
    c_up = _per_path_cva(replace(model, r0=model.r0 + h), swap, times, z, w)
    c_dn = _per_path_cva(replace(model, r0=model.r0 - h), swap, times, z, w)
    return _stats((c_up - c_dn) / (2 * h))


def delta_pathwise(model, swap, times, w, z) -> Estimate:
    """d/dr0 of D * V+ = dD * V+ + D * 1{V > 0} * dV  (indicator: the kink of max)."""
    rates = model.simulate_from_normals(times, z)
    values, dvalues = swap_mtm(swap, model, times, rates, with_delta=True)
    disc = discount_factors(times, rates)
    ddisc = discount_factors_delta(model, times, disc)
    pos = values > 0.0
    dc = (ddisc * np.maximum(values, 0.0) + disc * pos * dvalues) @ w
    return _stats(dc)


def delta_likelihood_ratio(model, swap, times, w, z) -> Estimate:
    """E[c * score], score = d log density / d r0.

    r0 only enters the law of r(t_1): N(mean, std^2) with d mean / d r0 = exp(-a t_1),
    so score = (r(t_1) - mean) / std^2 * exp(-a t_1) = z_1 / std * exp(-a t_1).
    No derivative of the payoff is needed (works for kinks and digitals), but the
    score blows up as the first time step shrinks: here std ~ sigma * sqrt(dt).
    """
    c = _per_path_cva(model, swap, times, z, w)
    _, std = model.first_step_moments(times[1])
    score = z[:, 0] / std * np.exp(-model.a * times[1])
    return _stats(c * score)


def gamma_bump_crn(model, swap, times, w, z, h=1e-4) -> Estimate:
    """Second difference: suffers from the max(V, 0) kink (1/h^2 amplification)."""
    c_up = _per_path_cva(replace(model, r0=model.r0 + h), swap, times, z, w)
    c_0 = _per_path_cva(model, swap, times, z, w)
    c_dn = _per_path_cva(replace(model, r0=model.r0 - h), swap, times, z, w)
    return _stats((c_up - 2 * c_0 + c_dn) / h**2)


def gamma_pathwise_crn(model, swap, times, w, z, h=1e-4) -> Estimate:
    """Mixed method: CRN central difference of the pathwise delta (1/h, not 1/h^2).

    It still meets the kink: the indicator 1{V > 0} flips on the paths that cross
    zero between the two bumps, so its variance also grows as h shrinks. The
    bump size is a bias / variance trade-off; smoothing the max or a likelihood
    ratio term are the standard fixes for a production gamma.
    """
    def per_path_delta(m):
        rates = m.simulate_from_normals(times, z)
        values, dvalues = swap_mtm(swap, m, times, rates, with_delta=True)
        disc = discount_factors(times, rates)
        ddisc = discount_factors_delta(m, times, disc)
        return (ddisc * np.maximum(values, 0.0) + disc * (values > 0) * dvalues) @ w

    d_up = per_path_delta(replace(model, r0=model.r0 + h))
    d_dn = per_path_delta(replace(model, r0=model.r0 - h))
    return _stats((d_up - d_dn) / (2 * h))
