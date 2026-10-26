"""Variance reduction for the CVA: control variate and Sobol QMC with PCA paths."""
import numpy as np
from scipy.stats import norm, qmc

from .swap import Swap, discount_factors, expected_discounted_value, swap_mtm
from .vasicek import Vasicek


# ------------------------------------------------------------ random inputs
def normals_mc(n_paths: int, dim: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_paths, dim))


def normals_antithetic(n_paths: int, dim: int, seed: int) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal((n_paths // 2, dim))
    return np.concatenate([z, -z])


def normals_sobol(n_paths: int, dim: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points mapped to N(0,1). n_paths should be a power of 2.

    Scrambling (Owen) keeps the low discrepancy but makes each replication an
    unbiased random estimate, so we can still compute error bars.
    """
    u = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n_paths)
    return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


# ------------------------------------------------------------ path construction
def pca_factor(model: Vasicek, times: np.ndarray) -> np.ndarray:
    """Matrix L with Cov(r(t_1..t_n)) = L L^T, columns sorted by explained variance.

    For Sobol, the first coordinates are the best distributed, so they must drive
    the directions that carry most of the variance. The step-by-step recursion
    wastes them on the first week; PCA puts them on the level / slope of the path.
    Vasicek covariance: Cov(r_s, r_t) = sigma^2/(2a) * exp(-a(t-s)) * (1 - exp(-2a s)), s <= t.
    """
    t = times[1:]
    s, u = np.minimum.outer(t, t), np.abs(np.subtract.outer(t, t))
    cov = model.sigma**2 / (2 * model.a) * np.exp(-model.a * u) * (1 - np.exp(-2 * model.a * s))
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    return eigvec[:, order] * np.sqrt(np.maximum(eigval[order], 0.0))


def rates_pca(model: Vasicek, times: np.ndarray, z: np.ndarray, factor: np.ndarray) -> np.ndarray:
    t = times[1:]
    mean = model.r0 * np.exp(-model.a * t) + model.b * (1 - np.exp(-model.a * t))
    rates = np.empty((z.shape[0], len(times)))
    rates[:, 0] = model.r0
    rates[:, 1:] = mean + z @ factor.T
    return rates


def explained_variance(factor: np.ndarray, k: int) -> float:
    var = (factor**2).sum(axis=0)
    return float(var[:k].sum() / var.sum())


# ------------------------------------------------------------ estimators
def cva_estimate(model: Vasicek, swap: Swap, times: np.ndarray, w: np.ndarray,
                 rates: np.ndarray, control_variate: bool = False) -> float:
    """CVA from simulated paths, optionally with a control variate per date.

    Control: X(t) = D(t) V(t), whose mean is known in closed form
    (expected_discounted_value). It is strongly correlated with
    Y(t) = D(t) V(t)+, so  Y_cv = mean(Y) - beta * (mean(X) - E[X])
    with beta = Cov(Y, X) / Var(X) estimated on the same paths.
    """
    values = swap_mtm(swap, model, times, rates)
    disc = discount_factors(times, rates)
    y = disc * np.maximum(values, 0.0)
    ee = y.mean(axis=0)
    if control_variate:
        x = disc * values
        mu = expected_discounted_value(swap, model, times)
        xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
        var_x = (xc**2).mean(axis=0)
        beta = np.divide((xc * yc).mean(axis=0), var_x, out=np.zeros_like(var_x), where=var_x > 0)
        ee = ee - beta * (x.mean(axis=0) - mu)
    return float(w @ ee)


METHODS = ("Plain MC", "Antithetic", "Control variate", "Sobol + PCA", "Sobol + PCA + CV")


def cva_by_method(method: str, model, swap, times, w, n_paths: int, seed: int,
                  factor: np.ndarray | None = None) -> float:
    dim = len(times) - 1
    if method == "Plain MC":
        rates = model.simulate_from_normals(times, normals_mc(n_paths, dim, seed))
        return cva_estimate(model, swap, times, w, rates)
    if method == "Antithetic":
        rates = model.simulate_from_normals(times, normals_antithetic(n_paths, dim, seed))
        return cva_estimate(model, swap, times, w, rates)
    if method == "Control variate":
        rates = model.simulate_from_normals(times, normals_mc(n_paths, dim, seed))
        return cva_estimate(model, swap, times, w, rates, control_variate=True)
    factor = pca_factor(model, times) if factor is None else factor
    rates = rates_pca(model, times, normals_sobol(n_paths, dim, seed), factor)
    return cva_estimate(model, swap, times, w, rates, control_variate=method.endswith("CV"))
