"""Unilateral CVA and DVA with a flat hazard rate (credit triangle)."""
import numpy as np


def hazard_from_spread(cds_spread: float, recovery: float) -> float:
    """Credit triangle: lambda = s / (1 - R)."""
    return cds_spread / (1.0 - recovery)


def survival(times: np.ndarray, hazard: float) -> np.ndarray:
    return np.exp(-hazard * times)


def cva_weights(times: np.ndarray, cds_spread: float, recovery: float) -> np.ndarray:
    """Weights w_k such that CVA = sum_k w_k * EE*(t_k).

    CVA = (1 - R) * sum_i EE*(t_i) * [S(t_{i-1}) - S(t_i)]   (exposure at period end).
    The right-endpoint rule gives t = 0 a zero weight: at inception a par swap
    has V(0) = 0 exactly, and max(V(0), 0) would put an artificial kink in
    CVA(r0) that breaks Greeks. With a weekly grid the rule choice is negligible.
    Being linear in EE*, the same weights give a per-path CVA contribution:
    c = (D * V+) @ w, whose mean is the CVA.
    """
    s = survival(times, hazard_from_spread(cds_spread, recovery))
    w = np.zeros(len(times))
    w[1:] = s[:-1] - s[1:]
    return (1.0 - recovery) * w


def cva(times: np.ndarray, discounted_ee: np.ndarray, cds_spread: float, recovery: float) -> float:
    """CVA from a discounted expected exposure profile. Positive = cost to us."""
    return float(cva_weights(times, cds_spread, recovery) @ discounted_ee)


def dva(times: np.ndarray, discounted_ene: np.ndarray, own_spread: float, own_recovery: float) -> float:
    """DVA: same formula on our own negative exposure. Positive = benefit to us."""
    return cva(times, -discounted_ene, own_spread, own_recovery)


def cs01(times: np.ndarray, discounted_ee: np.ndarray, cds_spread: float, recovery: float) -> float:
    """CVA change for +1bp of counterparty CDS spread, exact (no bump, no noise).

    Exposures do not depend on the spread here (no wrong-way risk), so only the
    default probabilities move: d/ds [S(t_{i-1}) - S(t_i)] with S = exp(-s t / (1-R)).
    """
    lam = hazard_from_spread(cds_spread, recovery)
    s = survival(times, lam)
    dw = np.zeros(len(times))
    dw[1:] = -times[:-1] * s[:-1] + times[1:] * s[1:]
    # the (1 - R) of the CVA formula cancels with dlambda/ds = 1 / (1 - R)
    return float(1e-4 * dw @ discounted_ee)
