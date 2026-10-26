"""Exposure profiles and the effect of variation margin."""
import numpy as np


def expected_exposure(values: np.ndarray, discount: np.ndarray | None = None) -> np.ndarray:
    """EE(t) = E[max(V(t), 0)], optionally discounted to today."""
    pos = np.maximum(values, 0.0)
    if discount is not None:
        pos = pos * discount
    return pos.mean(axis=0)


def expected_negative_exposure(values: np.ndarray, discount: np.ndarray | None = None) -> np.ndarray:
    """ENE(t) = E[min(V(t), 0)] (what we would owe if WE defaulted)."""
    neg = np.minimum(values, 0.0)
    if discount is not None:
        neg = neg * discount
    return neg.mean(axis=0)


def potential_future_exposure(values: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """PFE(t): high quantile of the positive exposure distribution."""
    return np.quantile(np.maximum(values, 0.0), quantile, axis=0)


def epe(times: np.ndarray, ee: np.ndarray, horizon: float = 1.0) -> float:
    """Expected Positive Exposure: time-average of EE over [0, horizon]."""
    mask = times <= horizon + 1e-12
    t, e = times[mask], ee[mask]
    return float(np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(t)) / (t[-1] - t[0]))


def collateralised_values(values: np.ndarray, mpor_steps: int, threshold: float = 0.0) -> np.ndarray:
    """Exposure net of variation margin, one-way (the counterparty posts).

    Collateral held at t reflects the value observed one margin period of risk
    (MPoR) earlier, minus the threshold: C(t) = max(V(t - MPoR) - H, 0).
    The residual risk is the move of the portfolio during the MPoR.
    """
    lagged = np.empty_like(values)
    lagged[:, :mpor_steps] = 0.0  # no collateral before the first margin call
    lagged[:, mpor_steps:] = values[:, :-mpor_steps]
    collateral = np.maximum(lagged - threshold, 0.0)
    return values - collateral
