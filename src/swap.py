"""Mark-to-market of a plain-vanilla interest rate swap along simulated paths."""
from dataclasses import dataclass

import numpy as np

from .vasicek import Vasicek


@dataclass(frozen=True)
class Swap:
    notional: float
    fixed_rate: float
    maturity: float      # in years
    period: float = 1.0  # payment frequency of both legs, in years
    payer: bool = True   # True = we pay fixed and receive floating

    @property
    def pay_dates(self) -> np.ndarray:
        n = int(round(self.maturity / self.period))
        return self.period * np.arange(1, n + 1)


def par_rate(model: Vasicek, maturity: float, period: float = 1.0) -> float:
    """Fixed rate that makes the swap worth zero today."""
    dates = period * np.arange(1, int(round(maturity / period)) + 1)
    discount = model.zcb(0.0, dates, model.r0)
    return (1.0 - discount[-1]) / (period * discount.sum())


def swap_mtm(swap: Swap, model: Vasicek, times: np.ndarray, rates: np.ndarray,
             with_delta: bool = False):
    """Value of the swap at every grid time and on every path.

    Convention: a coupon paid at T_j is no longer part of the value at T_j
    (exposure is measured just after payment). The floating coupon for the
    period (T_{j-1}, T_j] is fixed at T_{j-1} from the simulated curve, so
    between two dates the floating leg is worth
        N * [ (1 + L * delta) * P(t, T_j) - P(t, T_n) ],   1 + L*delta = 1 / P(T_{j-1}, T_j).
    Returns an array of shape (n_paths, len(times)).

    with_delta=True also returns the pathwise derivative dV(t)/dr0, computed
    exactly (tangent mode, as AAD would) using dr(t)/dr0 = exp(-a t) in Vasicek
    and dP(t,T)/dr = -B(T - t) * P(t,T).
    """
    dates = swap.pay_dates
    delta = swap.period
    n_paths = rates.shape[0]
    sign = swap.notional * (1.0 if swap.payer else -1.0)
    values = np.zeros((n_paths, len(times)))
    dvalues = np.zeros_like(values) if with_delta else None
    inv_p = d_inv_p = None
    next_idx = -1

    for k, t in enumerate(times):
        # first payment date strictly after t
        j = int(np.searchsorted(dates, t, side="right"))
        if j >= len(dates):
            break  # swap has matured: value stays 0
        if j != next_idx:
            # a new period starts at T_{j-1} (or at 0): fix the floating rate now
            start = dates[j - 1] if j > 0 else 0.0
            inv_p = 1.0 / model.zcb(start, dates[j], rates[:, k])
            d_inv_p = model.B(dates[j] - start) * np.exp(-model.a * start) * inv_p
            next_idx = j

        r = rates[:, k]
        p_rem = np.array([model.zcb(t, T, r) for T in dates[j:]])  # remaining discount bonds
        floating = inv_p * p_rem[0] - p_rem[-1]
        fixed = swap.fixed_rate * delta * p_rem.sum(axis=0)
        values[:, k] = sign * (floating - fixed)

        if with_delta:
            g = np.exp(-model.a * t)                      # dr(t)/dr0
            b_rem = model.B(dates[j:] - t)[:, None]       # B(T_i - t)
            dp_rem = -b_rem * p_rem * g                   # dP(t,T_i)/dr0
            d_float = d_inv_p * p_rem[0] + inv_p * dp_rem[0] - dp_rem[-1]
            d_fixed = swap.fixed_rate * delta * dp_rem.sum(axis=0)
            dvalues[:, k] = sign * (d_float - d_fixed)

    return (values, dvalues) if with_delta else values


def discount_factors(times: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Pathwise discount factor exp(-integral of r) using the trapezoid rule."""
    dt = np.diff(times)
    integral = np.concatenate(
        [np.zeros((rates.shape[0], 1)),
         np.cumsum(0.5 * (rates[:, 1:] + rates[:, :-1]) * dt, axis=1)],
        axis=1,
    )
    return np.exp(-integral)


def discount_factors_delta(model: Vasicek, times: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """Exact derivative of the trapezoid discount factor with respect to r0.

    D(t) = exp(-sum trapezoid(r)), dr(s)/dr0 = exp(-a s), so
    dD/dr0 = -D(t) * trapezoid integral of exp(-a s) over [0, t].
    """
    g = np.exp(-model.a * times)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(times))])
    return -disc * integral


def expected_discounted_value(swap: Swap, model: Vasicek, times: np.ndarray) -> np.ndarray:
    """Closed form of E[D(t) V(t)]: today's value of the cash flows still alive at t.

    The floating coupon fixed at T_{j-1} is worth P(0, T_{j-1}) today (it is a
    par floater at its fixing date). Used as the control variate's known mean.
    """
    dates = swap.pay_dates
    sign = swap.notional * (1.0 if swap.payer else -1.0)
    p0 = lambda T: model.zcb(0.0, T, model.r0)
    out = np.zeros(len(times))
    for k, t in enumerate(times):
        j = int(np.searchsorted(dates, t, side="right"))
        if j >= len(dates):
            break
        start = dates[j - 1] if j > 0 else 0.0
        fixed = swap.fixed_rate * swap.period * sum(p0(T) for T in dates[j:])
        out[k] = sign * (p0(start) - p0(dates[-1]) - fixed)
    return out
