"""Sanity checks: the Monte Carlo engine must agree with closed forms."""
import numpy as np

from src.cva import cva
from src.swap import Swap, discount_factors, par_rate, swap_mtm
from src.vasicek import Vasicek

MODEL = Vasicek(a=0.10, b=0.03, sigma=0.01, r0=0.025)
TIMES = np.linspace(0.0, 5.0, 5 * 52 + 1)
RATES = MODEL.simulate(TIMES, 20_000, seed=1)


def test_mc_bond_price_matches_closed_form():
    disc = discount_factors(TIMES, RATES)
    mc = disc[:, -1].mean()
    assert abs(mc - MODEL.zcb(0.0, 5.0, MODEL.r0)) < 5e-4


def test_par_swap_is_worth_zero_today():
    k = par_rate(MODEL, 5.0)
    values = swap_mtm(Swap(1e6, k, 5.0), MODEL, TIMES, RATES)
    assert abs(values[:, 0].mean()) < 1e-6


def test_discounted_swap_value_is_a_martingale():
    # Before the first coupon, E[D(t) V(t)] must stay ~0 for a par swap (no arbitrage)
    k = par_rate(MODEL, 5.0)
    values = swap_mtm(Swap(1e6, k, 5.0), MODEL, TIMES, RATES)
    disc = discount_factors(TIMES, RATES)
    assert np.abs((disc * values).mean(axis=0)[:52]).max() < 2_000  # < 0.2% of notional


def test_cva_is_zero_without_default_risk():
    ee = np.ones_like(TIMES)
    assert cva(TIMES, ee, cds_spread=0.0, recovery=0.4) == 0.0


def test_payer_and_receiver_exposures_mirror():
    k = par_rate(MODEL, 5.0)
    payer = swap_mtm(Swap(1e6, k, 5.0, payer=True), MODEL, TIMES, RATES)
    receiver = swap_mtm(Swap(1e6, k, 5.0, payer=False), MODEL, TIMES, RATES)
    assert np.allclose(payer, -receiver)
