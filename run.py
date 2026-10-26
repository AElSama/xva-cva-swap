"""End-to-end CVA / DVA of a 5-year payer swap. Run: python run.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cva import cva, dva
from src.exposure import (collateralised_values, epe, expected_exposure,
                          expected_negative_exposure, potential_future_exposure)
from src.swap import Swap, discount_factors, par_rate, swap_mtm
from src.vasicek import Vasicek

# ---------------------------------------------------------------- parameters
MODEL = Vasicek(a=0.10, b=0.03, sigma=0.01, r0=0.025)
NOTIONAL = 10_000_000
MATURITY = 5.0
STEPS_PER_YEAR = 52           # weekly grid
N_PATHS = 20_000
CPTY_SPREAD, CPTY_RECOVERY = 0.0150, 0.40   # counterparty: 150 bp CDS
OWN_SPREAD, OWN_RECOVERY = 0.0100, 0.40     # our bank: 100 bp CDS
MPOR_WEEKS = 2                              # margin period of risk (10 business days)
THRESHOLD = 0.0

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel("Years", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def main():
    times = np.linspace(0.0, MATURITY, int(MATURITY * STEPS_PER_YEAR) + 1)
    rates = MODEL.simulate(times, N_PATHS)
    disc = discount_factors(times, rates)

    k = par_rate(MODEL, MATURITY)
    swap = Swap(NOTIONAL, k, MATURITY)
    values = swap_mtm(swap, MODEL, times, rates)

    ee = expected_exposure(values)
    ee_disc = expected_exposure(values, disc)
    ene_disc = expected_negative_exposure(values, disc)
    pfe = potential_future_exposure(values, 0.95)

    coll = collateralised_values(values, MPOR_WEEKS, THRESHOLD)
    ee_coll_disc = expected_exposure(coll, disc)

    results = {
        "par_rate_pct": round(100 * k, 4),
        "mc_value_t0_check": round(float(values[:, 0].mean()), 2),
        "peak_EE": round(float(ee.max()), 0),
        "peak_PFE_95": round(float(pfe.max()), 0),
        "EPE_1y": round(epe(times, ee), 0),
        "CVA_uncollateralised": round(cva(times, ee_disc, CPTY_SPREAD, CPTY_RECOVERY), 0),
        "DVA": round(dva(times, ene_disc, OWN_SPREAD, OWN_RECOVERY), 0),
        "CVA_collateralised": round(cva(times, ee_coll_disc, CPTY_SPREAD, CPTY_RECOVERY), 0),
    }
    results["BCVA"] = results["CVA_uncollateralised"] - results["DVA"]
    results["CVA_bp_of_notional"] = round(1e4 * results["CVA_uncollateralised"] / NOTIONAL, 2)

    Path("figures").mkdir(exist_ok=True)
    Path("results.json").write_text(json.dumps(results, indent=2))

    # 1. sample short-rate paths
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    for i in range(30):
        ax.plot(times, 100 * rates[i], color=BLUE, alpha=0.25, linewidth=1)
    ax.plot(times, 100 * rates.mean(axis=0), color=INK, linewidth=2)
    ax.annotate("Mean of all paths", (times[-1], 100 * rates.mean(axis=0)[-1]),
                xytext=(-110, 12), textcoords="offset points", color=INK)
    style(ax, "Vasicek short rate: 30 simulated paths", "Short rate (%)")
    fig.tight_layout(); fig.savefig("figures/rate_paths.png", dpi=150); plt.close(fig)

    # 2. exposure profile
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.plot(times, pfe / 1e3, color=ORANGE, linewidth=2, label="PFE 95%")
    ax.plot(times, ee / 1e3, color=BLUE, linewidth=2, label="Expected exposure (EE)")
    ax.plot(times, -expected_negative_exposure(values) / 1e3, color=AQUA, linewidth=2,
            linestyle="--", label="Expected negative exposure (|ENE|)")
    ax.legend(frameon=False, labelcolor=INK)
    style(ax, "5y payer swap, 10M notional: exposure profile", "Exposure (k EUR)")
    fig.tight_layout(); fig.savefig("figures/exposure_profile.png", dpi=150); plt.close(fig)

    # 3. collateral effect
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.plot(times, ee / 1e3, color=BLUE, linewidth=2, label="Uncollateralised EE")
    ax.plot(times, expected_exposure(coll) / 1e3, color=ORANGE, linewidth=2,
            label=f"Collateralised EE (MPoR {MPOR_WEEKS} weeks)")
    ax.legend(frameon=False, labelcolor=INK)
    style(ax, "Variation margin cuts exposure to the MPoR move", "Exposure (k EUR)")
    fig.tight_layout(); fig.savefig("figures/collateral_effect.png", dpi=150); plt.close(fig)

    for key, val in results.items():
        print(f"{key:>24}: {val:,}")


if __name__ == "__main__":
    main()
