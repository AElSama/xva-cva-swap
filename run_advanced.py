"""Greeks of the CVA and variance-reduction benchmark. Run: python run_advanced.py

Takes a few minutes: the convergence study runs 5 methods x 6 sample sizes x 16 seeds.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import (AQUA, BLUE, CPTY_RECOVERY, CPTY_SPREAD, GRID, INK, INK2, MATURITY, MODEL,
                 NOTIONAL, ORANGE, STEPS_PER_YEAR, SURFACE, style)
from src import greeks
from src.cva import cs01, cva_weights
from src.swap import Swap, discount_factors, par_rate, swap_mtm
from src.variance_reduction import METHODS, cva_by_method, explained_variance, pca_factor

YELLOW, MAGENTA = "#eda100", "#e87ba4"
N_GREEKS = 32_768
SAMPLE_SIZES = [2**k for k in range(9, 15)]   # 512 ... 16,384 paths
N_SEEDS = 16


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


def make_figures(out):
    sizes = out["rmse_vs_n"]["n_paths"]
    colors = dict(zip(METHODS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for m in METHODS:
        ax.plot(sizes, out["rmse_vs_n"][m], color=colors[m], linewidth=2, marker="o",
                markersize=5, label=m)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    style(ax, "CVA error vs number of paths (RMSE over 16 seeds, log-log)", "RMSE of CVA (EUR)")
    ax.set_xlabel("Number of paths", color=INK2)
    fig.tight_layout(); fig.savefig("figures/convergence.png", dpi=150); plt.close(fig)

    # delta estimators: point estimate with 95% confidence interval
    rows = out["ir_delta"]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    truth = rows[-1]["delta_eur_per_bp"]
    ax.axvline(truth, color=GRID, linewidth=1.5, zorder=0)
    for i, r in enumerate(rows):
        ax.errorbar(r["delta_eur_per_bp"], i, xerr=1.96 * r["std_error"], fmt="o", color=BLUE,
                    ecolor=BLUE, elinewidth=2, capsize=4, markersize=7)
        ax.annotate(f"{r['delta_eur_per_bp']:.1f} \u00b1 {1.96 * r['std_error']:.1f}",
                    (r["delta_eur_per_bp"] + 1.96 * r["std_error"], i), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r["method"] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(right=max(r["delta_eur_per_bp"] + 1.96 * r["std_error"] for r in rows) + 45)
    style(ax, "CVA rate delta: same paths budget, four estimators (95% CI)", "")
    ax.set_xlabel("Delta (EUR per bp of r0)", color=INK2)
    ax.grid(True, axis="x", color=GRID); ax.grid(False, axis="y")
    fig.tight_layout(); fig.savefig("figures/delta_estimators.png", dpi=150); plt.close(fig)


def main():
    times = np.linspace(0.0, MATURITY, int(MATURITY * STEPS_PER_YEAR) + 1)
    swap = Swap(NOTIONAL, par_rate(MODEL, MATURITY), MATURITY)
    w = cva_weights(times, CPTY_SPREAD, CPTY_RECOVERY)
    z = np.random.default_rng(7).standard_normal((N_GREEKS, len(times) - 1))
    out = {}

    # ------------------------------------------------ 1. IR delta, four estimators
    bp = 1e-4
    estimators = {
        "Bump, independent seeds": lambda: greeks.delta_bump_independent(MODEL, swap, times, w, N_GREEKS),
        "Likelihood ratio": lambda: greeks.delta_likelihood_ratio(MODEL, swap, times, w, z),
        "Bump, common random numbers": lambda: greeks.delta_bump_crn(MODEL, swap, times, w, z),
        "Pathwise (tangent / AAD-style)": lambda: greeks.delta_pathwise(MODEL, swap, times, w, z),
    }
    delta_rows = []
    for name, f in estimators.items():
        est, secs = timed(f)
        delta_rows.append({"method": name, "delta_eur_per_bp": est.value * bp,
                           "std_error": est.std_error * bp, "seconds": secs})
    ref_var_time = delta_rows[-1]["std_error"] ** 2 * delta_rows[-1]["seconds"]
    for row in delta_rows:  # efficiency = 1 / (variance x time), relative to pathwise
        row["relative_efficiency"] = ref_var_time / (row["std_error"] ** 2 * row["seconds"])
    out["ir_delta"] = delta_rows

    # ------------------------------------------------ 2. CS01 (exact)
    rates = MODEL.simulate_from_normals(times, z)
    values = swap_mtm(swap, MODEL, times, rates)
    ee_disc = (discount_factors(times, rates) * np.maximum(values, 0.0)).mean(axis=0)
    out["cs01_eur_per_bp"] = cs01(times, ee_disc, CPTY_SPREAD, CPTY_RECOVERY)

    # ------------------------------------------------ 3. Gamma vs bump size
    gamma_rows = []
    for h in (1e-4, 1e-5, 1e-6):
        g_fd = greeks.gamma_bump_crn(MODEL, swap, times, w, z, h)
        g_mx = greeks.gamma_pathwise_crn(MODEL, swap, times, w, z, h)
        gamma_rows.append({"bump_bp": h / bp,
                           "second_difference": [g_fd.value * bp * bp, g_fd.std_error * bp * bp],
                           "pathwise_then_bump": [g_mx.value * bp * bp, g_mx.std_error * bp * bp]})
    out["gamma_eur_per_bp2"] = gamma_rows

    # ------------------------------------------------ 4. Convergence benchmark
    factor = pca_factor(MODEL, times)
    out["pca_explained_variance"] = {"1_factor": explained_variance(factor, 1),
                                     "3_factors": explained_variance(factor, 3)}
    ref = np.mean([cva_by_method("Sobol + PCA + CV", MODEL, swap, times, w, 2**15, 1000 + s, factor)
                   for s in range(16)])
    out["cva_reference"] = ref
    rmse = {m: [] for m in METHODS}
    for m in METHODS:
        for n in SAMPLE_SIZES:
            est = [cva_by_method(m, MODEL, swap, times, w, n, s, factor) for s in range(N_SEEDS)]
            rmse[m].append(float(np.sqrt(np.mean((np.array(est) - ref) ** 2))))
        print(f"{m:>18}: " + "  ".join(f"{x:7.1f}" for x in rmse[m]))
    out["rmse_vs_n"] = {"n_paths": SAMPLE_SIZES, **rmse}
    Path("results_advanced.json").write_text(json.dumps(out, indent=2))

    make_figures(out)
    print(json.dumps({k: v for k, v in out.items() if k != "rmse_vs_n"}, indent=2))


if __name__ == "__main__":
    import sys
    if "--plots-only" in sys.argv:
        make_figures(json.loads(Path("results_advanced.json").read_text()))
    else:
        main()
