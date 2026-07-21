#!/usr/bin/env python
"""Plot the QMC vs Chebyshev benchmark on the 24-site periodic square.

For each coupling sign, produces one figure with the correlations on top and the
pull (difference divided by the combined error bar) underneath, one column per
site, all inverse temperatures overlaid.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_vs_chebyshev import (SITES, SITE_LABEL, cheby_path, load_chebyshev,
                                    load_qmc, _fmt)


def qmc_path(data_dir, beta, J):
    return os.path.join(
        data_dir,
        f"ISO__Square_NN_PBC_N=24__sites=0-1-7__beta={_fmt(beta)}"
        f"__J={_fmt(J)}.hdf5")


def make_figure(betas, J, data_dir, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(betas)))

    any_data = False
    for ib, beta in enumerate(betas):
        qp, rp = qmc_path(data_dir, beta, J), cheby_path(beta, J)
        if not (os.path.exists(qp) and os.path.exists(rp)):
            continue
        any_data = True
        tau_q, qmc = load_qmc(qp)
        tau_r, ref = load_chebyshev(rp)
        mask = tau_q <= tau_r[-1] + 1e-12

        # Plotted against tau/beta so every temperature shares the same
        # horizontal range, [0, 1/2] (Chebyshev stores tau up to beta/2).
        x_q, x_r = tau_q[mask] / beta, tau_r / beta

        for col, site in enumerate(SITES):
            top, bottom = axes[0, col], axes[1, col]
            q, qerr = qmc[site][0][mask], qmc[site][1][mask]
            r, rerr = ref[site]

            top.plot(x_r, r, "-", color=colours[ib], lw=1.4,
                     label=rf"$\beta={beta:g}$" if col == 0 else None)
            top.errorbar(x_q, q, yerr=qerr, fmt="o", ms=3.0,
                         color=colours[ib], mfc="white", mew=0.9, lw=0.8,
                         capsize=1.5)

            # tau = 0 is dropped from the pull: C(0) = 1/4 is an operator
            # identity, both error bars vanish there, and the ratio diverges on
            # a fluctuation of no significance.
            inner = x_q > 0
            ri = np.interp(tau_q[mask], tau_r, r)
            rei = np.interp(tau_q[mask], tau_r, rerr)
            combined = np.sqrt(qerr**2 + rei**2)
            with np.errstate(divide="ignore", invalid="ignore"):
                pull = np.where(combined > 0, (q - ri) / combined, 0.0)
            bottom.plot(x_q[inner], pull[inner], "o-", ms=2.5, lw=0.8,
                        color=colours[ib])

    if not any_data:
        plt.close(fig)
        return None

    for col, site in enumerate(SITES):
        axes[0, col].set_title(f"site {site} — {SITE_LABEL[site]}")
        axes[1, col].set_xlabel(r"$\tau/\beta$")
        axes[1, col].axhspan(-2, 2, color="0.85", zorder=0)
        axes[1, col].axhline(0, color="0.4", lw=0.8, zorder=1)
        axes[1, col].set_ylim(-6, 6)
        axes[1, col].set_xlim(0, 0.5)
    axes[0, 0].set_ylabel(r"$C^{xx}(\tau)$")
    axes[1, 0].set_ylabel("pull")
    axes[0, 0].legend(fontsize=8, ncol=2)

    order = "antiferromagnetic" if J > 0 else "ferromagnetic"
    fig.suptitle(f"24-site periodic square (6x4), $J={J:+g}$ ({order}), "
                 f"$h_z=0$  —  lines: Chebyshev typicality, points: QMC",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return outpath


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", default="0.2,0.5,1,1.5,2,2.5,3")
    parser.add_argument("--couplings", default="0.5,-0.5")
    parser.add_argument("--data-dir", default="Data/benchmark_N24")
    parser.add_argument("--outdir", default="Plots")
    args = parser.parse_args(argv)

    betas = [float(b) for b in args.betas.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    written = []
    for J in [float(j) for j in args.couplings.split(",")]:
        tag = "afm" if J > 0 else "fm"
        path = make_figure(betas, J, args.data_dir,
                           os.path.join(args.outdir, f"benchmark_N24_{tag}.png"))
        if path:
            written.append(path)
            print(f"wrote {path}")
    if not written:
        print("no data found to plot")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
