#!/usr/bin/env python
"""Compare the local (on-site) correlation from QMC against spinDMFT.

QMC data is the N = 8000 cube (20x20x20) in Data/AFM_FM, both coupling signs;
the reference is the spinDMFT solution in Data/spinDMFT. Both use the same
normalisation, C(0) = 1/4, and the same coupling convention -- spinDMFT's
JQ = 1 corresponds to J = 1/sqrt(z) = 0.408248 on the z = 6 cube.

Top row: the correlations themselves, one panel per inverse temperature.
Bottom row: QMC minus spinDMFT, with the QMC error band around zero.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

QMC_DIR = "Data/AFM_FM"
DMFT_DIR = "Data/spinDMFT"
SITES = "0-1-21-421-2"          # shells 0..4 on the 20^3 cube
ROWS = ["xx", "xy", "yx", "zz"]

# Same palette and typography as plot_qmc_vs_spindmft_field.py, so the
# zero-field and finite-field figures can sit side by side.  Each series also
# carries a distinct mark, so identity never rests on colour alone.
C_DMFT, C_AFM, C_FM = "#51cf66", "#e81919", "#364fc7"

plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "DejaVu Serif",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

LEGEND_BOX = dict(frameon=True, facecolor="white", framealpha=0.93,
                  edgecolor="0.65", fancybox=True, borderpad=0.7,
                  labelspacing=0.6)
N_MARKS = 9


def _fmt(x):
    return f"{x:g}"


def load_qmc(beta, J):
    """Local correlation on the closed [0, beta]: (tau, C^xx, err, C^zz).

    The sampled grid is the half-open [0, beta) that DSQSS writes. The value
    at tau = beta is not an extrapolation: C is beta-periodic in imaginary
    time, so C(beta) = C(0) exactly, and the same holds for its error bar.
    Closing the interval lets the QMC curves meet spinDMFT, which is stored
    on the closed grid, at both ends.
    """
    path = os.path.join(
        QMC_DIR, f"ISO__Cube_NN_PBC_N=8000__sites={SITES}"
                 f"__beta={_fmt(beta)}__J={_fmt(J)}.hdf5")
    with h5py.File(path) as h:
        re = np.asarray(h["results/Re_correlation/0-0"])
        er = np.asarray(h["results/Re_stds/0-0"])
        ntau = int(h["parameters"].attrs["num_TimePoints"])
    tau = np.arange(ntau) * beta / ntau
    ixx, izz = ROWS.index("xx"), ROWS.index("zz")

    wrap = lambda a: np.append(a, a[0])
    return (np.append(tau, beta), wrap(re[ixx]), wrap(er[ixx]), wrap(re[izz]))


def load_dmft(beta):
    """Local correlation on the closed interval [0, beta]: (tau, C, err)."""
    path = os.path.join(DMFT_DIR, f"spinmodel=ISO__beta={_fmt(beta)}.hdf5")
    with h5py.File(path) as h:
        c = np.asarray(h["results/Re_correlation"])[0]
        e = np.asarray(h["runtimedata/Re_correlation_sample_stds"])[0]
    return np.linspace(0.0, beta, c.size), c, e


BETAS = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MARKERS = ["o", "s", "^", "D", "v", "P", "*"]


def _ms(mark):
    """The star needs more area than the others to read at the same weight."""
    return 9.0 if mark == "*" else 6.0


def _every(n, phase):
    """markevery for n points, shifted by `phase` of one marker spacing.

    QMC and spinDMFT sit on different grids (101 vs 200 points), so the offset
    is a fraction of the curve rather than an index count.
    """
    return (int((0.04 + phase / N_MARKS) * n), max(n // N_MARKS, 1))


def make_overlay(betas, out):
    """Everything on one axis: colour = method, dash = QMC, marker = beta."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(15.0, 6.3))

    series = [("spinDMFT", None, C_DMFT, "-"),
              ("QMC, AFM", 0.408248, C_AFM, "--"),
              ("QMC, FM", -0.408248, C_FM, "--")]

    for ib, beta in enumerate(betas):
        mark = MARKERS[ib]
        for k, (_, J, colour, dash) in enumerate(series):
            if J is None:
                tau, c, _ = load_dmft(beta)
            else:
                tau, c, _, _ = load_qmc(beta, J)
            x = tau / beta
            # QMC sits above spinDMFT at every temperature, so the dashed
            # curves are never hidden by a solid one from a colder panel.
            # The reference is drawn as a broad band underneath them.
            z, lw = (2, 3.4) if J is None else (3, 1.6)
            # Markers staggered per series so the three do not coincide.
            ax.plot(x, c, dash, color=colour, lw=lw, marker=mark, ms=_ms(mark),
                    markevery=_every(x.size, k / len(series)), mfc="white",
                    mec=colour, mew=1.6 if J is None else 1.3, zorder=z)

    ax.set_xlabel(r"$\tau/\beta$")
    ax.set_ylabel(r"$\mathrm{Re}\,g^{xx}(\tau)$")
    ax.set_xlim(0, 1)
    # No manufactured headroom: the legend plaquettes are opaque, so they can
    # sit over the curves and the axis can close on the data.
    ax.margins(y=0.02)
    ax.grid(alpha=0.25, lw=0.6)

    method_keys = [Line2D([], [], color=c, ls=d, marker="o", ms=6.0,
                          mfc="white", mec=c, lw=3.4 if J is None else 1.8,
                          mew=1.6 if J is None else 1.3, label=n)
                   for n, J, c, d in series]
    # spinDMFT sets J_Q = 1, so beta*J_Q is numerically beta.
    beta_keys = [Line2D([], [], color="0.35", ls="none", marker=m, ms=_ms(m),
                        mfc="white", mew=1.3, label=rf"$\beta J_Q={b:g}$")
                 for b, m in zip(betas, MARKERS)]

    first = ax.legend(handles=method_keys, loc="lower left", **LEGEND_BOX)
    ax.legend(handles=beta_keys, loc="lower right", ncol=2,
              columnspacing=1.2, handletextpad=0.5, **LEGEND_BOX)
    ax.add_artist(first)

    fig.subplots_adjust(left=0.082, right=0.990, top=0.985, bottom=0.132)
    fig.savefig(out, dpi=160)
    print("wrote", out)


def main():
    betas = BETAS

    fig, axes = plt.subplots(2, len(betas), figsize=(3.1 * len(betas), 7.6),
                             sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [3, 1.5]})

    for col, beta in enumerate(betas):
        top, bot = axes[0, col], axes[1, col]
        tau_d, c_d, e_d = load_dmft(beta)

        top.plot(tau_d / beta, c_d, "-", color=C_DMFT, lw=2.6, zorder=3,
                 label=r"$\mathrm{spinDMFT}$" if col == 0 else None)

        for J, colour, mark, name in (
                (0.408248, C_AFM, "o", r"$\mathrm{QMC}$  AFM  $J>0$"),
                (-0.408248, C_FM, "x", r"$\mathrm{QMC}$  FM  $J<0$")):
            tau_q, c_q, e_q, c_zz = load_qmc(beta, J)
            sub = slice(None, None, 3)   # thin the 100 points for legibility
            top.plot(tau_q[sub] / beta, c_q[sub], mark, color=colour, ms=5.0,
                     mfc="white" if mark == "o" else colour, mew=1.4,
                     zorder=4, label=name if col == 0 else None)

            resid = c_q - np.interp(tau_q, tau_d, c_d)
            # Both methods are stochastic, so the band is the combined error.
            # spinDMFT dominates it -- its sampling error is roughly an order
            # of magnitude above the QMC one from beta = 0.5 upward.
            comb = np.sqrt(e_q**2 + np.interp(tau_q, tau_d, e_d)**2)
            bot.axhline(0.0, color="0.45", lw=0.8, zorder=1)
            bot.fill_between(tau_q / beta, -comb, comb, color="0.85",
                             lw=0, zorder=0)
            bot.plot(tau_q / beta, resid, "-", color=colour, lw=1.6, zorder=3)

            print(f"beta={beta:<4g} J={J:+.6f}  max|xx-zz| = {np.abs(c_q - c_zz).max():.2e}"
                  f"   max|QMC-DMFT| = {np.abs(resid).max():.4f}"
                  f"   at tau/beta = {tau_q[np.argmax(np.abs(resid))] / beta:.2f}")

        top.set_title(rf"$\beta J_Q = {beta:g}$")
        top.grid(alpha=0.25, lw=0.6)
        bot.grid(alpha=0.25, lw=0.6)
        bot.set_xlabel(r"$\tau/\beta$")
        bot.set_xlim(0, 1)

    axes[0, 0].set_ylabel(r"$\mathrm{Re}\,g^{xx}(\tau)$")
    axes[1, 0].set_ylabel(r"$\mathrm{QMC}-\mathrm{spinDMFT}$", fontsize=14)
    axes[0, 0].legend(loc="lower left", fontsize=11, **LEGEND_BOX)

    fig.tight_layout()

    out = "Plots/qmc_N8000_vs_spindmft.png"
    os.makedirs("Plots", exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)

    make_overlay(betas, "Plots/qmc_N8000_vs_spindmft_overlay.png")


if __name__ == "__main__":
    main()
