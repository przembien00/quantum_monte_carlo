#!/usr/bin/env python
"""QMC vs spinDMFT for the finite-field data (h_z = 0.5).

Companion to plot_qmc_vs_spindmft.py, which does the same at zero field.
Two things differ once the field is on:

* spinDMFT is no longer blind to the sign of J.  The field polarises the
  spins, so the Weiss field z*J*<S^z> enters linearly and the two signs
  need separate solutions.  They are the JL = z*J = +-2.4495 files; JQ = 1
  still carries the quadratic coupling, |J| = 1/sqrt(6).
* C^xx and C^zz are no longer equal, so both are shown.

Encoding therefore changes: colour is the coupling sign (the quantity being
compared), linestyle is the method -- thick solid spinDMFT, thin dashed QMC.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import h5py

QMC_DIR = "Data/AFM_FM_B"
DMFT_DIR = "Data/spinDMFT"
SITES = "0-1-21-421-2"          # shells 0..4 on the 20^3 cube, N = 8000
ROWS = ["xx", "xy", "yx", "zz"]
H_Z = 0.5

BETAS = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MARKERS = ["o", "s", "^", "D", "v", "P", "*"]

C_AFM, C_FM = "#e81919", "#364fc7"
C_DMFT = "#51cf66"              # spinDMFT keeps the green of the zero-field plots
COUPLINGS = [(0.408248, C_AFM, r"AFM  $J>0$"),
             (-0.408248, C_FM, r"FM  $J<0$")]

Z_COORD = 6                     # simple cubic


def _fmt(x):
    return f"{x:g}"


def _ms(mark):
    return 9.0 if mark == "*" else 6.0


N_MARKS = 9                     # markers drawn per curve


def _every(n, phase):
    """markevery for a curve of n points, shifted by `phase` of one spacing.

    The two methods are stored on different grids (101 vs 200 points), so the
    offset has to be computed as a fraction of the curve rather than in index
    units for the symbols to interleave.
    """
    step = max(n // N_MARKS, 1)
    return (int((0.04 + phase / N_MARKS) * n), step)


def load_qmc(beta, J, row):
    """(tau, C, err) on the closed [0, beta].

    DSQSS writes the half-open [0, beta); the endpoint follows from the KMS
    relation C^ab(beta - tau) = C^ba(tau).  For xx and zz that is plain
    symmetry, so C(beta) = C(0).  For xy it gives C^xy(beta) = C^yx(0) =
    -C^xy(0): the transverse channel is *anti*symmetric about beta/2 and the
    endpoint flips sign.  The error bar is unchanged either way.
    """
    path = os.path.join(
        QMC_DIR, f"ISO__Cube_NN_PBC_N=8000__sites={SITES}__beta={_fmt(beta)}"
                 f"__J={_fmt(J)}__h_z={_fmt(H_Z)}.hdf5")
    part = "Im" if row in ("xy", "yx") else "Re"
    with h5py.File(path) as h:
        c = np.asarray(h[f"results/{part}_correlation/0-0"])[ROWS.index(row)]
        e = np.asarray(h[f"results/{part}_stds/0-0"])[ROWS.index(row)]
        nt = int(h["parameters"].attrs["num_TimePoints"])
    sign = -1.0 if row in ("xy", "yx") else 1.0
    tau = np.append(np.arange(nt) * beta / nt, beta)
    return tau, np.append(c, sign * c[0]), np.append(e, e[0])


def dmft_path(beta, J):
    """JL = z*J carries the coupling sign; JQ = 1 carries its magnitude."""
    return os.path.join(
        DMFT_DIR,
        f"spinmodel=ISO__JL={Z_COORD * J:.4f}__beta={_fmt(beta)}"
        f"__h=z_h_abs={_fmt(H_Z)}.hdf5")


def load_dmft(beta, J, row):
    """(tau, C, err) on the closed [0, beta], or None when the run is absent."""
    path = dmft_path(beta, J)
    if not os.path.exists(path):
        return None
    part = "Im" if row in ("xy", "yx") else "Re"
    with h5py.File(path) as h:
        c = np.asarray(h[f"results/{part}_correlation"])[ROWS.index(row)]
        e = np.asarray(h[f"runtimedata/{part}_correlation_sample_stds"])[
            ROWS.index(row)]
    return np.linspace(0.0, beta, c.size), c, e


CHANNELS = [("xx", r"$\mathrm{Re}\,g^{xx}(\tau)$"),
            ("xy", r"$\mathrm{Im}\,g^{xy}(\tau)$"),
            ("zz", r"$\mathrm{Re}\,g^{zz}(\tau)$")]

# Roman text inside mathtext (Re, Im, QMC, spinDMFT) is set in a serif face;
# italic symbols (g, tau, beta, J) stay in the sans face of the rest of the
# figure.  That needs the 'custom' fontset -- the stock ones tie both together.
plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "DejaVu Serif",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    # Tick numbers are plain text, so they follow font.family rather than the
    # mathtext settings above; serif here matches the Re/Im/QMC labels.
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


def make_sign_figure(J, colour, name, out, link=()):
    """One coupling sign; xx, xy and zz stacked on a shared tau/beta axis.

    `link` names channels that should share a common y range, so panels of
    comparable magnitude can be read against each other directly.
    """
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(13.0, 12.0),
                             sharex=True, gridspec_kw={"hspace": 0.0})

    # Only temperatures where both methods exist -- a lone QMC curve has
    # nothing to be compared against and just reads as an unmatched line.
    # Markers stay keyed to the global beta list so a given beta wears the
    # same symbol in both figures.
    shown = [(i, b) for i, b in enumerate(BETAS)
             if load_dmft(b, J, CHANNELS[0][0]) is not None]
    missing = [b for b in BETAS if b not in [s[1] for s in shown]]

    for ax, (row, ylab) in zip(axes, CHANNELS):
        for ib, beta in shown:
            mark = MARKERS[ib]
            tau_q, c_q, _ = load_qmc(beta, J, row)
            ax.plot(tau_q / beta, c_q, "--", color=colour, lw=1.6, marker=mark,
                    ms=_ms(mark), markevery=_every(tau_q.size, 0.0),
                    mfc="white", mew=1.3, zorder=3)

            tau_d, c_d, _ = load_dmft(beta, J, row)
            # Offset by half a marker spacing so the two methods' symbols
            # interleave along tau rather than sitting on top of each other.
            ax.plot(tau_d / beta, c_d, "-", color=C_DMFT, lw=3.4, marker=mark,
                    ms=_ms(mark), markevery=_every(tau_d.size, 0.5),
                    mfc="white", mec=C_DMFT, mew=1.6, zorder=2)

        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.6)
        # Only xy straddles zero, and only there does a zero line inform.  On
        # xx/zz it would drag the autoscale down to 0 and flatten the data.
        if row == "xy":
            ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
        # The panels share edges, so the extreme y ticks of neighbours would
        # collide; dropping them keeps the boundary clean.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    linked = [ax for ax, (row, _) in zip(axes, CHANNELS) if row in link]
    if linked:
        lo = min(ax.get_ylim()[0] for ax in linked)
        hi = max(ax.get_ylim()[1] for ax in linked)
        for ax in linked:
            ax.set_ylim(lo, hi)

    axes[-1].set_xlabel(r"$\tau/\beta$")
    axes[-1].set_xlim(0, 1)

    method_keys = [
        Line2D([], [], color=colour, ls="--", lw=1.8, marker="o", ms=6.0,
               mfc="white", mew=1.3, label=r"$\mathrm{QMC}$"),
        Line2D([], [], color=C_DMFT, ls="-", lw=3.4, marker="o", ms=6.0,
               mfc="white", mec=C_DMFT, mew=1.6, label=r"$\mathrm{spinDMFT}$")]
    # spinDMFT sets J_Q = 1, so beta*J_Q is numerically beta -- the label
    # states the dimensionless combination rather than the bare inverse
    # temperature.
    beta_keys = [Line2D([], [], color="0.35", ls="none", marker=MARKERS[i],
                        ms=_ms(MARKERS[i]), mfc="white", mew=1.3,
                        label=rf"$\beta J_Q={b:g}$") for i, b in shown]

    # The xy panel is empty in its upper-left corner in both figures -- xy
    # rises monotonically from its most negative value at tau = 0 -- so the
    # temperature key goes there instead of fighting the curves on xx.  A
    # translucent background is kept as insurance against near misses.
    box = dict(frameon=True, facecolor="white", framealpha=0.93,
               edgecolor="0.65", fancybox=True, borderpad=0.7,
               labelspacing=0.6)
    axes[0].legend(handles=method_keys, loc="lower left", **box)
    axes[1].legend(handles=beta_keys, loc="upper left", ncol=2,
                   columnspacing=1.2, handletextpad=0.5, **box)

    if missing:
        print(f"  {name}: omitted beta = "
              + ", ".join(f"{b:g}" for b in sorted(missing))
              + "  (no spinDMFT run)")
    # Margins are held at a fixed size in inches so the plot area, not the
    # whitespace, absorbs a change of figsize.
    fig.subplots_adjust(left=0.095, right=0.989, top=0.992, bottom=0.069,
                        hspace=0.0)
    fig.savefig(out, dpi=160)
    print("wrote", out)


def report():
    print("max |QMC - spinDMFT| over tau, on-site")
    for row, _ in CHANNELS:
        print(f"  channel {row}")
        for beta in BETAS:
            line = f"    beta={beta:<4g}"
            for J, _, name in COUPLINGS:
                d = load_dmft(beta, J, row)
                if d is None:
                    line += f"   {name}: --      "
                    continue
                tau_q, c_q, _ = load_qmc(beta, J, row)
                tau_d, c_d, _ = d
                dev = np.abs(c_q - np.interp(tau_q, tau_d, c_d)).max()
                line += f"   {name}: {dev:.4f}"
            print(line)


def main():
    os.makedirs("Plots", exist_ok=True)
    report()
    # xx and zz sit in the same range for the antiferromagnet, so a shared
    # scale makes the two channels directly comparable.  The ferromagnet's
    # zz spans a fortieth of its xx -- linking there would flatten it.
    make_sign_figure(0.408248, C_AFM, "Antiferromagnet  $J>0$",
                     "Plots/qmc_N8000_vs_spindmft_field_afm.png",
                     link=("xx", "zz"))
    make_sign_figure(-0.408248, C_FM, "Ferromagnet  $J<0$",
                     "Plots/qmc_N8000_vs_spindmft_field_fm.png")


if __name__ == "__main__":
    main()
