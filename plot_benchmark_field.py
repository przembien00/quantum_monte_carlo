#!/usr/bin/env python
"""Plot the finite-field QMC vs Chebyshev benchmark (24-site square, site 0).

One figure per (h_z, J): the three measurable components across the top and,
where the reference carries error bars, the pull underneath. Plotted against
tau/beta so every temperature shares the range [0, 1/2].

The h_z = 2 reference files at beta in {0.2, 0.4, 0.6} were produced by an older
version of the Chebyshev code that swapped the two coupling signs and inverted
C^xy. Those files are corrected on load (see KNOWN_BAD) and the figure is
annotated accordingly; the correction is a relabelling, not a fit.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_field import (AVAILABLE, cheby_path, load_chebyshev, load_qmc,
                             _fmt)

COMPONENTS = [("xx", 0, r"$C^{xx}(\tau)$"),
              ("zz", 0, r"$C^{zz}(\tau)$"),
              ("xy", 1, r"$\mathrm{Im}\,C^{xy}(\tau)$")]

# (h_z, beta) pairs whose reference files have swapped J labels and an
# inverted C^xy, confirmed by the user as output of superseded code.
KNOWN_BAD = {(2.0, 0.2), (2.0, 0.4), (2.0, 0.6)}


def qmc_path(data_dir, beta, J, h_z):
    return os.path.join(
        data_dir,
        f"ISO__Square_NN_PBC_N=24__beta={_fmt(beta)}__J={_fmt(J)}"
        f"__h_z={_fmt(h_z)}.hdf5")


def load_reference(beta, h_z, J):
    """Reference for physical coupling J, undoing the known labelling bug."""
    corrected = (h_z, beta) in KNOWN_BAD
    # The buggy files store physical J under the opposite label.
    tau, ref, has_err = load_chebyshev(cheby_path(beta, h_z, -J if corrected else J))
    if corrected:
        ref = dict(ref)
        re, im, ree, ime = ref["xy"]
        ref["xy"] = (re, -im, ree, ime)
    return tau, ref, has_err, corrected


def make_figure(h_z, J, data_dir, outpath):
    betas = AVAILABLE.get((h_z, J), [])
    if not betas:
        return None
    tmp = [b for b in betas
           if os.path.exists(qmc_path(data_dir, b, J, h_z))]
    if not tmp:
        return None
    betas = tmp

    _, _, has_err, _ = load_reference(betas[0], h_z, J)
    nrows = 2 if has_err else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 7.5 if has_err else 5.0),
                             sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]} if has_err else None)
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(betas)))
    corrected_any = False

    for ib, beta in enumerate(betas):
        tau_r, ref, _, corrected = load_reference(beta, h_z, J)
        corrected_any |= corrected
        tau_q, qmc = load_qmc(qmc_path(data_dir, beta, J, h_z))
        mask = tau_q <= tau_r[-1] + 1e-12
        x_q, x_r = tau_q[mask] / beta, tau_r / beta

        for col, (comp, part, _) in enumerate(COMPONENTS):
            top = axes[0, col]
            q, qerr = qmc[comp][part][mask], qmc[comp][2 + part][mask]
            r, rerr = ref[comp][part], ref[comp][2 + part]

            top.plot(x_r, r, "-", color=colours[ib], lw=1.4,
                     label=rf"$\beta={beta:g}$" if col == 0 else None)
            top.errorbar(x_q, q, yerr=qerr, fmt="o", ms=3.0, color=colours[ib],
                         mfc="white", mew=0.9, lw=0.8, capsize=1.5)

            if has_err:
                inner = x_q > 0
                ri = np.interp(tau_q[mask], tau_r, r)
                rei = np.interp(tau_q[mask], tau_r, rerr)
                combined = np.sqrt(qerr**2 + rei**2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pull = np.where(combined > 0, (q - ri) / combined, 0.0)
                axes[1, col].plot(x_q[inner], pull[inner], "o-", ms=2.5, lw=0.8,
                                  color=colours[ib])

    for col, (comp, _, ylabel) in enumerate(COMPONENTS):
        axes[0, col].set_title(comp)
        axes[0, col].set_ylabel(ylabel)
        bottom = axes[nrows - 1, col]
        bottom.set_xlabel(r"$\tau/\beta$")
        bottom.set_xlim(0, 0.5)
        if has_err:
            axes[1, col].axhspan(-2, 2, color="0.85", zorder=0)
            axes[1, col].axhline(0, color="0.4", lw=0.8, zorder=1)
            axes[1, col].set_ylim(-6, 6)
    if has_err:
        axes[1, 0].set_ylabel("pull")
    axes[0, 0].legend(fontsize=8, ncol=2)

    order = "antiferromagnetic" if J > 0 else "ferromagnetic"
    title = (f"24-site periodic square (6x4), site 0, $h_z={h_z:g}$, "
             f"$J={J:+g}$ ({order})  —  lines: Chebyshev, points: QMC")
    if not has_err:
        title += "\nreference stores no error bars; absolute comparison only"
    if corrected_any:
        title += ("\nreference relabelled: superseded code swapped the coupling "
                  "signs and inverted $C^{xy}$")
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if (corrected_any or not has_err) else 0.96))
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return outpath


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Data/benchmark_N24_field")
    parser.add_argument("--outdir", default="Plots")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    written = []
    for (h_z, J) in sorted(AVAILABLE):
        tag = f"h{_fmt(h_z)}_{'afm' if J > 0 else 'fm'}".replace(".", "p")
        path = make_figure(h_z, J, args.data_dir,
                           os.path.join(args.outdir, f"benchmark_N24_field_{tag}.png"))
        if path:
            written.append(path)
            print(f"wrote {path}")
    if not written:
        print("no data found to plot")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
