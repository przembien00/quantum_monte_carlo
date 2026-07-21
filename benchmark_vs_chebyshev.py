#!/usr/bin/env python
"""Benchmark QMC against Chebyshev typicality on the 24-site periodic square.

The lattice is the 6x4 torus, whose coupling matrix is byte-identical to
``Couplings/Square_NN_PBC_N=24.hdf5`` in the Chebyshev tree, so site indices
agree and sites 0, 1, 7 are the local, nearest-neighbour and next-nearest
(diagonal) correlations respectively.

Chebyshev stores tau on ``linspace(0, beta/2, num_TimePoints)`` (its ``Tmax`` is
beta/2), while QMC stores a uniform grid on [0, beta). The comparison is made at
the QMC grid points lying in [0, beta/2], with the dense Chebyshev curve
interpolated onto them -- interpolating the reference rather than the data keeps
the QMC error bars meaningful.
"""

import argparse
import os
import sys

import h5py
import numpy as np

from qmc import run as run_mod

CHEBY_DIR = "/Users/przembien/Projects/chebyshev_typicality/Data"
LATTICE = "square:6x4"
SITES = [0, 1, 7]
SITE_LABEL = {0: "local", 1: "nearest neighbour", 7: "next-nearest (diagonal)"}


def cheby_path(beta, J):
    """Chebyshev files name the coupling scale 'rescale', which equals J here."""
    return os.path.join(
        CHEBY_DIR,
        f"ISO__Square_NN_PBC_N=24__sites=0-1-7__beta={_fmt(beta)}"
        f"__rescale={_fmt(J)}.hdf5",
    )


def _fmt(x):
    x = float(x)
    return str(int(x)) if x == int(x) else repr(x)


def load_chebyshev(path):
    """Return tau grid and {site: (values, errors)}; class A stores one row."""
    with h5py.File(path, "r") as f:
        a = f["parameters"].attrs
        tau = np.linspace(0.0, float(a["Tmax"]), int(a["num_TimePoints"]))
        out = {}
        for site in SITES:
            key = f"{site}-0"
            out[site] = (f[f"results/Re_correlation/{key}"][0].copy(),
                         f[f"results/Re_stds/{key}"][0].copy())
    return tau, out


def load_qmc(path):
    """Return tau grid and {site: (values, errors)} for the xx row."""
    with h5py.File(path, "r") as f:
        a = f["parameters"].attrs
        beta, ntau = float(a["beta"]), int(a["num_TimePoints"])
        tau = np.arange(ntau) * beta / ntau
        rows = a["correlation_rows"].split(",")
        xx = rows.index("xx")
        out = {}
        for site in SITES:
            key = f"{site}-0"
            out[site] = (f[f"results/Re_correlation/{key}"][xx].copy(),
                         f[f"results/Re_stds/{key}"][xx].copy())
    return tau, out


def compare(beta, J, qmc_path, out=sys.stdout):
    """Compare one (beta, J) pair; returns a list of per-site result dicts."""
    ref_path = cheby_path(beta, J)
    if not os.path.exists(ref_path):
        print(f"  no Chebyshev reference for beta={beta}, J={J}", file=out)
        return []

    tau_ref, ref = load_chebyshev(ref_path)
    tau_qmc, qmc = load_qmc(qmc_path)

    # Compare only where the reference is defined, i.e. tau <= beta/2.
    mask = tau_qmc <= tau_ref[-1] + 1e-12
    tau = tau_qmc[mask]
    # tau = 0 is excluded from the pull statistic: there C^xx(0) = 1/4 is an
    # operator identity, both methods carry essentially zero error bar, and any
    # fluctuation divided by that is arbitrarily large while meaning nothing.
    # It is reported separately as an absolute deviation instead.
    interior = tau > 0

    rows = []
    for site in SITES:
        q, qerr = qmc[site][0][mask], qmc[site][1][mask]
        r = np.interp(tau, tau_ref, ref[site][0])
        rerr = np.interp(tau, tau_ref, ref[site][1])
        combined = np.sqrt(qerr**2 + rerr**2)
        dev = np.abs(q - r)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(combined > 0, dev / combined, 0.0)
        rows.append({
            "beta": beta, "J": J, "site": site,
            "max_abs": float(np.nanmax(dev[interior])),
            "rms": float(np.sqrt(np.nanmean(dev[interior] ** 2))),
            "max_sigma": float(np.nanmax(sigma[interior])),
            "dev_at_zero": float(dev[0]),
            "qmc_err": float(np.nanmedian(qerr)),
            "cheby_err": float(np.nanmedian(rerr)),
            "npoints": int(interior.sum()),
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", default="0.2,0.5,1,1.5,2,2.5,3")
    parser.add_argument("--couplings", default="0.5,-0.5")
    parser.add_argument("--ntau", type=int, default=64)
    parser.add_argument("--nmcs", type=int, default=50000)
    parser.add_argument("--nset", type=int, default=20)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--data-dir", default="Data/benchmark_N24")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse QMC output already present")
    args = parser.parse_args(argv)

    betas = [float(b) for b in args.betas.split(",")]
    couplings = [float(j) for j in args.couplings.split(",")]

    all_rows = []
    for J in couplings:
        for beta in betas:
            tag = f"beta={_fmt(beta)} J={_fmt(J)}"
            expected = os.path.join(
                args.data_dir,
                f"ISO__Square_NN_PBC_N=24__sites=0-1-7__beta={_fmt(beta)}"
                f"__J={_fmt(J)}.hdf5")
            if args.skip_existing and os.path.exists(expected):
                print(f"[{tag}] reusing {expected}")
                path = expected
            else:
                print(f"[{tag}] running")
                path = run_mod.run(
                    LATTICE, beta, args.ntau, spin_sites=SITES, h_z=0.0, J=J,
                    mc={"nmcs": args.nmcs, "nset": args.nset},
                    ncores=args.cores, data_dir=args.data_dir, progress=True,
                )
            all_rows.extend(compare(beta, J, path))

    if not all_rows:
        print("nothing to compare")
        return 1

    print("\n" + "=" * 86)
    print(" QMC vs Chebyshev typicality -- 24-site periodic square (6x4), h_z = 0")
    print("=" * 86)
    print(f"{'J':>5} {'beta':>5} {'site':>5} {'pair':<24} "
          f"{'max|dev|':>9} {'rms':>9} {'max pull':>9} "
          f"{'err_QMC':>9} {'err_Cheb':>9} {'dev@tau=0':>10}")
    for row in all_rows:
        print(f"{row['J']:>5.2f} {row['beta']:>5.2f} {row['site']:>5} "
              f"{SITE_LABEL[row['site']]:<24} "
              f"{row['max_abs']:>9.2e} {row['rms']:>9.2e} "
              f"{row['max_sigma']:>9.1f} "
              f"{row['qmc_err']:>9.1e} {row['cheby_err']:>9.1e} "
              f"{row['dev_at_zero']:>10.1e}")

    worst = max(all_rows, key=lambda r: r["max_sigma"])
    pulls = np.array([r["max_sigma"] for r in all_rows])
    print("-" * 86)
    print(f" largest pull: {worst['max_sigma']:.1f} sigma "
          f"({worst['max_abs']:.2e} absolute) at beta={worst['beta']}, "
          f"J={worst['J']}, site {worst['site']}")
    print(f" pulls: median {np.median(pulls):.1f}, "
          f"{int(np.sum(pulls > 3))}/{len(pulls)} series exceed 3 sigma")
    print(f" median QMC error {np.median([r['qmc_err'] for r in all_rows]):.1e}, "
          f"median Chebyshev error "
          f"{np.median([r['cheby_err'] for r in all_rows]):.1e}")
    print(" tau = 0 is excluded from the pull (C(0) = 1/4 is exact, so both "
          "error bars vanish there); it is listed as an absolute deviation.")

    np.save(os.path.join(args.data_dir, "comparison.npy"), all_rows,
            allow_pickle=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
