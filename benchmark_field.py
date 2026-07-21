#!/usr/bin/env python
"""Benchmark QMC against Chebyshev typicality at finite longitudinal field.

Same 24-site periodic square (6x4) as ``benchmark_vs_chebyshev.py``. The field
data is local only (site 0), but it carries all four correlation rows, so this
exercises the components that vanish at zero field:

  * ``C^xx`` and ``C^zz`` separately -- they split once h_z != 0
  * ``Im C^xy`` -- purely imaginary, antisymmetric about beta/2, and measurable
    only with the patched worm estimator (patches/dsqss_estimators.patch)

The C^xy comparison is the one that cannot be checked by exact diagonalization
at this size, so it is the substantive test of the patch.

Chebyshev field files predate the per-site group layout: ``results/Re_correlation``
is a flat (4, num_TimePoints) dataset rather than a group keyed by site.
"""

import argparse
import os
import sys

import h5py
import numpy as np

from qmc import run as run_mod

CHEBY_DIR = "/Users/przembien/Projects/chebyshev_typicality/Data"
LATTICE = "square:6x4"
ROWS = ["xx", "xy", "yx", "zz"]

# (h_z, J) -> betas present in the Chebyshev tree
AVAILABLE = {
    (0.5, 0.5): [0.2, 0.5, 1.0, 1.5, 2.0, 2.5],
    (0.5, -0.5): [0.2, 0.5, 1.0, 1.5, 2.0, 2.5],
    (2.0, 0.5): [0.2, 0.4, 0.6, 0.8, 1.0],
    (2.0, -0.5): [0.2, 0.4, 0.6],
}


def _fmt(x):
    x = float(x)
    return str(int(x)) if x == int(x) else repr(x)


def cheby_path(beta, h_z, J):
    """Locate a reference file; the h_z = 2 set lives in a Mag_Field subfolder."""
    name = (f"ISO__Square_NN_PBC_N=24__beta={_fmt(beta)}__h_z={_fmt(h_z)}"
            f"__rescale={_fmt(J)}.hdf5")
    for folder in (CHEBY_DIR, os.path.join(CHEBY_DIR, "Mag_Field")):
        candidate = os.path.join(folder, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(CHEBY_DIR, name)


def load_chebyshev(path):
    """Return tau, {component: (real, imag, real_err, imag_err)}, has_errors.

    Error datasets are named ``Re_stds`` in newer files and ``Re_stddev`` in
    older ones, and the h_z = 2 set carries none at all. Missing errors come
    back as zeros and are flagged, so the caller can fall back to a
    QMC-only comparison rather than silently inventing a reference error.
    """
    with h5py.File(path, "r") as f:
        a = f["parameters"].attrs
        tau = np.linspace(0.0, float(a["Tmax"]), int(a["num_TimePoints"]))
        re, im = f["results/Re_correlation"][:], f["results/Im_correlation"][:]
        keys = set(f["results"].keys())
        for stem in ("stds", "stddev"):
            if f"Re_{stem}" in keys:
                ree, ime = f[f"results/Re_{stem}"][:], f[f"results/Im_{stem}"][:]
                has_errors = True
                break
        else:
            ree = ime = np.zeros_like(re)
            has_errors = False
    return (tau,
            {r: (re[i], im[i], ree[i], ime[i]) for i, r in enumerate(ROWS)},
            has_errors)


def load_qmc(path):
    with h5py.File(path, "r") as f:
        a = f["parameters"].attrs
        beta, ntau = float(a["beta"]), int(a["num_TimePoints"])
        tau = np.arange(ntau) * beta / ntau
        order = a["correlation_rows"].split(",")
        re, im = f["results/Re_correlation/0-0"][:], f["results/Im_correlation/0-0"][:]
        ree, ime = f["results/Re_stds/0-0"][:], f["results/Im_stds/0-0"][:]
        out = {r: (re[i], im[i], ree[i], ime[i]) for i, r in enumerate(order)}
    return tau, out


def compare(beta, h_z, J, qmc_path):
    ref_path = cheby_path(beta, h_z, J)
    if not os.path.exists(ref_path):
        return []
    tau_r, ref, has_errors = load_chebyshev(ref_path)
    tau_q, qmc = load_qmc(qmc_path)
    mask = tau_q <= tau_r[-1] + 1e-12
    tau = tau_q[mask]
    interior = tau > 0

    rows = []
    # xx and zz live in the real part, xy in the imaginary part.
    for comp, part in [("xx", 0), ("zz", 0), ("xy", 1)]:
        q = qmc[comp][part][mask]
        qerr = qmc[comp][2 + part][mask]
        r = np.interp(tau, tau_r, ref[comp][part])
        rerr = np.interp(tau, tau_r, ref[comp][2 + part])
        combined = np.sqrt(qerr**2 + rerr**2)
        dev = np.abs(q - r)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(combined > 0, dev / combined, 0.0)
        rows.append({
            "beta": beta, "h_z": h_z, "J": J, "comp": comp,
            "max_abs": float(np.nanmax(dev[interior])),
            "max_sigma": float(np.nanmax(sigma[interior])),
            "amplitude": float(np.nanmax(np.abs(r))),
            "qmc_err": float(np.nanmedian(qerr)),
            "cheby_err": float(np.nanmedian(rerr)),
            "has_errors": has_errors,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", default="0.5,2.0")
    parser.add_argument("--ntau", type=int, default=64)
    parser.add_argument("--nmcs", type=int, default=50000)
    parser.add_argument("--nset", type=int, default=20)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--data-dir", default="Data/benchmark_N24_field")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    fields = [float(h) for h in args.fields.split(",")]
    all_rows = []
    for (h_z, J), betas in sorted(AVAILABLE.items()):
        if h_z not in fields:
            continue
        for beta in betas:
            expected = os.path.join(
                args.data_dir,
                f"ISO__Square_NN_PBC_N=24__beta={_fmt(beta)}__J={_fmt(J)}"
                f"__h_z={_fmt(h_z)}.hdf5")
            if args.skip_existing and os.path.exists(expected):
                path = expected
            else:
                print(f"[h_z={h_z} J={J} beta={beta}] running")
                path = run_mod.run(
                    LATTICE, beta, args.ntau, spin_sites=[0], h_z=h_z, J=J,
                    mc={"nmcs": args.nmcs, "nset": args.nset},
                    ncores=args.cores, data_dir=args.data_dir, progress=True)
            all_rows.extend(compare(beta, h_z, J, path))

    if not all_rows:
        print("nothing to compare")
        return 1

    print("\n" + "=" * 92)
    print(" QMC vs Chebyshev at finite field -- 24-site periodic square (6x4), site 0")
    print("=" * 92)
    print(f"{'h_z':>5} {'J':>6} {'beta':>5} {'comp':>5} "
          f"{'max|dev|':>10} {'max pull':>9} {'amplitude':>10} "
          f"{'err_QMC':>9} {'err_Cheb':>9}")
    for row in all_rows:
        pull = f"{row['max_sigma']:>9.1f}" if row["has_errors"] else f"{'n/a':>9}"
        cerr = f"{row['cheby_err']:>9.1e}" if row["has_errors"] else f"{'none':>9}"
        print(f"{row['h_z']:>5.1f} {row['J']:>6.1f} {row['beta']:>5.2f} "
              f"{row['comp']:>5} {row['max_abs']:>10.2e} "
              f"{pull} {row['amplitude']:>10.4f} "
              f"{row['qmc_err']:>9.1e} {cerr}")

    print("-" * 92)
    scored = [r for r in all_rows if r["has_errors"]]
    if scored:
        pulls = np.array([r["max_sigma"] for r in scored])
        worst = max(scored, key=lambda r: r["max_sigma"])
        print(f" with reference error bars ({len(scored)} series):")
        print(f"   largest pull {worst['max_sigma']:.1f} sigma at h_z={worst['h_z']},"
              f" J={worst['J']}, beta={worst['beta']}, {worst['comp']}")
        print(f"   median {np.median(pulls):.1f}, "
              f"{int(np.sum(pulls > 3))}/{len(pulls)} exceed 3 sigma")
        for comp in ["xx", "zz", "xy"]:
            sub = [r for r in scored if r["comp"] == comp]
            if sub:
                print(f"     {comp}: max pull {max(r['max_sigma'] for r in sub):.1f}, "
                      f"typical amplitude "
                      f"{np.median([r['amplitude'] for r in sub]):.4f}")
    unscored = [r for r in all_rows if not r["has_errors"]]
    if unscored:
        print(f" without reference error bars ({len(unscored)} series, h_z = 2):")
        print("   the Chebyshev files store no uncertainties, so only absolute "
              "agreement can be quoted.")
        for comp in ["xx", "zz", "xy"]:
            sub = [r for r in unscored if r["comp"] == comp]
            if sub:
                w = max(sub, key=lambda r: r["max_abs"])
                print(f"     {comp}: max|dev| {w['max_abs']:.2e} against amplitude "
                      f"{w['amplitude']:.4f} and QMC error {w['qmc_err']:.1e}")
    print(" tau = 0 excluded from the pull (error bars vanish there).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
