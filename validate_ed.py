#!/usr/bin/env python
"""Validate QMC output against exact diagonalization on a small chain.

Builds the same Hamiltonian the Chebyshev code uses,
H = sum_{i<j} J_ij S_i.S_j + h_z sum_i S^z_i, computes the imaginary-time
correlations exactly, and compares them with a QMC run on the same lattice.
"""

import argparse
import itertools
import sys

import numpy as np

from qmc import lattice as lattice_mod, run as run_mod

SX = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
SY = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)


def site_operator(op, site, nsites):
    """Embed a single-site operator into the full Hilbert space."""
    result = np.eye(1, dtype=complex)
    for i in range(nsites):
        factor = op if i == site else np.eye(2, dtype=complex)
        result = np.kron(result, factor)
    return np.ascontiguousarray(result, dtype=complex)


def hamiltonian(J_ij, h_z):
    nsites = J_ij.shape[0]
    dim = 2**nsites
    H = np.zeros((dim, dim), dtype=complex)
    ops = [[site_operator(o, i, nsites) for o in (SX, SY, SZ)] for i in range(nsites)]
    # errstate for the Accelerate BLAS false alarm; see exact_correlations.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i in range(nsites):
            for j in range(i + 1, nsites):
                if J_ij[i, j] == 0:
                    continue
                for a in range(3):
                    H += J_ij[i, j] * ops[i][a] @ ops[j][a]
            H += h_z * ops[i][2]
    if not np.all(np.isfinite(H)):
        raise RuntimeError("Hamiltonian contains non-finite entries")
    return H, ops


def exact_correlations(J_ij, beta, taus, h_z, sites):
    """Return {(site, 'xx'|'xy'|'zz'): complex array over taus}.

    The matrix products below are wrapped in errstate for the same reason as in
    qmc/correlations.py: numpy built against Accelerate reports floating-point
    exception flags left set by the BLAS even when every input and output is
    finite. Both are asserted finite here so nothing real is hidden.
    """
    if not np.all(np.isfinite(J_ij)):
        raise ValueError("coupling matrix contains non-finite entries")
    H, ops = hamiltonian(J_ij, h_z)
    evals, evecs = np.linalg.eigh(H)
    evals -= evals.min()
    weights = np.exp(-beta * evals)
    Z = weights.sum()

    def rotate(op):
        return evecs.conj().T @ op @ evecs

    out = {}
    components = {"xx": (0, 0), "xy": (0, 1), "zz": (2, 2)}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for site in sites:
            for name, (a, b) in components.items():
                A = rotate(ops[site][a])
                B = rotate(ops[0][b])
                values = np.empty(len(taus), dtype=complex)
                for it, tau in enumerate(taus):
                    # <A(tau) B(0)> = 1/Z sum_nm e^{-b E_n} e^{tau(E_n-E_m)} A_nm B_mn
                    prop = np.exp(np.subtract.outer(evals, evals) * tau)
                    values[it] = np.sum(weights[:, None] * prop * A * B.T) / Z
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"exact {name} correlation for site {site} is non-finite; "
                        "the reference itself is unreliable"
                    )
                out[(site, name)] = values
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", default="chain:8",
                        help="prefer a case with two or more sides of length >= 3 "
                             "(e.g. square:4x3 with --J=-1) when touching the "
                             "C^zz path: shorter sides make k -> -k trivial and "
                             "hide Brillouin-zone coverage bugs")
    parser.add_argument("--J", type=float, default=1.0,
                        help="coupling; negative (ferromagnet) lifts the "
                             "bipartite restriction")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--ntau", type=int, default=16)
    parser.add_argument("--h_z", type=float, default=0.0)
    parser.add_argument("--nmcs", type=int, default=50000)
    parser.add_argument("--nset", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="max allowed deviation in units of the QMC error bar")
    parser.add_argument("--atol", type=float, default=5e-4,
                        help="absolute deviation that always counts as agreement")
    parser.add_argument("--workdir", default=".qmc_validate")
    args = parser.parse_args(argv)

    lat = lattice_mod.build(args.lattice, J=args.J)
    if lat.nsites > 14:
        parser.error(f"{lat.nsites} sites is too many for exact diagonalization")
    if args.J > 0 and not lat.is_bipartite:
        parser.error(
            f"{lat.name} has an odd side length, so the antiferromagnet is "
            "frustrated and QMC has a sign problem. Validation would fail for "
            "physical reasons; use even side lengths, or --J=-1 for the "
            "ferromagnet, which is sign-free on any lattice."
        )
    sites = list(range(min(lat.nsites, 4)))

    outputs = run_mod.run_dsqss(
        lat, args.beta, args.ntau, args.workdir, spin_sites=sites,
        h_z=args.h_z, mc={"nmcs": args.nmcs, "nset": args.nset},
    )
    params, corr_re, corr_im, stds_re, stds_im = run_mod.convert(
        lat, args.beta, args.ntau, outputs, sites, h_z=args.h_z,
    )
    rows = params["correlation_rows"].split(",")

    taus = np.arange(args.ntau) * (args.beta / args.ntau)
    exact = exact_correlations(lat.coupling_matrix(), args.beta, taus, args.h_z, sites)

    print(f"{lat.name}  beta={args.beta}  h_z={args.h_z}  ntau={args.ntau}")
    print(f"{'site':>5} {'comp':>5} {'max|dev|':>10} {'max dev/err':>12}  status")
    worst = 0.0
    failures = 0
    for index, site in enumerate(sites):
        for name in ("xx", "zz", "xy"):
            if name not in rows:
                continue
            row = rows.index(name)
            # xy is purely imaginary; xx and zz purely real.
            if name == "xy":
                qmc, errs = corr_im[index, row], stds_im[index, row]
                ref = exact[(site, name)].imag
                if not np.any(errs):
                    continue  # unpatched build: channel not measured
            else:
                qmc, errs = corr_re[index, row], stds_re[index, row]
                ref = exact[(site, name)].real
            err = np.where(errs > 0, errs, np.inf)
            deviation = np.abs(qmc - ref)
            # tau=0 has a near-zero error bar, which makes a pure sigma test
            # meaningless there, so an absolute floor also counts as agreement.
            ratio = np.max(deviation / err)
            worst = max(worst, ratio)
            ok = bool(np.all((deviation <= args.tolerance * err)
                             | (deviation <= args.atol)))
            failures += 0 if ok else 1
            print(f"{site:>5} {name:>5} {np.max(deviation):>10.2e} {ratio:>12.2f}"
                  f"  {'ok' if ok else 'FAIL'}")

    print(f"\nworst deviation: {worst:.2f} sigma (tolerance {args.tolerance})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
