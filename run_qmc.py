#!/usr/bin/env python
"""Run worldline QMC on a periodic spin lattice, writing Chebyshev-format HDF5.

Example, mirroring the Chebyshev CLI:

    ./run_qmc.py --lattice=square:4x4 --beta=3 --num_TimePoints=100 \
                 --sites=0,1,5 --h_z=0.0 --cores=4
"""

import argparse
import sys

from qmc import lattice as lattice_mod, run as run_mod


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lattice", required=True,
                        help="chain:L, square:LxL or cube:LxLxL (periodic)")
    parser.add_argument("--beta", type=float, required=True, help="inverse temperature")
    parser.add_argument("--num_TimePoints", type=int, default=100,
                        dest="ntau", help="number of imaginary-time points")
    parser.add_argument("--sites", default="0",
                        help="comma-separated neighbour shells: 0 is the "
                             "on-site autocorrelation, 1 nearest neighbour, "
                             "2 next-nearest, and so on")
    parser.add_argument("--site-indices", action="store_true",
                        help="interpret --sites as raw site indices instead of "
                             "neighbour shells")
    parser.add_argument("--h_z", type=float, default=0.0, help="longitudinal field")
    parser.add_argument("--J", type=float, default=1.0,
                        help="coupling; positive is antiferromagnetic")
    parser.add_argument("--cores", type=int, default=1, help="MPI ranks")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--nmcs", type=int, default=20000, help="MC sweeps per set")
    parser.add_argument("--nset", type=int, default=10, help="number of MC sets")
    parser.add_argument("--ntherm", type=int, default=1000, help="thermalization sweeps")
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--project", default="", help="subdirectory under Data/")
    parser.add_argument("--extension", default="", help="filename suffix")
    parser.add_argument("--write-couplings", metavar="PATH", default=None,
                        help="also write the matching Chebyshev coupling file")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the parameter banner and progress output")
    args = parser.parse_args(argv)

    requested = [int(s) for s in args.sites.split(",") if s.strip() != ""]

    try:
        lat = lattice_mod.build(args.lattice, J=args.J)
        if args.write_couplings:
            lat.write_coupling_file(args.write_couplings)
            print(f"wrote coupling file {args.write_couplings}")

        # --sites names neighbour shells by default; the rest of the code works
        # in site indices, so translate once, here.
        shells = None if args.site_indices else requested
        sites = requested if args.site_indices else lat.shell_sites(requested)

        path = run_mod.run(
            args.lattice, args.beta, args.ntau, spin_sites=sites,
            spin_shells=shells, h_z=args.h_z,
            J=args.J, seed=args.seed, ncores=args.cores,
            mc={"nmcs": args.nmcs, "nset": args.nset, "ntherm": args.ntherm},
            data_dir=args.data_dir, project_name=args.project,
            extension=args.extension, keep_workdir=args.keep_workdir,
            progress=not args.quiet,
        )
    except ValueError as exc:
        # Configuration errors (non-bipartite lattice, bad spec, site out of
        # range) are the user's to fix; a traceback adds nothing.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.quiet:
        # Progress mode already reported the destination in context.
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
