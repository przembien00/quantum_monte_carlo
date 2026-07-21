"""Readers for the DSQSS/DLA output files.

Result lines everywhere look like ``R <name> = <mean> <error>``. The transverse
correlations land in ``cf.dat`` as ``C<kind>t<itau>``, the k-resolved S^zz in
``sample.log`` as ``S<ik>t<itau>``.
"""

import itertools
import re

import numpy as np

RESULT_RE = re.compile(r"^R\s+(\S+)\s*=\s*(\S+)\s+(\S+)")


def read_results(path):
    """Return {name: (mean, error)} for every ``R`` line in the file."""
    results = {}
    with open(path) as f:
        for line in f:
            m = RESULT_RE.match(line)
            if m:
                results[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return results


def read_displacement_kinds(path):
    """Return {(isite, jsite): kind} from a DSQSS ``disp.xml``.

    The file is not well-formed XML (it has bare comments and repeated tags at
    top level), so it is parsed with a regex rather than an XML parser.
    """
    kinds = {}
    tag = re.compile(r"<R>\s*(-?\d+)\s+(\d+)\s+(\d+)\s*</R>")
    with open(path) as f:
        for line in f:
            m = tag.search(line)
            if m:
                kind, isite, jsite = (int(m.group(i)) for i in (1, 2, 3))
                kinds[(isite, jsite)] = kind
    if not kinds:
        raise ValueError(f"no <R> displacement entries found in {path}")
    return kinds


def wavevector_list(size):
    """The full Brillouin zone, in integer units of 2*pi/L per dimension.

    DSQSS's own generator emits the product set ``{0..L/2}`` per dimension and
    relies on S(k) = S(-k) to cover the rest. That is a valid fundamental domain
    for k -> -k in one dimension, but *not* in two or more: on a 4x4 lattice the
    orbit {(1,3), (3,1)} lies entirely outside it, so those k are never measured
    and the back-transform to real space silently loses their contribution.

    The full zone is measured instead (see ``write_full_bz_wavevectors``), which
    makes the inverse transform exact with uniform weight and removes the need
    for any multiplicity bookkeeping.
    """
    axes = [list(range(length)) for length in size]
    return np.array(list(itertools.product(*axes)), dtype=np.int64)


def write_full_bz_wavevectors(path, size, coords, comment="full-BZ"):
    """Write a DSQSS ``wv.xml`` covering every k in the Brillouin zone.

    The file stores the precomputed phases cos/sin(2*pi*k.r/L) for each
    (site, k) pair, matching ``dsqss.wavevector.Wavevector.write_xml``. The k
    ordering is the one ``wavevector_list`` returns.
    """
    size = np.asarray(size, dtype=np.int64)
    kvecs = wavevector_list(size)
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<WaveVector>\n")
        f.write(f"<Comment> {comment} </Comment>\n")
        f.write(f"<NumberOfSites> {len(coords)} </NumberOfSites>\n")
        f.write(f"<NumberOfWaveVectors> {len(kvecs)} </NumberOfWaveVectors>\n")
        f.write("<!-- <RK> [phase(cos)] [phase(sin)] [isite] [kindx] </RK> -->\n")
        for ik, k in enumerate(kvecs):
            for isite, coord in enumerate(coords):
                phase = 2.0 * np.pi * float(np.sum(k * coord / size))
                f.write(f"<RK> {np.cos(phase):0< 18} {np.sin(phase):0< 18} "
                        f"{isite} {ik} </RK>\n")
        f.write("</WaveVector>\n")
    return kvecs


def collect_tau_series(results, prefix, nkinds, ntau):
    """Gather ``<prefix><kind>t<itau>`` results into (mean, err) arrays.

    Both have shape (nkinds, ntau). Missing entries are returned as NaN so that
    a partial run is visible rather than silently zero.
    """
    mean = np.full((nkinds, ntau), np.nan)
    err = np.full((nkinds, ntau), np.nan)
    for kind in range(nkinds):
        for itau in range(ntau):
            key = f"{prefix}{kind}t{itau}"
            if key in results:
                mean[kind, itau], err[kind, itau] = results[key]
    return mean, err
