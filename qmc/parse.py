"""Readers for the DSQSS/DLA output files.

Result lines everywhere look like ``R <name> = <mean> <error>``. With the patch
applied, ``cf.dat`` carries all three measured channels on the same displacement
classes: ``C<kind>t<itau>`` (transverse, symmetric), ``A<kind>t<itau>``
(transverse, antisymmetric) and ``D<kind>t<itau>`` (real-space S^z S^z).
"""

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


def write_site_displacements(path, lat, sites):
    """Write a ``disp.xml`` covering only the requested displacement classes.

    DSQSS's own generator enumerates every ordered site pair and bins them into
    roughly N displacement classes, so both the file and the per-step
    accumulator loop are O(N^2). Nothing needs that: a run asks for a handful of
    sites, and every other class is measured and discarded.

    Here class ``k`` is the displacement from site 0 to ``sites[k]``, and the
    file lists every ordered pair sharing that displacement -- N pairs per
    class on a translation-invariant lattice, so the site averaging and the
    ``NR[kind]`` normalization inside DSQSS are unchanged.
    """
    # Built by translating each source site by the requested displacement,
    # rather than scanning all N^2 pairs and testing each: the partner site is
    # determined by arithmetic, so this is O(N * len(sites)).
    size = np.asarray(lat.size, dtype=np.int64)
    strides = np.cumprod(np.concatenate(([1], size[:-1])))
    entries = []
    for kind, site in enumerate(sites):
        shifted = (lat.coords + np.asarray(lat.displacement_vector(0, site))) % size
        partners = shifted @ strides
        entries.extend((kind, i, int(j)) for i, j in enumerate(partners))

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<Displacements>\n")
        f.write(f"<Comment> {lat.name}, sites "
                f"{','.join(str(s) for s in sites)} </Comment>\n")
        f.write(f"<NumberOfKinds> {len(sites)} </NumberOfKinds>\n")
        f.write(f"<NumberOfSites> {lat.nsites} </NumberOfSites>\n")
        f.write("\n<!-- <R> [kind] [isite] [jsite] </R> -->\n\n")
        for kind, i, j in entries:
            f.write(f"<R> {kind} {i} {j} </R>\n")
        f.write("</Displacements>\n")
    return len(entries)


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
