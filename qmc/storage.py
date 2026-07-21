"""Write QMC results in the Chebyshev HDF5 schema.

Mirrors Algorithm/Storage_Concept/Storage_Concept.cpp so that the existing
Plot_*.py scripts read QMC output unchanged:

    /parameters                       (values stored as HDF5 attributes)
    /results/Re_correlation/<i>-0     (nrows, num_TimePoints)
    /results/Im_correlation/<i>-0
    /results/Re_stds/<i>-0
    /results/Im_stds/<i>-0
    /runtime_data
"""

import os

import h5py
import numpy as np


def build_filename(spin_model, couplings_name, spin_sites, beta, h_z=0.0,
                   rescale=1.0, extension="", J=1.0):
    """Reproduce Storage_Concept::create_file's naming scheme.

    ``J`` is not part of the C++ scheme (there the couplings live in the named
    source file), but it is a run parameter here, so it enters the name whenever
    it differs from 1 to keep a ferromagnet from overwriting an antiferromagnet.
    """
    name = f"{spin_model}__{couplings_name}"
    if not (len(spin_sites) == 1 and spin_sites[0] == 0):
        name += "__sites=" + "-".join(str(s) for s in spin_sites)
    name += f"__beta={_fmt(beta)}"
    if J != 1.0:
        name += f"__J={_fmt(J)}"
    if h_z != 0.0:
        name += f"__h_z={_fmt(h_z)}"
    if rescale != 1.0:
        name += f"__rescale={_fmt(rescale)}"
    if extension:
        name += f"__{extension}"
    return name + ".hdf5"


def _fmt(x):
    """Format like C++ ostream: integers without a trailing '.0'."""
    x = float(x)
    return str(int(x)) if x == int(x) else repr(x)


MAX_TRIES = 5


def resolve_path(path, max_tries=MAX_TRIES):
    """Pick a free filename, appending 'X' to the stem as the C++ writer does.

    Tries ``name.hdf5``, ``nameX.hdf5``, ... up to ``max_tries - 1`` X's and
    returns the first that does not exist. When all of them are taken the last
    candidate is returned and will be overwritten, so a runaway scan cannot
    accumulate files without bound. (The C++ writer throws at this point
    instead; overwriting is the behaviour requested here.)

    Returns ``(path, n_suffix)``.
    """
    root, ext = os.path.splitext(path)
    for n in range(max_tries):
        candidate = f"{root}{'X' * n}{ext}"
        if not os.path.exists(candidate):
            return candidate, n
    return f"{root}{'X' * (max_tries - 1)}{ext}", max_tries - 1


def store(path, params, spin_sites, corr_re, corr_im, stds_re, stds_im,
          runtime=None):
    """Write one QMC result file.

    corr_* are indexed [site, row, tau]; ``spin_sites`` names the site of each
    leading index.
    """
    corr_re, corr_im = np.asarray(corr_re), np.asarray(corr_im)
    stds_re, stds_im = np.asarray(stds_re), np.asarray(stds_im)
    if len(spin_sites) != corr_re.shape[0]:
        raise ValueError(
            f"{len(spin_sites)} sites requested but correlation array has "
            f"{corr_re.shape[0]} entries"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        pgroup = f.create_group("parameters")
        for key, value in params.items():
            pgroup.attrs[key] = value

        results = f.create_group("results")
        groups = {
            "Re_correlation": corr_re,
            "Im_correlation": corr_im,
            "Re_stds": stds_re,
            "Im_stds": stds_im,
        }
        infos = {
            "Re_correlation": "Real part of correlations <S^a_i(t)S^b_0(0)> for a given site",
            "Im_correlation": "Imaginary part of correlations <S^a_i(t)S^b_0(0)> for a given site",
            "Re_stds": "Standard deviations of the real part of correlations for a given site",
            "Im_stds": "Standard deviations of the imaginary part of correlations for a given site",
        }
        for name, array in groups.items():
            group = results.create_group(name)
            for index, site in enumerate(spin_sites):
                dset = group.create_dataset(
                    f"{site}-0", data=np.ascontiguousarray(array[index], dtype="float64")
                )
                dset.attrs["info"] = infos[name]

        rgroup = f.create_group("runtime_data")
        for key, value in (runtime or {}).items():
            rgroup.attrs[key] = value
    return path
