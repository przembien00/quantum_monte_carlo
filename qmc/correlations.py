"""Turn DSQSS estimators into the Chebyshev correlation tensor.

Conventions
-----------
``cf.dat`` holds the worm-measured transverse correlator. Empirically (see
``docs/estimators.md``) DSQSS averages over both worm-head types, so

    cf(r, tau) = 1/2 [ G^{+-}(r, tau) + G^{-+}(r, tau) ]

with G^{+-}(r, tau) = <S^+(r, tau) S^-(0, 0)>. Because
S^x S^x + S^y S^y = 1/2 (S^+S^- + S^-S^+),

    C^xx(r, tau) = C^yy(r, tau) = cf(r, tau) / 2.

The antisymmetric combination 1/2[G^{+-} - G^{-+}], which carries C^xy, is not
separable from the stock estimator; it requires the patch in
``patches/cf_antisymmetric.patch`` and appears as ``A<kind>t<itau>`` entries.

``sfoutfile`` holds S^zz(k, tau) on half the Brillouin zone; the real-space
C^zz(r, tau) is recovered by an inverse Fourier transform using S(k) = S(-k).
"""

import numpy as np

# Row order of Chebyshev's symmetry class 'C' (see Algorithm/Types/Tensors.h).
# Every run emits this layout, at any field: both channels are always measured,
# and the class-A collapse at zero field would discard the independent C^zz.
CLASS_C_ROWS = ("xx", "xy", "yx", "zz")


def transverse_from_cf(cf_mean, cf_err):
    """C^xx = C^yy from the stock cf.dat estimator. Shapes (nkinds, ntau)."""
    return 0.5 * cf_mean, 0.5 * np.abs(cf_err)


def _reverse_tau(a):
    """Map index it -> (ntau - it) % ntau along the last axis."""
    return np.roll(a[..., ::-1], 1, axis=-1)


def xy_from_antisymmetric(af_mean, af_err):
    """Im C^xy from the patched antisymmetric estimator.

    With af = 1/2 [G^{+-} - G^{-+}] and
    C^xy = (1/4i)[G^{-+} - G^{+-}] = (i/4)[G^{+-} - G^{-+}],
    the correlator is purely imaginary with Im C^xy = af / 2.

    The worm bins the head-tail time displacement with the opposite orientation
    to the convention used here, so the tau axis is reversed. That reversal is
    invisible in the symmetric channel (which is even about beta/2) but negates
    this one, so it must be undone explicitly. Verified against exact
    diagonalization in validate_ed.py.
    """
    return 0.5 * _reverse_tau(af_mean), 0.5 * np.abs(_reverse_tau(af_err))


def zz_from_structure_factor(sf_mean, sf_err, kvecs, size, coords):
    """Inverse-Fourier S^zz(k, tau) to C^zz(r_i, tau) for every site.

    ``kvecs`` must span the full Brillouin zone, so every k enters with weight
    one and no multiplicity bookkeeping is needed. Returns arrays of shape
    (nsites, ntau).
    """
    size = np.asarray(size, dtype=np.int64)
    nsites = int(np.prod(size))
    if len(kvecs) != nsites:
        raise ValueError(
            f"expected {nsites} wavevectors spanning the full Brillouin zone, "
            f"got {len(kvecs)}. A partial zone silently drops the k orbits it "
            "does not cover; see parse.wavevector_list."
        )
    ntau = sf_mean.shape[1]

    # Checked here rather than relying on floating-point warnings from the
    # matmul below, which on some BLAS backends fire spuriously (see errstate).
    if not np.all(np.isfinite(sf_mean)) or not np.all(np.isfinite(sf_err)):
        raise ValueError(
            "structure factor contains non-finite values, so C^zz cannot be "
            "computed. This usually means dla did not finish writing its "
            "output, or the run diverged."
        )

    czz = np.zeros((nsites, ntau))
    czz_err = np.zeros((nsites, ntau))
    # numpy built against Apple's Accelerate raises spurious divide/overflow/
    # invalid warnings from matmul: the library leaves floating-point exception
    # flags set by its vectorized kernels (padding lanes included) and numpy
    # reports them afterwards. Verified on finite random input that the warning
    # fires while the result still agrees with einsum to 7e-15. The inputs are
    # checked for finiteness above, so nothing real is being masked here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for isite in range(nsites):
            weight = np.cos(
                2.0 * np.pi * np.sum(kvecs * coords[isite] / size, axis=1)
            ) / nsites
            czz[isite] = weight @ sf_mean
            # Distinct k are independent accumulators, so errors add in quadrature.
            czz_err[isite] = np.sqrt((weight**2) @ (sf_err**2))

    if not np.all(np.isfinite(czz)) or not np.all(np.isfinite(czz_err)):
        raise RuntimeError(
            "the transform to real space produced non-finite C^zz from finite "
            "input, which should not happen; do not trust this run."
        )
    return czz, czz_err


def mirror_to_full_beta(values, antisymmetric=False):
    """Extend a [0, beta) series to its value at tau = beta.

    C^zz and C^xx satisfy C(beta - tau) = C(tau); C^xy is antisymmetric.
    """
    tail = -values[..., :1] if antisymmetric else values[..., :1]
    return np.concatenate([values, tail], axis=-1)


def build_tensor(nrows, ntau, nsites):
    """Zero-filled (nsites, nrows, ntau) real and imaginary tensors."""
    return (
        np.zeros((nsites, nrows, ntau)),
        np.zeros((nsites, nrows, ntau)),
    )
