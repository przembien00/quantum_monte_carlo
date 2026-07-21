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
``patches/dsqss_estimators.patch`` and appears as ``A<kind>t<itau>`` entries.

C^zz is measured directly in real space by the same patch, on the same
displacement classes, and appears as ``D<kind>t<itau>``. It needs no conversion
at all -- ``convert`` reads it straight through.

Earlier versions obtained C^zz by inverse-Fourier-transforming the structure
factor over the full Brillouin zone. That was correct but cost O(N^2) per
measurement and dominated everything past a few hundred sites; the real-space
estimator is linear in N. See docs/estimators.md.
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
