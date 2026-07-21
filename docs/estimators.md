# DSQSS estimators and the corrections applied

Notes on what DSQSS/DLA actually measures and the four corrections the
converter applies. Each was established empirically against exact
diagonalization, not assumed; `validate_ed.py` re-checks all of them.

## Where each component comes from

| Component | Source | File |
|---|---|---|
| `C^xx = C^yy` | worm head-tail displacement histogram | `cf.dat`, `C<kind>t<itau>` |
| `C^xy = -C^yx` | same histogram weighted by worm-head type (patched) | `cf.dat`, `A<kind>t<itau>` |
| `C^zz` | inverse Fourier transform of `S^zz(k, tau)` | structure-factor output, `C<ik>t<itau>` |

`disp.xml` assigns an ordered site pair to a displacement bin, and DSQSS
normalizes by the number of pairs per bin, so the measured quantity is already
averaged over reference sites.

## 1. The stock transverse estimator is symmetrized

DSQSS accumulates the head-tail displacement without recording whether the head
is `S+` or `S-`, so `cf.dat` holds

    cf(r, tau) = 1/2 [ G^{+-}(r, tau) + G^{-+}(r, tau) ] = 2 C^xx(r, tau)

Evidence: at `h_z = 1.5` on a 4-site chain with magnetization `<m_z> = 0.240`,
`cf(0, 0) = 0.4995`. The two orderings give `G^{+-}(0,0) = 1/2 + <m_z> = 0.740`
and `G^{-+}(0,0) = 1/2 - <m_z> = 0.260`; the measured value is their mean,
independent of field.

Consequence: `C^xx` is available from the stock build, but `C^xy` — the
antisymmetric combination, which is nonzero only at finite field — is not.
`patches/cf_antisymmetric.patch` adds it.

## 2. The Marshall gauge staggers the transverse correlator

`algorithm.py:255` takes `abs()` of the vertex weights, which is what makes a
bipartite antiferromagnet sign-free (`sign = 1.0` exactly). The worm therefore
measures the transverse correlator in the rotated frame, related to the physical
one by `(-1)^(sum of displacement components)`.

Evidence: at `h_z = 0` the isotropic model requires `C^xx = C^zz`. Before the
correction, on a chain, sites 0 and 2 agreed while site 1 had matching magnitude
and opposite sign (`+0.0678` vs `-0.0679`).

`C^zz` is diagonal and gauge-invariant, so it needs no correction. Applied in
`lattice.PeriodicLattice.marshall_sign`, only for `J > 0`.

## 3. The antisymmetric channel is time-reversed

The worm bins the head-tail time displacement with the opposite orientation to
the convention used here: measured `A[it]` corresponds to `tau = beta - it*dtau`.
This is invisible in the symmetric channel, which is even about `beta/2`, but it
negates the antisymmetric one.

Evidence: `A[(ntau - it) % ntau]` reproduces the exact `Im C^xy` at every `tau`,
while `A[it]` matches only at `tau = 0`. Applied in
`correlations._reverse_tau`.

## 4. The half Brillouin zone is not a fundamental domain in 2D or 3D

DSQSS's wavevector generator emits the product set `{0..L/2}` per dimension and
relies on `S(k) = S(-k)` to cover the rest. That is a valid fundamental domain
for `k -> -k` in one dimension, but **not** in two or more: on a 4x4 lattice the
orbit `{(1,3), (3,1)}` lies entirely outside the product set, so those two k are
never measured and the back-transform to real space silently loses their
contribution. On 3x3 the missing orbit is `{(1,2), (2,1)}`.

Evidence: at `h_z = 0` isotropy requires `C^xx = C^zz`. On `square:4x4` the two
differed by `0.021` against error bars of `1e-4` — a 261-sigma discrepancy,
visible in the third digit. With the full zone measured the same comparison
gives `0.0001`, or 1.5 sigma.

The fix measures **every** k in the zone (`parse.write_full_bz_wavevectors`
overwrites the `wv.xml` that `dla_pre` generates), which makes the inverse
transform exact with uniform weight and removes the multiplicity bookkeeping
entirely. A partially populated structure factor is now a hard error rather than
propagating NaN.

### Why this survived the first round of validation

The original 2D and 3D validation cases were `square:4x2` and `cube:2x2x2`.
Every potentially-problematic direction in those has length 2, where `k -> -k`
is trivial and the product set happens to cover the whole zone. They were
degenerate in exactly the way that hides this bug.

A meaningful test of the `C^zz` path needs **two or more sides of length >= 3**.
Since an odd side makes the antiferromagnet frustrated, the cheap option is the
ferromagnet, which is sign-free on any lattice: `square:3x3` with `--J=-1` is
9 sites, validates in about 20 seconds, and does miss an orbit under the old
scheme. It is now the standard 2D regression case.

## Bond multiplicity along short directions

Along a periodic direction of length 2 the `+1` and `-1` neighbours are the same
site, and DSQSS lays down two bonds — an effective `2J`. `coupling_matrix`
accumulates rather than assigns so the exported coupling file describes the same
model. Only relevant to small test lattices; production sizes have all side
lengths at least 4.

## The patch

`patches/cf_antisymmetric.patch` (against DSQSS 2.0.6) adds a second, signed
histogram to `CF`, giving `1/2 [G^{+-} - G^{-+}]` alongside the existing
symmetric one, and reported as `A<kind>t<itau>`.

The head type is the sign of the spin change across the head, evaluated at the
start of each step in `UP_ONESTEP` / `DOWN_ONESTEP` (before the move updates the
worm state):

```
UP:   headtype = c_S.X() - xinc     // traversed region below carries xinc
DOWN: headtype = xinc - c_S.X()     // traversed region above carries xinc
```

A first attempt used the direction of motion (`+1` up, `-1` down) instead. That
is wrong: a single worm moves both ways during its life, so the channel averaged
to zero (measured `6e-5` where exact diagonalization gives `-0.077`). The head
type is a property of the local worldline configuration, not of the direction of
travel.

Unpatched builds still work; the converter detects the missing `A` entries and
leaves `C^xy` zero.
