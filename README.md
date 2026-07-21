# Worldline QMC for spin systems

Continuous-time worldline quantum Monte Carlo for Heisenberg spin systems,
writing HDF5 output in the same schema as the Chebyshev typicality code
(`../chebyshev_typicality`) so the existing `Plot_*.py` scripts read it
unchanged.

The QMC engine is [DSQSS/DLA](https://github.com/issp-center-dev/dsqss) 2.0.6
(ISSP, GPLv3), vendored in `external_dsqss/` with one local patch. Everything in
`qmc/` is the conversion layer around it.

## Scope

Works on **periodic, bipartite** lattices: chains, squares and cubes with even
side lengths, antiferromagnetic (`J > 0`) or ferromagnetic (`J < 0`), with an
optional longitudinal field `h_z`. Frustrated and random-sign couplings have a
sign problem and are out of reach — keep using Chebyshev typicality for those.
QMC gives imaginary time only; real-time data needs analytic continuation.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install toml numpy scipy h5py matplotlib

git clone --depth 1 https://github.com/issp-center-dev/dsqss.git external_dsqss
cd external_dsqss
git apply ../patches/cf_antisymmetric.patch     # enables the C^xy channel
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=$PWD/../dsqss_install \
  -DPython3_EXECUTABLE=$PWD/../venv/bin/python \
  -DTesting=OFF
cmake --build build -j8 && cmake --install build
cd ..

# let the venv find the installed dsqss python package
echo "$PWD/dsqss_install/lib/python3.13/site-packages" \
  > "$(./venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/dsqss.pth"
```

## Running

```bash
./venv/bin/python run_qmc.py \
  --lattice=square:4x4 --beta=3 --num_TimePoints=100 \
  --sites=0,1,5 --h_z=0.0 --cores=4
```

Writes `Data/ISO__Square_NN_PBC_N=16__sites=0-1-5__beta=3.hdf5`.

### Where the data goes

Output is written to **`<data-dir>/<project>/<generated filename>`**, where
`--data-dir` defaults to `Data` **relative to the current working directory**.
Running from the repository root therefore fills
`quantum_monte_carlo/Data/`, which is a *different* directory from the Chebyshev
code's `Data/`. Two ways to put both methods' files side by side for comparison
plots:

```bash
# absolute path into the Chebyshev tree
--data-dir=/Users/przembien/Projects/chebyshev_typicality/Data --project=QMC

# or just run from there
cd ../chebyshev_typicality && ../quantum_monte_carlo/venv/bin/python \
    ../quantum_monte_carlo/run_qmc.py --lattice=square:4x4 --beta=3
```

Directories are created as needed. The filename is generated from the run
parameters by the same rule the C++ writer uses:

```
ISO__<lattice>[__sites=i-j-k]__beta=<b>[__J=<J>][__h_z=<h>][__<extension>].hdf5
```

Optional fragments appear only when they differ from the default (`sites` when
it is not just site 0, `J` when not 1, `h_z` when not 0), so an antiferromagnet
and a ferromagnet at the same `beta` do not collide.

**Existing files are not clobbered.** As in the Chebyshev writer, an `X` is
appended to the stem until a free name is found:

```
ISO__..._beta=3.hdf5  ->  ..._beta=3X.hdf5  ->  ...XX  ->  ...XXX  ->  ...XXXX
```

Once all five variants exist (base plus four `X`s) the last one is overwritten,
so a repeated scan cannot fill the disk. The run reports which name it chose.
Use `--extension` or `--project` to keep variants apart deliberately rather than
relying on the `X` chain.

(The C++ writer raises an error at this point instead of overwriting; the QMC
side overwrites the four-`X` file.)

### All options

| Option | Default | Meaning |
|---|---|---|
| `--lattice` | *required* | `chain:L`, `square:LxL`, `cube:LxLxL`. A single number expands over all dimensions (`square:4` = `square:4x4`); explicit forms may be anisotropic (`square:8x4`). Always periodic. |
| `--beta` | *required* | Inverse temperature. |
| `--num_TimePoints` | `100` | Number of imaginary-time points on `[0, beta)`; sets `delta_t = beta/ntau`. |
| `--sites` | `0` | Comma-separated target sites, each correlated with site 0. |
| `--h_z` | `0.0` | Uniform longitudinal field. Nonzero makes `C^xx != C^zz` and turns on the purely imaginary `C^xy`. |
| `--J` | `1.0` | Coupling. **Positive is antiferromagnetic**, negative is ferromagnetic. |
| `--cores` | `1` | MPI ranks (`mpirun -n`). Ranks are independent Markov chains, combined in the error analysis. |
| `--seed` | `31415` | Random seed. Vary it for independent repeats. |
| `--nmcs` | `20000` | Monte Carlo sweeps per set (measurement statistics). |
| `--nset` | `10` | Number of sets; error bars come from the scatter between sets, so keep this at 10 or more. |
| `--ntherm` | `1000` | Thermalization sweeps discarded before measurement. |
| `--data-dir` | `Data` | Output root; see above. |
| `--project` | *(none)* | Subdirectory under the output root. |
| `--extension` | *(none)* | Suffix appended to the filename. |
| `--write-couplings PATH` | *(none)* | Also write the matching `J_ij` file in Chebyshev `Couplings/` format. Same lattice object drives both, so site indexing agrees by construction. |
| `--keep-workdir` | off | Keep the scratch directory with the raw DSQSS input and output (`std.toml`, `cf.dat`, `disp.xml`, ...) instead of deleting it. Useful for debugging. |
| `--quiet` | off | Suppress the parameter banner and progress output; only the written path is printed. |

### Progress output

A run prints the parameters it is about to use, then per-set progress as the
sampler works, then the equal-time values:

```
========================================================================
 worldline QMC  (DSQSS/DLA)
========================================================================
  model           isotropic Heisenberg  S = 1/2
  lattice         Square_NN_PBC_N=16  4x4, periodic
  spins           16
  coupling        J = +1  antiferromagnetic
  field           h_z = 0.3
  temperature     beta = 2
  imaginary time  32 points  delta_tau = 0.0625
  sites           0, 1, 5  each correlated with site 0
  sampling        8 x 20000 sweeps  1000 thermalization
  parallel        4 MPI ranks
  seed            31415
========================================================================
  calibrated worm cycles per sweep: 4
  [##..................] set   1/8   12%  elapsed     1s  remaining     9s
  ...
  [####################] set   8/8  100%  elapsed    11s  remaining     0s
  sampling complete in 11s
  wrote Data/ISO__Square_NN_PBC_N=16__sites=0-1-5__beta=2__h_z=0.3.hdf5
```

Key values are bold, the heading and progress bar coloured, completion green and
warnings amber; everything else stays the terminal's default colour. There is
deliberately no faint/dim text, which is hard to read on low-contrast palettes.
Colour is dropped automatically for pipes and batch logs, and honours
[`NO_COLOR`](https://no-color.org) and `TERM=dumb`; styling is applied after
padding, so column alignment is identical either way.

The destination is reported **after** the run rather than up front, since the
name is only settled once the results are written (see the `X` suffix rule
above). In `--quiet` mode the bare path is printed on its own, so
`path=$(run_qmc.py ... --quiet)` works in scripts.

Progress comes from the sampler itself and appears **live**, one line per
completed set, including when stdout is a pipe or a batch log — DSQSS flushes
each line explicitly. There are no carriage returns or redraws, so HPC job logs
stay readable. Error bars only become meaningful once several sets have
finished, so `--nset` also sets the progress granularity.

A warning is printed if any error bar comes back non-finite. DSQSS forms the
error as `sqrt(<x^2> - <x>^2)`, and with few sets on a barely fluctuating
quantity that difference can go slightly negative through cancellation, making
the root NaN. The correlations themselves are unaffected — raise `--nset` for a
stable variance estimate. This is realistic at `--nset=2` and not seen at the
default of 10.

### Site numbering

`--sites` takes flat integer indices in `[0, N)`. Every site is correlated with
**site 0**, which is the origin, so `--sites=0,1,5` gives
`<S_0(tau) S_0(0)>`, `<S_1(tau) S_0(0)>` and `<S_5(tau) S_0(0)>`, stored as
datasets `0-0`, `1-0` and `5-0`.

The index is a **row-major (x-fastest) unpacking of the coordinates**:

```
index = x + Lx*y + Lx*Ly*z        <->        x = i % Lx,  y = (i // Lx) % Ly, ...
```

so for `square:4x4` the lattice reads left to right, bottom to top:

```
  y=3 | 12 13 14 15
  y=2 |  8  9 10 11
  y=1 |  4  5  6  7
  y=0 |  0  1  2  3
        x=0 x=1 x=2 x=3
```

This matches DSQSS's own `index2coord`, and — because the same
`PeriodicLattice` object also writes the Chebyshev coupling file via
`--write-couplings` — it matches the site numbering in that file too. The two
codes therefore agree on what "site 5" means by construction.

**Which sites to pick.** Correlations depend only on the displacement from site
0, so useful choices are one representative per distance. For `square:4x4`:

| `r^2` | Equivalent sites | Meaning |
|---|---|---|
| 0 | 0 | local (on-site) |
| 1 | **1**, 3, 4, 12 | nearest neighbour |
| 2 | **5**, 7, 13, 15 | diagonal |
| 4 | **2**, 8 | second neighbour along an axis |
| 5 | **6**, 9, 11, 14 | knight's-move |
| 8 | 10 | far corner |

`--sites=0,1,5,2,6,10` covers every distinct distance on this lattice.

Sites within a row are related by the lattice point group, so they are
*physically* identical but sit in **different displacement bins** and are
measured independently. Requesting several of them is therefore a free
consistency check (they must agree within error bars), and averaging them
improves statistics. Verified on `square:4x4` at `beta=2`: sites 1, 3, 4 and 12
returned `-0.10810`, `-0.10821`, `-0.10819`, `-0.10828` with error bars of
about `8e-5`.

To inspect the layout for any lattice:

```python
from qmc import lattice
lat = lattice.build("cube:4x4x4")
lat.coords[5]                    # coordinates of site 5
lat.displacement_vector(0, 5)    # minimum-image displacement from site 0
```

### Negative (ferromagnetic) couplings

Pass `--J` a negative value. Both argparse forms work, but prefer the `=` form,
since a bare `-1.0` after a space is only parsed correctly because it looks like
a number rather than a flag:

```bash
./venv/bin/python run_qmc.py --lattice=square:4x4 --beta=3 --J=-1.0
```

The sign propagates in two places, both handled automatically: the DSQSS
Hamiltonian is written with `Jz = Jxy = -J`, and the Marshall staggering
correction (`docs/estimators.md`) is applied **only** for `J > 0`, since the
ferromagnet needs no sublattice rotation to be sign-free. Validated against
exact diagonalization for both signs.

Note that a ferromagnet is sign-free on *any* lattice, so the bipartite
restriction below applies only to `J > 0`.

### Dimensionality

All three are implemented and validated against exact diagonalization:

| Spec | Example | Notes |
|---|---|---|
| 1D | `chain:16` | |
| 2D | `square:4x4`, `square:8x4` | |
| 3D | `cube:4x4x4` | Works; a 64-site cube at `beta=1`, `ntau=16` takes about 18 s on 4 ranks. |

Side lengths must be **even** for `J > 0` (odd rings close on a frustrated bond,
which is not bipartite and has a sign problem). This is checked and rejected with
an explanation rather than producing wrong numbers. A length-2 direction is
allowed but unusual: periodic boundaries then place two bonds between the same
pair, giving an effective `2J` there. The exported coupling file reflects this,
so both codes still see the same model.

## Output

Identical in structure to the Chebyshev output:

```
/parameters                     (attributes: beta, num_Spins, num_TimePoints, h_z, ...)
/results/Re_correlation/<i>-0   (nrows, num_TimePoints)
/results/Im_correlation/<i>-0
/results/Re_stds/<i>-0
/results/Im_stds/<i>-0
/runtime_data
```

Every run writes **four rows in the fixed order `[xx, xy, yx, zz]`**, at any
field. Both the transverse and longitudinal channels are measured on every run,
so collapsing to a single row at `h_z = 0` would throw away the independent
`C^zz` estimate — which is the cheapest check available on `C^xx`, since
isotropy requires the two to agree. The `correlation_rows` attribute states the
layout explicitly; nothing needs to be inferred from a symmetry class.

`C^xx` and `C^zz` are purely real and symmetric about `beta/2`; `C^xy` is purely
imaginary and *antisymmetric*, and vanishes at zero field (`C^yx = -C^xy`). Note
the antisymmetry when mirroring `[0, beta/2]` data to the full interval: the
`xy` row flips sign where the others do not.

Beyond the C++ attribute set, `correlation_rows`, `method`, `J` and
`lattice_size` are written so the layout and the model are self-describing. A
Chebyshev ISO run at zero field stores a single row, so code reading both
sources should branch on `correlation_rows` rather than assume a row count.

Correlations are site-averaged over all pairs sharing a displacement vector,
which is exact on a translation-invariant lattice and buys roughly a factor N in
statistics.

## Validation

`validate_ed.py` builds the same Hamiltonian by exact diagonalization and
compares every component:

```bash
./venv/bin/python validate_ed.py --lattice=chain:8 --beta=2.0 --h_z=1.0 --nmcs=200000
```

Agreement is at the 1e-4 level for `xx`, `zz` and `xy`, at zero and finite
field, for both signs of `J`, in 1D, 2D and 3D. See `docs/estimators.md` for the
estimator conventions and the four corrections the converter applies.

When touching the `C^zz` path, validate on a lattice with **two or more sides of
length >= 3** — `square:3x3 --J=-1` is the standard case. Shorter sides make
`k -> -k` trivial and hide Brillouin-zone coverage bugs; an earlier suite of
`square:4x2` and `cube:2x2x2` passed while `C^zz` was wrong.

## Troubleshooting

**`RuntimeWarning: divide by zero / overflow / invalid value encountered in
matmul`.** Spurious, and suppressed as of the current version. numpy built
against Apple's Accelerate BLAS (the default on macOS via pip) leaves
floating-point exception flags set inside its vectorized kernels, and numpy
reports them after the call even when every input and output is finite —
confirmed by feeding finite random input, where the warning fires and the result
still matches `einsum` to 7e-15. It only appears above the size where numpy
hands off to the BLAS, which is why small runs are quiet and larger ones are not.

The code now checks the structure factor for finiteness explicitly, and checks
the transformed result again afterwards, before silencing the warning in that
one block. So a genuine non-finite value still raises a clear error rather than
being masked. If you would rather not rely on that, installing a numpy built
against OpenBLAS removes the warning at source.
