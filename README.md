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
./install.sh
```

Builds everything into the repository: `external_dsqss/` (patched source),
`dsqss_install/` (binaries), `venv/` (Python). It checks prerequisites, applies
the estimator patch, builds, links the `dsqss` package into the venv, and
finishes with a smoke test that asserts `C^xx(0) = 1/4` exactly, `C^xx = C^zz`
at zero field, and that all three channels are present.

`JOBS=16 ./install.sh` sets the build parallelism; `PYTHON=python3.11
./install.sh` picks the interpreter; `DSQSS_VERSION=v2.0.6` pins the engine.

### On a compute cluster

Run the installer **on a login node** — compute nodes are usually offline and
the build clones DSQSS and pip-installs wheels. Load your site's modules first;
the names vary, so check `module avail`:

```bash
module load gcc openmpi cmake python     # site-specific names
mkdir -p logs && ./install.sh
```

Two properties of the build constrain how you use the result, and both bite
silently if ignored:

- **`dla` is dynamically linked against the MPI you build with.** Batch jobs
  must load the *same* MPI module as the install did. The installer prints the
  library it linked against; a mismatch usually shows up as a loader error, but
  can also produce a job that runs on one rank while claiming many.
- **The helper scripts carry an absolute shebang to `venv/bin/python`.** The
  directory cannot be moved after installing — rerun `./install.sh` instead.
  This also means a venv on `$HOME` and a repository on scratch will break if
  scratch is purged.

If your `$HOME` has a small quota, put the whole repository on a work
filesystem; the install is ~200 MB, dominated by the DSQSS checkout.

### Batch jobs

`submit_slurm.sh` is a working SLURM array-job template. Submit with:

```bash
mkdir -p logs && sbatch submit_slurm.sh
```

The structure follows from how the sampler parallelises. **MPI ranks are
independent Markov chains**, combined only at the end, so ranks buy statistics,
not speed — measured at 19.2 s on 1 rank versus 21.6 s on 4 for the same job.
Therefore:

| axis | use it for |
|---|---|
| ranks within a task (`--ntasks`) | error bars on **one** parameter point (~1/sqrt(ranks)) |
| array tasks (`--array`) | **different** parameter points, running concurrently |

Sixteen ranks gives 4x smaller error bars than one; much beyond that trades
poorly against queue wait. Seeds are drawn fresh per run and offset per rank, so
array tasks are independent without any action; set `--seed` only when you want
a scan to be reproducible.

## Running

```bash
./venv/bin/python run_qmc.py \
  --lattice=square:4x4 --beta=3 --num_TimePoints=100 \
  --sites=0,1,2 --h_z=0.0 --cores=4
```

Writes `Data/ISO__Square_NN_PBC_N=16__sites=0-1-5__beta=3.hdf5`
(shells 0, 1, 2 resolve to sites 0, 1, 5 on this lattice).

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
| `--sites` | `0` | Comma-separated **neighbour shells**: 0 on-site, 1 nearest, 2 next-nearest, ... See below. |
| `--h_z` | `0.0` | Uniform longitudinal field. Nonzero makes `C^xx != C^zz` and turns on the purely imaginary `C^xy`. |
| `--J` | `1.0` | Coupling. **Positive is antiferromagnetic**, negative is ferromagnetic. |
| `--cores` | `1` | MPI ranks (`mpirun -n`). Ranks are independent Markov chains, combined in the error analysis. |
| `--seed` | *fresh each run* | Random seed. Omit for an independent run; the value used is printed and stored in `parameters/seed`, so passing it back reproduces that run exactly. |
| `--nmcs` | `20000` | Monte Carlo sweeps per set (measurement statistics). |
| `--nset` | `10` | Number of sets; error bars come from the scatter between sets, so keep this at 10 or more. |
| `--ntherm` | `1000` | Thermalization sweeps discarded before measurement. |
| `--data-dir` | `Data` | Output root; see above. |
| `--project` | *(none)* | Subdirectory under the output root. |
| `--extension` | *(none)* | Suffix appended to the filename. |
| `--write-couplings PATH` | *(none)* | Also write the matching `J_ij` file in Chebyshev `Couplings/` format. Same lattice object drives both, so site indexing agrees by construction. |
| `--site-indices` | off | Interpret `--sites` as raw site indices instead of shells. |
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

### Seeds and reproducibility

Each run draws a fresh seed from OS entropy, so repeating a run genuinely adds
statistics rather than reproducing the same chain. The seed is printed in the
banner and stored in the output, so any run can be replayed exactly:

```bash
./venv/bin/python run_qmc.py ... --seed=748173299
```

MPI ranks are seeded independently — `dla` adds the rank index to the seed — so
the chains within a run are distinct as well.

A warning is printed if any error bar comes back non-finite. DSQSS forms the
error as `sqrt(<x^2> - <x>^2)`, and with few sets on a barely fluctuating
quantity that difference can go slightly negative through cancellation, making
the root NaN. The correlations themselves are unaffected — raise `--nset` for a
stable variance estimate. This is realistic at `--nset=2` and not seen at the
default of 10.

### Site numbering

`--sites` selects **neighbour shells** measured from site 0, ordered by
distance: `0` is the on-site autocorrelation, `1` nearest neighbour, `2`
next-nearest, and so on. This is lattice-independent — `--sites=0,1,2` means the
same physics on a chain, a square or a cube, with no need to know which index
happens to sit where.

Shells are ordered by squared Euclidean distance, the usual condensed-matter
convention, so on a square lattice shell 2 is the diagonal `(1,1)` rather than
`(2,0)` (which is shell 3).

The run prints the resolved mapping:

```
  correlations    site 0 = shell 0 (on-site) (0, 0)   displacement from site 0
                  site 1 = shell 1 (nearest) (1, 0)
                  site 7 = shell 2 (next-nearest) (1, 1)
```

**Output is still keyed by site index** — datasets are `0-0`, `1-0`, `7-0` and
the filename says `sites=0-1-7` — which keeps the files interchangeable with
Chebyshev output. Three attributes record all three views: `spin_sites`,
`spin_shells` and `spin_displacements`.

Use `--site-indices` to go back to raw indices, e.g. to target a specific site
on a lattice where the shell is not what you want.

#### A caveat on anisotropic lattices

A shell groups sites at equal distance, but on a lattice whose sides differ
those are not always related by a symmetry. On a `6x4` torus the nearest
neighbours `(1,0)` and `(0,1)` sit at the same distance yet are physically
distinct, because the two directions have different periodicities. The
measurement always uses the representative displacement shown in the banner, and
the run says so:

```
  note: on this lattice shell 1 contains displacements that are not
  symmetry-equivalent (the sides have different lengths); the one shown
  above is measured.
```

On a cube or square with equal sides this never arises.

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

## Performance and system size

The sampling is linear in the number of sites and linear in `beta`. MPI ranks
run **independent Markov chains**, so more cores buys smaller error bars
(~1/sqrt(ranks)), not shorter wall time -- measured at 19.2 s on 1 core versus
21.6 s on 4 for the same job.

Measured, 3D cube at `beta=1`, `ntau=16`, 1 rank: ~3.6 us per site per sweep,
flat from N = 64 to N = 1728. Extrapolated wall time for 10^6 sweeps:

| L (cube) | N | wall time |
|---|---|---|
| 4 | 64 | 4 min |
| 8 | 512 | 31 min |
| 16 | 4,096 | 4.1 h |
| 32 | 32,768 | 33 h |
| 64 | 262,144 | 11 days |

Multiply by `beta`. Memory is dominated by the per-site imaginary-time
magnetization, `N * ntau * 8` bytes (134 MB at L = 64, ntau = 64).

Earlier versions were O(N^2) because `C^zz` went through the structure factor
and `disp.xml` enumerated all site pairs; at L = 16 that cost 7.3 days rather
than 4.1 hours. See `docs/estimators.md`.

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
