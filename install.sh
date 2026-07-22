#!/usr/bin/env bash
# Build the QMC stack (DSQSS/DLA + patches + Python venv) in one place.
#
# Designed for a login node on an HPC cluster, but works anywhere. Run it from
# the repository root, after loading whatever modules your site needs:
#
#     module load gcc openmpi cmake python      # site-specific names!
#     ./install.sh
#
# Everything lands under the repository: external_dsqss/ (source),
# dsqss_install/ (binaries), venv/ (Python). Nothing is written outside.
#
# Two properties of the build are worth knowing because they constrain how you
# use the result:
#
#   * dla_pre and friends get an ABSOLUTE shebang pointing at venv/bin/python,
#     so the repository cannot be moved after installing. Reinstall instead.
#   * dla is dynamically linked against the MPI you build with, so compute jobs
#     must load the SAME MPI module as this script did.

set -euo pipefail

DSQSS_VERSION="${DSQSS_VERSION:-v2.0.6}"
DSQSS_REPO="${DSQSS_REPO:-https://github.com/issp-center-dev/dsqss.git}"
JOBS="${JOBS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4 )}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/external_dsqss"
PREFIX="$ROOT/dsqss_install"
VENV="$ROOT/venv"
PATCH="$ROOT/patches/dsqss_estimators.patch"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- prerequisites
say "Checking prerequisites"
for tool in cmake git; do
    command -v "$tool" >/dev/null || die "$tool not found. Try: module load $tool"
done
command -v mpicxx >/dev/null || die "mpicxx not found. Load an MPI module (openmpi, mpich, intel-mpi)."

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || die "$PYTHON not found. Try: module load python"
PYVER=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,6) else 1)' \
    || die "Python >= 3.6 required, found $PYVER"

printf '  cmake    %s\n' "$(cmake --version | head -1 | awk '{print $3}')"
printf '  mpicxx   %s\n' "$(command -v mpicxx)"
printf '  python   %s (%s)\n' "$PYVER" "$(command -v "$PYTHON")"
printf '  parallel build jobs: %s\n' "$JOBS"

# HDF5 is reached through h5py, not linked into dla, so it is a pip concern only.

# ------------------------------------------------------------------- python env
say "Creating the Python environment"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV" || die "venv creation failed (need python3-venv?)"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# scipy is needed by dsqss's own lattice generators, not by qmc/ itself.
"$VENV/bin/python" -m pip install --quiet toml numpy scipy h5py matplotlib \
    || die "pip install failed. On an offline node, pre-download wheels and use --find-links."
printf '  %s\n' "$("$VENV/bin/python" --version)"

# --------------------------------------------------------------------- sources
say "Fetching DSQSS $DSQSS_VERSION"
if [[ -d "$SRC/.git" ]]; then
    printf '  reusing existing checkout in %s\n' "$SRC"
    # Return to pristine sources so the patch always applies to a known state.
    # Both steps are needed: reset restores tracked files, clean removes the
    # files the patch adds. Doing only one leaves a half-patched tree that
    # still looks patched to a file-existence check.
    git -C "$SRC" reset -q --hard HEAD 2>/dev/null || true
    git -C "$SRC" clean -qfd src/ 2>/dev/null || true
else
    git clone --depth 1 --branch "$DSQSS_VERSION" "$DSQSS_REPO" "$SRC" 2>/dev/null \
        || git clone --depth 1 "$DSQSS_REPO" "$SRC" \
        || die "clone failed. Compute nodes are usually offline -- run this on a login node."
fi

say "Applying the estimator patch"
[[ -f "$PATCH" ]] || die "patch not found at $PATCH"
git -C "$SRC" apply --check "$PATCH" 2>/dev/null \
    || die "patch does not apply to this DSQSS checkout (expected $DSQSS_VERSION)"
git -C "$SRC" apply "$PATCH"
printf '  applied %s\n' "$(basename "$PATCH")"

# ----------------------------------------------------------------------- build
say "Building"
cmake -S "$SRC" -B "$SRC/build" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DPython3_EXECUTABLE="$VENV/bin/python" \
      -DCMAKE_BUILD_TYPE=Release \
      -DTesting=OFF > "$ROOT/install-cmake.log" 2>&1 \
    || { tail -25 "$ROOT/install-cmake.log"; die "cmake configure failed (see install-cmake.log)"; }
cmake --build "$SRC/build" -j"$JOBS" > "$ROOT/install-build.log" 2>&1 \
    || { tail -25 "$ROOT/install-build.log"; die "build failed (see install-build.log)"; }
cmake --install "$SRC/build" > /dev/null 2>&1 || die "install failed"
printf '  binaries in %s/bin\n' "$PREFIX"

# ------------------------------------------------------------- link the package
# The dsqss Python package installs under a version-specific directory, so the
# path is derived rather than hardcoded.
say "Linking the dsqss Python package into the venv"
DSQSS_PKG=$(find "$PREFIX/lib" -maxdepth 2 -name site-packages -type d | head -1)
[[ -n "$DSQSS_PKG" ]] || die "could not locate the installed dsqss package under $PREFIX/lib"
SITE=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
echo "$DSQSS_PKG" > "$SITE/dsqss.pth"
"$VENV/bin/python" -c 'import dsqss' || die "dsqss still not importable"
printf '  %s\n' "$DSQSS_PKG"

# ------------------------------------------------------------------ smoke test
say "Verifying the installation"
"$VENV/bin/python" - <<'PY' || die "smoke test failed"
import shutil, sys, tempfile, numpy as np
sys.path.insert(0, ".")
from qmc import lattice as L, run as R
work = tempfile.mkdtemp()
try:
    lat = L.build("chain:8", J=1.0)
    out = R.run_dsqss(lat, 1.0, 8, work, spin_sites=[0, 1],
                      mc={"nmcs": 500, "nset": 4}, ncores=1)
    p, cre, cim, sre, sim = R.convert(lat, 1.0, 8, out, [0, 1], h_z=0.0)
    rows = p["correlation_rows"].split(",")
    xx, zz = cre[0, rows.index("xx")], cre[0, rows.index("zz")]
    assert abs(xx[0] - 0.25) < 1e-6, f"C^xx(0) = {xx[0]}, expected 1/4 exactly"
    gap = np.max(np.abs(xx - zz))
    assert gap < 5e-3, f"isotropy violated at h_z=0: max|xx-zz| = {gap:.2e}"
    assert not np.all(np.isnan(sim)), "no C^xy channel -- is the patch applied?"
    print(f"  C^xx(0) = {xx[0]:.6f}   (exact: 0.25)")
    print(f"  max|C^xx - C^zz| = {gap:.1e}   (zero by isotropy at h_z = 0)")
    print("  all three channels present")
finally:
    shutil.rmtree(work, ignore_errors=True)
PY

MPILIB=$( (command -v ldd >/dev/null && ldd "$PREFIX/bin/dla" 2>/dev/null || otool -L "$PREFIX/bin/dla" 2>/dev/null) | grep -io '[^ ]*libmpi[^ ]*' | head -1 )

cat <<EOF

$(say "Done")
  run with:   $VENV/bin/python run_qmc.py --lattice=cube:8x8x8 --beta=1 ...
  validate:   $VENV/bin/python validate_ed.py --lattice=chain:8 --beta=2 --h_z=1

  Batch jobs must load the same modules used here; dla is linked against
    ${MPILIB:-your MPI runtime}
  and the helper scripts hardcode $VENV/bin/python, so do not move this
  directory -- rerun ./install.sh if you need it elsewhere.
EOF
