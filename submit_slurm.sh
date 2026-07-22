#!/usr/bin/env bash
#SBATCH --job-name=qmc
#SBATCH --array=0-13
#SBATCH --ntasks=16              # MPI ranks per array task
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/qmc-%A_%a.out
#SBATCH --error=logs/qmc-%A_%a.out
#
# Parameter scan as a SLURM array job.  Submit with:
#
#     mkdir -p logs && sbatch submit_slurm.sh
#
# WHY AN ARRAY JOB RATHER THAN ONE BIG MPI JOB
# --------------------------------------------
# DSQSS/DLA runs one independent Markov chain per rank and only combines them
# at the end.  Doubling the ranks therefore halves the error bar by sqrt(2); it
# does NOT make a run finish sooner (measured: 19.2 s on 1 rank vs 21.6 s on 4
# for the same job).  So:
#
#   * ranks within a task  -> statistics for ONE parameter point
#   * array tasks          -> different parameter points, running concurrently
#
# Sixteen ranks gives 4x smaller error bars than one.  Going much beyond that
# is usually a poor trade against queue wait, since the error only improves as
# 1/sqrt(ranks).
#
# SIZING THE RUN
# --------------
# Measured cost law (see docs/estimators.md):
#
#     error ~ A * N^-0.55 * beta^p / sqrt(nmcs * nset * ranks)
#
# with p ~ 0.7 for C^xx, ~0.2 for C^xy and ~0 for C^zz.  Because cost per sweep
# is linear in N while the error falls as N^-0.55, the wall time needed to hit a
# given error target is nearly independent of system size -- it is set by beta
# and by the target.  For N=24 at beta=3 the prefactor is A ~ 0.075 on the
# noisiest channel, i.e.
#
#     nmcs * nset * ranks  ~  (0.075 / target)^2
#
# The settings below target ~1e-4 with margin.  Calibrate once at your own size
# and beta with a short run, then rescale by 1/sqrt(sweeps).

set -euo pipefail

# ---- site-specific: replace with your cluster's module names ----------------
# Must match what install.sh was run with -- dla is dynamically linked to MPI.
# module load gcc/12 openmpi/4.1 python/3.11

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="$ROOT/venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at $PY -- run ./install.sh first" >&2; exit 1; }

# ---- the scan ---------------------------------------------------------------
# One line per array index; keep the count in sync with --array above.
BETAS=(0.2 0.5 1.0 1.5 2.0 2.5 3.0)
COUPLINGS=(0.5 -0.5)

i=${SLURM_ARRAY_TASK_ID:-0}
BETA=${BETAS[$(( i % ${#BETAS[@]} ))]}
J=${COUPLINGS[$(( i / ${#BETAS[@]} ))]}

RANKS=${SLURM_NTASKS:-1}

echo "host $(hostname)   array task $i   beta=$BETA   J=$J   ranks=$RANKS"

# --cores is passed through to mpirun -n.  On SLURM, srun is often preferred;
# if your site requires it, set QMC_MPIRUN=srun and adapt qmc/run.py, or simply
# run with --cores=1 under `srun -n $RANKS` and merge afterwards.
"$PY" run_qmc.py \
    --lattice=cube:8x8x8 \
    --beta="$BETA" \
    --J="$J" \
    --h_z=0.0 \
    --num_TimePoints=64 \
    --sites=0,1,7 \
    --nmcs=20000 \
    --nset=20 \
    --cores="$RANKS" \
    --project="scan_L8" \
    --seed=$(( 31415 + i ))     # distinct streams across the array

echo "done"
