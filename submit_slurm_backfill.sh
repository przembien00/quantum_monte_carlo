#!/usr/bin/env bash
# Backfill the missing L = 32 (N = 32768) points.  Ten holes remain:
#
#   AFM_FM   (h_z=0)    beta=4, 4.5, 5   AFM   and   beta=4, 4.5, 5   FM
#   AFM_FM_B (h_z=0.5)  beta=4, 4.5, 5   AFM   and   beta=1.5         FM
#
# The array is exactly these ten -- one task per real job, no no-op slots.
# Coupling is 1/sqrt(6) and the site token / ntau match the existing N=32768
# series.  The AFM 625 / FM 1250 sample split is the base count; for the
# coldest runs (beta > 3.5) it is slashed by 2/3 -- down to a third -- so the
# most expensive jobs have a chance to finish.  A skip guard still fires if a
# file has appeared since, so re-running never duplicates a good run.
#
#SBATCH --job-name=qmc-backfill
#SBATCH --array=0-9
#SBATCH --ntasks=16              # MPI ranks per array task
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/qmc-backfill-%A_%a.out
#SBATCH --error=logs/qmc-backfill-%A_%a.out


set -euo pipefail


ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="$ROOT/venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at $PY -- run ./install.sh first" >&2; exit 1; }

# ---- the missing points -----------------------------------------------------
# One entry per job: "PROJECT H_Z beta J".  All are L = 32.  Keep --array at
# 0 .. count-1.
POINTS=(
    "AFM_FM   0.0  4    0.408248"
    "AFM_FM   0.0  4.5  0.408248"
    "AFM_FM   0.0  5    0.408248"
    "AFM_FM   0.0  4    -0.408248"
    "AFM_FM   0.0  4.5  -0.408248"
    "AFM_FM   0.0  5    -0.408248"
    "AFM_FM_B 0.5  4    0.408248"
    "AFM_FM_B 0.5  4.5  0.408248"
    "AFM_FM_B 0.5  5    0.408248"
    "AFM_FM_B 0.5  1.5  -0.408248"
)

read -r PROJECT H_Z BETA J <<< "${POINTS[${SLURM_ARRAY_TASK_ID:-0}]}"

# L = 32 lattice metadata.
L=32; N=32768; TOK=0-1-33-1057-2; NTAU=100

# ---- sample count -----------------------------------------------------------
# Base: AFM 625 / FM 1250 (the antiferromagnet is slower, so sampled less).
if [[ "$J" == -* ]]; then NMCS=1250; else NMCS=625; fi
# Coldest runs are the most expensive; slash them by 2/3 (keep one third).
if awk "BEGIN{exit !($BETA > 3.5)}"; then
    NMCS=$(( NMCS / 3 ))
fi

# ---- skip if it already exists (safety against a race / re-run) -------------
FNAME="ISO__Cube_NN_PBC_N=${N}__sites=${TOK}__beta=${BETA}__J=${J}"
[[ "$H_Z" != "0.0" && "$H_Z" != "0" ]] && FNAME+="__h_z=${H_Z}"
FNAME+=".hdf5"
if [[ -f "$ROOT/Data/$PROJECT/$FNAME" ]]; then
    echo "task ${SLURM_ARRAY_TASK_ID:-0}: exists, skipping  $PROJECT/$FNAME"
    exit 0
fi

RANKS=${SLURM_NTASKS:-1}
echo "task ${SLURM_ARRAY_TASK_ID:-0}: host $(hostname)  L=$L N=$N beta=$BETA J=$J h_z=$H_Z  nmcs=$NMCS ntau=$NTAU  project=$PROJECT  ranks=$RANKS"

"$PY" run_qmc.py \
    --lattice="cube:${L}x${L}x${L}" \
    --beta="$BETA" \
    --J="$J" \
    --h_z="$H_Z" \
    --num_TimePoints="$NTAU" \
    --sites=0,1,2,3,4 \
    --nmcs="$NMCS" \
    --nset=10 \
    --cores="$RANKS" \
    --project="$PROJECT"

echo "task ${SLURM_ARRAY_TASK_ID:-0}: done"
