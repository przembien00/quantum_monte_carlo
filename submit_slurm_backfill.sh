#!/usr/bin/env bash
# Backfill the missing finite-size points at L > 20 (below L = 32, which has
# its own generator).  A bug left these six holes at N = 13824 and 21952:
#
#   AFM_FM   (h_z=0)    L=24  beta=1.5  AFM
#   AFM_FM   (h_z=0)    L=28  beta=0.2  AFM
#   AFM_FM_B (h_z=0.5)  L=28  beta=2    AFM
#   AFM_FM_B (h_z=0.5)  L=28  beta=2    FM
#   AFM_FM_B (h_z=0.5)  L=28  beta=3.5  AFM
#   AFM_FM_B (h_z=0.5)  L=28  beta=5    FM
#
# The array is exactly these six -- one task per real job, no no-op slots.
# Each knob (coupling, sample count, imaginary-time resolution) matches the
# existing files of the same size, so a backfilled point is consistent with
# the rest of its series.  A skip guard still fires if a file has appeared
# since, so re-running is safe and never duplicates a good run.
#
#SBATCH --job-name=qmc-backfill
#SBATCH --array=0-5
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
# One entry per job: "PROJECT H_Z L beta J".  Keep --array at 0 .. count-1.
POINTS=(
    "AFM_FM   0.0  24  1.5  0.408248"
    "AFM_FM   0.0  28  0.2  0.408248"
    "AFM_FM_B 0.5  28  2    0.408248"
    "AFM_FM_B 0.5  28  2    -0.408248"
    "AFM_FM_B 0.5  28  3.5  0.408248"
    "AFM_FM_B 0.5  28  5    -0.408248"
)

read -r PROJECT H_Z L BETA J <<< "${POINTS[${SLURM_ARRAY_TASK_ID:-0}]}"

# Side length -> total spins, filename site token, imaginary-time points.
case "$L" in
    24) N=13824; TOK=0-1-25-601-2; NTAU=100 ;;
    28) N=21952; TOK=0-1-29-813-2; NTAU=100 ;;
    *)  echo "no lattice metadata for L=$L" >&2; exit 1 ;;
esac

# Sample count: the big lattices use the AFM 625 / FM 1250 split (the
# antiferromagnet is slower, so it is sampled less to keep wall times in line).
if [[ "$J" == -* ]]; then NMCS=1250; else NMCS=625; fi

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
