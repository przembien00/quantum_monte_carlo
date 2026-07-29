#!/usr/bin/env bash
# Backfill the finite-size ladder for L < 32 (N < 32768), both fields.
#
# A bug left the smaller lattices missing points that the L = 32 runs have:
# the extended beta = 3.5..5 tail at every L <= 20, the whole L = 4 cube at
# h_z = 0.5, and a handful of single holes at L = 24 and 28.  This job sweeps
# the full ladder and SKIPS any (L, beta, J, h_z) whose output already exists,
# so it regenerates exactly the gaps and never clobbers or duplicates a good
# run (run_qmc.py would otherwise append an X on a name collision).
#
# Every knob -- coupling, sample counts, imaginary-time resolution -- is set to
# match the existing files of the same size, so a backfilled point is
# statistically consistent with the rest of its series.
#
#SBATCH --job-name=qmc-backfill
#SBATCH --array=0-87
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

# ---- the ladder -------------------------------------------------------------
# Restricted to L > 20 (below L = 32, which has its own generator).  SITE_TOK
# is the shell 0..4 site-index token run_qmc.py bakes into the filename for
# that lattice; NTAU = 100 matches the existing series at these sizes.
LS=(24       28)
NS=(13824    21952)
TOKS=(0-1-25-601-2  0-1-29-813-2)
NTAUS=(100      100)

BETAS=(0.2 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5)   # %g forms -- match the filenames
COUPLINGS=(0.408248 -0.408248)              # 1/sqrt(6); positive = AFM, negative = FM
PROJECTS=(AFM_FM   AFM_FM_B)                 # zero field, then h_z = 0.5
HZS=(0.0      0.5)

nL=${#LS[@]}; nB=${#BETAS[@]}; nS=${#COUPLINGS[@]}; nP=${#PROJECTS[@]}
# Keep --array in sync: it must be 0 .. nL*nB*nS*nP - 1  (2*11*2*2 = 88).

# ---- decode the array index into (size, beta, sign, project) ----------------
i=${SLURM_ARRAY_TASK_ID:-0}
bi=$(( i % nB ))
li=$(( (i / nB) % nL ))
si=$(( (i / (nB*nL)) % nS ))
pi=$(( (i / (nB*nL*nS)) % nP ))

L=${LS[$li]}; N=${NS[$li]}; TOK=${TOKS[$li]}; NTAU=${NTAUS[$li]}
BETA=${BETAS[$bi]}
J=${COUPLINGS[$si]}
PROJECT=${PROJECTS[$pi]}; H_Z=${HZS[$pi]}

# ---- sample count: match the size's series ----------------------------------
# L <= 20 were run at 2500 sweeps for both signs; the big lattices use the
# AFM 625 / FM 1250 split (the antiferromagnet is slower, so it is sampled
# less to keep wall times in line).
if (( N <= 8000 )); then
    NMCS=2500
elif [[ "$J" == -* ]]; then
    NMCS=1250            # ferromagnet
else
    NMCS=625             # antiferromagnet
fi

# ---- skip points that already exist -----------------------------------------
# Reconstruct the exact name run_qmc.py would write; the h_z fragment only
# appears when the field is nonzero.
FNAME="ISO__Cube_NN_PBC_N=${N}__sites=${TOK}__beta=${BETA}__J=${J}"
[[ "$H_Z" != "0.0" && "$H_Z" != "0" ]] && FNAME+="__h_z=${H_Z}"
FNAME+=".hdf5"
TARGET="$ROOT/Data/$PROJECT/$FNAME"

if [[ -f "$TARGET" ]]; then
    echo "task $i: exists, skipping  $PROJECT/$FNAME"
    exit 0
fi

RANKS=${SLURM_NTASKS:-1}
echo "task $i: host $(hostname)  L=$L N=$N beta=$BETA J=$J h_z=$H_Z  nmcs=$NMCS ntau=$NTAU  project=$PROJECT  ranks=$RANKS"

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

echo "task $i: done"
