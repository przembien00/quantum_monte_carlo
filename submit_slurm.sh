#!/usr/bin/env bash
#SBATCH --job-name=qmc
#SBATCH --array=0-21
#SBATCH --ntasks=16              # MPI ranks per array task
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/qmc-%A_%a.out
#SBATCH --error=logs/qmc-%A_%a.out


set -euo pipefail


ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
PY="$ROOT/venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at $PY -- run ./install.sh first" >&2; exit 1; }

# ---- the scan ---------------------------------------------------------------
# One line per array index; keep the count in sync with --array above.
BETAS=(0.2 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4 4.5 5)
COUPLINGS=(0.5 -0.5)

i=${SLURM_ARRAY_TASK_ID:-0}
BETA=${BETAS[$(( i % ${#BETAS[@]} ))]}
J=${COUPLINGS[$(( i / ${#BETAS[@]} ))]}

RANKS=${SLURM_NTASKS:-1}

echo "host $(hostname)   array task $i   beta=$BETA   J=$J   ranks=$RANKS"

"$PY" run_qmc.py \
    --lattice=cube:32x32x32 \
    --beta="$BETA" \
    --J="$J" \
    --h_z=0.5 \
    --num_TimePoints=100 \
    --sites=0,1,2,3,4 \
    --nmcs=1250 \
    --nset=10 \
    --cores="$RANKS" \
    --project="AFM_FM_B"

echo "done"
