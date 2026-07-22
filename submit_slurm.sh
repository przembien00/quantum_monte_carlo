#!/usr/bin/env bash
#SBATCH --job-name=qmc
#SBATCH --array=0-13
#SBATCH --ntasks=16              # MPI ranks per array task
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/qmc-%A_%a.out
#SBATCH --error=logs/qmc-%A_%a.out


set -euo pipefail


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

# --sites names neighbour shells: 0 on-site, 1 nearest, 2 next-nearest.
# --seed differs per array task so the chains are independent.
# No trailing comments inside the continued command: a '#' after a backslash
# silently truncates the argument list rather than failing.
"$PY" run_qmc.py \
    --lattice=cube:8x8x8 \
    --beta="$BETA" \
    --J="$J" \
    --h_z=0.0 \
    --num_TimePoints=64 \
    --sites=0,1,2 \
    --nmcs=20000 \
    --nset=20 \
    --cores="$RANKS" \
    --project="scan_L8" \
    --seed=$(( 31415 + i ))

echo "done"
