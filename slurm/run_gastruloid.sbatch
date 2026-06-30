#!/bin/bash
# SLURM job array for the architecture-gastruloid study.
#
# The study defines 12 experiments (4 architectures x 3 seeds). This array runs
# each experiment as its own task: SLURM_ARRAY_TASK_ID selects one experiment
# (by YAML order) inside main.py via select_experiments().
#
# Submit from the repo root (after running slurm/setup_env.sh once):
#   sbatch slurm/run_gastruloid.sbatch
#
# ============================ EDIT THESE ====================================
#SBATCH --account=CHANGE_ME              # your allocation/account  (sacctmgr show user $USER)
#SBATCH --partition=CHANGE_ME_GPU        # a GPU partition          (sinfo -s)
# ============================================================================
#SBATCH --job-name=simenc-gastruloid
#SBATCH --array=0-11%4                    # 12 experiments; run at most 4 at once
#SBATCH --gres=gpu:1                      # 1 GPU per task (some clusters use: --gpus=1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00                   # per-task wall time; tune to your queue limits
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm

# --- Load cluster modules (uncomment/adjust to match your cluster) ----------
# module load cuda/12.4 || true
# module load python/3.11 || true

# --- Activate the environment built by slurm/setup_env.sh -------------------
source .venv/bin/activate

# Limit thread oversubscription to the cores SLURM granted this task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# main.py reads SLURM_ARRAY_TASK_ID and runs exactly one experiment.
srun python src/simulation_encoder/main.py
