#!/bin/bash
#SBATCH --job-name=rl_3epoch_v2       # Job name
#SBATCH --output=job_%j.out         # Standard output log
#SBATCH --error=job_%j.err          # Standard error log
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=64G                   # Memory per node (bumped for 3-epoch run)
#SBATCH --time=40:00:00             # Time limit hrs:min:sec (Bumped for 6 epochs/8 gens)
#SBATCH --partition=workq             # Partition/queue name (adjust as needed)
#SBATCH --gres=gpu:1                # Request 1 GPU (adjust as needed)

# Load your personal Miniconda environment
source ~/miniconda/bin/activate

# Navigate to the project directory
cd ~/RL_project

# Activate your virtual environment if you have one
source activate rl_env

# Set cache directories to local /tmp to avoid NFS/vLLM file lock crashes
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export VLLM_CONFIG_ROOT=/tmp/$USER/vllm_config
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torch_cache

# Run the python script
echo "Starting job at $(date)"
python RL_trainer.py
echo "Job finished at $(date)"
