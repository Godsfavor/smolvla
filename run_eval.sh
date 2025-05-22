#!/usr/bin/env bash
#SBATCH --job-name=SMOLVLA_Eval
#SBATCH --partition=mundus
#SBATCH --gres=gpu:a100-20:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mundus/gugwugab432/projects/smolvla/output/eval_output_%j.log
#SBATCH --error=/home/mundus/gugwugab432/projects/smolvla/output/eval_error_%j.log

echo "Starting evaluation job at $(date)"

# Load modules
module purge
echo "Modules purged"
module load cuda/11.8
echo "Loaded cuda/11.8"
module load python/3.9
echo "Loaded python/3.9"

# Define project directory
PROJECT_DIR="/home/mundus/gugwugab432/projects/smolvla"

# Navigate to the project directory
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "Changed to project directory: $PROJECT_DIR"
else
    echo "Project directory does not exist: $PROJECT_DIR"
    exit 1
fi

# Activate the virtual environment
source env/bin/activate
echo "Activated virtual environment at $(which python)"

# Ensure all required packages are installed
echo "Checking required packages..."
pip install torch torchvision torchaudio packaging wheel accelerate datasets peft bitsandbytes transformers tensorflow Pillow numpy wandb matplotlib

# Set WandB API key (replace with your actual key)
export WANDB_API_KEY=abc123

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Define the checkpoint path - update this as needed
CHECKPOINT_PATH="./SmolVLM-Base-Fixed"
echo "Using checkpoint: $CHECKPOINT_PATH"

# Run the evaluation script
echo "Starting evaluation..."
python quick_start_infer.py || {
    echo "Evaluation script failed with exit code $?"
    exit 1
}

echo "Evaluation completed successfully at $(date)"