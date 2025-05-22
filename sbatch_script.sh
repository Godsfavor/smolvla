#!/usr/bin/env bash
#SBATCH --job-name=Run_Quick_Start-full_ft
#SBATCH --partition=mundus
#SBATCH --gres=gpu:a100-20:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=100:00:00
#SBATCH --output=/home/mundus/gugwugab432/projects/smolvla/output/run_output_%j.log
#SBATCH --error=/home/mundus/gugwugab432/projects/smolvla/output/run_error_%j.log

echo "Starting job at $(date)"

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

# Upgrade pip
pip install --upgrade pip

# Install all required packages including WandB
pip install torch torchvision torchaudio packaging wheel accelerate datasets peft bitsandbytes tensorboard flash-attn transformers tensorflow Pillow numpy wandb matplotlib

# Set WandB API key (replace with your actual key)
export WANDB_API_KEY=abc123

# Run the Python script with basic error handling
python quick_start.py || {
    echo "Python script failed with exit code $?"
    exit 1
}
echo "Ran quick_start.py"

echo "Job finished at $(date)"