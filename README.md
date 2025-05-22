VLA Training with Bridge Data


This repository contains code for training a Visual Language Action (VLA) model using the Bridge Data dataset. The model predicts robotic actions based on visual and language inputs, leveraging state-of-the-art transformer architectures like Idefics3 and SmolVLM with efficient fine-tuning techniques such as LoRA and QLoRA.
Table of Contents

Introduction
Prerequisites
Setup
Data Preparation
Running the Code
Understanding the Results
Contributing
License
Contact

Introduction
This project aims to train a Visual Language Action (VLA) model to predict robotic actions based on visual observations (images) and language instructions. It uses the Bridge Data dataset, which contains 7,200 demonstrations of a WidowX250 robot arm performing 71 kitchen tasks across 10 environments with varying lighting, robot positions, and backgrounds. The model leverages transformer architectures (Idefics3 or SmolVLM) and employs efficient fine-tuning methods like LoRA and QLoRA to achieve generalization across tasks and domains. The training pipeline is built using PyTorch and the Hugging Face Transformers library, with data loaded from TFRecord files using TensorFlow.
The project is inspired by approaches like OpenVLA, incorporating features such as freezing the vision encoder, mixed supervision with language embeddings, and options for discrete or continuous action prediction. The training process logs metrics and predictions to Weights & Biases (WandB) for monitoring and analysis.


Prerequisites
To run this project, you need the following:
Software

Python: Version 3.9 or higher
PyTorch: Version 1.12 or higher
Transformers: Hugging Face Transformers library
TensorFlow: For loading TFRecord files
CUDA: Version 11.8 for GPU acceleration
Additional Libraries: Listed in the setup section below

Hardware

GPU: At least 32GB memory (e.g., NVIDIA A100)
CPU: Multiple cores recommended for data loading
Memory: At least 32GB RAM

Environment

SLURM Cluster: Optional, for running on a cluster with SLURM job scheduler
WandB Account: For logging training metrics and visualizations

Setup
Follow these steps to set up the project environment:

Clone the Repository: Clone the project from your GitHub repository (replace Godsfavor with your actual GitHub username).
git clone https://github.com/Godsfavor/smolvla.git
cd smolvla


Create a Virtual Environment: Set up a Python virtual environment to isolate dependencies.
python -m venv env
source env/bin/activate


Install Dependencies: Upgrade pip and install the required Python packages.
pip install --upgrade pip
pip install torch torchvision torchaudio packaging wheel accelerate datasets peft bitsandbytes tensorboard flash-attn transformers tensorflow Pillow numpy wandb matplotlib


Set Up WandB: The project uses Weights & Biases (WandB) for logging training progress and metrics. Create an account at WandB and obtain your API key.
export WANDB_API_KEY=your_api_key



Data Preparation
The model is trained on the Bridge Data dataset, a large multi-domain and multi-task dataset containing 7,200 demonstrations of a WidowX250 robot arm performing 71 kitchen tasks across 10 environments. The dataset is stored in TFRecord format and must be placed in a specific directory structure for the code to access it.
Steps to Prepare the Dataset

Create the Directory Structure: Create the directory where the dataset will be stored, relative to the project root.
mkdir -p data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0


Download the Dataset: The dataset is available at Bridge Data. It consists of 1024 TFRecord files for the training split and 128 for the validation split. Download these files into the 1.0.0 directory.
cd data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0
wget -r -l1 -np -nH --cut-dirs=5 https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/


Explanation of wget options:
-r: Recursive download
-l1: Limit recursion to one level (only download files in the specified directory)
-np: No parent directories (stay within the specified directory)
-nH: No host directories (avoid creating extra directories)
--cut-dirs=5: Remove five leading directories from the URL path to match the local directory structure


Note: This step may require significant time and bandwidth due to the large number of files (approximately 387.49 GiB total, as per TensorFlow Datasets documentation).



Alternative: Using TensorFlow Datasets (TFDS)
Instead of manually downloading the TFRecord files, you can use TensorFlow Datasets to download the dataset, but you’ll need to adjust the code to match the TFDS file naming convention. To download using TFDS:
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('bridge', download=True)"

This downloads the dataset to ~/tensorflow_datasets/bridge/1.0.0/. However, the code expects files named bridge_dataset-<split>.tfrecord-<index>-of-<total>, while TFDS may use bridge-<split>.tfrecord-<index>-of-<total>. To use TFDS:

Locate the downloaded files in ~/tensorflow_datasets/bridge/1.0.0/.

Modify the create_tf_dataset function in train_vla.py to use the correct file pattern:
file_pattern = f"{base_path}/{split}-{{:05d}}-of-0{num_files:04d}.tfrecord"

Set base_path = os.path.expanduser('~/tensorflow_datasets/bridge/1.0.0/').


Due to the hardcoded path in the provided code, manually downloading the TFRecord files is recommended for simplicity.
Running the Code
The project includes two scripts:

train_vla.py: The main Python script that defines the model, dataset, and training pipeline.
train_vla.sh: A bash script for submitting the training job to a SLURM cluster.

Running on a SLURM Cluster
The train_vla.sh script is configured for a cluster with an A100 GPU (20GB variant). To run:
sbatch train_vla.sh

Note: Adjust the SLURM directives in train_vla.sh (e.g., --partition, --gres, --mem) to match your cluster’s configuration. For example:



Directive
Description
Example Value



--job-name
Name of the job
Run_Quick_Start


--partition
Cluster partition
mundus


--gres
GPU resources
gpu:a100-20:1


--cpus-per-task
Number of CPU cores
4


--mem
Memory allocation
32G


--time
Maximum job runtime
100:00:00


Running Locally
If you’re not using a SLURM cluster, run the Python script directly:
python train_vla.py

Ensure your system has a compatible GPU and sufficient memory. The script uses 8-bit optimizers and bfloat16 precision to reduce memory usage, but a high-end GPU is still required.
Configuration Options
The train_vla.py script includes several configuration flags at the top:



Flag
Description
Default Value



USE_LORA
Enable LoRA fine-tuning
False


USE_QLORA
Enable QLoRA fine-tuning
True


SMOL
Use SmolVLM-Base instead of Idefics3
True


FREEZE_VISION_ENCODER
Freeze the vision encoder
True


MIX_SUPERVISION
Use mixed supervision with language embeddings
True


DISCRETE_ACTIONS
Use discrete action prediction
False


NUM_DISCRETE_BINS
Number of bins for discrete actions
10 (if DISCRETE_ACTIONS is True)


Modify these flags in train_vla.py to experiment with different configurations.
Understanding the Results
Output Directory
Trained models are saved in the outputs directory with a timestamp, e.g., outputs/SmolVLM-Base_20250510_051229. The directory contains:

Model checkpoints (saved every 500 steps, with a limit of 3 checkpoints).
The best model based on validation MAE (saved as best_model).
The final model (saved as final_model).

Evaluation Metrics
The model is evaluated using:

Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual actions.
Mean Squared Error (MSE): Measures the average squared difference, emphasizing larger errors. These metrics are computed during evaluation (every 100 steps) and logged to WandB. Lower values indicate better performance.

WandB Logging
The training process logs:

Training loss and learning rate every 50 steps.
Evaluation metrics (MAE, MSE) every 100 steps.
Example predictions and target actions for validation examples. Access these logs on the WandB dashboard using your API key. The project is named smolvla-training, with run names like SmolVLM-Base-OpenVLA-style.

Example Predictions
After training, the script generates predictions for five validation examples, showing:

The language instruction.
Predicted actions (as a 7-dimensional vector or discrete bin indices).
Actual actions.
Per-example error (mean absolute error). These are printed to the console and logged to WandB under final_eval/example_<i>_prediction.

Contributing
Contributions are welcome! To contribute:

Open an issue to discuss proposed changes or report bugs.
Submit a pull request with your changes, ensuring:
Code follows PEP 8 style guidelines.
New features include tests or documentation.
Changes align with the project’s goal of improving robotic action prediction.


Test your changes locally or on a small subset of the dataset to ensure compatibility.

License
This project is licensed under the MIT License. See the LICENSE file for details.


Key Citations

Bridge Data Dataset: Source for the Bridge Data dataset used in training.
WandB: Platform for logging and monitoring training progress.
TensorFlow Datasets Bridge Catalog: Documentation for the Bridge dataset in TFDS.
BridgeData V2 GitHub Repository: Repository for BridgeData V2, providing context for dataset usage.
Bridge Data: Boosting Generalization of Robotic Skills: Research paper introducing the Bridge Data dataset.

