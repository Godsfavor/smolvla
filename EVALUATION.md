VLA Model Evaluation with Bridge Data


This repository contains scripts to evaluate a pre-trained Visual Language Action (VLA) model on the validation set of the Bridge Data dataset. The model predicts robotic actions based on visual observations (images) and language instructions, leveraging transformer architectures like SmolVLM or Idefics3. The evaluation pipeline, implemented in eval_vla.py and supported by run_eval.sh, assesses model performance and logs comprehensive metrics and visualizations to Weights & Biases (WandB).
Table of Contents

Introduction
Prerequisites
Setup
Data Preparation
Model Checkpoint
Running the Evaluation
Understanding the Results
Contributing
License
Contact

Introduction
This project provides a framework for evaluating a Visual Language Action (VLA) model trained to predict robotic actions from visual and language inputs. The evaluation uses the validation split of the Bridge Data dataset, which includes 128 TFRecord files containing robotic demonstrations for kitchen tasks. The eval_vla.py script loads a pre-trained model, processes the dataset, and computes metrics such as Mean Squared Error (MSE) and R² scores. Visualizations, including action comparisons and 3D trajectories, are logged to WandB for detailed analysis. The run_eval.sh script facilitates running the evaluation on a SLURM cluster.
Prerequisites
To run the evaluation, ensure you have the following:
Software

Python: Version 3.9 or higher
PyTorch: Version 1.12 or higher
Transformers: Hugging Face Transformers library
TensorFlow: For loading TFRecord files
CUDA: Version 11.8 for GPU acceleration
Additional Libraries: torchvision, torchaudio, packaging, wheel, accelerate, datasets, peft, bitsandbytes, Pillow, numpy, wandb, matplotlib

Hardware

GPU: At least 20GB memory (e.g., NVIDIA A100)
CPU: Multiple cores for efficient data loading
Memory: At least 32GB RAM

Environment

SLURM Cluster: Optional, for running on a cluster with SLURM job scheduler
WandB Account: Required for logging metrics and visualizations (WandB)

Setup
Follow these steps to set up the project environment:

Clone the Repository:Clone the project from your GitHub repository (replace Godsfavor with your actual GitHub username).
git clone https://github.com/Godsfavor/smolvla.git
cd smolvla


Create a Virtual Environment:Set up a Python virtual environment to isolate dependencies.
python -m venv env
source env/bin/activate


Install Dependencies:Upgrade pip and install the required Python packages.
pip install --upgrade pip
pip install torch torchvision torchaudio packaging wheel accelerate datasets peft bitsandbytes transformers tensorflow Pillow numpy wandb matplotlib


Set Up WandB:Create an account at WandB and obtain your API key. Set it as an environment variable.
export WANDB_API_KEY=your_api_key



Data Preparation
The evaluation uses the validation set of the Bridge Data dataset, which contains 128 TFRecord files with robotic demonstrations for kitchen tasks across various environments.
Steps to Prepare the Dataset

Create the Directory Structure:Create the directory where the dataset will be stored, relative to the project root.
mkdir -p data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0


Download the Dataset:Download the validation set from Bridge Data into the 1.0.0 directory.
cd data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0
wget -r -l1 -np -nH --cut-dirs=5 https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/


Explanation of wget options:
-r: Recursive download
-l1: Limit recursion to one level
-np: No parent directories
-nH: No host directories
--cut-dirs=5: Remove five leading directories from the URL path




Update Dataset Path:The dataset path in eval_vla.py is hardcoded as:
base_path = "/home/mundus/gugwugab432/projects/smolvla/data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"

If the dataset is stored elsewhere, update this path in the script.


Alternative: Using TensorFlow Datasets (TFDS)
You can use TensorFlow Datasets to download the dataset, but the file naming may differ. Install TFDS and download the dataset:
pip install tensorflow-datasets
python -c "import tensorflow_datasets as tfds; tfds.load('bridge', download=True)"

The dataset will be saved to ~/tensorflow_datasets/bridge/1.0.0/. Update the create_tf_dataset function in eval_vla.py to use the correct file pattern:
file_pattern = f"{base_path}/{split}-{{:05d}}-of-0{num_files:04d}.tfrecord"

Set base_path = os.path.expanduser('~/tensorflow_datasets/bridge/1.0.0/').
Model Checkpoint
The evaluation script loads a pre-trained model checkpoint from:
checkpoint_path = "./SmolVLM-Base-Fixed/checkpoint-1000/"

Ensure this path points to your trained model checkpoint. If the checkpoint is located elsewhere, update the checkpoint_path variable in eval_vla.py. The script assumes the model was trained with QLoRA and a specific architecture (e.g., SmolVLM-Base). If your model has different configurations, you may need to adjust the model loading logic.
Running the Evaluation
The project includes two scripts:

eval_vla.py: The main Python script for evaluation.
run_eval.sh: A bash script for submitting the evaluation job to a SLURM cluster.

Running on a SLURM Cluster
The run_eval.sh script is configured for a cluster with an A100 GPU (20GB). To run:
sbatch run_eval.sh

Adjust the SLURM directives in run_eval.sh to match your cluster’s configuration:



Directive
Description
Example Value



--job-name
Name of the job
SMOLVLA_Eval


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
24:00:00


Running Locally
Run the Python script directly:
python eval_vla.py

Ensure your system has a compatible GPU and sufficient memory. The script uses 4-bit quantization to reduce memory usage.
Configuration Notes

Model Selection: The script uses SMOL = True to load SmolVLM-Base. If your model is different (e.g., Idefics3), set SMOL = False in eval_vla.py.
Batch Size: The evaluation uses a batch size of 8, adjustable in eval_vla.py based on your GPU memory.
Sample Limit: The script evaluates up to 1000 samples, adjustable via max_samples in the EvalIterableDataset class.

Understanding the Results
The evaluation script logs results to WandB under the project smolvla-evaluation with a run name like SmolVLM-Base-eval-1000. Results include metrics, visualizations, and example predictions.
Metrics

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual actions, both overall and per dimension.
R² Scores: Indicates the proportion of variance in each action dimension explained by the model.
Additional Metrics: Maximum, minimum, and median sample errors.

Visualizations
The script generates comprehensive visualizations, logged to WandB:

Action Plots: Bar charts comparing predicted and target actions for individual samples and batches.
Error Distribution: Histogram of per-sample MSEs.
Scatter Plots: Target vs. predicted values for each action dimension, with R² scores.
3D Trajectories: Visualizations of action trajectories in 3D space for selected dimensions.
Temporal Evolution: Line plots showing predicted and target action values over time steps.
Correlation Heatmap: Matrix showing correlations between predicted and target action dimensions.

Tables

Example Predictions: A table with up to 20 examples, including predicted actions, target actions, and per-sample MSEs.

Accessing Results
View results on the WandB dashboard. Key artifacts include:

eval_mse: Overall MSE
dimension_mse: MSE per action dimension
dimension_r2: R² per action dimension
dimension_errors: Bar chart of MSE by dimension
example_predictions: Table of example predictions

Output Files
Evaluation logs are saved to /home/mundus/gugwugab432/projects/smolvla/output/ as specified in run_eval.sh. Check eval_output_%j.log and eval_error_%j.log for console output and errors.
Contributing
Contributions are welcome! To contribute:

Open an issue on GitHub to discuss changes or report bugs.
Submit a pull request with changes, ensuring:
Code follows PEP 8 style guidelines.
New features include tests or documentation.
Changes align with improving robotic action prediction evaluation.


Test changes locally or on a small dataset subset for compatibility.

License
This project is licensed under the MIT License. See the LICENSE file for details.
