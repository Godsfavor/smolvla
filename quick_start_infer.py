import torch
from torch.nn import Linear, MSELoss
from peft import LoraConfig, PeftModel
import tensorflow as tf
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
import io
import numpy as np
import wandb
import sys
import transformers
import random
import matplotlib.pyplot as plt
import os

# Clear cached transformers module if needed
if 'transformers' in sys.modules:
    del sys.modules['transformers']

# Import transformers components
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, BitsAndBytesConfig

# Configuration
SMOL = True  # Set to match your training setting
model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"
checkpoint_path = "./SmolVLM-Base-Fixed/checkpoint-1000/"  # Path to your saved checkpoint

# Initialize WandB with proper config tracking for evaluation
wandb.init(
    project="smolvla-evaluation", 
    name=f"{model_id.split('/')[-1]}-eval-1000",
    config={
        "model": model_id.split('/')[-1],
        "evaluation_only": True,
        "checkpoint_path": checkpoint_path,
    }
)

# Set up processor
processor = AutoProcessor.from_pretrained(model_id)

# TFRecord feature description (same as in your training script)
feature_description = {
    "steps/action": tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
    "steps/language_embedding": tf.io.FixedLenSequenceFeature([512], tf.float32, allow_missing=True),
    "steps/language_instruction": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/is_terminal": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "steps/is_last": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "steps/is_first": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "steps/reward": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    "steps/discount": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    "steps/observation/image_0": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/image_1": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/image_2": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/image_3": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "steps/observation/state": tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
    "episode_metadata/episode_id": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_0": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_1": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_2": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_image_3": tf.io.FixedLenFeature([], tf.int64),
    "episode_metadata/has_language": tf.io.FixedLenFeature([], tf.int64),
}

# Helper functions for dataset handling (reused from training script)
def parse_example(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    steps = {
        "action": parsed["steps/action"],
        "language_embedding": parsed["steps/language_embedding"],
        "language_instruction": parsed["steps/language_instruction"],
        "is_terminal": tf.cast(parsed["steps/is_terminal"], tf.bool),
        "is_last": tf.cast(parsed["steps/is_last"], tf.bool),
        "is_first": tf.cast(parsed["steps/is_first"], tf.bool),
        "reward": parsed["steps/reward"],
        "discount": parsed["steps/discount"],
        "observation": {
            "image_0": parsed["steps/observation/image_0"],
            "image_1": parsed["steps/observation/image_1"],
            "image_2": parsed["steps/observation/image_2"],
            "image_3": parsed["steps/observation/image_3"],
            "state": parsed["steps/observation/state"],
        }
    }
    episode_metadata = {
        "file_path": parsed["episode_metadata/file_path"],
        "episode_id": parsed["episode_metadata/episode_id"],
        "has_image_0": tf.cast(parsed["episode_metadata/has_image_0"], tf.bool),
        "has_image_1": tf.cast(parsed["episode_metadata/has_image_1"], tf.bool),
        "has_image_2": tf.cast(parsed["episode_metadata/has_image_2"], tf.bool),
        "has_image_3": tf.cast(parsed["episode_metadata/has_image_3"], tf.bool),
        "has_language": tf.cast(parsed["episode_metadata/has_language"], tf.bool),
    }
    return {"steps": steps, "episode_metadata": episode_metadata}

def flatten_episode(episode):
    steps = episode["steps"]
    num_steps = tf.shape(steps["action"])[0]
    def get_step(i):
        return {
            "action": steps["action"][i],
            "language_instruction": steps["language_instruction"][i],
            "image_0": steps["observation"]["image_0"][i],
            "language_embedding": steps["language_embedding"][i],
            "is_terminal": steps["is_terminal"][i],
            "is_last": steps["is_last"][i],
            "is_first": steps["is_first"][i],
            "reward": steps["reward"][i],
            "discount": steps["discount"][i],
        }
    indices = tf.range(num_steps)
    return tf.data.Dataset.from_tensor_slices(indices).map(get_step)

def transform_step(step):
    image_bytes = step["image_0"].numpy()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Extract only the instruction part
    raw_instruction = step["language_instruction"].numpy().decode("utf-8")
    if "Action:" in raw_instruction:
        question = raw_instruction.split("Action:")[0].strip()
    else:
        question = raw_instruction
        
    action = step["action"].numpy()
    language_embedding = step["language_embedding"].numpy()
    reward = step["reward"].numpy().item()
    discount = step["discount"].numpy().item()
    is_terminal = bool(step["is_terminal"].numpy().item())
    is_last = bool(step["is_last"].numpy().item())
    is_first = bool(step["is_first"].numpy().item())
    return {
        "image": image,
        "question": question,
        "language_embedding": language_embedding,
        "action": action,
        "reward": reward,
        "discount": discount,
        "is_terminal": is_terminal,
        "is_last": is_last,
        "is_first": is_first,
    }

def create_tf_dataset():
    # Let's use validation dataset for evaluation
    base_path = "/home/mundus/gugwugab432/projects/smolvla/data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"
    num_files = 128
    file_pattern = f"{base_path}/bridge_dataset-val.tfrecord-{{:05d}}-of-00128"
    
    tfrecord_files = [file_pattern.format(i) for i in range(num_files)]
    
    raw_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    raw_dataset = raw_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    flattened_dataset = parsed_dataset.flat_map(flatten_episode)
    flattened_dataset = flattened_dataset.prefetch(tf.data.AUTOTUNE)
    
    return flattened_dataset

def tf_dataset_to_pytorch_generator(tf_dataset):
    for example in tf_dataset:
        yield transform_step(example)

class EvalIterableDataset(IterableDataset):
    def __init__(self, tf_dataset, max_samples=500):
        self.tf_dataset = tf_dataset.take(max_samples)
        self.max_samples = max_samples
        
    def __iter__(self):
        return tf_dataset_to_pytorch_generator(self.tf_dataset)

def collate_fn(examples):
    texts = []
    images = []
    actions = []
    for example in examples:
        images.append([example["image"]])
        questions = example["question"]
        
        # Ensure consistent message formatting with no leakage
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What action should the robot take to "},
                {"type": "image"},
                {"type": "text", "text": questions}
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors=None)
        texts.append(text)
        actions.append(example["action"])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    batch["actions"] = torch.tensor(np.array(actions), dtype=torch.float32)
    return batch

# Function to log action comparison plots
def log_action_plots(predicted, target, step=0, prefix="eval"):
    """Create and log matplotlib plots comparing predicted and target actions"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart comparing predicted vs target for each action dimension (first sample)
    ax1 = axes[0]
    x = np.arange(7)  # 7 dimensions
    width = 0.35
    ax1.bar(x - width/2, predicted[0], width, label='Predicted')
    ax1.bar(x + width/2, target[0], width, label='Target')
    ax1.set_xlabel('Action Dimension')
    ax1.set_ylabel('Value')
    ax1.set_title('Action Comparison - Sample 1')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Dim {i}' for i in range(7)])
    ax1.legend()
    
    # Plot 2: Error per dimension - all samples in batch
    errors = np.square(np.array(predicted) - np.array(target))
    mean_errors = errors.mean(axis=0)
    ax2 = axes[1]
    ax2.bar(x, mean_errors)
    ax2.set_xlabel('Action Dimension')
    ax2.set_ylabel('MSE')
    ax2.set_title(f'Mean Squared Error by Dimension (Batch)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Dim {i}' for i in range(7)])
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({
        f"{prefix}_action_plots": wandb.Image(fig),
        "step": step
    })
    
    plt.close(fig)
    
    # Also create a line plot comparing predictions vs targets across samples
    if len(predicted) > 1:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # First 7 plots for each dimension
        for i in range(7):
            ax = axes[i]
            sample_range = min(len(predicted), 10)  # Up to 10 samples
            x = np.arange(sample_range)
            pred_vals = [predicted[j][i] for j in range(sample_range)]
            target_vals = [target[j][i] for j in range(sample_range)]
            
            ax.plot(x, pred_vals, 'bo-', label='Predicted')
            ax.plot(x, target_vals, 'ro-', label='Target')
            ax.set_title(f'Action Dim {i}')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')
            if i == 0:
                ax.legend()
                
        # Last plot for overall MSE
        total_mse = np.mean(errors, axis=1)
        axes[7].bar(np.arange(min(len(total_mse), 10)), total_mse[:10])
        axes[7].set_title('Total MSE per Sample')
        axes[7].set_xlabel('Sample')
        axes[7].set_ylabel('MSE')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({
            f"{prefix}_dimension_plots": wandb.Image(fig),
            "step": step
        })
        
        plt.close(fig)

# Plot action trajectory in 3D
def plot_action_trajectory(predictions, targets, action_dims=[0, 1, 2], step=0):
    """Create a 3D visualization of predicted vs target action trajectories"""
    if len(predictions) < 5:
        return  # Need at least a few samples
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates for selected dimensions
    x_dim, y_dim, z_dim = action_dims
    
    # Plot predicted trajectory
    xs_pred = [p[x_dim] for p in predictions]
    ys_pred = [p[y_dim] for p in predictions]
    zs_pred = [p[z_dim] for p in predictions]
    
    # Plot target trajectory
    xs_target = [t[x_dim] for t in targets]
    ys_target = [t[y_dim] for t in targets]
    zs_target = [t[z_dim] for t in targets]
    
    # Plot both trajectories
    ax.plot(xs_pred, ys_pred, zs_pred, 'b-o', label='Predicted')
    ax.plot(xs_target, ys_target, zs_target, 'r-o', label='Target')
    
    # Connect corresponding points with lines
    for i in range(len(xs_pred)):
        ax.plot([xs_pred[i], xs_target[i]], 
                [ys_pred[i], ys_target[i]], 
                [zs_pred[i], zs_target[i]], 
                'k--', alpha=0.3)
    
    ax.set_xlabel(f'Action Dim {x_dim}')
    ax.set_ylabel(f'Action Dim {y_dim}')
    ax.set_zlabel(f'Action Dim {z_dim}')
    ax.set_title('3D Action Trajectory Comparison')
    ax.legend()
    
    wandb.log({
        f"action_trajectory_3d_dims_{x_dim}_{y_dim}_{z_dim}": wandb.Image(fig),
        "step": step
    })
    plt.close(fig)

# Load the model
print("Loading model and checkpoint...")

# First load the base model with BNB config (for quantized model)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    output_hidden_states=True
)

# Define the same model architecture as used during training
class VLAWithActionHead(torch.nn.Module):
    def __init__(self, base_model, action_head):
        super().__init__()
        self.base_model = base_model
        self.action_head = action_head

    def forward(self, input_ids, pixel_values, attention_mask, pixel_attention_mask=None, actions=None):
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        action_logits = self.action_head(last_hidden_state[:, -1, :])

        loss = None
        if actions is not None:
            loss_fct = MSELoss()
            loss = loss_fct(action_logits, actions)

        return {"loss": loss, "logits": action_logits}
        
    # Helper method to get device
    @property
    def device(self):
        # Find a parameter and get its device
        for param in self.parameters():
            return param.device
        # Fallback
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the action head
action_head = Linear(base_model.config.text_config.hidden_size, out_features=7)

# Load the PEFT (LoRA) model
print("Loading PEFT model...")

# Try to find the adapter path in the checkpoint directory
adapter_path = checkpoint_path
if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
    # Check if there's an adapter_model directory
    if os.path.exists(os.path.join(checkpoint_path, "adapter_model")):
        adapter_path = os.path.join(checkpoint_path, "adapter_model")

# Load the PEFT adapter onto the base model
print(f"Loading adapter from: {adapter_path}")
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

# Create your VLAWithActionHead with the PEFT model as the base
model = VLAWithActionHead(peft_model, action_head)

# Move the action head to the same device as the model and match the dtype
device = next(model.parameters()).device
# Get the dtype of the model parameters
model_dtype = next(peft_model.parameters()).dtype
print(f"Model is using dtype: {model_dtype}")
# Convert action head to same dtype and device
model.action_head = model.action_head.to(device).to(model_dtype)
print(f"Model loaded on device: {device} with dtype: {model_dtype}")


# Set the model to evaluation mode
model.eval()

# Create the evaluation dataset
print("Creating evaluation dataset...")
tf_dataset = create_tf_dataset()
eval_dataset = EvalIterableDataset(tf_dataset, max_samples=1000)  # Adjust max_samples as needed

# Create a PyTorch DataLoader for the evaluation dataset
from torch.utils.data import DataLoader
batch_size = 8  # Adjust based on your memory constraints
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Run evaluation
print("Starting evaluation...")
model.eval()

all_predictions = []
all_targets = []
all_losses = []
step = 0

with torch.no_grad():
    for batch in eval_dataloader:
        # Extract actions from the batch
        actions = batch.pop("actions").to(device)
        
        # Move remaining inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get the model outputs manually
        base_outputs = model.base_model(**inputs, output_hidden_states=True)
        last_hidden_state = base_outputs.hidden_states[-1]
        predictions = model.action_head(last_hidden_state[:, -1, :])
        
        # Calculate loss
        loss_fct = MSELoss()
        loss = loss_fct(predictions, actions)
        
        # Convert to numpy arrays
        predictions_np = predictions.cpu().numpy()
        targets_np = actions.cpu().numpy()
        
        # Store predictions and targets for later analysis
        all_predictions.extend(predictions_np)
        all_targets.extend(targets_np)
        all_losses.append(loss.item())
        
        # Log current batch metrics
        if step % 5 == 0:  # Log every 5 batches
            print(f"Processing batch {step}, total samples so far: {len(all_predictions)}")
            log_action_plots(predictions_np, targets_np, step, prefix="eval_batch")
        
        step += 1
        
        # Break after a reasonable number of batches to avoid overwhelming WandB
        if step >= 50:  # Adjust as needed
            break



# Convert to numpy arrays for analysis
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Calculate overall MSE
mse = np.mean(np.square(all_predictions - all_targets))
print(f"Overall MSE: {mse:.6f}")

# Log overall evaluation metrics
wandb.log({
    "eval_mse": mse,
    "eval_loss_mean": np.mean(all_losses) if all_losses else None,
})

# Generate detailed plots and visualizations
print("Generating comprehensive evaluation visualizations...")

# Log action plots with more samples
log_action_plots(all_predictions[:100], all_targets[:100], prefix="overall_eval")

# Create visualization of errors by dimension
dim_errors = np.square(all_predictions - all_targets).mean(axis=0)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(7)  # 7 dimensions
ax.bar(x, dim_errors)
ax.set_xlabel('Action Dimension')
ax.set_ylabel('MSE')
ax.set_title('MSE by Action Dimension')
ax.set_xticks(x)
ax.set_xticklabels([f'Dim {i}' for i in range(7)])

for i, v in enumerate(dim_errors):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

wandb.log({"dimension_errors": wandb.Image(fig)})
plt.close(fig)

# Create error distribution histogram
all_sample_errors = np.mean(np.square(all_predictions - all_targets), axis=1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(all_sample_errors, bins=30)
ax.set_xlabel('Mean Squared Error')
ax.set_ylabel('Count')
ax.set_title('Distribution of Sample MSEs')
wandb.log({"error_distribution": wandb.Image(fig)})
plt.close(fig)

# Create scatter plots for each dimension
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

r2_values = []
for i in range(7):
    ax = axes[i]
    x = [t[i] for t in all_targets]
    y = [p[i] for p in all_predictions]
    
    ax.scatter(x, y, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_title(f'Dimension {i}: Target vs Prediction')
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    
    # Calculate R²
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    r2 = correlation_xy**2
    r2_values.append(r2)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes)

# Add R² values in the last subplot
axes[7].bar(range(7), r2_values)
axes[7].set_title('R² by Dimension')
axes[7].set_xlabel('Dimension')
axes[7].set_ylabel('R²')
axes[7].set_ylim(0, 1)

for i, v in enumerate(r2_values):
    axes[7].text(i, v + 0.05, f'{v:.4f}', ha='center')

plt.tight_layout()
wandb.log({"dimension_scatter_plots": wandb.Image(fig)})
plt.close(fig)

# Create 3D trajectory visualizations
sample_size = min(30, len(all_predictions))
for dims in [[0, 1, 2], [3, 4, 5], [0, 2, 4]]:
    plot_action_trajectory(all_predictions[:sample_size], all_targets[:sample_size], dims)

# Create a temporal evolution plot for each dimension
fig, axes = plt.subplots(7, 1, figsize=(12, 24))
time_steps = range(min(30, len(all_predictions)))

for dim in range(7):
    ax = axes[dim]
    pred_vals = [all_predictions[t][dim] for t in time_steps]
    target_vals = [all_targets[t][dim] for t in time_steps]
    
    ax.plot(time_steps, pred_vals, 'b-o', label='Predicted')
    ax.plot(time_steps, target_vals, 'r-o', label='Target')
    ax.set_title(f'Dimension {dim} Over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    if dim == 0:
        ax.legend()

plt.tight_layout()
wandb.log({"temporal_action_evolution": wandb.Image(fig)})
plt.close(fig)

# Create a correlation heatmap across dimensions
corr_matrix = np.zeros((14, 14))

# Convert predictions and targets to arrays if they aren't already
all_predictions_array = np.array(all_predictions)
all_targets_array = np.array(all_targets)

# Create combined data
# We need to convert the data to the right shape
all_data = np.vstack([
    all_predictions_array.T,  # shape: (7, n_samples)
    all_targets_array.T       # shape: (7, n_samples)
])

# Calculate correlations
for i in range(14):
    for j in range(14):
        if i < all_data.shape[0] and j < all_data.shape[0]:  # Make sure indices are valid
            corr_matrix[i, j] = np.corrcoef(all_data[i], all_data[j])[0, 1]
        else:
            corr_matrix[i, j] = 0  # Default value if dimensions are missing


            
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(14))
ax.set_yticks(np.arange(14))
ax.set_xticklabels([f'P{i}' for i in range(7)] + [f'T{i}' for i in range(7)])
ax.set_yticklabels([f'P{i}' for i in range(7)] + [f'T{i}' for i in range(7)])
plt.colorbar(im)
ax.set_title('Correlation Matrix Between Predicted and Target Dimensions')

# Add correlation values in the cells
for i in range(14):
    for j in range(14):
        color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center", color=color, fontsize=7)

plt.tight_layout()
wandb.log({"correlation_heatmap": wandb.Image(fig)})
plt.close(fig)

# Create a final summary table with key metrics
wandb.log({
    "overall_mse": mse,
    "dimension_mse": {f"dim_{i}_mse": dim_errors[i] for i in range(7)},
    "dimension_r2": {f"dim_{i}_r2": r2_values[i] for i in range(7)},
    "max_sample_error": np.max(all_sample_errors),
    "min_sample_error": np.min(all_sample_errors),
    "median_sample_error": np.median(all_sample_errors),
})

# Create example predictions table
example_table = wandb.Table(columns=["example_id"] + 
                           [f"pred_dim_{i}" for i in range(7)] + 
                           [f"target_dim_{i}" for i in range(7)] + 
                           ["sample_mse"])

for i in range(min(20, len(all_predictions))):
    row = [i]
    # Add predicted values
    for dim in range(7):
        row.append(float(all_predictions[i][dim]))
    # Add target values
    for dim in range(7):
        row.append(float(all_targets[i][dim]))
    # Add sample MSE
    row.append(float(all_sample_errors[i]))
    example_table.add_data(*row)

wandb.log({"example_predictions": example_table})

print("Evaluation complete! Results logged to WandB.")
wandb.finish()