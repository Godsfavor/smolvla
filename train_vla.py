import torch
from torch.nn import Linear, MSELoss
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import tensorflow as tf
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
import io
import numpy as np
import wandb
import sys
import transformers
import random  # Added for proper validation split
import matplotlib.pyplot as plt

# Clear cached transformers module if needed
if 'transformers' in sys.modules:
    del sys.modules['transformers']

# Import transformers components
from transformers import AutoProcessor, Idefics3ForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer, TrainerCallback, __version__ as transformers_version

# Print transformers version to confirm
print(f"Transformers version: {transformers_version}")

USE_LORA = False
USE_QLORA = False
SMOL = True

model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"

processor = AutoProcessor.from_pretrained(model_id)

# Update the model loading section to freeze vision model but train language model
# This replaces the code block where model is loaded and parameters are set

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False

    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            _attn_implementation="flash_attention_2",
            device_map="auto",
            output_hidden_states=True
        )
        
        # Explicitly freeze the vision model parameters
        for param in base_model.model.vision_model.parameters():
            param.requires_grad = False
            
        # Make sure language model parameters are trainable (no change needed for QLORA)
        # In QLORA, language model parameters will be trained through the LoRA adapters

        action_head = Linear(base_model.config.text_config.hidden_size, out_features=7)

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

            def gradient_checkpointing_enable(self, *args, **kwargs):
                self.base_model.gradient_checkpointing_enable(*args, **kwargs)

        model = prepare_model_for_kbit_training(base_model)
        model = VLAWithActionHead(model, action_head)
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

else:
    # For full-parameter training (non-LoRA)
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")
    
    # Freeze the vision model parameters
    for param in base_model.model.vision_model.parameters():
        param.requires_grad = False
    
    # Ensure text model parameters are trainable
    for param in base_model.model.text_model.parameters():
        param.requires_grad = True
        
    # Create action head
    action_head = Linear(base_model.config.text_config.hidden_size, out_features=7)
    
    # Wrap in the same custom model as for LoRA
    model = VLAWithActionHead(base_model, action_head)

# TFRecord feature description
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
    
    # Extract only the instruction part - ensure no leakage from potential answers
    raw_instruction = step["language_instruction"].numpy().decode("utf-8")
    # Assuming format might be "Instruction: X. Action: Y" - only keep instruction part
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

# Create TF dataset from multiple files with proper caching and buffering
def create_tf_dataset(split):
    base_path = "/home/mundus/gugwugab432/projects/smolvla/data/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"
    if split == "train":
        num_files = 1024
        file_pattern = f"{base_path}/bridge_dataset-train.tfrecord-{{:05d}}-of-01024"
    elif split == "val":
        num_files = 128
        file_pattern = f"{base_path}/bridge_dataset-val.tfrecord-{{:05d}}-of-00128"
    else:
        raise ValueError("Invalid split")
    
    # Shuffle the file list to ensure different order each epoch
    tfrecord_files = [file_pattern.format(i) for i in range(num_files)]
    if split == "train":
        random.shuffle(tfrecord_files)
    
    # Use interleave for parallel file reading (better performance)
    raw_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    raw_dataset = raw_dataset.shuffle(buffer_size=len(tfrecord_files)) if split == "train" else raw_dataset
    
    raw_dataset = raw_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply parsing and use proper buffering for shuffling
    parsed_dataset = raw_dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if split == "train":
        parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
    
    flattened_dataset = parsed_dataset.flat_map(flatten_episode)
    
    # Add prefetching for better performance
    flattened_dataset = flattened_dataset.prefetch(tf.data.AUTOTUNE)
    
    return flattened_dataset

# Generator to convert TF dataset to PyTorch with proper separation
def tf_dataset_to_pytorch_generator(tf_dataset):
    for example in tf_dataset:
        yield transform_step(example)

# Use consistent approaches for both train and validation
class TFRecordIterableDataset(IterableDataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset

    def __iter__(self):
        return tf_dataset_to_pytorch_generator(self.tf_dataset)

# Use IterableDataset for validation too to maintain consistency
class ValIterableDataset(IterableDataset):
    def __init__(self, tf_dataset, max_samples=1000):
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

# Create datasets with consistent handling approach
train_tf_dataset = create_tf_dataset("train")
train_dataset = TFRecordIterableDataset(train_tf_dataset)

val_tf_dataset = create_tf_dataset("val")
# Use IterableDataset for validation too
eval_dataset = ValIterableDataset(val_tf_dataset, max_samples=1000)

# Training arguments with improved settings
model_name = model_id.split("/")[-1]
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,  # Added explicit eval batch size
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=5e-5,  # Reduced initial learning rate to prevent too fast descent
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir=f"./{model_name}-Fixed-full_finetuning",
    hub_model_id=f"{model_name}-Fixed-full_finetuning",
    report_to="wandb",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    label_names=["actions"],
    max_steps=1000,
    eval_strategy="steps",
    eval_steps=50,
    # Added seed for reproducibility
    seed=42,
    # Added fp16 auto mixed precision
    fp16_full_eval=True,
)

# Initialize WandB with proper config tracking
wandb.init(
    project="smolvla-training-fixed", 
    name=f"{model_name}-run-fixed-full_finetuning",
    config={
        "model": model_name,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "use_qlora": USE_QLORA,
        "use_lora": USE_LORA,
    }
)

# Function to create and log action comparison plots
def log_action_plots(predicted, target, step, prefix="eval"):
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

# Custom callback that uses separate eval data and generates plots
class LogActionCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.eval_batches = None
        self.train_batches = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Pre-generate a few validation batches to use consistently
        eval_dataloader = self.trainer.get_eval_dataloader()
        self.eval_batches = []
        for i, batch in enumerate(eval_dataloader):
            if i >= 2:  # Store 2 batches
                break
            self.eval_batches.append(batch)
            
        # Also get a few training batches for consistent monitoring
        train_dataloader = self.trainer.get_train_dataloader()
        self.train_batches = []
        for i, batch in enumerate(train_dataloader):
            if i >= 2:  # Store 2 batches
                break
            self.train_batches.append(batch)
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            trainer = self.trainer
            
            # Get device properly
            device = self.get_model_device(trainer.model)
            
            # Process validation batch
            if self.eval_batches:
                batch = self.eval_batches[0]  # Use first validation batch
                
                # Move batch to the appropriate device - FIXED LINE
                batch = {k: v.to(device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = trainer.model(**batch)
                
            # Rest of the method remains the same...
                
                # Fixed: Log scalar values, not media
                predicted_actions = outputs['logits'].cpu().numpy()
                target_actions = batch["actions"].cpu().numpy()
                
                # Log individual action components as scalars
                for i in range(min(2, len(predicted_actions))):
                    for j in range(7):  # 7 action dimensions
                        wandb.log({
                            f"eval_sample_{i}_dim_{j}_pred": float(predicted_actions[i][j]),
                            f"eval_sample_{i}_dim_{j}_target": float(target_actions[i][j]),
                            "step": state.global_step
                        })
                
                # Log action prediction plots
                log_action_plots(predicted_actions, target_actions, state.global_step, prefix="eval")
                
                # Add MSE per action component
                for i in range(7):  # 7 action dimensions
                    mse = ((outputs['logits'][:, i] - batch["actions"][:, i])**2).mean().item()
                    wandb.log({f"eval_action_{i}_mse": mse, "step": state.global_step})
            
            # Process training batch for comparison
            # Process training batch for comparison
            if self.train_batches:
                batch = self.train_batches[0]  # Use first training batch
                
                # Move batch to the appropriate device - FIXED LINE
                batch = {k: v.to(device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = trainer.model(**batch)
                
                # Rest of the code remains the same...
                
                # Fixed: Log scalar values, not media
                predicted_actions = outputs['logits'].cpu().numpy()
                target_actions = batch["actions"].cpu().numpy()
                
                # Log action prediction plots for training data
                log_action_plots(predicted_actions, target_actions, state.global_step, prefix="train")


    # Add this helper method to the LogActionCallback class
    def get_model_device(self, model):
        """Helper to get device from PEFT models which might not have .device attribute"""
        # Try different ways to access device
        if hasattr(model, 'device'):
            return model.device
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
            return model.base_model.device
        elif hasattr(model, 'module') and hasattr(model.module, 'device'):
            return model.module.device
        else:
            # Find a parameter and get its device
            for param in model.parameters():
                return param.device
            # Fallback
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Initialize callback and trainer
callback = LogActionCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    callbacks=[callback],
)
callback.trainer = trainer

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()
print("Training completed successfully")
print("Evaluation results:", eval_results)

# Add final model quality check with detailed plots
def final_eval_with_details():
    eval_dataloader = trainer.get_eval_dataloader()
    full_eval_results = []
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    for batch in eval_dataloader:
        batch = {k: v.to(trainer.model.device) if hasattr(v, 'to') else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = trainer.model(**batch)
        
        # Calculate per-dimension MSE
        batch_results = []
        predictions = outputs['logits'].cpu().numpy()
        targets = batch["actions"].cpu().numpy()
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        
        for i in range(len(predictions)):
            pred = predictions[i]
            true = targets[i]
            dim_errors = ((pred - true)**2)
            
            batch_results.append({
                "prediction": pred.tolist(),
                "target": true.tolist(),
                "mse_per_dim": dim_errors.tolist(),
                "total_mse": np.mean(dim_errors)
            })
        
        full_eval_results.extend(batch_results)
        if len(full_eval_results) >= 100:  # Limit to 100 examples for detailed analysis
            break
    
    # Log example predictions and distribution of errors
    all_errors = np.array([res["total_mse"] for res in full_eval_results])
    dim_errors = np.array([res["mse_per_dim"] for res in full_eval_results])
    
    # Create final evaluation plots with more samples
    all_predictions_array = np.array(all_predictions[:100])
    all_targets_array = np.array(all_targets[:100])
    log_action_plots(all_predictions_array, all_targets_array, trainer.state.global_step, prefix="final_eval")
    
    # Create distribution plot of errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_errors, bins=20)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of MSE across samples')
    wandb.log({
        "final_error_distribution": wandb.Image(fig),
    })
    plt.close(fig)
    
    # Create heatmap of predictions vs targets
    if len(all_predictions) >= 20:
        fig, ax = plt.subplots(figsize=(12, 8))
        sample_count = min(20, len(all_predictions))
        
        # Create heatmap data
        heatmap_data = np.zeros((sample_count, 14))  # 7 dims for pred, 7 for target
        for i in range(sample_count):
            heatmap_data[i, :7] = all_predictions[i]
            heatmap_data[i, 7:] = all_targets[i]
        
        im = ax.imshow(heatmap_data, cmap='coolwarm')
        ax.set_yticks(np.arange(sample_count))
        ax.set_yticklabels([f'Sample {i}' for i in range(sample_count)])
        ax.set_xticks(np.arange(14))
        ax.set_xticklabels([f'P{i}' for i in range(7)] + [f'T{i}' for i in range(7)])
        plt.colorbar(im)
        ax.set_title('Prediction vs Target Heatmap')
        
        wandb.log({
            "prediction_target_heatmap": wandb.Image(fig),
        })
        plt.close(fig)
    
    # Create scatter plots for each dimension
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i in range(7):
        ax = axes[i]
        x = [t[i] for t in all_targets[:100]]
        y = [p[i] for p in all_predictions[:100]]
        
        ax.scatter(x, y, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f'Dimension {i}: Target vs Prediction')
        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
    
    # Add R² values in the last subplot
    r2_values = []
    for i in range(7):
        x = np.array([t[i] for t in all_targets[:100]])
        y = np.array([p[i] for p in all_predictions[:100]])
        
        # Calculate R²
        correlation_matrix = np.corrcoef(x, y)
        correlation_xy = correlation_matrix[0, 1]
        r2 = correlation_xy**2
        r2_values.append(r2)
    
    axes[7].bar(range(7), r2_values)
    axes[7].set_title('R² by Dimension')
    axes[7].set_xlabel('Dimension')
    axes[7].set_ylabel('R²')
    axes[7].set_ylim(0, 1)
    
    plt.tight_layout()
    wandb.log({
        "final_scatter_plots": wandb.Image(fig),
    })
    plt.close(fig)
    
    wandb.log({
        "final_eval_mean_mse": np.mean(all_errors),
        "final_eval_median_mse": np.median(all_errors),
        "final_eval_max_mse": np.max(all_errors),
        "final_eval_example_predictions": wandb.Table(
            columns=["prediction", "target", "mse"],
            data=[[str(res["prediction"]), str(res["target"]), res["total_mse"]] 
                 for res in full_eval_results[:10]]  # Show first 10
        )
    })
    
    # Per dimension analysis
    for dim in range(7):
        dim_mse = np.mean(dim_errors[:, dim])
        wandb.log({f"final_eval_dim_{dim}_mse": dim_mse})
    
    return np.mean(all_errors)

# Run detailed final evaluation
final_mse = final_eval_with_details()
print(f"Final detailed MSE: {final_mse}")

# Additional visualization: Create 3D plot of action trajectories over time
def plot_action_trajectory(predictions, targets, action_dims=[0, 1, 2]):
    """Create a 3D visualization of predicted vs target action trajectories"""
    # We'll visualize the first 3 dimensions of the action space by default
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
        "action_trajectory_3d": wandb.Image(fig)
    })
    plt.close(fig)

# Get a larger batch of validation data for trajectory visualization
def visualize_trajectory():
    eval_dataloader = trainer.get_eval_dataloader()
    model.eval()
    
    predictions_over_time = []
    targets_over_time = []
    
    # Get predictions for consecutive samples
    for i, batch in enumerate(eval_dataloader):
        batch = {k: v.to(trainer.model.device) if hasattr(v, 'to') else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = trainer.model(**batch)
        
        predictions = outputs['logits'].cpu().numpy()
        targets = batch["actions"].cpu().numpy()
        
        for j in range(len(predictions)):
            predictions_over_time.append(predictions[j])
            targets_over_time.append(targets[j])
        
        if len(predictions_over_time) >= 30:
            break
    
    if len(predictions_over_time) >= 10:
        # Generate different trajectory visualizations
        for dims in [[0, 1, 2], [3, 4, 5], [0, 2, 4]]:
            plot_action_trajectory(predictions_over_time[:20], targets_over_time[:20], dims)
        
        # Create a temporal evolution plot for each dimension
        fig, axes = plt.subplots(7, 1, figsize=(12, 24))
        time_steps = range(len(predictions_over_time[:30]))
        
        for dim in range(7):
            ax = axes[dim]
            pred_vals = [p[dim] for p in predictions_over_time[:30]]
            target_vals = [t[dim] for t in targets_over_time[:30]]
            
            ax.plot(time_steps, pred_vals, 'b-o', label='Predicted')
            ax.plot(time_steps, target_vals, 'r-o', label='Target')
            ax.set_title(f'Dimension {dim} Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            if dim == 0:
                ax.legend()
        
        plt.tight_layout()
        wandb.log({
            "temporal_action_evolution": wandb.Image(fig)
        })
        plt.close(fig)

# Run the trajectory visualization
visualize_trajectory()

# Log final model metadata
wandb.run.summary.update({
    "total_training_steps": trainer.state.global_step,
    "final_model_loss": eval_results["eval_loss"] if "eval_loss" in eval_results else "N/A",
    "final_mse": final_mse,
    "model_type": "SmolVLM" if SMOL else "Idefics3-8B",
    "used_qlora": USE_QLORA,
    "training_completed": True
})

# Save a sample artifact with example code for prediction
sample_usage = """
# Sample code to use the trained model
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
base_model = Idefics3ForConditionalGeneration.from_pretrained("HuggingFaceTB/SmolVLM-Base")
model = PeftModel.from_pretrained(base_model, "./SmolVLM-Base-Fixed-full_finetuning")

def predict_action(image, instruction):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "What action should the robot take to "},
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors=None)
    inputs = processor(text=text, images=[[image]], return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states[-1]
    predicted_action = model.action_head(hidden_states[:, -1, :])
    return predicted_action.cpu().numpy()[0]
"""

wandb.run.log_text("model_usage_sample", sample_usage)

# Close the wandb run
wandb.finish()
