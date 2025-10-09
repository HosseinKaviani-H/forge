# Complete Guide: Interactive Configuration Notebook

This guide explains step-by-step how to use the interactive configuration notebook for SFT training.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Notebook Step-by-Step](#notebook-step-by-step)
4. [Utility Functions Explained](#utility-functions-explained)
5. [How to Run](#how-to-run)
6. [Common Scenarios](#common-scenarios)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The interactive configuration notebook (`interactive_config_notebook.ipynb`) allows you to:
- Configure SFT training **without YAML files**
- Define configuration interactively in separate cells
- Easily modify parameters and experiment
- Use pre-built templates for common scenarios

### What Problem Does This Solve?

**Before**: You had to edit YAML files, which required:
- External file management
- Reloading files after changes
- Difficult to experiment with different configs

**After**: You can:
- Define everything in the notebook
- Change values in cells and re-run
- See all configurations clearly
- No external file management needed

---

## Architecture Components

Before diving into the notebook, let's understand the components:

### 1. BaseForgeActor (`actor.py`)

**What it is**: An abstract base class that defines the contract for all actors.

**What it does**:
- Handles distributed initialization (sets up multi-GPU environment)
- Manages common attributes (model, optimizer, checkpointer, etc.)
- Defines three required methods that subclasses must implement:
  - `setup()` - Initialize data, checkpoints, etc.
  - `run()` - Main execution logic
  - `cleanup()` - Resource cleanup

**Why it matters**: Provides a consistent interface for different actor types (Trainer, Evaluator, Inferencer, etc.)

### 2. TrainerActor (`trainer_actor.py`)

**What it is**: A concrete implementation of BaseForgeActor for training.

**What it does**:
- Implements the training loop
- Handles forward/backward passes
- Manages checkpointing
- Supports various parallelism strategies (FSDP, Pipeline Parallel, Tensor Parallel)

**Key Methods**:
- `setup()` - Loads tokenizer, dataset, and checkpoints
- `run()` - Executes the training loop
- `forward_backward()` - Performs forward and backward passes
- `train_step()` - Single training step
- `cleanup()` - Closes resources

### 3. SpawnActor (`spawn_actor.py`)

**What it is**: An orchestrator that manages actor lifecycle.

**What it does**:
- Creates actor instances
- Manages the lifecycle: spawn → setup → run → cleanup
- Provides error handling and cleanup guarantees

**Key Methods**:
- `spawn()` - Creates the actor instance
- `setup()` - Calls actor's setup
- `run()` - Calls actor's run
- `cleanup()` - Calls actor's cleanup and stops the mesh
- `run_full_lifecycle()` - Executes all phases automatically

**Why it matters**: Simplifies actor management and ensures proper resource cleanup.

### 4. Utility Functions (`utils.py`)

Helper functions for common operations. See [Utility Functions Explained](#utility-functions-explained) section below.

---

## Notebook Step-by-Step

### Step 1: Import Dependencies

```python
import asyncio
import logging
from omegaconf import OmegaConf, DictConfig

from forge.apps.sft_v2.trainer_actor import TrainerActor
from forge.apps.sft_v2.spawn_actor import SpawnActor, run_actor
```

**What this does**:
- `asyncio` - For async/await operations (actors run asynchronously)
- `logging` - For logging training progress
- `OmegaConf` - For managing configurations (converts dicts to config objects)
- `TrainerActor` - The training actor we'll use
- `SpawnActor`, `run_actor` - For managing actor lifecycle

**Why we need it**: These are the core dependencies for running the actor-based training.

---

### Step 2: Configure Model Settings

```python
model_config = {
    "name": "llama3",
    "flavor": "8B",
    "hf_assets_path": "/tmp/Meta-Llama-3.1-8B-Instruct"
}
```

**What this does**:
- `name` - Model architecture type (e.g., "llama3", "llama2")
- `flavor` - Model size (e.g., "8B", "70B", "405B")
- `hf_assets_path` - Path to the model files (tokenizer, weights, config)

**How to modify**:
- Change `flavor` to use different model sizes
- Update `hf_assets_path` to point to your model location
- Make sure the path contains `tokenizer.json`, `tokenizer_config.json`, and model weights

**Example variations**:
```python
# For a 70B model
model_config = {
    "name": "llama3",
    "flavor": "70B",
    "hf_assets_path": "/path/to/Meta-Llama-3.1-70B"
}
```

---

### Step 3: Configure Process Settings

```python
processes_config = {
    "procs": 8,        # Number of processes
    "with_gpus": True  # Use GPUs
}
```

**What this does**:
- `procs` - Number of parallel processes (usually = number of GPUs)
- `with_gpus` - Whether to use GPUs or CPUs

**How to modify**:
- For single GPU: `"procs": 1`
- For 4 GPUs: `"procs": 4`
- For CPU training: `"with_gpus": False` (not recommended for LLMs)

**Important**: Set `procs` to match your available GPUs!

---

### Step 4: Configure Optimizer Settings

```python
optimizer_config = {
    "name": "AdamW",
    "lr": 1e-5,    # Learning rate
    "eps": 1e-8
}
```

**What this does**:
- `name` - Optimizer type (AdamW is recommended for LLMs)
- `lr` - Learning rate (how fast the model learns)
- `eps` - Epsilon for numerical stability

**How to modify**:
- **Lower learning rate** (e.g., `1e-6`) for fine-tuning
- **Higher learning rate** (e.g., `5e-5`) for pre-training (use with caution)
- Typical range for fine-tuning: `1e-6` to `1e-4`

**Tips**:
- Start conservative with `1e-5` or `2e-5`
- If loss explodes, reduce learning rate
- If training is too slow, slightly increase learning rate

---

### Step 5: Configure Learning Rate Scheduler

```python
lr_scheduler_config = {
    "warmup_steps": 200  # Number of warmup steps
}
```

**What this does**:
- `warmup_steps` - Number of steps to gradually increase learning rate from 0 to `lr`

**Why warmup**: Prevents training instability at the beginning by starting with a low learning rate.

**How to modify**:
- For short training (< 1000 steps): use 10-50 warmup steps
- For medium training (1000-5000 steps): use 100-200 warmup steps
- For long training (> 5000 steps): use 200-500 warmup steps
- Rule of thumb: ~5-10% of total training steps

---

### Step 6: Configure Training Settings

```python
training_config = {
    "local_batch_size": 1,  # Batch size per GPU
    "seq_len": 2048,         # Sequence length
    "max_norm": 1.0,         # Gradient clipping
    "steps": 1000,           # Total training steps
    "compile": False,        # PyTorch compilation
    "dataset": "c4"          # Dataset name
}
```

**What this does**:
- `local_batch_size` - Number of samples per GPU per step
- `seq_len` - Maximum sequence length (in tokens)
- `max_norm` - Gradient clipping threshold (prevents exploding gradients)
- `steps` - Total number of training steps
- `compile` - Enable PyTorch 2.0 compilation (experimental)
- `dataset` - Dataset identifier

**How to modify**:

**For Memory Issues**:
- Reduce `seq_len` (e.g., from 2048 to 1024)
- Reduce `local_batch_size` (e.g., from 2 to 1)
- Both reduce memory usage

**For Faster Training**:
- Increase `local_batch_size` if you have memory
- Use shorter `seq_len` for tasks that don't need long context

**For Quick Testing**:
- Set `steps` to 10-100 for quick validation

**Global batch size** = `local_batch_size` × `procs` × `data_parallel_shard_degree`

---

### Step 7: Configure Parallelism Settings

```python
parallelism_config = {
    "data_parallel_replicate_degree": 1,
    "data_parallel_shard_degree": -1,  # -1 = use all GPUs for FSDP
    "tensor_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "context_parallel_degree": 1,
    "expert_parallel_degree": 1,
    "disable_loss_parallel": False
}
```

**What this does**:

- **Data Parallel Shard Degree (FSDP)**: Splits model parameters across GPUs
  - `-1` means use all available GPUs
  - `8` means split across 8 GPUs
  - Most common strategy for fine-tuning

- **Tensor Parallel Degree**: Splits individual layers across GPUs
  - Use for very large models that don't fit on single GPU even with FSDP
  - `1` means no tensor parallelism

- **Pipeline Parallel Degree**: Splits model into sequential stages
  - Use for extremely large models
  - `1` means no pipeline parallelism

- **Context Parallel Degree**: Splits sequence dimension
  - For very long sequences
  - `1` means no context parallelism

**Common Configurations**:

**Single GPU**:
```python
"data_parallel_shard_degree": 1
```

**8 GPUs with FSDP (recommended)**:
```python
"data_parallel_shard_degree": -1  # or 8
```

**Large Model (70B+) with Tensor Parallelism**:
```python
"data_parallel_shard_degree": 4,
"tensor_parallel_degree": 2
```

---

### Step 8: Configure Checkpoint Settings

```python
checkpoint_config = {
    "enable": True,
    "folder": "/tmp/Meta-Llama-3.1-8B-Instruct/saved_checkpoints",
    "initial_load_path": "/tmp/Meta-Llama-3.1-8B-Instruct/",
    "initial_load_in_hf": True,
    "last_save_in_hf": True,
    "interval": 500,           # Save every N steps
    "async_mode": "disabled"
}
```

**What this does**:
- `enable` - Whether to enable checkpointing
- `folder` - Where to save checkpoints
- `initial_load_path` - Where to load initial weights from
- `initial_load_in_hf` - Load weights in HuggingFace format
- `last_save_in_hf` - Save final checkpoint in HuggingFace format
- `interval` - How often to save (in steps)
- `async_mode` - Async saving mode (use "disabled" for simplicity)

**How to modify**:
- **Save more frequently**: Reduce `interval` (e.g., 100)
- **Save less frequently**: Increase `interval` (e.g., 1000)
- **Resume training**: Point `initial_load_path` to your checkpoint folder

**Important**: Make sure `folder` path exists and has enough disk space!

---

### Step 9: Configure Activation Checkpointing

```python
activation_checkpoint_config = {
    "mode": "selective",
    "selective_ac_option": "op"
}
```

**What this does**:
- Saves memory by recomputing activations during backward pass instead of storing them
- `mode` - Checkpointing mode ("selective" or "full")
- `selective_ac_option` - Which operations to checkpoint

**Memory vs Speed Trade-off**:
- **Activation checkpointing ON**: Lower memory, slower training
- **Activation checkpointing OFF**: Higher memory, faster training

**When to use**: Enable when running out of memory.

---

### Step 10: Configure Communication Settings

```python
comm_config = {
    "trace_buf_size": 0
}
```

**What this does**:
- Configuration for distributed communication (required by TorchTitan)
- Usually you don't need to modify this

---

### Step 11: Combine All Configurations

```python
complete_config = {
    "comm": comm_config,
    "model": model_config,
    "processes": processes_config,
    "optimizer": optimizer_config,
    "lr_scheduler": lr_scheduler_config,
    "training": training_config,
    "parallelism": parallelism_config,
    "checkpoint": checkpoint_config,
    "activation_checkpoint": activation_checkpoint_config
}

cfg = OmegaConf.create(complete_config)
```

**What this does**:
- Combines all configuration sections into one complete config
- Converts to OmegaConf format (allows dot notation access)

**Prints**: The complete configuration in YAML format for review

---

### Step 12: Run Training (Simple Way)

```python
await run_actor(TrainerActor, cfg)
```

**What this does**:
- Spawns the trainer actor
- Runs setup (loads data, model, checkpoints)
- Runs training loop
- Cleans up resources
- All in one line!

**When to use**: When you want fully automatic training with no manual intervention.

---

### Alternative: Manual Lifecycle Control

For more control over the training process:

#### Create and Spawn the Actor

```python
spawner = SpawnActor(TrainerActor, cfg)
actor = await spawner.spawn()
```

**What this does**:
- Creates a spawner with your config
- Spawns the actor instance (allocates resources, initializes distributed environment)

#### Setup the Actor

```python
await spawner.setup()
```

**What this does**:
- Loads tokenizer from `hf_assets_path`
- Loads training dataset
- Initializes model
- Loads checkpoint if specified

**At this point**: You could inspect the actor state before training:
```python
print(f"Current step: {actor.current_step}")
print(f"Device: {actor.device}")
```

#### Run Training

```python
await spawner.run()
```

**What this does**:
- Executes the training loop
- Iterates through batches
- Performs forward/backward passes
- Updates weights
- Saves checkpoints at intervals

#### Cleanup

```python
await spawner.cleanup()
```

**What this does**:
- Closes checkpointer
- Closes logger
- Stops the actor mesh
- Frees resources

**When to use manual control**:
- When you want to inspect state between phases
- When you want to modify configuration between setup and run
- For debugging purposes

---

## Utility Functions Explained

The `utils.py` module provides reusable helper functions:

### 1. `setup_tokenizer()`

```python
def setup_tokenizer(
    hf_assets_path: str,
    tokenizer_filename: str = "tokenizer.json",
    tokenizer_config_filename: str = "tokenizer_config.json",
    generation_config_filename: str = "generation_config.json",
) -> HuggingFaceModelTokenizer
```

**What it does**:
- Loads a HuggingFace tokenizer from the model assets directory
- Initializes tokenizer with config and generation settings

**Parameters**:
- `hf_assets_path` - Path to directory containing tokenizer files
- Other parameters are filenames (usually don't need to change)

**Returns**: Initialized `HuggingFaceModelTokenizer` object

**Example**:
```python
tokenizer = setup_tokenizer("/tmp/Meta-Llama-3.1-8B-Instruct")
```

**When to use**: If you need to use the tokenizer independently (e.g., for preprocessing data)

---

### 2. `setup_sft_dataloader()`

```python
def setup_sft_dataloader(
    tokenizer: HuggingFaceModelTokenizer,
    dataset_path: str,
    dataset_split: str,
    target_tokens_per_pack: int,
    batch_size: int,
    device: torch.device,
    padding_idx: int = 0,
    message_transform: Optional[Any] = None,
) -> StatefulDataLoader
```

**What it does**:
- Creates a dataloader for supervised fine-tuning
- Handles data loading, tokenization, and packing
- Returns a StatefulDataLoader (can save/restore state for checkpointing)

**Parameters**:
- `tokenizer` - Tokenizer to use for text processing
- `dataset_path` - HuggingFace dataset name (e.g., "yahma/alpaca-cleaned")
- `dataset_split` - Which split to use ("train", "validation", "test")
- `target_tokens_per_pack` - Sequence length (same as `seq_len` in config)
- `batch_size` - Batch size (same as `local_batch_size` in config)
- `device` - Which device to move tensors to
- `padding_idx` - Token ID for padding (usually 0)
- `message_transform` - Transform to convert dataset format (default: AlpacaToMessages)

**Returns**: Configured `StatefulDataLoader`

**Example**:
```python
dataloader = setup_sft_dataloader(
    tokenizer=tokenizer,
    dataset_path="yahma/alpaca-cleaned",
    dataset_split="train",
    target_tokens_per_pack=2048,
    batch_size=4,
    device=torch.device("cuda"),
)
```

**When to use**: If you want to create a custom dataloader outside of TrainerActor

---

### 3. `create_context_parallel_context()`

```python
def create_context_parallel_context(
    parallel_dims: ParallelDims,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model_parts: list,
    rotate_method: str,
)
```

**What it does**:
- Creates context for context parallelism (splits sequence across GPUs)
- Returns None if context parallelism is disabled

**Parameters**:
- `parallel_dims` - Parallel dimensions configuration
- `inputs` - Input tensor
- `labels` - Label tensor
- `model_parts` - List of model parts
- `rotate_method` - Rotation method for context parallel

**Returns**: Context parallel context or None

**When to use**: Internally used by TrainerActor. You rarely need to call this directly.

---

### 4. `move_batch_to_device()`

```python
def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]
```

**What it does**:
- Moves all tensors in a batch dictionary to the specified device
- Leaves non-tensor values unchanged

**Parameters**:
- `batch` - Dictionary containing batch data
- `device` - Target device (e.g., `torch.device("cuda")`)

**Returns**: Batch with tensors moved to device

**Example**:
```python
batch = {"tokens": tensor, "labels": tensor, "metadata": "some_string"}
batch = move_batch_to_device(batch, torch.device("cuda"))
```

**When to use**: Useful when manually processing batches

---

### 5. `log_training_step()`

```python
def log_training_step(
    step: int,
    total_steps: int,
    loss: torch.Tensor,
    logger: logging.Logger,
)
```

**What it does**:
- Logs training progress in a formatted way
- Shows current step, total steps, and loss value

**Parameters**:
- `step` - Current training step
- `total_steps` - Total number of training steps
- `loss` - Current loss tensor
- `logger` - Logger instance

**Example output**:
```
Step 100/1000 | Loss: 2.3456
```

**When to use**: Internally used by TrainerActor. You can use it for custom logging.

---

## How to Run

### Prerequisites

1. **Download Model**:
```bash
export HF_HUB_DISABLE_XET=1
forge download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir /tmp/Meta-Llama-3.1-8B-Instruct
```

2. **Check GPU Availability**:
```bash
nvidia-smi  # Should show your GPUs
```

### Running the Notebook

#### Option 1: Using Jupyter Notebook

1. **Start Jupyter**:
```bash
cd /home/hosseinkh/TorchForge/forge
jupyter notebook
```

2. **Open the notebook**:
   - Navigate to `apps/sft_v2/interactive_config_notebook.ipynb`
   - Click to open

3. **Run cells sequentially**:
   - Click on first cell, press `Shift + Enter` to run
   - Continue through all cells
   - Modify configuration cells as needed
   - Run Step 12 to start training

#### Option 2: Using VS Code

1. **Open notebook in VS Code**:
   - File → Open → `interactive_config_notebook.ipynb`

2. **Select Python kernel**:
   - Click "Select Kernel" in top right
   - Choose your Python environment

3. **Run cells**:
   - Click "Run Cell" button on each cell
   - Or press `Shift + Enter`

#### Option 3: Using Command Line (with simplified entry point)

```bash
cd /home/hosseinkh/TorchForge/forge
python -m apps.sft_v2.notebook_main --config apps/sft_v2/llama3_8b.yaml
```

Note: This uses a YAML file, but you can use the notebook for interactive config.

---

## Common Scenarios

### Scenario 1: Quick Test (1 GPU, 100 steps)

```python
# Modify these cells:
processes_config = {"procs": 1, "with_gpus": True}
training_config = {
    "local_batch_size": 1,
    "seq_len": 1024,
    "steps": 100,  # Just 100 steps
    ...
}
```

**Expected time**: 5-10 minutes on A100

### Scenario 2: Full Training (8 GPUs, 5000 steps)

```python
processes_config = {"procs": 8, "with_gpus": True}
training_config = {
    "local_batch_size": 2,
    "seq_len": 2048,
    "steps": 5000,
    ...
}
parallelism_config = {
    "data_parallel_shard_degree": -1,  # Use all 8 GPUs
    ...
}
```

**Expected time**: Several hours depending on hardware

### Scenario 3: Memory-Constrained Training

```python
training_config = {
    "local_batch_size": 1,  # Small batch
    "seq_len": 1024,         # Shorter sequence
    ...
}
activation_checkpoint_config = {
    "mode": "selective",  # Enable AC for memory savings
    ...
}
```

**Use when**: Running out of GPU memory

### Scenario 4: Resume from Checkpoint

```python
checkpoint_config = {
    "enable": True,
    "folder": "/path/to/previous/checkpoints",
    "initial_load_path": "/path/to/previous/checkpoints/step_1000",
    "interval": 500,
    ...
}
```

**Use when**: Continuing training from a saved checkpoint

---

## Troubleshooting

### Problem: "CUDA out of memory"

**Solutions**:
1. Reduce `seq_len` (e.g., from 2048 to 1024)
2. Reduce `local_batch_size` (e.g., from 2 to 1)
3. Enable activation checkpointing
4. Use more GPUs with FSDP

### Problem: "Loss is NaN or exploding"

**Solutions**:
1. Reduce learning rate (e.g., from `1e-5` to `1e-6`)
2. Increase gradient clipping (`max_norm` from 1.0 to 0.5)
3. Increase warmup steps

### Problem: "Training is too slow"

**Solutions**:
1. Increase `local_batch_size` if memory allows
2. Use more GPUs
3. Reduce `seq_len` if your task doesn't need long context
4. Enable compilation (`compile: True`)

### Problem: "Cannot find tokenizer files"

**Solutions**:
1. Check `hf_assets_path` is correct
2. Ensure path contains `tokenizer.json` and `tokenizer_config.json`
3. Re-download model if files are missing

### Problem: "Actor spawning fails"

**Solutions**:
1. Check you have enough GPUs for `procs`
2. Verify CUDA is available (`torch.cuda.is_available()`)
3. Check no other processes are using GPUs

---

## Summary

**Key Takeaways**:

1. **Interactive Configuration**: Define all settings in notebook cells, no YAML needed
2. **Step-by-Step**: Configure model, processes, optimizer, training, parallelism, checkpoints separately
3. **Two Ways to Run**: Simple (`run_actor()`) or manual (lifecycle control)
4. **Utility Functions**: Helper functions for tokenization, data loading, device management
5. **Templates Provided**: Quick test, multi-GPU, memory-efficient configs ready to use
6. **Flexible**: Easy to modify parameters and experiment

**Next Steps**:
1. Download your model
2. Open the notebook
3. Modify configuration cells for your needs
4. Run Step 12 to start training
5. Monitor logs for progress

Happy Training! 🚀
