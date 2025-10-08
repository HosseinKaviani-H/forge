# 🚀 SFT Training Notebook Guide

This directory contains an interactive Jupyter notebook experience for training Language Models with Supervised Fine-Tuning (SFT).

## 📁 Files

### Core Files
- **`sft_training_notebook.ipynb`** - Main Jupyter notebook for interactive training
- **`notebook_utils.py`** - Utility functions for notebook-based training
- **`main.py`** - Original command-line training script (unchanged)

### Configuration Files
- **`llama3_8b.yaml`** - Original single-node config
- **`llama3_8b_single_node.yaml`** - Single-node config without provisioner
- **`llama3_8b_slurm_multinode.yaml`** - Multi-node config with SLURM
- **`llama3_8b_local.yaml`** - Local testing config

## 🎯 Quick Start

### 1. Open the Notebook

```bash
cd /home/hosseinkh/forge
jupyter notebook apps/sft_v2/sft_training_notebook.ipynb
```

Or in VS Code:
- Open `apps/sft_v2/sft_training_notebook.ipynb`
- Select Python kernel
- Run cells sequentially

### 2. Configure Training

The notebook is organized into sections:

1. **📦 Model Configuration** - Choose model and path
2. **⚙️ Training Configuration** - Set hyperparameters
3. **🔧 Optimizer Configuration** - Configure optimizer and LR scheduler
4. **🔀 Parallelism Configuration** - Set distributed training strategy
5. **💾 Checkpoint Configuration** - Configure checkpointing
6. **🖥️ Resource Configuration** - Set number of GPUs/nodes
7. **☁️ Provisioner Configuration** (optional) - For multi-node SLURM

### 3. Run Training

Execute the "Run Training!" cell to start training with your configuration.

## 📚 Using the Utility Library

The `notebook_utils.py` module provides a clean API for training:

### Configuration Builders

```python
from apps.sft_v2 import notebook_utils as nb

# Create model config
model_config = nb.create_model_config(
    name="llama3",
    flavor="8B",
    hf_assets_path="/path/to/model"
)

# Create training config
training_config = nb.create_training_config(
    steps=1000,
    local_batch_size=1,
    seq_len=2048
)

# Create optimizer config
optimizer_config = nb.create_optimizer_config(
    name="AdamW",
    lr=1e-5
)

# ... configure other components

# Build complete config
config = nb.build_config(
    model_config=model_config,
    training_config=training_config,
    optimizer_config=optimizer_config,
    # ... other configs
)
```

### Training Functions

```python
# Simple: run everything
nb.train(config)

# Advanced: step-by-step control
import asyncio

async def custom_training():
    # Initialize
    await nb.initialize_provisioner(config)

    # Create and setup
    recipe = await nb.create_recipe(config)
    await nb.setup_recipe(recipe)

    # Train
    await nb.train_recipe(recipe)

    # Cleanup
    await nb.cleanup_recipe(recipe)
    await nb.shutdown_provisioner(config)

asyncio.run(custom_training())
```

### Display Utilities

```python
# Print summary
nb.summarize_config(config)

# Print full YAML
nb.print_config(config, title="My Config")
```

## 🔧 Configuration Functions Reference

### Model Configuration

```python
nb.create_model_config(
    name: str = "llama3",
    flavor: str = "8B",
    hf_assets_path: str = "/tmp/Meta-Llama-3.1-8B-Instruct"
)
```

### Training Configuration

```python
nb.create_training_config(
    local_batch_size: int = 1,
    seq_len: int = 2048,
    max_norm: float = 1.0,
    steps: int = 1000,
    dataset: str = "c4",
    compile: bool = False
)
```

### Optimizer Configuration

```python
nb.create_optimizer_config(
    name: str = "AdamW",
    lr: float = 1e-5,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999)
)
```

### LR Scheduler Configuration

```python
nb.create_lr_scheduler_config(
    warmup_steps: int = 200,
    decay_steps: Optional[int] = None,
    min_lr: float = 0.0
)
```

### Parallelism Configuration

```python
nb.create_parallelism_config(
    data_parallel_replicate_degree: int = 1,
    data_parallel_shard_degree: int = -1,  # -1 = auto (FSDP)
    tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    context_parallel_degree: int = 1,
    expert_parallel_degree: int = 1,
    disable_loss_parallel: bool = False
)
```

### Checkpoint Configuration

```python
nb.create_checkpoint_config(
    enable: bool = True,
    folder: str = "/tmp/checkpoints",
    initial_load_path: Optional[str] = None,
    initial_load_in_hf: bool = True,
    last_save_in_hf: bool = True,
    interval: int = 500,
    async_mode: str = "disabled"
)
```

### Activation Checkpoint Configuration

```python
nb.create_activation_checkpoint_config(
    mode: str = "selective",  # 'selective', 'full', 'none'
    selective_ac_option: str = "op"
)
```

### Process Configuration

```python
# Single node
nb.create_process_config(
    procs: int = 8,
    with_gpus: bool = True,
    hosts: Optional[int] = None
)

# Multi-node
nb.create_process_config(
    procs: int = 8,
    with_gpus: bool = True,
    hosts: int = 4  # 4 nodes
)
```

### Provisioner Configuration (Multi-Node Only)

```python
nb.create_provisioner_config(
    launcher: str = "slurm",
    job_name: str = "sft_training",
    partition: Optional[str] = None,
    time: Optional[str] = None,
    account: Optional[str] = None
)
```

## 📖 Example Configurations

### Quick Test (Single GPU, 10 steps)

```python
model_config = nb.create_model_config(
    name="llama3",
    flavor="8B",
    hf_assets_path="/path/to/model"
)

training_config = nb.create_training_config(
    steps=10,
    local_batch_size=1
)

process_config = nb.create_process_config(procs=1)

# ... configure other components with defaults
```

### Single Node, 8 GPUs, FSDP

```python
parallelism_config = nb.create_parallelism_config(
    data_parallel_shard_degree=-1  # Use all 8 GPUs with FSDP
)

process_config = nb.create_process_config(procs=8)

# No provisioner needed
provisioner_config = None
```

### Multi-Node, 4×8 GPUs, Tensor Parallel

```python
parallelism_config = nb.create_parallelism_config(
    data_parallel_shard_degree=16,  # 32 GPUs / 2 TP = 16 FSDP
    tensor_parallel_degree=2
)

process_config = nb.create_process_config(
    procs=8,
    hosts=4
)

provisioner_config = nb.create_provisioner_config(
    launcher="slurm",
    job_name="sft_multinode",
    partition="gpu_partition",
    time="24:00:00"
)
```

## 🎓 Advanced Usage

### Custom Training Loop

You can modify the training loop by creating your own recipe class:

```python
from apps.sft_v2.main import ForgeSFTRecipe

class CustomRecipe(ForgeSFTRecipe):
    async def train(self):
        # Custom training logic
        dataloader = iter(self.train_dataloader)

        for step in range(self.num_training_steps):
            batch = next(dataloader)
            # Custom batch processing
            self.train_step(batch)
```

### Experiment Tracking

Integrate with your favorite tracking tool:

```python
import wandb

# Initialize tracking
wandb.init(project="sft-training", config=config)

# Train
nb.train(config)

# Log results
wandb.log({"final_step": config.training.steps})
```

### Config Variations

Generate multiple configs for hyperparameter sweeps:

```python
learning_rates = [1e-5, 5e-5, 1e-4]
configs = []

for lr in learning_rates:
    optimizer_config = nb.create_optimizer_config(lr=lr)
    config = nb.build_config(
        # ... other configs
        optimizer_config=optimizer_config
    )
    configs.append(config)

# Train all configs
for config in configs:
    nb.train(config)
```

## 🔍 Debugging Tips

### Start Simple

1. **Use 1 GPU first**:
   ```python
   process_config = nb.create_process_config(procs=1)
   ```

2. **Run few steps**:
   ```python
   training_config = nb.create_training_config(steps=10)
   ```

3. **Disable compilation**:
   ```python
   training_config = nb.create_training_config(compile=False)
   ```

### Common Issues

**Memory Errors:**
- Reduce batch size or sequence length
- Enable FSDP: `data_parallel_shard_degree=-1`
- Enable activation checkpointing: `mode="selective"` or `"full"`

**Slow Training:**
- Increase batch size if memory allows
- Enable compilation: `compile=True`
- Use tensor parallelism for large models

**Actor Timeout Errors:**
- Make sure you're not using provisioner config on single node
- Check SLURM availability with `sinfo`
- See `TROUBLESHOOTING_MULTINODE.md` for details

## 📦 Saving and Loading Configs

### Save Config

```python
from omegaconf import OmegaConf

config_path = "my_config.yaml"
with open(config_path, 'w') as f:
    OmegaConf.save(config, f)
```

### Load Config

```python
from omegaconf import OmegaConf

config = OmegaConf.load("my_config.yaml")
nb.train(config)
```

## 🚀 Next Steps

1. **Start with the notebook**: Open `sft_training_notebook.ipynb` and follow along
2. **Try a test run**: Configure for 10 steps with 1 GPU
3. **Scale up**: Increase to 8 GPUs with FSDP
4. **Go multi-node**: Configure SLURM provisioner for cluster training

## 📚 Additional Resources

- **`MULTINODE_SFT_V2_GUIDE.md`** - Detailed guide on multi-node training
- **`TROUBLESHOOTING_MULTINODE.md`** - Troubleshooting guide for multi-node issues
- **`main.py`** - Original implementation for reference

## 🤝 Contributing

To add new configuration options:

1. Add a `create_*_config()` function in `notebook_utils.py`
2. Update `build_config()` to include the new config
3. Add a new cell in the notebook to configure it
4. Update this README

## ⚖️ License

Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the BSD-style license found in the LICENSE file.
