# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for notebook-based SFT training.
This module provides a clean API for interactive training in Jupyter notebooks.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import torch

from apps.sft_v2.main import ForgeSFTRecipe
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Configuration Builders
# ============================================================================


def create_model_config(
    name: str = "llama3",
    flavor: str = "8B",
    hf_assets_path: str = "/tmp/Meta-Llama-3.1-8B-Instruct",
) -> Dict[str, Any]:
    """
    Create model configuration.

    Args:
        name: Model architecture name (e.g., 'llama3', 'llama2')
        flavor: Model size (e.g., '8B', '70B')
        hf_assets_path: Path to HuggingFace model assets

    Returns:
        Dictionary with model configuration
    """
    return {
        "name": name,
        "flavor": flavor,
        "hf_assets_path": hf_assets_path,
    }


def create_optimizer_config(
    name: str = "AdamW",
    lr: float = 1e-5,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
) -> Dict[str, Any]:
    """
    Create optimizer configuration.

    Args:
        name: Optimizer name (e.g., 'AdamW', 'Adam', 'SGD')
        lr: Learning rate
        eps: Epsilon for numerical stability
        weight_decay: L2 regularization coefficient
        betas: Coefficients for computing running averages

    Returns:
        Dictionary with optimizer configuration
    """
    return {
        "name": name,
        "lr": lr,
        "eps": eps,
        "weight_decay": weight_decay,
        "betas": list(betas),
    }


def create_lr_scheduler_config(
    warmup_steps: int = 200,
    decay_steps: Optional[int] = None,
    min_lr: float = 0.0,
) -> Dict[str, Any]:
    """
    Create learning rate scheduler configuration.

    Args:
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps (None = no decay)
        min_lr: Minimum learning rate

    Returns:
        Dictionary with LR scheduler configuration
    """
    config = {"warmup_steps": warmup_steps}
    if decay_steps is not None:
        config["decay_steps"] = decay_steps
    if min_lr > 0:
        config["min_lr"] = min_lr
    return config


def create_training_config(
    local_batch_size: int = 1,
    seq_len: int = 2048,
    max_norm: float = 1.0,
    steps: int = 1000,
    dataset: str = "c4",
    compile: bool = False,
) -> Dict[str, Any]:
    """
    Create training configuration.

    Args:
        local_batch_size: Batch size per GPU
        seq_len: Sequence length
        max_norm: Gradient clipping max norm
        steps: Total training steps
        dataset: Dataset name
        compile: Whether to use torch.compile

    Returns:
        Dictionary with training configuration
    """
    return {
        "local_batch_size": local_batch_size,
        "seq_len": seq_len,
        "max_norm": max_norm,
        "steps": steps,
        "dataset": dataset,
        "compile": compile,
    }


def create_parallelism_config(
    data_parallel_replicate_degree: int = 1,
    data_parallel_shard_degree: int = -1,
    tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    context_parallel_degree: int = 1,
    expert_parallel_degree: int = 1,
    disable_loss_parallel: bool = False,
) -> Dict[str, Any]:
    """
    Create parallelism configuration.

    Args:
        data_parallel_replicate_degree: Data parallel replication
        data_parallel_shard_degree: Data parallel sharding (FSDP), -1 = auto
        tensor_parallel_degree: Tensor parallelism degree
        pipeline_parallel_degree: Pipeline parallelism degree
        context_parallel_degree: Context parallelism degree
        expert_parallel_degree: Expert parallelism degree (for MoE)
        disable_loss_parallel: Whether to disable loss parallelism

    Returns:
        Dictionary with parallelism configuration
    """
    return {
        "data_parallel_replicate_degree": data_parallel_replicate_degree,
        "data_parallel_shard_degree": data_parallel_shard_degree,
        "tensor_parallel_degree": tensor_parallel_degree,
        "pipeline_parallel_degree": pipeline_parallel_degree,
        "context_parallel_degree": context_parallel_degree,
        "expert_parallel_degree": expert_parallel_degree,
        "disable_loss_parallel": disable_loss_parallel,
    }


def create_checkpoint_config(
    enable: bool = True,
    folder: str = "/tmp/checkpoints",
    initial_load_path: Optional[str] = None,
    initial_load_in_hf: bool = True,
    last_save_in_hf: bool = True,
    interval: int = 500,
    async_mode: str = "disabled",
) -> Dict[str, Any]:
    """
    Create checkpoint configuration.

    Args:
        enable: Whether to enable checkpointing
        folder: Path to save checkpoints
        initial_load_path: Path to load initial checkpoint from
        initial_load_in_hf: Load initial checkpoint in HF format
        last_save_in_hf: Save last checkpoint in HF format
        interval: Steps between checkpoints
        async_mode: Async checkpoint mode ('disabled', 'async', etc.)

    Returns:
        Dictionary with checkpoint configuration
    """
    return {
        "enable": enable,
        "folder": folder,
        "initial_load_path": initial_load_path,
        "initial_load_in_hf": initial_load_in_hf,
        "last_save_in_hf": last_save_in_hf,
        "interval": interval,
        "async_mode": async_mode,
    }


def create_activation_checkpoint_config(
    mode: str = "selective",
    selective_ac_option: str = "op",
) -> Dict[str, Any]:
    """
    Create activation checkpointing configuration.

    Args:
        mode: Activation checkpoint mode ('selective', 'full', 'none')
        selective_ac_option: Selective AC option ('op', 'layer', etc.)

    Returns:
        Dictionary with activation checkpoint configuration
    """
    return {
        "mode": mode,
        "selective_ac_option": selective_ac_option,
    }


def create_process_config(
    procs: int = 8,
    with_gpus: bool = True,
    hosts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create process configuration.

    Args:
        procs: Number of processes per host
        with_gpus: Whether to use GPUs
        hosts: Number of hosts (None = single node)

    Returns:
        Dictionary with process configuration
    """
    config = {
        "procs": procs,
        "with_gpus": with_gpus,
    }
    if hosts is not None:
        config["hosts"] = hosts
    return config


# ============================================================================
# Configuration Assembly
# ============================================================================


def build_config(
    model_config: Dict[str, Any],
    optimizer_config: Dict[str, Any],
    lr_scheduler_config: Dict[str, Any],
    training_config: Dict[str, Any],
    parallelism_config: Dict[str, Any],
    checkpoint_config: Dict[str, Any],
    activation_checkpoint_config: Dict[str, Any],
    process_config: Dict[str, Any],
) -> DictConfig:
    """
    Build complete configuration from component configs.

    Args:
        model_config: Model configuration
        optimizer_config: Optimizer configuration
        lr_scheduler_config: LR scheduler configuration
        training_config: Training configuration
        parallelism_config: Parallelism configuration
        checkpoint_config: Checkpoint configuration
        activation_checkpoint_config: Activation checkpoint configuration
        process_config: Process configuration

    Returns:
        Complete OmegaConf DictConfig
    """
    config = {
        "comm": {"trace_buf_size": 0},
        "model": model_config,
        "optimizer": optimizer_config,
        "lr_scheduler": lr_scheduler_config,
        "training": training_config,
        "parallelism": parallelism_config,
        "checkpoint": checkpoint_config,
        "activation_checkpoint": activation_checkpoint_config,
        "processes": process_config,
    }

    return OmegaConf.create(config)


# ============================================================================
# Training Functions
# ============================================================================


async def create_recipe(config: DictConfig):
    """
    Create and return a ForgeSFTRecipe actor.

    Args:
        config: Complete configuration

    Returns:
        ForgeSFTRecipe actor instance
    """
    process_cfg = config.pop("processes")
    recipe = await ForgeSFTRecipe.options(**process_cfg).as_actor(config)
    logger.info("Recipe created successfully")
    return recipe


async def setup_recipe(recipe):
    """
    Setup the recipe (load model, initialize data loaders, etc.).

    Args:
        recipe: ForgeSFTRecipe actor instance
    """
    logger.info("Setting up recipe...")
    await recipe.setup.call()
    logger.info("Recipe setup complete")


async def train_recipe(recipe):
    """
    Run training on the recipe.

    Args:
        recipe: ForgeSFTRecipe actor instance
    """
    logger.info("Starting training...")
    await recipe.train.call()
    logger.info("Training complete")


async def cleanup_recipe(recipe):
    """
    Cleanup recipe resources.

    Args:
        recipe: ForgeSFTRecipe actor instance
    """
    logger.info("Cleaning up...")
    await recipe.cleanup.call()
    await recipe.mesh.stop()
    logger.info("Cleanup complete")


# ============================================================================
# High-Level Training API
# ============================================================================


async def run_training(config: DictConfig):
    """
    Run complete training pipeline with the given configuration.

    Args:
        config: Complete configuration

    Raises:
        Exception: If training fails
    """
    # Create recipe
    recipe = await create_recipe(config)

    # Setup
    await setup_recipe(recipe)

    # Train
    await train_recipe(recipe)

    # Cleanup
    await cleanup_recipe(recipe)


def train(config: DictConfig):
    """
    Synchronous wrapper for run_training.

    Args:
        config: Complete configuration
    """
    asyncio.run(run_training(config))


# ============================================================================
# Display Utilities
# ============================================================================


def print_config(config: DictConfig, title: str = "Configuration"):
    """
    Pretty print configuration.

    Args:
        config: Configuration to print
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(OmegaConf.to_yaml(config))
    print(f"{'='*60}\n")


def summarize_config(config: DictConfig):
    """
    Print a summary of the configuration.

    Args:
        config: Configuration to summarize
    """
    print("\n" + "=" * 60)
    print("Configuration Summary".center(60))
    print("=" * 60)

    print(f"\n📦 Model:")
    print(f"  • Name: {config.model.name}")
    print(f"  • Flavor: {config.model.flavor}")
    print(f"  • Path: {config.model.hf_assets_path}")

    print(f"\n⚙️  Training:")
    print(f"  • Steps: {config.training.steps}")
    print(f"  • Batch Size: {config.training.local_batch_size}")
    print(f"  • Sequence Length: {config.training.seq_len}")
    print(f"  • Dataset: {config.training.dataset}")

    print(f"\n🔧 Optimizer:")
    print(f"  • Name: {config.optimizer.name}")
    print(f"  • Learning Rate: {config.optimizer.lr}")
    print(f"  • Warmup Steps: {config.lr_scheduler.warmup_steps}")

    print(f"\n🔀 Parallelism:")
    print(
        f"  • Data Parallel (Replicate): {config.parallelism.data_parallel_replicate_degree}"
    )
    print(
        f"  • Data Parallel (Shard/FSDP): {config.parallelism.data_parallel_shard_degree}"
    )
    print(f"  • Tensor Parallel: {config.parallelism.tensor_parallel_degree}")
    print(f"  • Pipeline Parallel: {config.parallelism.pipeline_parallel_degree}")

    print(f"\n💾 Checkpointing:")
    print(f"  • Enabled: {config.checkpoint.enable}")
    print(f"  • Folder: {config.checkpoint.folder}")
    print(f"  • Interval: {config.checkpoint.interval} steps")

    print(f"\n🖥️  Resources:")
    if "hosts" in config.processes:
        print(f"  • Hosts: {config.processes.hosts}")
    print(f"  • Processes per host: {config.processes.procs}")
    print(f"  • GPUs: {config.processes.with_gpus}")

    print("\n" + "=" * 60 + "\n")
