#!/bin/bash
# Multi-node SFT Training Script for Qwen3-32B
# Usage: bash apps/sft/run_training.sh

# Exit on error
set -e

echo "=== Multi-Node Training Setup ==="

# Set HuggingFace offline mode BEFORE Python starts
# This prevents HuggingFace from trying to download datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# SLURM configuration for 32-node allocation
# Topology: 6 blocks with 18 nodes each
# 32 nodes need: 18 (block 1) + 14 (block 2) = 2 blocks minimum
# SLURM_SWITCHES=2 optimizes for network locality
export SLURM_SWITCHES=1

# Optional: WandB configuration
# Uncomment and set your API key if using WandB
# export WANDB_API_KEY="your_api_key_here"

# Optional: Additional NCCL debugging (only if needed)
# export NCCL_DEBUG=INFO

echo "Environment Variables Set:"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
echo "  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  SLURM_SWITCHES=$SLURM_SWITCHES"
echo ""

# Run training
cd /home/ubuntu/Hosseinkh/Forge_Branches/multinode/forge
python -m apps.sft.main --config apps/sft/llama3_8b_multinode.yaml

echo ""
echo "=== Training Complete ==="
