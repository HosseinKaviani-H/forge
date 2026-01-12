# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypedDict

import torch


class TextTrainBatch(TypedDict, total=False):
    """A batch of text training data for forward_backward.

    This TypedDict defines the standard format for text training batches across all
    Forge text trainers. Using TypedDict instead of dataclass allows for more
    flexible serialization and easier integration with existing dict-based code.

    Required Keys:
        input_ids: Input token IDs. Shape: [batch_size, seq_len]
        target_ids: Target token IDs for loss computation. Shape: [batch_size, seq_len]

    Optional Keys:
        target_mask: Mask indicating which tokens to compute loss on.
            Shape: [batch_size, seq_len]. Values are 0 (ignore) or 1 (compute loss).
            If not provided, computes loss on all tokens.
        target_weights: Per-token weights for loss computation.
            Shape: [batch_size, seq_len]. Used for importance weighting, such as
            advantages in RL (GRPO, PPO) or custom loss weighting schemes.
            If not provided, all tokens have weight 1.0.
        extra: Additional tensors for specialized use cases (e.g., ref_logprobs for GRPO).
            Keys are field names, values are tensors. This allows extending the batch
            format without breaking the protocol interface.

    Example:
        >>> batch: TextTrainBatch = {
        >>>     "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        >>>     "target_ids": torch.tensor([[2, 3, 4, 5, 6]]),
        >>>     "target_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        >>>     "target_weights": torch.tensor([[0, 0, 1.0, 0.8, 1.2]]),
        >>>     "extra": {"ref_logprobs": ref_logprobs_tensor},
        >>> }
        >>> result = await trainer.forward_backward(batch)
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    target_weights: torch.Tensor
    extra: dict[str, torch.Tensor]


# Type alias for loss functions used in forward_backward
LossFn = Callable[[dict[str, Any], TextTrainBatch], torch.Tensor]
"""Loss function type for forward_backward.

A loss function takes:
    - outputs: Dict containing model outputs (e.g., {"logits": tensor})
    - batch: The TextTrainBatch that was passed to forward_backward

Returns:
    - loss: A scalar tensor representing the loss to be backpropagated
"""


@dataclass
class ForwardBackwardResult:
    """Result of a forward_backward call.

    Attributes:
        loss: The computed loss value (scalar float).
        metrics: Optional dictionary of additional metrics computed during
            the forward/backward pass (e.g., accuracy, perplexity).
    """

    loss: float
    metrics: dict[str, float]


@dataclass
class OptimStepResult:
    """Result of an optim_step call.

    Attributes:
        step: The current training step number after this optimizer step.
        learning_rate: The current learning rate after this step.
        accumulated_microbatches: Number of microbatches that were accumulated
            before this optimizer step was applied.
    """

    step: int
    learning_rate: float
    accumulated_microbatches: int


@dataclass
class ParallelismConfig:
    """Configuration for distributed parallelism.

    Describes how the model is distributed across devices for training.
    """

    dp_degree: int  # Data parallel degree
    tp_degree: int  # Tensor parallel degree
    pp_degree: int  # Pipeline parallel degree
    cp_degree: int  # Context parallel degree
    ep_degree: int  # Expert parallel degree
    world_size: int  # Total number of processes
    dp_rank: int  # This worker's data parallel rank
    tp_rank: int  # This worker's tensor parallel rank
    device: str  # Device string (e.g., "cuda:0")


@dataclass
class TrainerConfig:
    """Static configuration for a trainer.

    Contains model and parallelism configuration that doesn't change during training.
    """

    model_name: str  # Model identifier (e.g., "meta-llama/Llama-3.1-8B")
    model_config: dict[str, Any]  # Model-specific configuration
    parallelism: ParallelismConfig  # Parallelism settings


@dataclass
class TrainerStatus:
    """Current runtime status of a trainer.

    Contains dynamic information about the current training state.
    """

    step: int  # Current training step
    accumulated_microbatches: int  # Microbatches accumulated since last optim_step


class Trainer(Protocol):
    """Protocol defining the standard interface for Forge trainers.

    This protocol enables swappable trainers across different training recipes
    (SFT, GRPO, PPO, etc.) by defining a common set of methods that all trainers
    must implement.

    The key design principle is separation of concerns:
    - forward_backward: Computes gradients (can be called multiple times for accumulation)
    - optim_step: Applies gradients and updates model weights
    - Custom loss functions are passed as parameters, not baked into the trainer

    Example usage:
        >>> # Forward/backward with custom loss
        >>> result = await trainer.forward_backward.call(batch, custom_loss_fn)
        >>> print(f"Loss: {result.loss}")
        >>>
        >>> # Optimizer step
        >>> step_result = await trainer.optim_step.call()
        >>> print(f"Step: {step_result.step}, LR: {step_result.learning_rate}")
    """

    async def forward_backward(
        self, batch: TextTrainBatch, loss_fn: LossFn | None = None
    ) -> ForwardBackwardResult:
        """Execute forward pass and backward pass for one batch of data.

        This method computes the forward pass through the model, calculates the loss
        using either the provided loss_fn or a default loss, and backpropagates
        the gradients. It does NOT apply the optimizer step.

        Multiple calls to forward_backward accumulate gradients, enabling gradient
        accumulation across microbatches.

        Args:
            batch: TextTrainBatch containing input_ids, target_ids, and optional
                target_mask/target_weights/extra tensors.
            loss_fn: Optional custom loss function. If None, uses the trainer's
                default loss function.

        Returns:
            ForwardBackwardResult containing the loss value and optional metrics.
        """
        ...

    async def optim_step(self) -> OptimStepResult:
        """Apply optimizer step using accumulated gradients, then clear gradients.

        This method applies the optimizer update using all gradients accumulated
        from previous forward_backward calls, steps the learning rate scheduler,
        clears the gradients, and optionally saves a checkpoint.

        Returns:
            OptimStepResult containing step number, learning rate, and
            the number of accumulated microbatches.
        """
        ...

    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying optimizer step.

        Use this to discard accumulated gradients, for example when an
        error occurs during gradient accumulation and you want to restart.
        """
        ...

    async def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass only, without backward pass (for evaluation/inference).

        Args:
            inputs: Dictionary containing model inputs (e.g., {"input_ids": tensor}).

        Returns:
            Model output logits.
        """
        ...

    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        """Save trainer state or weights to persistent storage.

        Args:
            name: Optional checkpoint name/identifier.
            path: Optional base directory or URI.
            weights_only: If True, saves only model weights (not optimizer state).

        Returns:
            Full path/URI where checkpoint was saved.
        """
        ...

    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint.

        Args:
            path: Optional path or URI to the checkpoint to load.
                If None, loads from the default checkpoint location.

        Returns:
            Path/URI that was loaded.
        """
        ...

    async def get_config(self) -> TrainerConfig:
        """Get static trainer and model configuration.

        Returns:
            TrainerConfig containing model name, model config, and parallelism settings.
        """
        ...

    async def get_status(self) -> TrainerStatus:
        """Get current runtime status of the trainer.

        Returns:
            TrainerStatus containing current step and accumulated microbatch count.
        """
        ...
