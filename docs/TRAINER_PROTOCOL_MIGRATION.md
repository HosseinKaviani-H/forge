# Trainer Protocol Migration Guide

This document explains the Trainer protocol implementation in `TitanTrainer` and how GRPO was migrated to use the new protocol methods.

## Overview

The Trainer protocol (defined in `forge/api/trainer.py`) provides a standardized interface for trainers, enabling:
- **Swappable trainers** across different training recipes (SFT, GRPO, PPO)
- **Consistent API** for forward/backward passes, optimizer steps, and checkpointing
- **Decoupled loss functions** from the trainer implementation

## Protocol Methods

The Trainer protocol defines these core methods:

| Method | Purpose |
|--------|---------|
| `forward_backward(batch, loss_fn)` | Execute forward + backward pass |
| `optim_step()` | Apply optimizer step, update LR scheduler |
| `clear_gradients()` | Zero gradients without stepping |
| `forward(inputs)` | Inference-only forward pass |
| `save(name, path, weights_only)` | Save checkpoint |
| `load(path)` | Load checkpoint |
| `get_config()` | Get model/parallelism configuration |
| `get_status()` | Get current training status |

## TitanTrainer Implementation

### File: `src/forge/actors/trainer/titan.py`

TitanTrainer implements the Trainer protocol on top of TorchTitan's ForgeEngine.

### Key Components

#### 1. forward_backward Method

```python
@endpoint
async def forward_backward(
    self, batch: TextTrainBatch, loss_fn: LossFn | None = None
) -> ForwardBackwardResult:
    """Execute forward pass and backward pass for one batch of data."""

    # Move batch tensors to device
    input_ids = batch.input_ids.to(self.engine.device)
    target_ids = batch.target_ids.to(self.engine.device)

    # Move extra tensors to device (for RL: ref_logprobs, etc.)
    if batch.extra is not None:
        for key, value in batch.extra.items():
            if isinstance(value, torch.Tensor):
                batch.extra[key] = value.to(self.engine.device)

    with self.engine.train_context(optional_context_parallel_ctx):
        logits = model_parts[0](input_ids)

        # Use custom loss_fn if provided
        if loss_fn is not None:
            outputs = {"logits": logits}
            loss = loss_fn(outputs, batch)
        else:
            loss = self.loss(logits, **kwargs)

        loss.backward()

    # All-reduce loss across DP ranks
    torch.distributed.all_reduce(loss)

    return ForwardBackwardResult(loss=loss_val, metrics={})
```

**Key Features:**
- Accepts `TextTrainBatch` (standardized batch format)
- Accepts optional `loss_fn` (allows custom loss functions like GRPO loss)
- Handles `batch.extra` for RL-specific tensors
- Returns `ForwardBackwardResult` with loss and metrics

#### 2. optim_step Method

```python
@endpoint
async def optim_step(self) -> OptimStepResult:
    """Apply optimizer step using accumulated gradients."""

    current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
    accumulated = self._accumulated_microbatches

    self.engine.optimizers.step()
    self.engine.lr_schedulers.step()
    self.engine.optimizers.zero_grad()

    self.step += 1
    self._accumulated_microbatches = 0

    # Save checkpoint if needed
    self.engine.checkpointer.save(...)

    return OptimStepResult(
        step=self.step,
        learning_rate=current_lr,
        accumulated_microbatches=accumulated,
    )
```

**Key Features:**
- Applies optimizer step and LR scheduler step
- Clears gradients
- Handles automatic checkpointing
- Returns step info for logging

## GRPO Migration

### Before: Legacy `train_step` Approach

```python
# Old GRPO main.py (simplified)
async def continuous_training():
    while training_step < max_steps:
        inputs, targets = await replay_buffer.sample.call_one(...)

        # Legacy: single train_step combines everything
        loss = await trainer.train_step.call(inputs, targets)

        await trainer.push_weights.call(training_step)
        await policy.update_weights.fanout(training_step)
```

**Problems:**
- `train_step` is monolithic - combines forward, backward, optimizer step
- `inputs` and `targets` are raw dicts - no type safety
- Loss function is baked into the trainer
- Hard to reuse trainer for different training algorithms

### After: Protocol-Based Approach

#### Step 1: Standardize Batch Format

**File: `src/forge/rl/collate.py`**

```python
def collate(batches: list[Group]) -> list[TextTrainBatch]:
    """Collates episode groups into TextTrainBatch objects."""

    for batch in batches:
        # Build TextTrainBatch with RL tensors in extra
        text_batch = TextTrainBatch(
            input_ids=torch.cat([request, response], dim=1),
            target_ids=response,
            target_mask=padding_mask,
            target_weights=advantages.unsqueeze(-1).expand_as(response),
            extra={
                "ref_logprobs": ref_logprobs,
                "response": response,
                "padding_mask": padding_mask,
                "advantages": advantages,
            },
        )
        result.append(text_batch)

    return result
```

**Key Change:** Returns `list[TextTrainBatch]` instead of `tuple[dict, dict]`

#### Step 2: Define Protocol-Compliant Loss Function

**File: `apps/grpo/main.py`**

```python
def grpo_loss_fn(outputs: dict, batch: TextTrainBatch) -> torch.Tensor:
    """GRPO loss function conforming to the Trainer LossFn protocol."""

    logits = outputs["logits"]
    extra = batch.extra

    # Extract RL-specific tensors from batch.extra
    response = extra["response"]
    ref_logprobs = extra["ref_logprobs"]
    advantages = extra["advantages"]
    padding_mask = extra["padding_mask"]

    return simple_grpo_loss(
        logits=logits,
        response=response,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        padding_mask=padding_mask,
    )
```

**Key Change:** Loss function signature is `(outputs: dict, batch: TextTrainBatch) -> Tensor`

#### Step 3: Update Training Loop

**File: `apps/grpo/main.py`**

```python
async def continuous_training():
    # Get DP rank from trainer config
    trainer_config = await trainer.get_config.call_one()
    dp_rank = trainer_config.parallelism.dp_rank

    while training_step < max_steps:
        # Sample returns list[TextTrainBatch]
        batches = await replay_buffer.sample.call_one(...)

        # Get local batch for this DP rank
        local_batch = batches[dp_rank]

        # Forward/backward using protocol method
        result = await trainer.forward_backward.call(local_batch, grpo_loss_fn)
        record_metric("trainer/loss", result.loss, Reduce.MEAN)

        # Optimizer step using protocol method
        step_result = await trainer.optim_step.call()
        training_step = step_result.step

        # Push weights (RL-specific extension)
        await trainer.push_weights.call(training_step)
        await policy.update_weights.fanout(training_step)
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRPO Training Loop                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  ReplayBuffer.sample() → list[TextTrainBatch]                   │
│                                                                  │
│  TextTrainBatch:                                                 │
│  ├── input_ids: [batch, req_len + res_len]                      │
│  ├── target_ids: [batch, res_len]                               │
│  ├── target_mask: [batch, res_len]                              │
│  ├── target_weights: [batch, res_len] (expanded advantages)     │
│  └── extra: {ref_logprobs, response, advantages, padding_mask}  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  trainer.forward_backward(batch, grpo_loss_fn)                  │
│                                                                  │
│  1. Move tensors to device (including batch.extra)              │
│  2. logits = model(input_ids)                                   │
│  3. loss = grpo_loss_fn({"logits": logits}, batch)              │
│  4. loss.backward()                                              │
│  5. all_reduce(loss)                                             │
│                                                                  │
│  Returns: ForwardBackwardResult(loss=..., metrics={})           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  trainer.optim_step()                                            │
│                                                                  │
│  1. optimizer.step()                                             │
│  2. lr_scheduler.step()                                          │
│  3. optimizer.zero_grad()                                        │
│  4. checkpointer.save() (if interval reached)                   │
│                                                                  │
│  Returns: OptimStepResult(step=..., learning_rate=...)          │
└─────────────────────────────────────────────────────────────────┘
```

## Benefits of Migration

| Aspect | Before | After |
|--------|--------|-------|
| **Batch Format** | Raw dicts `(inputs, targets)` | Type-safe `TextTrainBatch` |
| **Loss Function** | Baked into trainer | Passed as parameter |
| **Trainer Swapping** | Requires code changes | Just swap trainer class |
| **Gradient Accumulation** | Manual | Built into protocol |
| **Metrics** | Custom per-recipe | Standardized `ForwardBackwardResult` |

## RL-Specific Extensions

TitanTrainer includes RL-specific methods beyond the base protocol:

```python
@endpoint
async def push_weights(self, policy_version: int) -> None:
    """Push weights to TorchStore for vLLM policy synchronization."""
    # Converts model state dict to HF format
    # Saves via DCP or TorchStore based on configuration
```

These extensions are called using the same `.call()` pattern:
```python
await trainer.push_weights.call(training_step)
```

## Backward Compatibility

The `train_step` convenience method is preserved for gradual migration:

```python
@endpoint
async def train_step(
    self, inputs: list[dict], targets: list[dict]
) -> float:
    """Convenience wrapper for legacy code."""
    # Internally uses same forward/backward logic
```

New code should use `forward_backward` + `optim_step` directly.

## Summary

The Trainer protocol migration:

1. **Standardizes batch format** via `TextTrainBatch` with `extra` for RL tensors
2. **Decouples loss functions** via `LossFn` protocol
3. **Separates concerns** with `forward_backward` and `optim_step`
4. **Enables swappable trainers** across SFT, GRPO, and future algorithms
5. **Maintains backward compatibility** via convenience methods
