# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torch.distributed.checkpoint as dcp
import torchstore as ts

from forge.actors._torchstore_utils import (
    DcpHandle,
    get_dcp_whole_state_dict_key,
    get_param_key,
    rdma_available,
)
from forge.api.types import (
    ForwardBackwardResult,
    LossFn,
    OptimStepResult,
    ParallelismConfig,
    TextTrainBatch,
    TrainerConfig,
    TrainerStatus,
)
from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

from monarch.actor import endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TitanTrainer(ForgeActor):
    """A trainer actor implementing the Trainer protocol on top of TorchTitan.

    This trainer implements the Trainer protocol defined in forge.api.trainer,
    providing a standard interface for forward/backward passes, optimizer steps,
    checkpointing, and configuration access.

    The trainer supports distributed training strategies including tensor parallelism,
    data parallelism, and FSDP (Fully Sharded Data Parallel).
    """

    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"
    use_dcp: bool = not rdma_available()
    dcp_path: str = "forge_dcp_tmp"

    def __post_init__(self):
        super().__init__()
        if self.use_dcp:
            torch.serialization.set_crc32_options(False)

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 1
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        self._accumulated_microbatches = 0
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "state_dict_key",
            "use_dcp",
            "dcp_path",
        }:
            engine_config.pop(key)
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.step)
        self.engine.optimizers.zero_grad()

    # =========================================================================
    # Trainer Protocol Implementation
    # =========================================================================

    @endpoint
    async def forward_backward(
        self, batch: TextTrainBatch, loss_fn: LossFn | None = None
    ) -> ForwardBackwardResult:
        """Execute forward pass and backward pass for one batch of data.

        Args:
            batch: TextTrainBatch containing input_ids, target_ids, and optional
                target_mask/target_weights.
            loss_fn: Optional custom loss function. If None, uses the default loss.

        Returns:
            ForwardBackwardResult containing loss and metrics
        """
        t = Tracer("trainer_perf/forward_backward", timer="gpu", track_memory=True)
        t.start()

        self.engine.gc_handler.run(self.step)

        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        optional_context_parallel_ctx = None

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")

        # Move batch tensors to device
        input_ids = batch.input_ids.to(self.engine.device)
        target_ids = batch.target_ids.to(self.engine.device)
        target_mask = (
            batch.target_mask.to(self.engine.device)
            if batch.target_mask is not None
            else None
        )
        target_weights = (
            batch.target_weights.to(self.engine.device)
            if batch.target_weights is not None
            else None
        )

        # Move extra tensors to device (for RL, contains ref_logprobs, etc.)
        if batch.extra is not None:
            for key, value in batch.extra.items():
                if isinstance(value, torch.Tensor):
                    batch.extra[key] = value.to(self.engine.device)

        with self.engine.train_context(optional_context_parallel_ctx):
            assert len(model_parts) == 1
            with self.engine.maybe_enable_amp:
                logits = model_parts[0](input_ids)

                # Use custom loss_fn if provided
                if loss_fn is not None:
                    outputs = {"logits": logits}
                    loss = loss_fn(outputs, batch)
                else:
                    # Default loss computation using self.loss
                    kwargs = {"labels": target_ids}
                    if target_mask is not None:
                        kwargs["target_mask"] = target_mask
                    if target_weights is not None:
                        kwargs["target_weights"] = target_weights
                    loss = self.loss(logits, **kwargs)

            del logits
            loss.backward()

        t.step("forward_backward")

        # All-reduce loss across data parallel ranks
        torch.distributed.all_reduce(loss)

        self._accumulated_microbatches += 1
        loss_val = loss.detach().item()

        t.stop()

        return ForwardBackwardResult(loss=loss_val, metrics={})

    @endpoint
    async def optim_step(self) -> OptimStepResult:
        """Apply optimizer step using accumulated gradients, then clear gradients.

        Returns:
            OptimStepResult containing step number, learning rate, and accumulated batch count
        """
        t = Tracer("trainer_perf/optim_step", timer="gpu", track_memory=True)
        t.start()

        current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        accumulated = self._accumulated_microbatches

        record_metric("trainer/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.lr_schedulers.step()
        self.engine.optimizers.zero_grad()
        t.step("optimizer_step")

        self.step += 1
        self._accumulated_microbatches = 0

        # Save checkpoint if needed
        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()

        return OptimStepResult(
            step=self.step,
            learning_rate=current_lr,
            accumulated_microbatches=accumulated,
        )

    @endpoint
    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying them."""
        self.engine.optimizers.zero_grad()
        self._accumulated_microbatches = 0

    @endpoint
    async def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass only, without backward pass (for evaluation/inference).

        Args:
            inputs: Dictionary containing model inputs (e.g., input_ids).

        Returns:
            Model output logits.
        """
        model_parts = self.engine.model_parts

        # Move inputs to device
        device_inputs = {k: v.to(self.engine.device) for k, v in inputs.items()}

        with torch.no_grad():
            with self.engine.maybe_enable_amp:
                assert len(model_parts) == 1
                logits = model_parts[0](**device_inputs)

        return logits

    @endpoint
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
            weights_only: If True, saves only model weights.

        Returns:
            Full path/URI where checkpoint was saved
        """
        if name is None:
            name = f"step-{self.step}" if not weights_only else f"weights-step-{self.step}"

        checkpoint_path = path or self.engine.checkpointer.folder
        full_path = os.path.join(checkpoint_path, name)

        if weights_only:
            sd = self.engine.checkpointer.states["model"].state_dict()
            self.engine.checkpointer.dcp_save(
                sd,
                checkpoint_id=full_path,
                async_mode=self.engine.checkpointer.async_mode,
            )
        else:
            self.engine.checkpointer.save(
                curr_step=self.step,
                last_step=False,
            )
            full_path = self.engine.checkpointer._create_checkpoint_id(self.step)

        return full_path

    @endpoint
    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint.

        Args:
            path: Optional path or URI to the checkpoint to load.

        Returns:
            Path/URI that was loaded
        """
        if path is not None:
            states = self.engine.checkpointer._states_to_load(model_only=False)
            self.engine.checkpointer.dcp_load(
                states,
                checkpoint_id=path,
                from_hf=False,
                from_quantized=False,
            )
            return path
        else:
            self.engine.checkpointer.load(step=-1)
            return self.engine.checkpointer.folder

    @endpoint
    async def get_config(self) -> TrainerConfig:
        """Get static trainer and model configuration.

        Returns:
            TrainerConfig containing model name, model_config, and parallelism settings
        """
        parallel_dims = self.engine.parallel_dims

        parallelism_config = ParallelismConfig(
            dp_degree=parallel_dims.dp_shard if parallel_dims.dp_shard_enabled else 1,
            tp_degree=parallel_dims.tp if parallel_dims.tp_enabled else 1,
            pp_degree=parallel_dims.pp if parallel_dims.pp_enabled else 1,
            cp_degree=parallel_dims.cp if parallel_dims.cp_enabled else 1,
            ep_degree=parallel_dims.ep if parallel_dims.ep_enabled else 1,
            world_size=parallel_dims.world_size,
            dp_rank=self.engine.dp_rank,
            tp_rank=0,
            device=str(self.engine.device),
        )

        model_config = {
            "name": self.model.name,
            "flavor": self.model.flavor,
        }

        return TrainerConfig(
            model_name=self.model.hf_assets_path or f"{self.model.name}-{self.model.flavor}",
            model_config=model_config,
            parallelism=parallelism_config,
        )

    @endpoint
    async def get_status(self) -> TrainerStatus:
        """Get current runtime status of the trainer.

        Returns:
            TrainerStatus containing current step and accumulated batch count
        """
        return TrainerStatus(
            step=self.step,
            accumulated_microbatches=self._accumulated_microbatches,
        )

    @endpoint
    async def get_tokenizer(self):
        """Get the tokenizer associated with this model.

        Returns:
            The tokenizer for this model
        """
        raise NotImplementedError("get_tokenizer not yet implemented")

    # =========================================================================
    # RL-Specific Extensions (for weight synchronization with vLLM)
    # =========================================================================

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to TorchStore for vLLM policy synchronization.

        Args:
            policy_version: Version number for the policy weights
        """
        t = Tracer("trainer_perf/push_weights", timer="gpu", track_memory=True)
        t.start()

        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        t.step("flatten_state_dict")

        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        t.step("to_hf")

        if self.use_dcp:
            key = get_dcp_whole_state_dict_key(policy_version)
            dcp_id = f"{self.dcp_path}/{key}"
            storage_writer = torch.distributed.checkpoint.FileSystemWriter(
                dcp_id, single_file_per_rank=False, thread_count=8
            )
            metadata = dcp.save(storage_writer=storage_writer, state_dict=hf_state_dict)
            dcp_handle = DcpHandle(
                checkpoint_id=dcp_id,
                metadata=metadata,
                param_names=hf_state_dict.keys(),
            )
            await ts.put(key, dcp_handle)
            t.step("dcp_save")
        else:
            for name, param in hf_state_dict.items():
                key = get_param_key(policy_version, name)
                await ts.put(key, param)
            t.step("ts_save")

        t.stop()
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        """Clean up trainer resources."""
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
