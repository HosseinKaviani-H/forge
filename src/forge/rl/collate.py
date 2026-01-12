# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from forge.api.types import TextTrainBatch
from forge.rl.types import Group


def collate(batches: list[Group]) -> list[TextTrainBatch]:
    """Collates a list of episode groups into TextTrainBatch objects.

    Each batch (Group) is a list of episodes from one DP rank.
    Returns a list of TextTrainBatch objects (one per DP rank).

    The TextTrainBatch format:
    - input_ids: Full sequence [request + response] for model forward pass
    - target_ids: Response tokens for loss computation
    - target_mask: Padding mask for response tokens
    - target_weights: Per-token advantages (expanded from per-sequence)
    - extra: Additional RL-specific tensors (ref_logprobs)

    Args:
        batches: List of episode groups (one per DP rank)

    Returns:
        List of TextTrainBatch objects (one per DP rank)
    """
    result = []
    for batch in batches:
        request = torch.stack([e.request_tensor for e in batch])  # [b, req_len]
        response = torch.stack([e.response_tensor for e in batch])  # [b, res_len]
        ref_logprobs = torch.stack([e.ref_logprobs for e in batch]).squeeze()  # [b, res_len]
        advantages = torch.tensor([e.advantage for e in batch])  # [b]

        pad_id = batch[0].pad_id
        padding_mask = (response != pad_id).float()  # [b, res_len]

        # Expand advantages to per-token weights (same weight for all tokens in response)
        # Shape: [b] -> [b, res_len]
        target_weights = advantages.unsqueeze(-1).expand_as(response).float()

        # Full input sequence for model forward pass
        input_ids = torch.cat([request, response], dim=1)  # [b, req_len + res_len]

        text_batch: TextTrainBatch = {
            "input_ids": input_ids,
            "target_ids": response,
            "target_mask": padding_mask,
            "target_weights": target_weights,
            "extra": {
                "ref_logprobs": ref_logprobs,
                "response": response,
                "padding_mask": padding_mask,
                "advantages": advantages,
            },
        }
        result.append(text_batch)

    return result
