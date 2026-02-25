# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRPODataProducer — bridges TRL's GRPO generation pipeline with the
transformers DataProducer protocol for async rollout prefetching.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from transformers.data_producer import BaseDataProducer, DataProducerCallback, ProducerConfig

from .utils import RepeatSampler, split_pixel_values_by_grid, shuffle_sequence_dict, split_tensor_dict, unsplit_pixel_values_by_grid


logger = logging.getLogger(__name__)


class RolloutDataset(Dataset):
    """Map-style dataset wrapping a dict of tensors from a single GRPO rollout.

    Each item is a dict slice ``{key: tensor[idx]}`` suitable for the
    GRPO loss computation in ``GRPOTrainer._compute_loss``.

    Keys whose original values are 0-dim tensors or non-tensors are treated as
    *shared* metadata (e.g. ``num_items_in_batch``).  Keys whose original values
    have a batch dimension (``dim() > 0``) are *per-sample* — they are indexed
    in ``__getitem__`` and must be stacked by the collator even though they
    become 0-dim scalars after indexing (e.g. ``advantages``).
    """

    def __init__(self, data: dict[str, torch.Tensor | Any]):
        self._data = data
        # Keys that are shared across all samples (0-dim tensors or non-tensors).
        # Everything else is per-sample and will be indexed in __getitem__.
        self._shared_keys: set[str] = set()
        found_len = False
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                if not found_len:
                    self._len = v.size(0)
                    found_len = True
            else:
                self._shared_keys.add(k)
        if not found_len:
            raise ValueError("RolloutDataset requires at least one tensor value with dim > 0")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        result = {}
        for k, v in self._data.items():
            if k in self._shared_keys:
                result[k] = v
            else:
                result[k] = v[idx]
        return result


def make_rollout_collator(shared_keys: set[str]):
    """Return a collator that knows which keys are shared metadata.

    Args:
        shared_keys: Keys whose values are identical across all samples
            (e.g. ``num_items_in_batch``, ``_deferred_logps``).  These are
            passed through as-is.  All other tensor keys are stacked along
            a new batch dimension — even 0-dim scalars (e.g. per-sample
            ``advantages`` values).
    """

    def rollout_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            values = [item[key] for item in batch]
            if key in shared_keys:
                # Shared metadata — take first value (all identical)
                collated[key] = values[0]
            elif isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated

    return rollout_collator


class GRPODataProducer(BaseDataProducer, DataProducerCallback):
    """Produces GRPO rollout datasets using the transformers DataProducer protocol.

    This producer holds the prompt dataset and a back-reference to the
    GRPOTrainer.  Each ``produce()`` call:

    1. Samples a batch of prompts from the prompt dataset
    2. Delegates to ``trainer._generate_and_score_completions()``
    3. Wraps the result in a :class:`RolloutDataset`

    The generation methods remain on the trainer to minimise code churn;
    this class handles scheduling and the DataProducer interface.
    """

    def __init__(
        self,
        config: ProducerConfig,
        *,
        prompt_dataset: HFDataset,
        grpo_config: Any,
        num_generations: int,
    ):
        super().__init__(config)
        self._prompt_dataset = prompt_dataset
        self._grpo_config = grpo_config
        self._num_generations = num_generations

        # Back-reference to the trainer, set in on_train_begin
        self._trainer = None

        # Prompt sampler — created lazily once we know the batch size
        self._prompt_sampler = None
        self._prompt_iter = None
        self._rollout_count = 0

    # ------------------------------------------------------------------
    # Trainer callback hooks — get reference to trainer
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        """Validate that the trainer reference was set."""
        # The trainer reference is set explicitly via set_trainer() in
        # GRPOTrainer.__init__, not via callback kwargs (which only has
        # the model, not the trainer itself).

    def set_trainer(self, trainer):
        """Explicitly set the trainer back-reference.

        Called from ``GRPOTrainer.__init__`` after super().__init__().
        """
        self._trainer = trainer

    # ------------------------------------------------------------------
    # Prompt sampling
    # ------------------------------------------------------------------

    def _ensure_prompt_sampler(self):
        """Create the prompt sampler on first use."""
        if self._prompt_sampler is not None:
            return

        trainer = self._trainer
        args = self._grpo_config

        self._prompt_sampler = RepeatSampler(
            data_source=self._prompt_dataset,
            mini_repeat_count=self._num_generations,
            batch_size=args.generation_batch_size // self._num_generations,
            repeat_count=1,  # One pass per produce() call; we sample fresh each rollout
            shuffle=args.shuffle_dataset,
            seed=args.seed,
        )
        self._prompt_iter = iter(self._prompt_sampler)

    def _sample_prompts(self) -> list[dict]:
        """Sample a generation batch of prompts."""
        self._ensure_prompt_sampler()

        trainer = self._trainer
        args = self._grpo_config
        generation_batch_size = args.per_device_train_batch_size * args.steps_per_generation

        indices = []
        for _ in range(generation_batch_size):
            try:
                idx = next(self._prompt_iter)
            except StopIteration:
                # Reset sampler for next epoch
                self._prompt_iter = iter(self._prompt_sampler)
                idx = next(self._prompt_iter)
            indices.append(idx)

        return [self._prompt_dataset[i] for i in indices]

    # ------------------------------------------------------------------
    # DataProducer protocol
    # ------------------------------------------------------------------

    def produce(
        self,
        model: Any,
        global_step: int,
        *,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        **kwargs,
    ) -> Dataset:
        """Generate a fresh GRPO rollout dataset.

        Samples prompts, runs generation + reward scoring via the trainer's
        existing pipeline, and wraps the output tensors in a
        :class:`RolloutDataset`.
        """
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError(
                "GRPODataProducer.set_trainer() must be called before produce(). "
                "This is normally done in GRPOTrainer.__init__."
            )

        # Sample prompts
        inputs = self._sample_prompts()

        # Run the full GRPO generation + scoring pipeline
        generation_output = trainer._generate_and_score_completions(inputs)

        # Shuffle and split into per-step chunks, then recombine as flat dataset
        generation_output = split_pixel_values_by_grid(generation_output)
        generation_output = shuffle_sequence_dict(generation_output)

        # The output is a dict of tensors with batch dim = generation_batch_size.
        # We wrap it directly as a dataset — the dataloader will batch it at
        # per_device_train_batch_size.
        generation_output = unsplit_pixel_values_by_grid(generation_output)

        self._rollout_count += 1
        return RolloutDataset(generation_output)
