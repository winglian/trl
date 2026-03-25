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
DataProducer for the experimental AsyncGRPOTrainer.

Bridges the AsyncRolloutWorker's queue-based architecture with the transformers
DataProducer protocol. The producer batch-collects scored samples from the
rollout queue into a fixed-size Dataset, avoiding per-sample blocking in the
training loop.

## Performance design

Baseline (IterableDataset):
    For each micro-batch, the DataLoader calls queue.get() → blocks until
    sample available. Weight sync happens via callback at step boundary,
    pausing generation. Timeline:
        [queue.get×4] → [fwd+bwd×4] → [weight_sync] → repeat

DataProducer path:
    produce() batch-collects all samples for one training step upfront.
    With AsyncDataProducer wrapping, the NEXT batch is collected in a background
    thread while the CURRENT batch trains. Weight sync moves into produce() so
    it overlaps with the background collection wait. Timeline:
        [produce(BG: collect N samples)] ← overlaps with ↓
        [train on current batch: fwd+bwd×4] → [produce.result()] → repeat
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

import torch
from torch.utils.data import Dataset

from transformers.data_producer import BaseDataProducer, ProducerConfig

from trl.trainer.utils import pad


logger = logging.getLogger(__name__)


class RolloutBatchDataset(Dataset):
    """A pre-collated batch of rollout samples as a Dataset.

    Stores N samples, each containing input_ids, completion_mask, old_log_probs,
    advantage, and metrics. The DataLoader + collator handle padding at batch time.
    """

    def __init__(self, samples: list[dict[str, Any]]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]


class AsyncGRPODataProducer(BaseDataProducer):
    """DataProducer that batch-collects samples from the AsyncRolloutWorker's queue.

    Collects exactly `samples_per_rollout` samples per produce() call, filtering
    stale samples. Returns a finite Dataset that the training loop iterates over.

    With AsyncDataProducer wrapping, the collection happens in a background thread
    while training proceeds on the previous batch — this is the key performance win
    over the IterableDataset approach.

    Args:
        config: ProducerConfig controlling mini_epochs, async_prefetch, etc.
        samples_per_rollout: Number of samples to collect per produce() call.
            Should equal per_device_train_batch_size * gradient_accumulation_steps
            * num_processes for one full step, or a multiple for multi-step rollouts.
        max_staleness: Maximum model version lag to accept.
        timeout: Seconds to wait per individual sample before giving up.
    """

    def __init__(
        self,
        config: ProducerConfig,
        samples_per_rollout: int = 4,
        max_staleness: int = 4,
        timeout: float = 120.0,
    ):
        super().__init__(config)
        self.samples_per_rollout = samples_per_rollout
        self.max_staleness = max_staleness
        self.timeout = timeout

        # Injected by the trainer after construction
        self._rollout_queue: queue.Queue | None = None
        self._model_version_fn = None

    def set_queue(self, rollout_queue: queue.Queue, model_version_fn):
        """Inject the rollout queue and model version function from the trainer."""
        self._rollout_queue = rollout_queue
        self._model_version_fn = model_version_fn

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
        """Batch-collect samples from the rollout queue.

        This is designed to run either on the main thread (sync) or in a
        background thread (when wrapped with AsyncDataProducer). No GPU ops
        are performed here — only queue.get() calls and Python dict construction.
        """
        if self._rollout_queue is None:
            raise RuntimeError("set_queue() must be called before produce()")

        samples = []
        t0 = time.time()
        dropped = 0
        total_wait = 0.0

        while len(samples) < self.samples_per_rollout:
            t_get = time.time()
            remaining = max(1.0, self.timeout - (time.time() - t0))
            try:
                sample = self._rollout_queue.get(timeout=remaining)
            except queue.Empty:
                if samples:
                    logger.warning(
                        f"Queue timeout after {time.time() - t0:.1f}s, "
                        f"using partial batch ({len(samples)}/{self.samples_per_rollout})"
                    )
                    break
                raise TimeoutError(
                    f"Rollout queue empty for {self.timeout}s, no samples collected."
                )
            wait_this = time.time() - t_get
            total_wait += wait_this

            # Staleness filter
            if self._model_version_fn is not None:
                staleness = self._model_version_fn() - sample.model_version
                if staleness > self.max_staleness:
                    dropped += 1
                    continue

            samples.append({
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "advantage": sample.advantage,
                "metrics": {
                    **sample.metrics,
                    "queue_wait_time_s": wait_this,
                },
            })

        elapsed = time.time() - t0
        qsize = self._rollout_queue.qsize()

        logger.info(
            f"DataProducer: collected {len(samples)} samples in {elapsed:.3f}s "
            f"(queue_wait={total_wait:.3f}s, dropped_stale={dropped}, qsize={qsize})"
        )

        return RolloutBatchDataset(samples)
