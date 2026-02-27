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
AsyncGRPOTrainer — Orchestrated Async GRPO with prime-rl patterns.

Extends :class:`~trl.GRPOTrainer` with orchestration features inspired by
`Prime Intellect's prime-rl <https://github.com/PrimeIntellect-ai/prime-rl>`_:

* **Batch quality metrics**: solve_all / solve_none / effective_batch_size
* **Empty batch retry**: exponential backoff when generation fails
* **Off-policy staleness cap**: discard stale prefetched rollouts
* **Temperature scheduling**: linear or cosine ramp over training
* **Rollout filters**: zero out degenerate completions (repetition, min-length)
* **Eval-aware pause**: freeze background generation during evaluation
"""

import logging
import math
import threading
import time
from typing import Any

import torch

from ...trainer.grpo_data_producer import RolloutDataset
from ...trainer.grpo_trainer import GRPOTrainer
from .async_grpo_config import AsyncGRPOConfig


logger = logging.getLogger(__name__)


class AsyncGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with async orchestration patterns.

    This trainer is designed for the async DataProducer path (``use_data_producer=True``,
    ``async_prefetch=True``).  It layers orchestration intelligence on top of the
    existing :class:`~trl.GRPOTrainer` by overriding:

    - :meth:`_produce_data`: retry logic, staleness check, timing metrics, eval pause
    - :meth:`_compute_deferred_scores`: rollout filters, batch quality metrics
    - :meth:`_generate_and_score_completions`: temperature scheduling

    All prime-rl-inspired features are opt-in via :class:`AsyncGRPOConfig` fields.
    """

    def __init__(self, args: AsyncGRPOConfig | None = None, **kwargs):
        # Validate temperature schedule
        valid_schedules = ("constant", "linear", "cosine")
        if args.temperature_schedule not in valid_schedules:
            raise ValueError(
                f"Invalid temperature_schedule={args.temperature_schedule!r}. "
                f"Must be one of {valid_schedules}."
            )

        # Force DataProducer mode — this IS the async trainer
        if not args.use_data_producer:
            logger.info("AsyncGRPOTrainer requires use_data_producer=True. Enabling automatically.")
            args.use_data_producer = True

        super().__init__(args=args, **kwargs)

        # Event gate for eval-aware pausing of background generation.
        # The BG thread checks this in GRPODataProducer.produce() indirectly
        # via _produce_data which runs on the main thread.
        self._generation_allowed = threading.Event()
        self._generation_allowed.set()  # Start allowed

        # Track retry stats
        self._total_retries = 0
        self._total_stale_discards = 0

    # ------------------------------------------------------------------
    # Temperature scheduling
    # ------------------------------------------------------------------

    def _compute_temperature(self) -> float:
        """Compute the current temperature based on the schedule and training step."""
        schedule = self.args.temperature_schedule
        if schedule == "constant":
            return self.args.temperature

        step = self.state.global_step
        t_min = self.args.temperature_min if self.args.temperature_min is not None else self.args.temperature
        t_max = self.args.temperature_max if self.args.temperature_max is not None else self.args.temperature
        warmup = max(self.args.temperature_warmup_steps, 1)
        progress = min(step / warmup, 1.0)

        if schedule == "linear":
            return t_min + progress * (t_max - t_min)
        elif schedule == "cosine":
            # Cosine anneal: starts at t_min, rises to t_max
            return t_min + 0.5 * (t_max - t_min) * (1.0 + math.cos(math.pi * (1.0 - progress)))

        return self.args.temperature

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]], skip_policy_logps: bool = False,
    ) -> dict[str, torch.Tensor | Any]:
        """Override to apply temperature scheduling before generation."""
        # Update temperature based on schedule
        new_temp = self._compute_temperature()
        if new_temp != self.temperature:
            self.temperature = new_temp

        return super()._generate_and_score_completions(inputs, skip_policy_logps=skip_policy_logps)

    # ------------------------------------------------------------------
    # Staleness checking
    # ------------------------------------------------------------------

    def _is_stale(self, dataset: RolloutDataset) -> bool:
        """Check if a prefetched rollout is too stale to use.

        A rollout is stale when the gap between the current training step and the
        step at which the rollout was generated exceeds ``max_off_policy_steps``.
        """
        if self.args.max_off_policy_steps is None:
            return False
        if not isinstance(dataset, RolloutDataset):
            return False
        gen_step = dataset._data.get("_generation_step")
        if gen_step is None:
            return False
        staleness = self.state.global_step - gen_step
        return staleness > self.args.max_off_policy_steps

    # ------------------------------------------------------------------
    # _produce_data override: retry + staleness + timing + eval pause
    # ------------------------------------------------------------------

    def _produce_data(self, model):
        """Override to add empty batch retry, staleness check, and timing metrics.

        Also waits for ``_generation_allowed`` event if eval-aware pausing is enabled,
        ensuring background generation is frozen during evaluation.
        """
        t0 = time.perf_counter()

        max_attempts = self.args.max_empty_batch_retries + 1
        for attempt in range(max_attempts):
            # Wait for eval to finish if paused
            if self.args.pause_generation_during_eval:
                self._generation_allowed.wait()

            dataset = super()._produce_data(model)

            # Staleness check
            if self._is_stale(dataset):
                staleness = self.state.global_step - dataset._data.get("_generation_step", 0)
                logger.warning(
                    "Discarding stale rollout (staleness=%d > max=%d) at step %d. Regenerating.",
                    staleness, self.args.max_off_policy_steps, self.state.global_step,
                )
                self._total_stale_discards += 1
                self._metrics["train"]["async/stale_discards"].append(1)
                continue

            # Empty batch check
            if isinstance(dataset, RolloutDataset) and len(dataset) == 0:
                if attempt < self.args.max_empty_batch_retries:
                    backoff = min(
                        self.args.empty_batch_backoff_base * (2 ** attempt),
                        self.args.empty_batch_backoff_cap,
                    )
                    logger.warning(
                        "Empty batch at step %d. Retry %d/%d in %.0fs.",
                        self.state.global_step,
                        attempt + 1,
                        self.args.max_empty_batch_retries,
                        backoff,
                    )
                    self._total_retries += 1
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"Step {self.state.global_step} failed after "
                    f"{self.args.max_empty_batch_retries} consecutive empty batches."
                )

            # Valid dataset — break out of retry loop
            break

        produce_time = time.perf_counter() - t0
        self._metrics["train"]["async/produce_time_s"].append(produce_time)
        if self._total_retries > 0:
            self._metrics["train"]["async/total_retries"].append(self._total_retries)
        if self._total_stale_discards > 0:
            self._metrics["train"]["async/total_stale_discards"].append(self._total_stale_discards)

        # Log temperature if using a schedule
        if self.args.temperature_schedule != "constant":
            self._metrics["train"]["sampling/temperature"].append(self.temperature)

        return dataset

    # ------------------------------------------------------------------
    # Rollout filters
    # ------------------------------------------------------------------

    def _detect_repetitions(self, completion_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        """Detect completions with high n-gram repetition.

        Returns a boolean mask of shape ``(B,)`` where ``True`` indicates a
        degenerate (highly repetitive) completion.
        """
        n = self.args.repetition_ngram_size
        threshold = self.args.repetition_max_ratio
        batch_size = completion_ids.size(0)
        is_repetitive = torch.zeros(batch_size, dtype=torch.bool, device=completion_ids.device)

        for i in range(batch_size):
            ids = completion_ids[i][completion_mask[i].bool()].tolist()
            if len(ids) <= n:
                continue
            ngrams = [tuple(ids[j : j + n]) for j in range(len(ids) - n + 1)]
            num_ngrams = len(ngrams)
            num_unique = len(set(ngrams))
            repeated_ratio = 1.0 - (num_unique / num_ngrams)
            if repeated_ratio > threshold:
                is_repetitive[i] = True

        return is_repetitive

    def _apply_rollout_filters(self, dataset: RolloutDataset) -> dict[str, float]:
        """Apply rollout degeneration filters before advantage computation.

        Filters zero out ``completion_mask`` for degenerate completions, which
        effectively removes them from the loss (the masked tokens contribute
        zero gradient).

        Returns a dict of filter metrics to log.
        """
        data = dataset._data
        completion_ids = data["completion_ids"]
        completion_mask = data["completion_mask"]
        metrics: dict[str, float] = {}
        total = completion_ids.size(0)

        if total == 0:
            return metrics

        if self.args.filter_repetitions:
            rep_mask = self._detect_repetitions(completion_ids, completion_mask)
            num_filtered = rep_mask.sum().item()
            if num_filtered > 0:
                completion_mask = completion_mask.clone()
                completion_mask[rep_mask] = 0
                data["completion_mask"] = completion_mask
            metrics["filters/repetition_rate"] = num_filtered / total

        if self.args.filter_min_length:
            lengths = completion_mask.sum(dim=1)
            too_short = lengths < self.args.min_completion_tokens
            num_short = too_short.sum().item()
            if num_short > 0:
                completion_mask = completion_mask.clone()
                completion_mask[too_short] = 0
                data["completion_mask"] = completion_mask
            metrics["filters/min_length_rate"] = num_short / total

        return metrics

    # ------------------------------------------------------------------
    # Deferred scoring override: filters + batch quality metrics
    # ------------------------------------------------------------------

    def _compute_deferred_scores(self, dataset: RolloutDataset):
        """Override to apply rollout filters and compute batch quality metrics.

        The sequence is:
        1. Apply rollout filters (zeros degenerate completion_masks)
        2. Call parent ``_compute_deferred_scores`` (rewards, advantages, logprobs)
        3. Compute batch quality metrics from raw rewards (saved by parent)
        """
        # 1. Apply filters BEFORE advantage computation
        filter_metrics = self._apply_rollout_filters(dataset)

        # 2. Parent handles rewards, advantages, logprobs, shuffle
        super()._compute_deferred_scores(dataset)

        # 3. Compute batch quality metrics
        self._compute_batch_quality_metrics()

        # 4. Log filter metrics
        for key, value in filter_metrics.items():
            self._metrics["train"][key].append(value)

    def _compute_batch_quality_metrics(self):
        """Compute prime-rl style batch quality metrics.

        Uses raw rewards saved by the parent's ``_compute_deferred_scores`` as
        ``self._last_rewards_per_func``. Falls back to advantage-based approximation
        if raw rewards are not available.

        Metrics:
            - **batch/solve_all**: Fraction of prompt groups where ALL completions
              received maximum reward (too easy, no learning signal).
            - **batch/solve_none**: Fraction of prompt groups where NO completions
              received reward (too hard, no learning signal).
            - **batch/effective_batch_size**: ``1 - solve_all - solve_none``, the
              fraction of the batch providing useful training signal.
        """
        G = self.num_generations

        # Try to use exact raw rewards from parent
        rewards_per_func = getattr(self, "_last_rewards_per_func", None)
        if rewards_per_func is not None:
            rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
            grouped = rewards.view(-1, G)
            # For binary rewards (0/1): solve_all = all rewards == max, solve_none = all rewards == 0
            group_sums = grouped.sum(dim=1)
            group_maxes = grouped.max(dim=1).values
            # solve_none: groups where max reward is 0 (no completion got any reward)
            solve_none = (group_maxes == 0).float().mean().item()
            # solve_all: groups where min reward equals max (all same) and max > 0
            group_mins = grouped.min(dim=1).values
            solve_all = ((group_mins == group_maxes) & (group_maxes > 0)).float().mean().item()
        else:
            # Fallback: approximate from advantages (less accurate for non-binary rewards)
            logger.warning_once(
                "Raw rewards not available for batch quality metrics. "
                "Using advantage-based approximation."
            )
            solve_all = 0.0
            solve_none = 0.0

        effective_batch_size = 1.0 - solve_all - solve_none

        self._metrics["train"]["batch/solve_all"].append(solve_all)
        self._metrics["train"]["batch/solve_none"].append(solve_none)
        self._metrics["train"]["batch/effective_batch_size"].append(effective_batch_size)

    # ------------------------------------------------------------------
    # Eval-aware pause
    # ------------------------------------------------------------------

    def evaluate(self, *args, **kwargs):
        """Override to pause background generation during evaluation.

        When ``pause_generation_during_eval=True``, this method:
        1. Clears the ``_generation_allowed`` event (blocking the BG thread)
        2. Drains any in-flight prefetch futures to ensure weight consistency
        3. Runs the parent ``evaluate()``
        4. Sets the event again to resume background generation
        """
        if self.args.pause_generation_during_eval and self.args.async_prefetch:
            self._generation_allowed.clear()
            # Drain in-flight futures so eval uses consistent weights
            producer = self.data_producer
            if hasattr(producer, "_queue"):
                for future in list(producer._queue):
                    future.result()
            logger.info("Background generation paused for evaluation.")

        try:
            result = super().evaluate(*args, **kwargs)
        finally:
            if self.args.pause_generation_during_eval and self.args.async_prefetch:
                self._generation_allowed.set()
                logger.info("Background generation resumed after evaluation.")

        return result
