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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


@dataclass
class AsyncGRPOConfig(GRPOConfig):
    r"""
    Configuration for [`AsyncGRPOTrainer`].

    Extends [`GRPOConfig`] with orchestration parameters inspired by
    [Prime Intellect's prime-rl](https://github.com/PrimeIntellect-ai/prime-rl):
    temperature scheduling, off-policy staleness control, empty batch retry,
    rollout degeneration filters, batch quality metrics, and eval-aware async
    pausing.

    Parameters:
        temperature_schedule (`str`, *optional*, defaults to `"constant"`):
            Temperature schedule type. One of `"constant"`, `"linear"`, or `"cosine"`.
            - `"constant"`: Use `temperature` (from GRPOConfig) for all steps.
            - `"linear"`: Linearly interpolate from `temperature_min` to `temperature_max`
              over `temperature_warmup_steps`.
            - `"cosine"`: Cosine anneal from `temperature_min` to `temperature_max`
              over `temperature_warmup_steps`.
        temperature_min (`float` or `None`, *optional*, defaults to `None`):
            Minimum (starting) temperature for scheduled modes. If `None`, defaults
            to the base `temperature` value.
        temperature_max (`float` or `None`, *optional*, defaults to `None`):
            Maximum (ending) temperature for scheduled modes. If `None`, defaults
            to the base `temperature` value.
        temperature_warmup_steps (`int`, *optional*, defaults to `0`):
            Number of steps over which to ramp the temperature schedule. After this
            many steps the temperature stays at `temperature_max`.
        max_off_policy_steps (`int` or `None`, *optional*, defaults to `None`):
            Maximum number of training steps a prefetched rollout can lag behind
            the current training step before being discarded and regenerated.
            `None` disables staleness checking.
        max_empty_batch_retries (`int`, *optional*, defaults to `3`):
            Maximum number of retries when a generation step produces an empty batch
            (e.g., inference server temporarily unavailable). Uses exponential backoff.
        empty_batch_backoff_base (`float`, *optional*, defaults to `10.0`):
            Base backoff time in seconds for empty batch retries. Doubles each attempt.
        empty_batch_backoff_cap (`float`, *optional*, defaults to `120.0`):
            Maximum backoff time in seconds for empty batch retries.
        filter_repetitions (`bool`, *optional*, defaults to `False`):
            Enable n-gram repetition filter. Completions where more than
            `repetition_max_ratio` of n-grams are repeated have their
            `completion_mask` zeroed out (effectively removing them from the loss).
        repetition_ngram_size (`int`, *optional*, defaults to `3`):
            N-gram size for the repetition filter.
        repetition_max_ratio (`float`, *optional*, defaults to `0.5`):
            Maximum fraction of repeated n-grams before a completion is filtered.
            E.g., `0.5` means if more than 50% of n-grams are duplicates, the
            completion is considered degenerate.
        filter_min_length (`bool`, *optional*, defaults to `False`):
            Enable minimum-length filter. Completions shorter than
            `min_completion_tokens` have their `completion_mask` zeroed out.
        min_completion_tokens (`int`, *optional*, defaults to `4`):
            Minimum number of non-padding tokens in a completion. Only applies
            when `filter_min_length=True`.
        pause_generation_during_eval (`bool`, *optional*, defaults to `True`):
            When using async prefetch, pause the background generation thread
            during evaluation to ensure consistent model weights during eval.
    """

    temperature_schedule: str = field(
        default="constant",
        metadata={
            "help": "Temperature schedule type: 'constant', 'linear', or 'cosine'.",
            "choices": ["constant", "linear", "cosine"],
        },
    )
    temperature_min: float | None = field(
        default=None,
        metadata={"help": "Minimum temperature for scheduled modes. Defaults to base temperature if None."},
    )
    temperature_max: float | None = field(
        default=None,
        metadata={"help": "Maximum temperature for scheduled modes. Defaults to base temperature if None."},
    )
    temperature_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps over which to ramp the temperature schedule."},
    )
    max_off_policy_steps: int | None = field(
        default=None,
        metadata={
            "help": "Max staleness (in training steps) for prefetched rollouts. "
            "Stale rollouts are discarded and regenerated. None disables the check."
        },
    )
    max_empty_batch_retries: int = field(
        default=3,
        metadata={"help": "Max retries for empty batches before raising an error."},
    )
    empty_batch_backoff_base: float = field(
        default=10.0,
        metadata={"help": "Base backoff time in seconds for empty batch retries."},
    )
    empty_batch_backoff_cap: float = field(
        default=120.0,
        metadata={"help": "Maximum backoff time in seconds for empty batch retries."},
    )
    filter_repetitions: bool = field(
        default=False,
        metadata={"help": "Enable n-gram repetition filter to zero out degenerate completions."},
    )
    repetition_ngram_size: int = field(
        default=3,
        metadata={"help": "N-gram size for the repetition filter."},
    )
    repetition_max_ratio: float = field(
        default=0.5,
        metadata={"help": "Max fraction of repeated n-grams before filtering a completion."},
    )
    filter_min_length: bool = field(
        default=False,
        metadata={"help": "Enable minimum-length filter to zero out very short completions."},
    )
    min_completion_tokens: int = field(
        default=4,
        metadata={"help": "Minimum completion length (in tokens) when filter_min_length=True."},
    )
    pause_generation_during_eval: bool = field(
        default=True,
        metadata={"help": "Pause background generation during evaluation for weight consistency."},
    )
