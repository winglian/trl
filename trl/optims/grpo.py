# Copyright 2026 Axolotl AI. All rights reserved.
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

from ..import_utils import _is_package_available

if _is_package_available("triton"):
    import torch
    import triton
    from .grpo_kernels import _SelectiveLogSoftmaxTriton, _entropy_online_kernel, _entropy_online_kernel_strided
    from trl.trainer.utils import selective_log_softmax as selective_log_softmax_eager


    def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        """Triton-fused entropy (online single-pass). Handles non-contiguous tensors without copying."""
        original_shape = logits.shape[:-1]
        V = logits.shape[-1]
        N = 1
        for s in original_shape:
            N *= s

        if not logits.is_cuda:
            # CPU fallback: stable entropy via log_softmax
            logp = F.log_softmax(logits.float(), dim=-1)
            ent = -(logp.exp() * logp).sum(dim=-1)
            return ent.to(logits.dtype).reshape(original_shape)

        output = torch.empty(N, device=logits.device, dtype=torch.float32)

        BLOCK_V = 4096
        MAX_GRID_CONTIG = 8192
        MAX_GRID_STRIDED = 2048

        # Vocab (last) dim must be contiguous for coalesced loads
        if logits.stride(-1) != 1:
            logits = logits.contiguous()

        if logits.is_contiguous():
            flat_logits = logits.reshape(-1, V)
            stride = flat_logits.stride(0)
            for start in range(0, N, MAX_GRID_CONTIG):
                n_rows = min(MAX_GRID_CONTIG, N - start)
                _entropy_online_kernel[(n_rows,)](
                    flat_logits[start], output[start], stride, V=V, BLOCK_V=BLOCK_V
                )
        elif logits.ndim == 3:
            stride_outer = logits.stride(0)
            stride_inner = logits.stride(1)
            n_inner = logits.shape[1]
            for start in range(0, N, MAX_GRID_STRIDED):
                n_rows = min(MAX_GRID_STRIDED, N - start)
                _entropy_online_kernel_strided[(n_rows,)](
                    logits,
                    output[start],
                    stride_outer,
                    stride_inner,
                    n_inner,
                    start,
                    V=V,
                    BLOCK_V=BLOCK_V,
                )
        else:
            logits = logits.contiguous()
            flat_logits = logits.reshape(-1, V)
            stride = flat_logits.stride(0)
            for start in range(0, N, MAX_GRID_CONTIG):
                n_rows = min(MAX_GRID_CONTIG, N - start)
                _entropy_online_kernel[(n_rows,)](
                    flat_logits[start], output[start], stride, V=V, BLOCK_V=BLOCK_V
                )

        return output.to(logits.dtype).reshape(original_shape)

    def selective_log_softmax(logits, index) -> torch.Tensor:
        """
        Fused selective_log_softmax with Triton forward+backward kernels.

        Equivalent to: torch.gather(logits.log_softmax(-1), dim=-1, index=index)
        """
        squeeze = index.ndim == logits.ndim - 1
        if squeeze:
            index = index.unsqueeze(-1)

        if not logits.is_cuda or logits.dtype == torch.float64:
            # Triton kernel computes in float32; fall back for float64 and CPU
            return selective_log_softmax_eager(
                logits, index.squeeze(-1) if squeeze else index
            )

        V = logits.shape[-1]
        K = index.shape[-1]
        original_index_shape = index.shape

        flat_logits = logits.reshape(-1, V).contiguous()
        flat_index = index.reshape(-1, K).contiguous()

        BLOCK_V = 4096
        MAX_GRID = 8192
        K_BLOCK = max(1, triton.next_power_of_2(K))

        output = _SelectiveLogSoftmaxTriton.apply(
            flat_logits, flat_index, K, K_BLOCK, V, BLOCK_V, MAX_GRID
        )

        if K_BLOCK != K:
            output = output[:, :K]

        per_token_logps = output.to(logits.dtype).reshape(original_index_shape)

        if squeeze:
            per_token_logps = per_token_logps.squeeze(-1)

        return per_token_logps
else:
    from trl.trainer.utils import entropy_from_logits, selective_log_softmax
