from typing import Tuple

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.sampler import (
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils.common import is_cuda

if is_cuda():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float = 0.0,
    ) -> torch.Tensor:
        """Sample tokens from logits.

        Args:
            logits: Tensor of shape [N, V] (N positions, V vocab size).
            temperature: Temperature for sampling.
            top_k: Top-k filtering. <= 0 means disabled, <= 1 means greedy.
            top_p: Top-p (nucleus) filtering. >= 1.0 means disabled.
            min_p: Min-p filtering. <= 0 means disabled.
        Returns:
            token_ids: Tensor of shape [N] with sampled token indices.
        """
        if top_k <= 0:
            top_k = TOP_K_ALL
        if top_k <= 1:
            return torch.argmax(logits, dim=-1)

        probs = F.softmax(logits / temperature, dim=-1)
        return self._sample_from_probs(probs, top_k=top_k, top_p=top_p, min_p=min_p)

    def _sample_from_probs(
        self,
        probs: torch.Tensor,
        top_k: int,
        top_p: float,
        min_p: float = 0.0,
    ) -> torch.Tensor:
        """Sample from probabilities with fast backend dispatch."""
        if top_k <= 0:
            top_k = TOP_K_ALL

        simple_sampling_case = top_k == TOP_K_ALL and top_p >= 1.0 and min_p <= 0.0
        if simple_sampling_case:
            return sampling_from_probs_torch(
                probs, sampling_seed=None, positions=None
            ).view(-1)

        n = probs.shape[0]
        device = probs.device
        top_ks = torch.full((n,), top_k, device=device, dtype=torch.int64)
        top_ps = torch.full((n,), top_p, device=device, dtype=probs.dtype)

        server_args = get_global_server_args()
        backend = server_args.sampling_backend if server_args is not None else "pytorch"

        if backend == "flashinfer" and is_cuda():
            if min_p > 0.0:
                filtered_probs = top_k_renorm_prob(probs, top_ks)
                filtered_probs = top_p_renorm_prob(filtered_probs, top_ps)
                min_ps = torch.full((n,), min_p, device=device, dtype=probs.dtype)
                return min_p_sampling_from_probs(filtered_probs, min_ps).view(-1)
            return top_k_top_p_sampling_from_probs(
                probs.contiguous(),
                top_ks,
                top_ps,
                filter_apply_order="joint",
                check_nan=False,
            ).view(-1)

        # Fall back to the torch implementation for non-flashinfer backends.
        # This keeps dLLM compatible with custom sampler backends registered via
        # `register_sampler_backend`.
        min_ps = torch.full((n,), min_p, device=device, dtype=probs.dtype)
        return top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            top_ks,
            top_ps,
            min_ps,
            need_min_p_sampling=min_p > 0.0,
            sampling_seed=None,
            positions=None,
        )

    def _sample_from_logits_with_confidence(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits and return their confidence scores.

        Args:
            logits: Tensor of shape [N, V] (N positions, V vocab size).
            temperature: Temperature for sampling.
            top_k: Top-k filtering. <= 0 means disabled, <= 1 means greedy.
            top_p: Top-p (nucleus) filtering. >= 1.0 means disabled.
            min_p: Min-p filtering. <= 0 means disabled.

        Returns:
            token_ids: Tensor of shape [N] with sampled token indices.
            confidence: Tensor of shape [N] with the probability of each
                        sampled token (used as confidence score).
        """
        if top_k <= 0:
            top_k = TOP_K_ALL
        if top_k <= 1:
            token_ids = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            confidence = torch.gather(
                probs, dim=-1, index=token_ids.unsqueeze(-1)
            ).squeeze(-1)
            return token_ids, confidence

        probs = F.softmax(logits / temperature, dim=-1)
        token_ids = self._sample_from_probs(
            probs, top_k=top_k, top_p=top_p, min_p=min_p
        )

        # Confidence is gathered from the original (unfiltered) probs
        confidence = torch.gather(
            probs, dim=-1, index=token_ids.unsqueeze(-1).long()
        ).squeeze(-1)
        return token_ids, confidence

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
