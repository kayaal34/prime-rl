"""Inkling 2 (text tower) for PrimeRL training.

Reference implementations:
  * transformers 5.23 ``transformers/models/inkling2/modeling_inkling2.py``
  * the vLLM port (vllm/models/inkling2)

Structure per layer: block attention residual -> attention (sliding window or
global with a hidden-conditioned relative bias, per-token sinks, output gate,
q/rel L2 clips, unweighted value norm, log scaling on global layers) -> short
convolution -> block attention residual -> MoE (384 experts, top-8 sigmoid
router with a selection bias, softcapped SwiGLU) -> short convolution.

Context parallelism splits the packed row into equal shards, one per rank. Everything is per token except attention,
which the ranks compute head-sharded over the whole row (Ulysses), and the short convolutions, which read the tokens
before a shard from the rank that holds them.
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import AuxRequest, BlockMask
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL, TextModelOutput
from prime_rl.trainer.models.inkling2.configuration_inkling2 import Inkling2Config, Inkling2TextConfig
from prime_rl.trainer.models.inkling2.converting_inkling2 import conversion_chain
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.models.layers.moe import MoE, MoEArgs
from prime_rl.trainer.models.layers.norms import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.ulysses_attn import (
    _all_to_all_head_to_seq,
    _all_to_all_seq_to_head,
    _replicate_kv_heads,
)
from prime_rl.utils.cp import CPContext, gather_for_cp
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

# Eager flex_attention materializes the full [heads, seq, seq] score matrix through the vmap
# reference path (48 x 8192 x 8192 fp32 = 24 GiB at 8K). The compiled kernel fuses score_mod
# into the block-sparse loop: measured peak 0.32 GiB at 8K and 0.56 GiB at 16K for a 48-head,
# 8-KV-head attention with the relative-bias score_mod. Shapes are static (padding to seq_len).
flex_attention = torch.compile(_flex_attention, dynamic=False)
# flex_attention's tile: a block mask classifies each tile of this many queries and keys.
BLOCK_SIZE = 128


class Inkling2UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class Inkling2L2Clip(nn.Module):
    """Scales each head vector down to ``threshold`` when its L2 norm is larger."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


class Inkling2ScaledL2Clip(nn.Module):
    """``Inkling2L2Clip`` applied after a learned per-(head, dim) scale."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * self.scale.float()
        norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
        return (hidden_states * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


def _tokens_before(x: Tensor, count: int, cp_context: CPContext) -> Tensor:
    """The ``count`` tokens before ``x [B, T, ...]`` in the row. Under context parallelism ``x`` is
    this rank's shard, and they end the previous rank's shard. The row's first tokens get its last
    ones, the way ``torch.roll`` wraps, and the document mask drops them."""
    tail = x[:, x.shape[1] - count :]
    if not cp_context.cp_enabled:
        return tail
    tails = gather_for_cp(tail, cp_context.cp_group)
    previous_rank = (cp_context.cp_rank - 1) % cp_context.cp_world_size
    return tails[:, previous_rank * count : (previous_rank + 1) * count]


def _shifted_within_documents(x: Tensor, before: Tensor, lag: int, positions: Tensor) -> Tensor:
    """``x[t - lag]`` where ``t - lag`` is in the same document, else zeros. ``before`` holds the
    tokens before ``x`` (``_tokens_before``), and ``positions [T]`` counts tokens inside each document."""
    if lag == 0:
        return x
    shifted = torch.cat([before[:, before.shape[1] - lag :], x[:, : x.shape[1] - lag]], dim=1)
    return shifted * (positions >= lag).view(1, -1, *([1] * (x.dim() - 2)))


class Inkling2ShortConvolution(nn.Module):
    """Static depthwise causal short convolution with a renormalized residual.

    Implemented as a lag-accumulation (one shift-and-add per tap) so it stays a
    tensor program under ``torch.compile`` and cannot leak across packed
    documents. Computed in fp32 like the reference.
    """

    cp_context: CPContext = CPContext()

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(kernel_size + 1)

    def forward(self, hidden_states: Tensor, positions: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        before = _tokens_before(x, self.kernel_size - 1, self.cp_context)
        weight = self.conv1d.weight.float().view(-1, self.kernel_size)
        out = torch.zeros_like(x)
        for lag in range(self.kernel_size):
            # PyTorch's conv1d is a cross-correlation: tap `j` weighs lag `K-1-j`.
            out = out + _shifted_within_documents(x, before, lag, positions) * weight[:, self.kernel_size - 1 - lag]
        return ((out + x) * self.scale).to(dtype)


class Inkling2DynamicShortConvolution(nn.Module):
    """Short causal conv whose taps are static per-channel weights plus per-token
    weights projected from the same hidden states, shared within each KV head."""

    cp_context: CPContext = CPContext()

    def __init__(self, hidden_size: int, input_size: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.static_weight = nn.Parameter(torch.zeros(kernel_size, hidden_size))
        self.weight_proj = nn.Linear(input_size, num_heads * kernel_size, bias=False)
        self.scale = 1.0 / math.sqrt(2 * kernel_size + 1)

    def forward(self, hidden_states: Tensor, dynamic_input: Tensor, positions: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.float()
        before = _tokens_before(x, self.kernel_size - 1, self.cp_context)
        dynamic = (
            self.weight_proj(dynamic_input).float().view(x.shape[0], x.shape[1], self.num_heads, 1, self.kernel_size)
        )
        static = self.static_weight.float().T.view(self.num_heads, self.head_dim, self.kernel_size)
        taps = (static + dynamic).reshape(x.shape[0], x.shape[1], -1, self.kernel_size)  # [B, S, C, K]
        out = torch.zeros_like(x)
        for lag in range(self.kernel_size):
            out = out + _shifted_within_documents(x, before, lag, positions) * taps[..., lag]
        return ((out + x) * self.scale).to(dtype)


class Inkling2Attention(nn.Module):
    """Sliding-window or global attention with relative bias, sinks and output gate.

    Under context parallelism the attention kernel runs head-sharded (Ulysses): an all-to-all gives
    each rank the whole row for its share of the heads, and a second one returns the outputs to the
    sequence shards. Everything around the kernel stays per token."""

    cp_context: CPContext = CPContext()

    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_kv_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.scaling = self.head_dim**-0.5
        self.d_rel = config.d_rel

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.r_proj = nn.Linear(hidden, self.num_heads * self.d_rel, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.sink_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.q_l2_clip = Inkling2ScaledL2Clip(self.num_heads, self.head_dim, config.qk_l2_clip_threshold)
        self.k_l2_clip = Inkling2L2Clip(config.qk_l2_clip_threshold)
        self.rel_l2_clip = Inkling2ScaledL2Clip(self.num_heads, config.d_rel, config.rel_l2_clip_threshold)
        self.rel_logits_proj = nn.Parameter(torch.empty(config.d_rel, self.rel_extent))
        self.v_norm = Inkling2UnweightedRMSNorm(eps=config.rms_norm_eps)
        if self.is_sliding:
            kv_size = self.num_kv_heads * self.head_dim
            self.k_sconv = Inkling2ShortConvolution(kv_size, config.sconv_kernel_size)
            self.v_sconv = Inkling2ShortConvolution(kv_size, config.sconv_kernel_size)
        else:
            kv_size = self.num_kv_heads * self.head_dim
            self.k_sconv = Inkling2DynamicShortConvolution(kv_size, hidden, self.num_kv_heads, config.sconv_kernel_size)
            self.v_sconv = Inkling2DynamicShortConvolution(kv_size, hidden, self.num_kv_heads, config.sconv_kernel_size)

        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha

    def forward(
        self,
        hidden_states: Tensor,
        key_hidden_states: Tensor,
        value_hidden_states: Tensor,
        positions: Tensor,
        block_mask: BlockMask,
    ) -> Tensor:
        """``positions [T]`` counts tokens inside each document. ``block_mask`` covers the whole row,
        which under context parallelism is longer than this rank's ``T`` tokens."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        if self.is_sliding:
            key_states = self.k_sconv(self.k_proj(key_hidden_states), positions)
            value_states = self.v_sconv(self.v_proj(value_hidden_states), positions)
        else:
            key_states = self.k_sconv(self.k_proj(key_hidden_states), key_hidden_states, positions)
            value_states = self.v_sconv(self.v_proj(value_hidden_states), value_hidden_states, positions)
        relative_states = self.r_proj(hidden_states)
        gate = self.g_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)  # [B, T, H]

        query_states = self.q_l2_clip(query_states.view(hidden_shape))
        key_states = self.k_l2_clip(key_states.view(hidden_shape))
        value_states = value_states.view(hidden_shape)
        value_states = self.v_norm(value_states.float()).to(value_states.dtype)
        relative_states = self.rel_l2_clip(relative_states.view(*input_shape, self.num_heads, -1))

        if not self.is_sliding:
            tau = 1.0 + self.log_scaling_alpha * torch.log(
                ((positions + 1).float() / self.log_scaling_n_floor).clamp(min=1.0)
            )
            tau = tau.view(1, -1, 1, 1)
            query_states = (query_states.float() * tau).to(query_states.dtype)
            relative_states = (relative_states.float() * tau).to(relative_states.dtype)

        cp = self.cp_context
        if cp.cp_enabled:
            # Query heads are grouped by KV head, so a rank's query heads read its KV heads. With fewer
            # KV heads than ranks, each KV head is repeated for the ranks whose query heads read it.
            if self.num_kv_heads < cp.cp_world_size:
                key_states, value_states = (
                    _replicate_kv_heads(states[0], cp.cp_world_size).unsqueeze(0)
                    for states in (key_states, value_states)
                )
            query_states, key_states, value_states, relative_states = (
                _all_to_all_seq_to_head(states[0], cp.cp_world_size, cp.cp_group).unsqueeze(0)
                for states in (query_states, key_states, value_states, relative_states)
            )

        # Hidden-conditioned relative bias: one profile per (query, head, distance).
        rel_logits = torch.matmul(relative_states, self.rel_logits_proj)  # [B, S, H, rel_extent]

        out, aux = flex_attention(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            score_mod=_relative_bias_score_mod(rel_logits, self.rel_extent),
            block_mask=block_mask,
            scale=self.scaling,
            enable_gqa=query_states.shape[2] != key_states.shape[2],
            return_aux=AuxRequest(lse=True),
        )
        out, lse = out.transpose(1, 2), aux.lse.transpose(1, 2)  # [B, S, H, D], [B, S, H]
        if cp.cp_enabled:
            out = _all_to_all_head_to_seq(out[0], cp.cp_world_size, cp.cp_group).unsqueeze(0)
            lse = _all_to_all_head_to_seq(lse[0].unsqueeze(-1), cp.cp_world_size, cp.cp_group).view(
                1, -1, self.num_heads
            )
        attn_output = (out * torch.sigmoid(lse - sinks.float()).unsqueeze(-1)).to(out.dtype)
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)
        return self.o_proj(attn_output)


def _relative_bias_score_mod(rel_logits: Tensor, rel_extent: int):
    def score_mod(score, b, h, q_idx, kv_idx):
        dist = q_idx - kv_idx
        return score + rel_logits[b, q_idx, h, dist.clamp(0, rel_extent - 1)] * (dist < rel_extent)

    return score_mod


class Inkling2BlockAttentionResidual(nn.Module):
    """Pseudo-query attention over the block states (the previous block outputs)
    plus the current partial state."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, query: Tensor, block_states: Tensor, partial: Tensor) -> Tensor:
        """``query``: ``[num_queries, H]``; returns ``[B, S, num_queries, H]``."""
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float()
        stacked = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + self.eps)
        q = query.float().reshape(-1, self.hidden_size)
        # Products and sums, not matmuls: Inductor rewrites matmul-softmax-matmul into SDPA, whose memory-efficient
        # backward allocates [S, 128, H] fp32 for these at most 3 queries (72 GiB for 32K tokens).
        scores = (q.unsqueeze(-2) * stacked.unsqueeze(-3)).sum(-1) * self.hidden_size**-0.5
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * stacked.unsqueeze(-3)).sum(-2)


class Inkling2AttnRes(nn.Module):
    """The per-layer queries of the block attention residuals."""

    def __init__(self, config: Inkling2TextConfig):
        super().__init__()
        local_ids = set(config.local_layer_ids)
        # The release stores these in FP32; parameters use the model dtype (FSDP requires
        # one storage dtype) and the conversion chain casts them back to FP32 for HF export.
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(config.hidden_size) if (i + 1) in local_ids else torch.zeros(3, config.hidden_size),
                requires_grad=True,
            )
            for i in range(config.num_hidden_layers - 1)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(config.hidden_size), requires_grad=True) for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=True)


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        hidden = config.hidden_size

        self.self_attn = Inkling2Attention(config, layer_idx)
        if self.is_sliding:
            self.input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
        else:
            self.q_input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
            self.k_input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
            self.v_input_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(RMSNormConfig(hidden, config.rms_norm_eps))
        self.post_attention_convolution = Inkling2ShortConvolution(hidden, config.sconv_kernel_size)
        self.post_mlp_convolution = Inkling2ShortConvolution(hidden, config.sconv_kernel_size)
        self.attention_input = Inkling2BlockAttentionResidual(hidden, config.rms_norm_eps) if layer_idx > 0 else None
        self.mlp_input = Inkling2BlockAttentionResidual(hidden, config.rms_norm_eps)

        moe_args = MoEArgs(
            num_experts=config.n_routed_experts,
            expert_type="gated",
            activation="softcapped_swiglu",
            score_func="sigmoid",
            route_norm=True,
            route_scale=config.route_scale,
            score_before_experts=False,
            top_k=config.num_experts_per_tok,
            load_balance_coeff=1e-3,
        )
        self.mlp = MoE.from_args(moe_args, dim=hidden, hidden_dim=config.moe_intermediate_size, shared_expert=None)

    def forward(
        self,
        hidden_states: Tensor,
        block_states: Tensor,
        attn_query: Tensor | None,
        mlp_query: Tensor,
        positions: Tensor,
        block_masks: dict[str, BlockMask],
        routed_experts: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        dtype = hidden_states.dtype
        block_residual = hidden_states

        if self.attention_input is not None:
            q_hidden = self.attention_input(attn_query, block_states, hidden_states)
            if self.is_sliding:
                hidden_states = q_hidden[..., 0, :]
            else:
                q_hidden, k_hidden, v_hidden = q_hidden.unbind(-2)

        if self.is_sliding:
            attn_input = self.input_layernorm(hidden_states).to(dtype)
            attn_out = self.self_attn(attn_input, attn_input, attn_input, positions, block_masks["sliding"])
        else:
            attn_out = self.self_attn(
                self.q_input_layernorm(q_hidden).to(dtype),
                self.k_input_layernorm(k_hidden).to(dtype),
                self.v_input_layernorm(v_hidden).to(dtype),
                positions,
                block_masks["full"],
            )
        attn_out = self.post_attention_convolution(attn_out, positions)

        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
            residual = attn_out
        else:
            residual = (block_residual.float() + attn_out.float()).to(dtype)

        mlp_in = self.mlp_input(mlp_query, block_states, residual)[..., 0, :]
        mlp_out = self.mlp(self.post_attention_layernorm(mlp_in).to(dtype), routed_experts=routed_experts)
        mlp_out = self.post_mlp_convolution(mlp_out, positions)
        return ((residual.float() + mlp_out.float()).to(dtype), block_states)


def _block_mask(first_key: Tensor) -> BlockMask:
    """The flex_attention mask in which token ``q`` of the row attends keys ``first_key[q]`` to ``q``.

    ``first_key`` does not decrease along the row, so the tiles of queries and keys are classified
    from it alone. ``create_block_mask`` would evaluate the mask on every query and key pair: 64 GiB
    of booleans for a 256K row. The mask is head-independent (the relative bias lives in score_mod),
    and flex_attention broadcasts it over the query heads.
    """
    row = first_key.shape[0]
    starts = torch.arange(0, row, BLOCK_SIZE, device=first_key.device)
    ends = (starts + BLOCK_SIZE).clamp(max=row) - 1
    q_start, q_end, k_start, k_end = starts[:, None], ends[:, None], starts[None], ends[None]
    # The first query of the tile at or after the key tile's start reaches furthest into the key tile.
    first_query = torch.maximum(q_start, k_start)
    visible = (first_query <= q_end) & (first_key[first_query] <= k_end)
    # Like create_block_mask, a tile that the row's end cuts short is never full.
    complete = ends - starts == BLOCK_SIZE - 1
    full = (k_end <= q_start) & (first_key[q_end] <= k_start) & complete[:, None] & complete[None]

    def ordered(tiles: Tensor) -> tuple[Tensor, Tensor]:
        tiles = tiles.to(torch.int32)
        num_tiles = tiles.sum(-1, dtype=torch.int32)
        indices = torch.argsort(tiles, dim=-1, descending=True, stable=True).to(torch.int32)
        return num_tiles[None, None], indices[None, None]

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx >= first_key[q_idx]) & (kv_idx <= q_idx)

    return BlockMask.from_kv_blocks(
        *ordered(visible & ~full),
        *ordered(full),
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(row, row),
    )


class Inkling2PreTrainedModel(PreTrainedModelPrimeRL):
    config: Inkling2Config

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, MoE):
            buffer_device = module.tokens_per_expert.device
            if buffer_device.type == "meta":
                buffer_device = torch.device("cpu")
            module.init_weights(std, buffer_device=buffer_device)
        elif isinstance(module, Inkling2AttnRes):
            for param in module.parameters():
                nn.init.zeros_(param)
        elif isinstance(module, Inkling2ScaledL2Clip):
            nn.init.ones_(module.scale)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Inkling2DynamicShortConvolution):
            nn.init.zeros_(module.static_weight)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, Inkling2Attention):
            nn.init.normal_(module.rel_logits_proj, mean=0.0, std=std)

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(name.startswith("model.llm.") for name in state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any("mlp.experts.gate_proj" in name for name in state_dict)

    @classmethod
    def conversion_chain(cls, config):
        return conversion_chain(config)

    @classmethod
    def keep_in_fp32_for_weight_transfer(cls, name: str) -> bool:
        return "attn_res." in name or name.endswith("mlp.router.selection_bias")

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        # The Ulysses constraint, on the global and the sliding-window attention.
        text_config = getattr(config, "text_config", config)
        for prefix in ("", "swa_"):
            q_heads = getattr(text_config, f"{prefix}num_attention_heads")
            kv_heads = getattr(text_config, f"{prefix}num_key_value_heads")
            if q_heads % cp != 0 or (kv_heads % cp != 0 and cp % kv_heads != 0):
                return (
                    f"cp={cp} must divide the {q_heads} {prefix}num_attention_heads and divide or be a multiple of "
                    f"the {kv_heads} {prefix}num_key_value_heads"
                )
        return None


class Inkling2Model(Inkling2PreTrainedModel):
    def __init__(self, config: Inkling2TextConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = RMSNorm(RMSNormConfig(config.hidden_size, config.rms_norm_eps))
        self.layers = nn.ModuleList([Inkling2DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(RMSNormConfig(config.hidden_size, config.rms_norm_eps))
        self.attn_res = Inkling2AttnRes(config)
        self.final_input = Inkling2BlockAttentionResidual(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        routed_experts: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> TextModelOutput:
        """``seq_lens`` describes the whole packed row. Under context parallelism
        (``seq_lens_are_pre_shard``) the other inputs are this rank's shard of it. Positions restart at
        each document and come from ``seq_lens``, so ``position_ids`` is not read."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        cp = self.cp_context
        if seq_lens_are_pre_shard != cp.cp_enabled:
            raise ValueError(
                f"seq_lens_are_pre_shard={seq_lens_are_pre_shard} does not match context parallelism "
                f"(cp={cp.cp_world_size}): the inputs are a shard of the row exactly when it is on"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.embed_norm(inputs_embeds)

        num_tokens = inputs_embeds.shape[1]
        reach = self.config.sconv_kernel_size - 1
        if cp.cp_enabled and num_tokens < reach:
            raise ValueError(
                f"A context-parallel shard of {num_tokens} tokens is shorter than the {reach} tokens the short "
                "convolutions of the next shard read"
            )
        # Attention sees the whole row, everything else this rank's tokens of it.
        row_tokens = num_tokens * cp.cp_world_size
        device = inputs_embeds.device
        seq_lens = seq_lens.to(device=device)
        cu_seqlens, _ = get_cu_seqlens_from_seq_lens(seq_lens, total_tokens=row_tokens)
        token = torch.arange(row_tokens, device=device)
        document_start = torch.repeat_interleave(cu_seqlens[:-1].long(), seq_lens.long(), output_size=row_tokens)
        window_start = token - (self.config.sliding_window_size - 1)
        block_masks = {
            "full": _block_mask(document_start),
            "sliding": _block_mask(torch.maximum(document_start, window_start)),
        }
        # A copy: a compiled layer's input must not be a view of a larger tensor.
        positions = (token - document_start)[cp.cp_rank * num_tokens : (cp.cp_rank + 1) * num_tokens].clone()

        hidden_states = inputs_embeds
        B, S, _ = hidden_states.shape
        block_states = hidden_states.new_zeros(B, S, 0, self.config.hidden_size)

        for layer_idx, layer in enumerate(self.layers):
            attn_query = self.attn_res.attn_queries[layer_idx - 1] if layer_idx > 0 else None
            hidden_states, block_states = layer(
                hidden_states,
                block_states,
                attn_query,
                self.attn_res.mlp_queries[layer_idx],
                positions,
                block_masks,
                routed_experts[:, :, layer_idx] if routed_experts is not None else None,
            )

        hidden_states = self.final_input(self.attn_res.output_query, block_states, hidden_states)[..., 0, :]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return TextModelOutput(last_hidden_state=hidden_states)


class SoftcapLinear(nn.Linear):
    """``nn.Linear`` that carries the final-logit softcap into the injected LM head."""

    def __init__(self, in_features: int, out_features: int, softcap: float):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap


class Inkling2ForCausalLM(Inkling2PreTrainedModel):
    def __init__(self, config: Inkling2Config | Inkling2TextConfig):
        text_config = getattr(config, "text_config", None)
        if isinstance(text_config, dict):
            text_config = Inkling2TextConfig(**text_config)
        if text_config is None:
            text_config = config
        super().__init__(text_config)
        self.model = Inkling2Model(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = SoftcapLinear(
            text_config.hidden_size, text_config.vocab_size, text_config.final_logit_softcapping
        )
        self.lm_head.unpadded_vocab_size = text_config.unpadded_vocab_size
        self.router_aux_loss_coef = 0.0
        self.num_experts = text_config.n_routed_experts
        self.num_experts_per_tok = text_config.num_experts_per_tok
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        logits_to_keep: Union[int, Tensor] = 0,
        temperature: Tensor | None = None,
        routed_experts: Tensor | None = None,
        *,
        seq_lens: Tensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> PrimeLmOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
        )

    def init_buffers_post_meta(self) -> None:
        for module in self.modules():
            if isinstance(module, MoE):
                if module.tokens_per_expert.device.type != "meta":
                    module.tokens_per_expert.zero_()
                    module.routing_confidence_sum.zero_()
                    if module.router.selection_bias is not None:
                        module.router.selection_bias.zero_()


__all__ = ["Inkling2Config", "Inkling2TextConfig", "Inkling2ForCausalLM", "Inkling2Model", "Inkling2PreTrainedModel"]
