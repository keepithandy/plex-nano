"""Review-only, from-scratch ~600M-parameter Plex decoder.

Separate from the frozen 27M PlexLanguageModel, its ModelConfig, training
CLI, tokenizer, checkpoints, and evaluation artifacts. This module contains
no download, checkpoint loading, training execution, or file-writing logic.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Plex600Config:
    """Model dimensions; the default configuration has 600,372,480 parameters."""

    vocab_size: int = 32_768
    context_length: int = 1_024
    width: int = 1_280
    layers: int = 24
    attention_heads: int = 20
    key_value_heads: int = 5
    feed_forward_width: int = 4_992
    rope_theta: float = 10_000.0
    norm_epsilon: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        dimensions = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "width": self.width,
            "layers": self.layers,
            "attention_heads": self.attention_heads,
            "key_value_heads": self.key_value_heads,
            "feed_forward_width": self.feed_forward_width,
        }
        if any(type(value) is not int or value <= 0 for value in dimensions.values()):
            raise ValueError("All model dimensions must be positive exact integers")
        if self.width % self.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("attention_heads must be divisible by key_value_heads")
        if (self.width // self.attention_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        for name in ("rope_theta", "norm_epsilon", "dropout"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.rope_theta <= 0 or self.norm_epsilon <= 0:
            raise ValueError("rope_theta and norm_epsilon must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.width // self.attention_heads

    def parameter_count(self) -> int:
        """Count all trainable parameters in this specific architecture.

        This design uses RMSNorm (scale only), no linear biases, and a tied
        token embedding / LM output projection. RoPE has no trained weights.
        """
        q_and_output = 2 * self.width * self.width
        k_and_v = 2 * self.width * self.head_dim * self.key_value_heads
        swiglu = 3 * self.width * self.feed_forward_width
        norms_per_block = 2 * self.width
        return (
            self.vocab_size * self.width
            + self.layers * (q_and_output + k_and_v + swiglu + norms_per_block)
            + self.width
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, hidden: Tensor) -> Tensor:
        # Float32 variance for numerical stability with BF16 activations.
        normal = hidden.float()
        normal = normal * torch.rsqrt(normal.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return normal.to(dtype=hidden.dtype) * self.weight.to(dtype=hidden.dtype)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_frequency = 1.0 / (theta ** (indices / head_dim))
        self.register_buffer("inv_frequency", inv_frequency, persistent=False)

    def forward(self, length: int, device: torch.device) -> tuple[Tensor, Tensor]:
        positions = torch.arange(length, dtype=torch.float32, device=device)
        angles = torch.outer(positions, self.inv_frequency.to(device=device))
        # (1, 1, tokens, head_dim / 2), shared by every attention layer.
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rotary(hidden: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate adjacent even/odd channels in (B, H, T, D) attention tensors."""
    pairs = hidden.float().reshape(*hidden.shape[:-1], -1, 2)
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.reshape_as(hidden).to(dtype=hidden.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Plex600Config) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.width, config.width, bias=False)
        self.k_proj = nn.Linear(config.width, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.width, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.width, config.width, bias=False)

    def forward(self, hidden: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        batch, tokens, _ = hidden.shape

        def split(projection: nn.Linear, heads: int) -> Tensor:
            return projection(hidden).reshape(batch, tokens, heads, self.head_dim).transpose(1, 2)

        query = apply_rotary(split(self.q_proj, self.heads), cosine, sine)
        key = apply_rotary(split(self.k_proj, self.kv_heads), cosine, sine)
        value = split(self.v_proj, self.kv_heads)

        # Portable GQA implementation for CPU and CUDA PyTorch SDPA kernels.
        # Only keys/values are expanded; the learned K/V projection stays small.
        groups = self.heads // self.kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        combined = attended.transpose(1, 2).contiguous().reshape(batch, tokens, -1)
        return self.o_proj(combined)


class SwiGLU(nn.Module):
    def __init__(self, config: Plex600Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.width, config.feed_forward_width, bias=False)
        self.up_proj = nn.Linear(config.width, config.feed_forward_width, bias=False)
        self.down_proj = nn.Linear(config.feed_forward_width, config.width, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Plex600Block(nn.Module):
    def __init__(self, config: Plex600Config) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.norm_epsilon)
        self.attention = GroupedQueryAttention(config)
        self.feed_forward_norm = RMSNorm(config.width, config.norm_epsilon)
        self.feed_forward = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        hidden = hidden + self.residual_dropout(
            self.attention(self.attention_norm(hidden), cosine, sine)
        )
        return hidden + self.residual_dropout(self.feed_forward(self.feed_forward_norm(hidden)))


class Plex600Model(nn.Module):
    """Causal LM with scratch weights, tied embeddings and optional checkpointing.

    `targets` are already next-token aligned with `input_ids`: this forward
    method does NOT shift labels. -100 may be used as the ignored label.
    No data access, optimizer, training loop, or checkpoint selection is done.
    """

    def __init__(self, config: Plex600Config | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else Plex600Config()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.width)
        self.rotary = RotaryPositionEmbedding(self.config.head_dim, self.config.rope_theta)
        self.blocks = nn.ModuleList(Plex600Block(self.config) for _ in range(self.config.layers))
        self.final_norm = RMSNorm(self.config.width, self.config.norm_epsilon)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.gradient_checkpointing = False
        self.apply(self._initialize)
        # Scale deep residual branches at scratch initialization.
        residual_std = 0.02 / math.sqrt(2 * self.config.layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def gradient_checkpointing_enable(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        target_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must have shape (batch, sequence) and dtype long")
        batch, sequence = input_ids.shape
        if batch <= 0 or not (1 <= sequence <= self.config.context_length):
            raise ValueError("input dimensions exceed the configured context")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("input token id is outside vocabulary")
        if targets is None and target_weights is not None:
            raise ValueError("target_weights require targets")
        if targets is not None:
            if (targets.shape != input_ids.shape or targets.dtype != torch.long
                    or targets.device != input_ids.device):
                raise ValueError("targets must be long tensors aligned with input_ids")
            if bool(((targets != -100) & ((targets < 0) | (targets >= self.config.vocab_size))).any()):
                raise ValueError("target token id is outside vocabulary")
            if not bool((targets != -100).any()):
                raise ValueError("at least one target must be supervised")
            if target_weights is not None:
                if (target_weights.shape != targets.shape or target_weights.device != targets.device
                        or not torch.is_floating_point(target_weights)):
                    raise ValueError("target_weights must be floating tensors aligned with targets")
                if not bool(torch.isfinite(target_weights).all()) or bool((target_weights < 0).any()):
                    raise ValueError("target_weights must be finite and nonnegative")
                if not bool(((targets != -100) & (target_weights > 0)).any()):
                    raise ValueError("at least one supervised target must have positive weight")

        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        cosine, sine = self.rotary(sequence, input_ids.device)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, cosine, sine, use_reentrant=False)
            else:
                hidden = block(hidden, cosine, sine)
        logits = F.linear(self.final_norm(hidden), self.token_embedding.weight)
        loss: Tensor | None = None
        if targets is not None:
            token_losses = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none" if target_weights is not None else "mean",
            )
            if target_weights is None:
                loss = token_losses
            else:
                weights = target_weights.to(dtype=token_losses.dtype, device=logits.device)
                weights = weights * (targets != -100)
                loss = (token_losses.reshape_as(weights) * weights).sum() / weights.sum()
        return logits, loss


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
