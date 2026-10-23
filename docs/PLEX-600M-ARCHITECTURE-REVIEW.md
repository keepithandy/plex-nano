# Plex 600M — first architecture review

**Review state:** Architecture code + lightweight tests only. **No training performed, no trained weights, no tokenizer fitted, no benchmark pass, no model checkpoint.** This is **not** a roadmap or an experiment authorization.

## Purpose and isolation

The owner chose a **scratch-trained** ~600M-parameter coding/language model to develop independently of Plex Nano's 27,566,080-parameter research baseline. The new implementation is deliberately isolated:

- **New model:** [`training/src/plex_training/plex600.py`](../training/src/plex_training/plex600.py)
- **Small CPU test suite:** [`training/tests/test_plex600.py`](../training/tests/test_plex600.py)
- **Unchanged:** `training/src/plex_training/model.py`, `config.py`, `checkpoint.py`, the 27M training CLI, existing historical checkpoints and all experimental training evidence.

The existing `ModelConfig` and `PlexLanguageModel` still describe **only** the 27M baseline. Do not load, overwrite, or reinterpret those checkpoints as 600M checkpoints. This separate module intentionally does not register a training command or add any weights to the repository.

## Architecture for code review

| Component | Chosen value |
|---|---:|
| Model | Decoder-only autoregressive Transformer |
| Trainable parameters | **600,372,480** (analytic count) |
| Transformer blocks | 24 |
| Hidden width | 1,280 |
| Query attention heads | 20 |
| Key/value heads | 5 (grouped-query attention) |
| Head dimension | 64 |
| SwiGLU intermediate width | 4,992 |
| Vocabulary slots | 32,768 |
| Context length | 1,024 tokens |
| Position encoding | Rotary position embeddings (RoPE) |
| Normalization | RMSNorm |
| Output embedding | Tied to token input embedding |
| Attention | Causal `torch.nn.functional.scaled_dot_product_attention` |
| Weight initialization | Random normal (`std=0.02`) for linear + embedding matrices |
| Activation checkpointing | Optional, disabled until explicitly enabled |
| Pretrained weights | **None** |
| Tokenizer | **Not yet built** for this model |

The configuration formula (no linear biases) is:

```text
embedding:     vocab_size * width
each block:    (2 * width^2 + 2 * width * head_dim * kv_heads)
             + (3 * width * ffn_width)
             + (2 * width)                          # RMSNorm scales
final norm:    width
```

For these values that is exactly **600,372,480** trainable parameters. RoPE has no trainable weights. All code accepts a much smaller test configuration to exercise the real implementation without allocating the full model in CI.

## Forward pass and safety

- Input shape: `(batch, tokens)`; dtype: `torch.long`. A token must be in range `0...vocab_size-1`.
- Output: `(logits, loss)`, where logits have shape `(batch, tokens, vocab_size)`.
- Optional `targets` are **already shifted/aligned by the caller**; this method does not shift labels. `-100` marks ignored tokens.
- Optional `target_weights` may zero out unsupervised positions. Positive-weight supervision is required.
- Causal masking prevents future-token leakage. Evaluation disables dropout.
- RoPE and RMSNorm use float32 intermediates for more stable numerical calculations.
- Grouped-query attention expands K/V attention views with `repeat_interleave` for compatibility, so memory usage must be measured rather than assumed to equal an optimized fused GQA kernel.
- Gradient checkpointing can be turned on with `model.gradient_checkpointing_enable()` once an independently approved training runner exists.

**No data fetch, generation CLI, optimizer, checkpoint write, or experiment is triggered merely by importing this module.**

## How to inspect it safely

From the repository root, with the existing Python training environment available:

```powershell
# Inspect the configuration and exact analytic size: does NOT allocate 600M weights.
uv run --project training python -c "from plex_training.plex600 import Plex600Config; c = Plex600Config(); print(c.to_dict()); print(c.parameter_count())"

# Exercise tiny instances of the same code, including forward, backward, causal,
# label-masking, and gradient-checkpointing behavior. No full 600M allocation.
uv run --project training python -m pytest training/tests/test_plex600.py -q
```

Opening `plex600.py` in GitHub or VS Code is sufficient for the owner's source review.

## 16 GB RTX 4080 SUPER considerations

At 600.37M parameters, approximate raw tensor storage before activations, logits, temporary buffers and CUDA workspaces:

| Item | Approximate size |
|---|---:|
| BF16 weights (2 bytes/parameter) | 1.20 GB |
| BF16 gradients, if kept at BF16 (2 bytes/parameter) | 1.20 GB |
| Two FP32 Adam moments (8 bytes/parameter) | 4.80 GB |
| Optional FP32 master weights (4 bytes/parameter) | 2.40 GB |

These are **illustrative components**, not a verified sum for a specific PyTorch optimizer and autocast implementation. In conventional PyTorch AMP, model parameters and gradients may remain FP32, so actual allocations differ. AdamW can also allocate temporary tensors. A safe training plan requires an actual **read-only resource estimate and separately approved CUDA allocation/one-step test** before claiming this fits on 16 GB.

## What this first review does not deliver

- No text generation or conversational quality is demonstrated; random weights cannot meaningfully code.
- No vocabulary/tokenizer training. The frozen 16,384-token Nano BPE **must not** be silently reused to certify a 32,768-slot tokenizer.
- No curated new-scale training corpus, training loop, optimizer selection, LR schedule, GPU step, or checkpoint-resume integration.
- No 600M hardware benchmark, throughput or memory figures.
- No final evaluation holdout access, training authorization, or claims of coding skill.
- The separate 600M weights, if ever trained, require a versioned and independently reviewed checkpoint/manifest contract distinct from Nano's.

**Review decision requested:** inspect code/configuration and automated tests. The next implementation decision belongs to the owner; no roadmap is created here.
