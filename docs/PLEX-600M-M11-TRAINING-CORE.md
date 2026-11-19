# M11 training-core implementation: source-conditioned copying (review only)

**Status: implementation preparation, not training-ready or approved.** This
branch adds a deterministic, CPU-verifiable training core for the next
600M experiment. It does not run an optimizer, load model weights, import
CUDA on its default CLI path, create approval files, or mutate any M9/M10
artifact. The project's final holdout remains closed.

The verified M10 delimiter experiment prevented immediate EOS but produced
the same `= 0;\n` on all 18 prompts and 0/18 exact copies. The M11 objective
is to establish **source-conditioned content**, rather than optimizing the
stopping metric by itself.

## Exact data and update order

The proposed candidate pair is `let score = 0;\n` and
`let score = 1;\n`, with the M10 copying instruction and a **final
space** at the prompt boundary. The answer's terminal newline and EOS token
are unchanged. A separate `let value = 0/1;\n` pair is **development-only**
and has zero scheduled optimizer exposure; it is not an untouched final
holdout.

`exact_schedule()` defines **200 alternating single-example updates**:
100 presentations per training candidate, no gradient accumulation. The
predeclared planned milestones are 25, 50, 100 and 200 updates, plus an
update-zero read-only baseline when a runner is later implemented.

Both `candidateSha256` and `scheduleSha256` are printed by the default
plan. Any eventual owner approval must bind these exact values, the M9
checkpoint and tokenizer identities, optimizer parameters, arm definitions,
execution time and storage bounds. **The old M10 owner approval is not M11
authorization.**

## Training primitives implemented

`packed_training_samples(tokenizer)` audits the M7-fitted tokenizer,
including the single-token source contrast, exact text round trip,
maximum lengths and masked prompt. Only copied answer tokens and the final
EOS are supervised. The development probes are never packed for training.

`weighted_copy_loss(torch, logits, targets, eos_weight)` accepts only
the *proposed* arms 1.0 (M10-style terminal EOS loss) and 0.25 (reduced
terminal weight). It computes unreduced cross entropy, weights only the
terminal EOS position and divides by the **original count of supervised
positions**, not the sum of weights. For identical logits, the gradient
for each content position is therefore the same in both arms; only the
terminal EOS contribution differs. CPU tests compare these gradients.

`source_pair_measurement(...)` measures actual model logits under the
correct gold answer prefix for each input, requiring both the full-vocab
greedy token and the 0/1 margin to switch correctly when the source changes.
This is **not autonomous copying**. Separately,
`score_unverified_generations(...)` scores all four outputs under both
decoder policies, requires exact bytes and EOS, and rejects duplicate
IDs or mismatched source/checkpoint hashes. These are **caller-supplied**
records and are deliberately flagged unverified. A later GPU worker must
independently bind raw outputs to hash-verified snapshots before calling
any results model-generated.

## Mandatory gates before any real training

- A separate CPU-only exact-scope M11 approval/admission path, with verified
  independent M9 backup, frozen bundle/source SHA, and new output path.
- Hard subprocess timeout, GPU memory/VRAM and storage admission, fresh
  optimizer, exact step receipt and early-stop provenance.
- Create-only SHA-checked model snapshots, CPU save/reload parity without
  disturbing live CUDA parameters, plus independently rescored 2-decoder
  raw generation evidence at every milestone.
- Separate train/development scorecards: no perfect-score claim based on
  terminal EOS after teacher forcing, constant outputs, or source-insensitive
  generations. Do not access the project final holdout.

The existing M10 worker shows patterns for these protections, but its
existing authorization contract **must not be reused** to execute M11.

## Read-only verification

```bash
PYTHONPATH=training/src python -m plex_training.plex600_m11_training_core
PYTHONPATH=training/src python -m plex_training.plex600_m11_training_core --show-schedule
PYTHONPATH=training/src python -m plex_training.plex600_m11_training_core \
  --audit-frozen-tokenizer --bundle training/artifacts/plex600-fresh-tokenizer
PYTHONPATH=training/src python -m unittest training/tests/test_plex600_m11_training_core.py
```

There is **no --execute-approved-run** or --train option in this module.
The full 200-update two-arm 600M experiment has **not** been authorized,
implemented as an executable worker, or performed.
