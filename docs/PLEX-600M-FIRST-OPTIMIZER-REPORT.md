# Plex 600M · First optimizer-feasibility report

**October 31, 2026. Status: runner prepared; full-size optimizer result pending local GPU execution.**

This report is the first evidence gate before a bounded real-data training
pilot, not another architecture redesign. The fixed M6 protocol is implemented
in [plex600_optimizer_feasibility.py](../training/src/plex_training/plex600_optimizer_feasibility.py).
It runs a *single synthetic optimizer update* in a supervised subprocess on
the owner's RTX 4080 SUPER. It does not require or read the unresolved M4
candidate corpus. A pass means one optimizer step fit that exact configuration;
it **does not** prove sustained training or coding skill.

## Evidence already measured, not a new result

The previous [GPU forward/backward report](../training/pretraining/plex600-cuda-feasibility-2026-10-31-result.json)
records this configuration on Windows 11, RTX 4080 SUPER (16,376 MiB), CUDA 12.6,
and PyTorch 2.14.0+cu126:

| Observation | Earlier actual measurement |
| --- | ---: |
| Model parameters | 600,372,480 |
| Batch / sequence | 1 / 1,024 |
| Synthetic loss (random weights) | 10.654517 |
| GPU peak allocated | 4,947.25 MiB |
| GPU peak reserved | 6,140 MiB |
| Gradient tensors present and finite | 218 / 218 |
| **Optimizer updates** | **0** |

The peak VRAM includes gradients and activations, **not** the AdamW first-step
moments or optimizer work buffers. The earlier measured loss is **not** a
validation score or evidence of coding behavior.

The separate 27M experiment remains a failed development baseline: **0/36**
exact code edits and **0/36** unrelated code preservation at its frozen
100-step endpoint. See
[retained development summary](../training/pretraining/p3-02-development-gate-result-summary.json).
No comparison is yet possible for untrained Plex 600M.

## Fixed one-step M6 protocol

| Setting | Fixed value |
| --- | --- |
| Weights | Random from scratch, seed 17 |
| Architecture | 600,372,480 parameters, 24 blocks, width 1,280 |
| GPU | NVIDIA GeForce RTX 4080 SUPER, native BF16 |
| Batch / context | 1 × 1,024 next-token synthetic IDs |
| Model and gradients | FP32 |
| Forward autocast | BF16 |
| Activation checkpointing | Enabled |
| Optimizer | AdamW, FP32 first/second moments |
| LR / weight decay | 0.0001 / 0 |
| PyTorch implementation | foreach=False, fused=False |
| Update count | **Exactly one** |
| Free-VRAM minimum | 14,000 MiB |
| CUDA allocator fraction | 0.80 |
| Worker wall-time limit | 600 seconds |
| Checkpoint / dataset / final holdout | None |

The checks refuse a wrong GPU, insufficient free VRAM, unpinned/dirty model
or runner source, and non-finite gradients/optimizer state. The worker is killed
on timeout. A failed or inconclusive result is still a valid research result
and must be retained, not hidden or labeled as a pass.

## Run on the owner's RTX 4080 SUPER

The following commands run from a clean checkout of master in PowerShell.
The **first command does not touch CUDA**. The second performs the
**owner-authorized one-update check**. Do not run it while other workloads
occupy the GPU.

~~~powershell
uv run --project training --no-sync python -m plex_training.plex600_optimizer_feasibility
New-Item -ItemType Directory -Force training/artifacts | Out-Null
uv run --project training --no-sync python -m plex_training.plex600_optimizer_feasibility --execute-optimizer-check --output training/artifacts/plex600-m6-optimizer-first-report.json
~~~

The local report is create-only: choose a new path if it already exists.
The report captures device, driver, PyTorch version, model/runner source hashes,
finite loss and gradient checks, parameter change, AdamW state checks, per-phase
seconds, peak allocated/reserved bytes, and worker status. This report should be
reviewed **before** deciding whether more updates or precision/optimizer
changes are warranted. GitHub's standard Linux and Windows CI jobs do not
provide this specific GPU; they can validate only portable safeguards.

## Next recorded outcome

The next report must contain the real local JSON result, identified by its
SHA-256 and repository revision. **No 600M optimizer measurement has been made
by this commit.** Once that is measured, and after the pending M4 dataset-rights
and independent split decisions, propose a separately bounded real-data pilot:
fixed token budget, seeded sampler, update count, validation schedule, memory
ceiling, and checkpoint retention. That pilot is a distinct training
authorization, not permission to run uncontrolled updates.

Never substitute a lower training loss for exact-code edit, behavioral or
unrelated-code preservation evaluations. The protected final holdout remains
sealed.
