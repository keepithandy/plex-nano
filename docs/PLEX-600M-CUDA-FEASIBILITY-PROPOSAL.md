# Plex 600M CUDA feasibility check — draft proposal

**Status: design only; not approved or executed.** This proposal measures
whether one fixed full-model forward/backward workload can complete on the
available GPU. It does not test coding skill, optimizer memory, an optimizer
step, or sustained training. Preparation and execution require separate
decisions; this document does not grant execution permission.

## Fixed workload proposed for review

| Item | Proposed setting |
|---|---|
| Source | `training/src/plex_training/plex600.py`, SHA-256 `e7ad49628b1ebcf083991c385a9c55cff159c5567a6a0f6062d48418f26d1af2`; refuse a mismatch |
| Model | Default `Plex600Config`, 600,372,480 trainable parameters, freshly initialized from seed 17; no checkpoint load |
| Device | A single explicitly selected NVIDIA RTX 4080 SUPER with CUDA and BF16 support; refuse otherwise |
| Shape | One synthetic sequence, batch 1 × 1,024 tokens, vocabulary capacity 32,768 |
| Synthetic inputs | IDs `0..1023`; targets are IDs `1..1023` followed by `-100`, all `torch.long` on the selected device |
| Precision | FP32 parameters and gradients; CUDA BF16 autocast during the forward/loss calculation; no gradient scaler |
| Activation checkpointing | Enabled; model in training mode so the existing checkpointing path is active; dropout is zero in the default config |
| Work | Exactly one forward/loss and one backward; no optimizer object, updates, generation, tokenizer or dataset |
| Memory guard | Require at least 14,000 MiB free immediately before model allocation; limit PyTorch's process allocator to 80% of device total memory; abort on allocation failure |
| Time guard | One supervised worker process, terminated by its parent after 600 seconds, including initialization; no automatic retries or changed settings |
| Outputs | Report metadata and measurements only; no weights, gradients, logits, source data, checkpoint, or model-generated text written to disk |

The free-memory check is a snapshot, and the allocator cap is not a system-wide
GPU memory limit: other processes and CUDA allocations outside PyTorch's caching
allocator can change the outcome. Use a fresh process and do not run concurrent
GPU work during the check. An OOM, timeout, device mismatch or non-finite value
is a recorded failure or inconclusive result, not a reason to silently reduce
the workload or retry.

## Preflight and outcome

Before allocation, verify the source hash and configuration, CUDA availability,
selected device name and BF16 support, available memory, and allocator cap.
Record the exact Git commit, whether the model source is clean, operating
system, Python and PyTorch versions, CUDA runtime, GPU name, driver, total/free
memory, seed and all settings. Refuse execution if the hash differs, the source
is locally modified, the GPU does not match, BF16 is unsupported, or the free
memory threshold is not met. Do not fetch packages or data during the check.

Reset CUDA peak-memory statistics before model allocation. Synchronize at
measurement boundaries and record elapsed initialization, forward and backward
times, PyTorch peak allocated and reserved memory, finite scalar loss, and
whether every trainable parameter has a present, finite gradient. End the worker
and release its process-owned resources on success or failure. If killed by the
parent, report timeout rather than asserting cleanup or successful completion.

**Pass for this one workload** requires successful initialization, the exact
parameter count and shape, finite loss, successful backward, and present finite
gradients for all trainable parameters within the memory and time limits. Any
failed prerequisite is a refusal, not a model result. Preserve OOM and other
runtime errors as failures, and missing/untrustworthy measurements as
inconclusive. Report the actual settings and outcome; a pass does not establish
that Adam state, a training update, other sequence lengths, or repeated updates
fit, nor that the untrained model can edit code.

## Work needed before a decision to execute

1. Review these choices and their suitability for the target hardware.
2. Implement a separate guarded, single-use worker and parent timeout, with no
   implicit execution on import; keep it isolated from the Nano baseline.
3. Review and test refusal paths and process cleanup without running the 600M
   model; preserve the existing model and test results.
4. Present the exact command, source identities, settings and safety behavior
   for fresh explicit authorization of this particular CUDA check. A separate
   authorization would be required for optimizer updates or training.

## Evidence available while drafting

On October 31, 2026, read-only `nvidia-smi` reported an NVIDIA GeForce RTX 4080
SUPER, 16,376 MiB total, 14,732 MiB free and driver 610.60. Free memory is not
reserved and must be checked again at execution. The model source was inspected
without constructing a model. No CUDA model work, checkpoint access, dataset
access, final holdout access, or training was performed for this proposal.
