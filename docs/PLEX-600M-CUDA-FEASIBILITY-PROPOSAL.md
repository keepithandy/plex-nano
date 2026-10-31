# Plex 600M CUDA feasibility check — draft proposal

**Status: one authorized CUDA check completed and passed on October 31, 2026.** This proposal measures
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

- [x] Implement the separate [guarded runner](../training/src/plex_training/plex600_feasibility.py), with lazy PyTorch imports, fixed settings and a parent-enforced worker timeout.
- [x] Test its refusal paths, mocked workload, result validation, OOM reporting and process cleanup without running the 600M model.
- [x] Add the [guard tests](../training/tests/test_plex600_feasibility.py) to Linux and Windows CI. Remote CI has not been run for this change.
- [x] Obtain fresh explicit authorization and run this particular CUDA check once. Optimizer updates and training require a separate decision.

## Commands and local verification

From the repository root, using the existing training environment:

```powershell
# Plan only: no PyTorch import, CUDA access, model construction or artifact write.
& .\training\.venv\Scripts\python.exe -m plex_training.plex600_feasibility --device 0

# Historical command for the single check authorized on October 31, 2026.
# A further execution requires a fresh decision:
& .\training\.venv\Scripts\python.exe -m plex_training.plex600_feasibility --device 0 --execute-cuda-check

# Guards, mocks and short test processes only; does not run the 600M model:
& .\training\.venv\Scripts\python.exe -m pytest training/tests/test_plex600_feasibility.py training/tests/test_ci_test_inventory.py -q
```

The default command prints JSON with `status: plan`. Execution prints a JSON
result with a source/runner SHA, Git commit, settings, environment, measurements
and outcome. Failures return a nonzero exit code; the worker is not retried.
The runner writes no result file; its stdout can be preserved separately after
review. A timeout is reported even if a partial worker message exists, and a
malformed or mismatched worker result is inconclusive.

**Local guard verification, October 31, 2026:** 22 tests passed on Windows with Python
3.12.10. Tests used mocks for model/CUDA behavior and real short-lived processes
for default-command and timeout checks. Linux and Windows CI entries are
configured, with no new remote results claimed.

## Authorized CUDA result — October 31, 2026

The owner authorized one check. It completed with **status `pass`**, exit code 0,
and no optimizer updates. The full model's single 1,024-token forward/backward
produced finite loss **10.65451717376709** and **218/218 present, finite FP32
gradient tensors**. PyTorch reported peak allocated **5,187,562,496 bytes**
(4,947.25 MiB) and peak reserved **6,438,256,640 bytes** (6,140 MiB). The
worker took **8.36796179995872 seconds**; initialization, forward, and backward
times were **4.193430999992415**, **1.6220829000230879**, and
**0.13697680016048253 seconds**, respectively. These are single observations,
not a throughput benchmark.

The [result record](../training/pretraining/plex600-cuda-feasibility-2026-10-31-result.json)
transcribes the JSON printed by the runner, including the optional NumPy warning,
GPU memory and environment details. It was saved after the run from the tool
output; it is not a byte-for-byte captured stdout file. The model source was
clean at Git commit `9261946267dae6c995aacc22061b5e4244cadc25`; the runner was
untracked at execution and is identified by its recorded SHA-256, not that
commit. No checkpoint, model output, or optimizer state was saved. This result
establishes only that this fixed forward/backward workload fit once; optimizer
state, an optimizer update, sustained training and coding ability remain
unmeasured. The final holdout remains closed.

After the run, the runner was changed to retain a narrowly checked, explicitly
unverified source-refusal hint if the worker cannot establish its identity; the
outcome remains inconclusive in that case. The runner's current bytes therefore
have a different SHA from the historical run. This reporting change does not
alter or re-execute the recorded check.

## Evidence available while drafting

On October 31, 2026, read-only `nvidia-smi` reported an NVIDIA GeForce RTX 4080
SUPER, 16,376 MiB total, 14,732 MiB free and driver 610.60. Free memory is not
reserved and must be checked again at execution. The model source was inspected
without constructing a model. No CUDA model work, checkpoint access, dataset
access, final holdout access, or training was performed for this proposal.
