# Plex Nano 600M — M7 four-update experiment contract

**Status: implementation prepared; NO M7 OPTIMIZER TRAINING EXECUTED, NO APPROVAL ISSUED.**

## Fixed experiment

| Property | Contract |
| --- | --- |
| Architecture | Pinned 600,372,480-parameter model |
| GPU | RTX 4080 SUPER with native BF16, >=14,000 MiB free |
| Dtype / checkpointing | FP32 parameters and gradients, BF16 autocast, activation checkpointing |
| Updates | Exactly 4 AdamW updates, no extension |
| Optimizer | LR 0.0001, epsilon 1e-8, weight decay 0; foreach/fused disabled |
| Seed | 17 |
| Sampling | Four distinct shuffled training records, no crossing record boundaries |
| Batch / context | 1 record, up to 128 next-token positions per update |
| Validation | All 24 held-out internal records before and after, no gradients |
| Reporting | Each pre-update loss, post-training selected-window loss, validation losses, throughput, GPU memory peaks |
| Checkpoint | One model-only weights file, SHA-256/size check, atomic create-only publication |
| Disk / runtime | >=5 GiB free, max 3 GiB checkpoint, supervised 900-second worker |
| External six-source candidate / old P2 validation / final holdout | Never accessed |
| Authorization | Explicit separate owner JSON decision and execution flag; neither supplied today |

The 120 training records contain 6,970 tokens and the 24 internal-validation
records contain 1,845 tokens. These 144 owner-authored examples were drawn
only from a former P2 *training* split. The 72 historical P2-validation records
and all six rights-unresolved external repositories remain excluded.

This experiment could show mechanics and overfitting, not useful coding skill.
There is still no independently designed coding-editing development benchmark.

## Pinned identities from local October 31 tokenizer report

- Source manifest SHA-256: 4bb5742df59f0c56aaac34fc3129012b4617acf7d86a1804e47c5aa69e541a07
- Tokenizer SHA-256: 9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2
- Train packed tokens SHA-256: 3819b4fe0939e5f12b82c5a0e78d536ccc746fb558e419647eb28d0969886500
- Validation packed tokens SHA-256: 57cea37870cc9b5fde4e2bc525b778282951c13a06c87762ca18065a53d42999
- Actual vocabulary size 1,191, within fixed 32,768-capacity model.

The local bundle manifest SHA-256 must be supplied independently; these
four identities alone are not enough. M7 invokes the existing exact
source-binding check, validates original authored data/owner P2 approvals,
inspects split record IDs and EOS framing, and refuses mismatches *before*
GPU allocation.

## Allowed now: plan and read-only preflight

Run from a clean Windows checkout. These commands update ZERO parameters.

~~~powershell
git pull --ff-only origin master

uv run --project training --no-sync python -m plex_training.plex600_m7

$bundle = "training/artifacts/plex600-owner-authored-pilot-tokenizer-v1"
$bundleSha = (Get-FileHash "$bundle/manifest.json" -Algorithm SHA256).Hash.ToLowerInvariant()

uv run --project training --no-sync python -m plex_training.plex600_m7 --preflight --bundle-dir $bundle --bundle-manifest-sha256 $bundleSha
~~~

The preflight result should say status source-bound-plan-only,
trainingAuthorized false and optimizerUpdatesCompleted zero.

## Future owner approval: separate decision, NOT yet granted

A future *local, uncommitted* approval would need all fields shown below and
a new explicit authorization from the owner for this exact four-update run.

~~~json
{
  "kind": "plex600-m7-owner-training-approval-v1",
  "approvalStatus": "approved",
  "approvedBy": "repository-owner",
  "trainingScope": "local-four-update-m7-owner-authored-pilot-only",
  "approvedUpdates": 4,
  "sourceManifestSha256": "4bb5742df59f0c56aaac34fc3129012b4617acf7d86a1804e47c5aa69e541a07",
  "bundleManifestSha256": "the-independently-verified-local-bundle-manifest-sha256",
  "finalHoldoutAllowed": false
}
~~~

This is a **schema example, not an approval**. Do not generate it or use
the future execution flag until explicit training consent is recorded.

The guarded execution branch requires BOTH the approval file and
--execute-approved-run. It rechecks the approval and exact source identity
inside a bounded child process. Successful execution produces exactly one
model-only checkpoint (weights.pt), SHA-256 and report.json in one atomic,
create-only directory under training/artifacts. The optimizer state is
intentionally NOT saved, so resume is not supported. Timeout or failure
removes temporary files, leaving no published model. No external sources,
historical validation or final project holdout are read.

## Verification boundaries

CI must test plans, refusal paths, approval content checks, deterministic
sampling, source binding (without CUDA), output constraints, and checkpoint
hash detection. CI must NOT execute 600M optimizer steps.

Once an independently authorized local run is eventually performed, its
actual four-step loss curve, before/after internal validation, throughput,
VRAM peaks, checksum and training report must be inspected. **The M6
single-step hardware feasibility report is not an M7 training result.**
