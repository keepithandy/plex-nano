# M11 CPU admission gate and remaining training blockers

**Status: review-only, no GPU training, no owner approval.** This work is stacked on draft PR #211. It cannot import CUDA, load model weights for training, execute optimizer steps, write checkpoints or create an approval file. M9/M10 weights and the project's final holdout remain untouched.

## Implemented now

The standalone M11 CPU admission module checks source provenance before any eventual GPU execution:

- A separate M9 backup and report are verified using the existing original M9 backup checker. It also audits the frozen M7 tokenizer and M11 source/answer pairs, capturing the exact source, candidate, schedule, model and tokenizer SHA-256 bindings.
- It requires a distinct **M11 space-delimited four-case, two-decoder M9 read-only baseline**. It explicitly rejects the M10 18-case newline baseline. The M11 baseline must have eight raw responses, with hashes and scores independently recomputable from those responses. Receipt consistency alone does not prove actual inference.
- It prints an owner-review template bound to the 200 alternating updates, exact selected examples, answer-newline and EOS behavior, checkpoint milestones 25/50/100/200, optimizer settings, candidate EOS arm (1.0 or 0.25), decoder policies, M9 backup, and baseline.
- Owner consent is separate for each EOS arm, must be provided as an external file, and cannot reuse M10 approval. The module never grants or creates approval. Changed examples, source hashes, schedule, arm or baseline invalidate prior consent.
- The proposed space check demands at least **16 GiB free per arm**, enough for four maximum-size model-only checkpoints, one temporary save and 1 GiB safety headroom. This is a disk-space check, **not** a measured GPU feasibility result.
- Even a successful source/consent audit reports trainingAuthorized=false because the independent M11 GPU worker is not implemented.

The two train candidates are the score 0/1 pair. The value 0/1 pair remains visible development-only with zero training exposures, and is not an untouched final holdout.

## Next implementation, in priority order

**First: implement a read-only M11 M9 baseline generator.** It must verify the original backup/tokenizer, infer on the exact four space-delimited prompts under both decoders, and produce create-only report.json and responses.jsonl for all eight outputs. Those receipts must be independently re-scored. M10 baseline receipts cannot be substituted.

**Second: owner reviews the exact examples, two EOS arm scopes, budget and evidence.** The new CPU gate can check a distinct external owner-signed JSON artifact, but no M11 approval has been issued by this implementation.

**Third: implement a bounded M11 GPU worker.** Adapt M10's hard subprocess timeout, verified checkpoint/reload parity, source and optimizer identity checks, independent scored raw generation receipts and safe partial-output handling. Each EOS arm must begin from identical independently verified M9 model weights and its own fresh optimizer. Final project holdout access remains forbidden.

No 200-update experiment is authorized or currently running. The research branches should remain draft until these requirements pass.

## Future M11 baseline receipt contract

- Four cases under both decoder policies, eight JSONL responses, each with exactly: id, split, decodingPolicy, output, terminationReason, generatedTokens, promptSha256, checkpointSha256.
- Report protocol: plex600-m11-m9-space-prompt-read-only-baseline-v1.
- Report status: complete-m11-m9-space-read-only-inference.
- Required report evidence: checkpointSha256, m9ReportSha256, bundleManifestSha256, tokenizerSha256, candidateSha256, scheduleSha256, sourceHashes, responsesSha256, selectedCaseIds, decodingPolicies, responseCount=8, optimizerUpdates=0, readOnlyGenerationExecuted=true, trainingAuthorized=false, checkpointWrites=false, finalHoldoutUsed=false, scores. Scores must equal recomputation from the actual raw responses.
- Hash-bound reports are not a replacement for proof of the generation process. The future evaluator must actually run inference and the parent must independently verify the receipts.

## Read-only CLI

Default: python -m plex_training.plex600_m11_admission

CPU audit source inputs: --audit-sources, with --bundle-dir, --bundle-manifest-sha256, --m9-report, --m9-weights.

Generate unapproved template: --show-unapproved-template, adding --baseline-report, --baseline-responses, and --eos-weight (1.0 or 0.25).

Review an owner-provided consent without training: --audit-admission, additionally specifying --approval-file and --output-dir.

There is no --train or --execute-approved-run mode. Linux and Windows CI include the CPU-only refusal and source-binding tests.
