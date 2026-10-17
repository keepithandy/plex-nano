# P3-03 — Source Preservation and Request-Value Binding

**Status: preparation complete; evaluation is not authorized.** This bounded diagnostic measures how the 27,566,080-parameter Plex Nano coding model handles a request to edit one supplied HTML, CSS, or JavaScript file. It separates source-name preservation, requested-value selection, unrelated-content preservation, exact-file correctness, and contrast sensitivity. These are observable output categories; they do not establish an internal cause.

## Frozen artifacts

- [Task manifest](p3-03-source-preservation-v1.json) — 72 tasks, 24 three-request contrast groups, 12 fixed request templates, exact prompts, source files, expected files, target spans, and structural name roles.
- [Manifest builder](prepare_manifest.py) — verifies the authored P3-02 candidate identity, checks complete-file overlap against its 144-record training split, then creates the manifest without overwriting an existing file.
- [Read-only response analyzer](analyze_responses.py) — scores a separately authorized prediction file, writes a new Markdown report and separate JSON audit, and never loads or invokes a model.

Manifest SHA-256: `119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578`.

Preparation code SHA-256 values: builder `6c47c22c2d9e0b2331ad99e113b0bd1861c4bb704df9af5fd71278cd7cbf3127`; analyzer `f4d02a5a35d6e5247121119843f3951628c17864a8d4ee521ea69461c1ca6a8d`.

The prospective overlap check found **zero** matches in all four comparisons: P3-03 inputs against P3-02 training inputs and targets, and P3-03 expected files against P3-02 training inputs and targets. No examples were replaced. The frozen P3-02 candidate hash was verified as `98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403`.

## Baseline and future evaluation configuration

The historical baseline remains the failed P3-02 step-100 development result: 0/36 exact files, 0/36 preservation passes, 0/12 fully correct contrast groups, 36/36 outputs losing `account`, 25/36 outputs matching a training target, and 8/12 groups with identical outputs for all three requests. The fixed checkpoint is SHA-256 `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`; the frozen tokenizer is `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`.

The proposal's P3-02 inference settings are recorded in the manifest: greedy fixed-endpoint generation, seed 1337, temperature 0, and at most 192 new tokens. The prompt serialization is the exact P3-02 convention. **The checkpoint file was not opened or re-hashed during this preparation.** Any future evaluation must independently verify it and record the actual inference code revision, tokenizer/bundle identities, and configuration before the analyzer will accept predictions.

## Run boundary

No checkpoint or tokenizer was loaded, no responses were generated, no optimizer was created, no training was performed, the official P3-02 scorer and artifacts were not modified, and the final project holdout was not accessed. The analyzer requires a response envelope that records separate evaluation authorization and matching execution provenance. Its inputs and outputs are distinct from the official P3-02 artifacts.

After separate authorization and a complete 72-response run, the expected envelope contains:

- `schemaVersion`: integer `1`.
- `manifestSha256`: the frozen manifest hash above.
- `evaluationAuthorization`: `{ "status": "separately-authorized", "evidenceReference": "..." }`.
- `execution`: the checkpoint step/hash and `checkpointHashVerified: true`; tokenizer and P3-02 bundle hashes; policy, seed, temperature, token cap; `checkpointReselectionAllowed: false`; `finalProjectHoldoutOpened: false`; the prompt-template hash; and inference `codeRevision`.
- `execution` must also record `trainingPerformedDuringDiagnostic: false`, `optimizerUpdatesDuringDiagnostic: 0`, and `officialP302ScorerModified: false`.
- `responses`: exactly 72 objects of `{ "taskId": "...", "actual": "complete decoded file text" }`.

The analyzer invocation after those gates are met is:

```powershell
python training/phase3/diagnostics/p3-03-source-preservation-v1/analyze_responses.py `
  --responses path\to\authorized-p3-03-responses.json `
  --report training/phase3/diagnostics/p3-03-source-preservation-v1/P3-03-SOURCE-PRESERVATION-DIAGNOSTIC.md `
  --audit training/phase3/diagnostics/p3-03-source-preservation-v1/P3-03-SOURCE-PRESERVATION-DIAGNOSTIC-AUDIT.json
```

It refuses a modified manifest, mismatched checkpoint/configuration metadata, incomplete or duplicate tasks, and output paths that already exist. A full diagnostic pass would support only a proposal for further independent evaluation; it cannot change the failed P3-02 gate or authorize training.

## Workspace provenance

The manifest records preparation against local commit `94ef152d5ffbdac370110a7c1b3e4f397d03f701`. The checkout was already 14 commits behind `origin/master` and had uncommitted P3-02 documentation changes and `training/.test-tmp/`. Those owner changes were preserved; no synchronization was attempted.
