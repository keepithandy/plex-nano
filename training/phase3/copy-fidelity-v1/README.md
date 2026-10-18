# Phase 3 — Copy-fidelity recovery curriculum v1

**Current research closeout (2026-10-17):** one authorized 100-step run completed and independently checksum-verified; loss fell **7.1167 → 2.7984**, but frozen generated exact-file evaluation **FAILED all gates (0/24 validation copies, 0/72 edits, 0/6 developer copies, 0/36 developer edits; 138/138 reached EOS/full prompt visibility)**. The historical preparation/authorization sections below describe earlier phases and must not be interpreted as new permission. Preserve the original artifacts, no further training and no final holdout. [Completed research issue #181](https://github.com/keepithandy/plex-nano/issues/181) → [P3-SB source-binding recovery roadmap](../../../docs/PHASE-3-SOURCE-BINDING-RECOVERY-ROADMAP.md) / [issue #189](https://github.com/keepithandy/plex-nano/issues/189).

**New read-only diagnostic:** `diagnose_saved_outputs.py` checks the owner's existing `evaluation-result.json` against the frozen candidate/control manifests without importing PyTorch, running inference, writing files, or changing the checkpoint. It emits the source report SHA and explicitly labels namespace/literal occurrence as heuristics, not exact copied-file success.

**Source-binding technical review:** [Recorded SB-00 evidence and SB-01–05 preparation](../../../docs/PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md). The existing report is now measured: all 138 omit their supplied namespace and contain training-name text. The unchanged 48-case fixture passes frozen-tokenizer audit; SB-02–04 designs and SB-05 synthetic scorer are ready for review. No new model execution is authorized.

**Status: deterministic candidate frozen; neither training nor inference is authorized by this preparation.** This is replacement recovery, separate from formal roadmap P3-03 Insert and Delete.

## Evidence and diagnosis

The owner reported the October 17, 2026 fixed step-100 42-task evaluation: **0/42 exact full files, 0/6 exact unchanged copies, 0/18 minimal edits, 0/18 contextual edits, 0/12 complete three-request contrast groups**. All 42 responses terminated at EOS. The edited target values were **36/36 unscorable independently** because source boundaries were not retained. The full response envelope is on the owner's Windows PC, not GitHub.

All six inspected copies replaced unseen cobalt/spruce identifiers with prior-style cart/settings identifiers, sometimes producing malformed code. In the earlier P3-02 authored training split, cart occurs in 36/144 records and settings in 36/144. **Template fallback is an evidence-based hypothesis, not a proven internal mechanism.**

The P3-02 generator creates 216 authored edit examples (144 train, 36 validation, 36 development) from six repeating namespaces, with **zero exact-copy requests**. Its answer_only_target_mask supervises complete edited answers and EOS while masking the prompt. This is valid causal language-model training, but not enough to establish source-copy generalization; earlier decreased validation loss did not pass P3-02's 0/36 exact-edit development gate.

The [six-case copy-output audit](COPY-FAILURE-AUDIT.md) preserves the owner-reported output patterns, syntax anomalies, and comparisons to the original P3-02 training records without claiming independent access to the local response envelope.

## Frozen candidate and data split

Manifest: [copy-fidelity-candidate-v1.json](copy-fidelity-candidate-v1.json). Generator: [prepare_copy_fidelity.py](prepare_copy_fidelity.py).

**SHA-256, canonical UTF-8 LF:** 9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a

| Example condition | Train | Validation |
|---|---:|---:|
| Return supplied file unchanged | 72 | 24 |
| Replace exactly one source literal | 216 | 72 |
| **Total** | **288** | **96** |

There are 96 train + 32 validation examples per language (HTML, CSS, JavaScript). Each source group contains one copy request and three different literal-edit requests on the *same source file*, retained within a single split (96 groups). Half the examples use short minimal code, half have unrelated context; two source shapes per language vary selector/identifier syntax. Complete-file outputs and EOS are the intended training target, not short diff fragments.

Train-only namespaces: **cedar, lunar, prism, amber, zephyr, meadow**. Validation-only namespaces: **granite, saffron**. Previously evaluated controls use **cobalt, spruce**, and remain excluded from train and validation. No complete-file inputs or outputs overlap those in the prior P3-02 authored candidate, frozen 72-task diagnostic or frozen 42-task control. Train and validation also have **zero group, prompt, full-input and full-output overlap**; train/validation share broad natural-language request patterns intentionally, so this is an *identifier-transfer* experiment, not a held-out request-paraphrase test.

All candidate text is ASCII. Maximum prompt length is **313 bytes**, maximum answer is **127 bytes**. Even a conservative byte-BPE upper bound gives 313 prompt + 192 generation + 1 step = **506/512**. Actual tokenizer SHA, encode/decode round-trip and packed-token checks still require an independent local review before model work. Unit tests do not exercise the model.

Verify immutable data from the repository root using these PowerShell commands:

    python training/phase3/copy-fidelity-v1/prepare_copy_fidelity.py --verify
    uv run --project training python -m pytest training/tests/test_p3_copy_fidelity_curriculum.py -q

The builder refuses to overwrite an existing file.

**Optional local tokenizer review (no generation, no training):** the owner can run the following in the same Windows working environment that passed the earlier frozen preflight. It verifies true encode/decode roundtrips, exact answer-and-EOS next-token masks, complete-record fit and prompt/output ceilings against the copied original 16K tokenizer; it never loads a checkpoint or creates an optimizer.

    uv run --project training python training/phase3/copy-fidelity-v1/review_local_tokenizer.py

Its output should say **copy-fidelity-local-tokenizer-review-passed-no-model-run**. The local output has not been observed by this PR; passing CI's deliberately worst-case one-byte-token mock is not the same as verifying the owner's actual frozen tokenizer.

## Proposed measurements and decision gates

**Primary validation copy requirement:** 24/24 complete byte-exact files (8/8 each language), identifiers unchanged, EOS present.

**Validation edit requirement:** at least 60/72 complete exact edits and at least 20/24 per language. Unrelated bytes must remain exact on every credited edit; requested value must be called **unscorable** when its source boundaries were altered.

**Contrast sensitivity:** at least 18/24 validation three-request edit groups must all be correct. Record syntax-check status independently; simple structural validation of these restricted HTML/CSS/JavaScript shapes is not the same as general language parser verification.

**Historical controlled development comparison:** the old 42 tasks remain unchanged, with baseline 0/6 copy and 0/42 exact. For proposed advancement, require 6/6 copy and >=30/36 edits, all supplied names preserved. These controls were seen by the curriculum designer and are **not a pristine final held-out test**. Require genuinely fresh, independently reserved confirmation before claiming broadly improved model capability; leave the final project holdout closed.

Record exact files, EOS, full-source preservation, per-language and per-condition counts, complete edit contrasts, syntactic completeness, token loss and generated outputs. A lower training or validation loss alone is insufficient.

## Proposed one-run training ceiling — NOT AUTHORIZED

| Property | Fixed proposal |
|---|---|
| Initial model weights | P3-02 step-100; SHA-256 020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d |
| Model/tokenizer | 27,566,080 parameters, 16,384 byte-BPE; frozen SHA 2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697 |
| Context / output ceiling | 512 input-context tokens / max 192 newly generated |
| Loss | Supervise complete answer + EOS, mask prompt and padding, as P3-02 |
| Optimizer | Fresh AdamW; do not silently reuse previous optimizer state |
| LR / seed | 0.0001 / 1337 (review before authorizing) |
| Batch / accumulation | 1 / 16 |
| Max optimizer updates / wall time | 100 / 600 seconds |
| Validation steps | 0, 25, 50, 75, 100 |
| Proposed fixed endpoint | Step 100 only if the bounded run completes, no post-hoc checkpoint selection |
| Development inference | Greedy, temperature 0, seed 1337, <=192 output tokens |
| Auto resume / retries | Forbidden |
| Training authorized / performed | **NO / NO** |
| Evaluation authorized / performed for this candidate | **NO / NO** |
| Project final holdout opened | **NO** |

**Blocking prerequisites:** execute the new read-only frozen tokenizer and whole-record review locally; validate next-token target mask and train sampler/weighting; verify actual original checkpoint identity and storage budget; review independent validation and inference contract; implement safe stage and training commands; obtain fresh explicit owner approval for a bounded optimizer run. This PR deliberately adds **no trainer or optimizer entry point** and does not reuse P3-02 training authorization.


## Packed bundle and zero-update stage gates (preparation only)

The training data can now be packed and re-inspected without **any model training**. The bundle module uses the owner-verified frozen tokenizer under `$HOME\PlexArtifacts\p3-02-training-bundle` and writes only a new directory under `$HOME\PlexArtifacts\file-edits\p3-copy-fidelity-v1-bundle`. Files are partitioned strictly into `train` and `validation`, each with JSONL complete examples, little-endian uint16 token IDs, uint8 next-token masks, and a byte-indexed JSON record index. There is no development/final-holdout data in the bundle. It verifies exact source-dataset SHA, tokenizer identity, full encoding roundtrip, **7,300/2,420** owner-observed supervised targets, actual record order, all persisted file bytes and SHA-256, and deterministic **full-coverage 1,600-draw sampler proposal**. The sampler is **not executed by a trainer**; training code must independently implement and verify the exact selected schedule before authorization.

All commands below run at the repository root on the owner's Windows machine. The default pack command is read-only:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/pack_bundle.py
```

Only after the read-only preflight passes, create the packed artifact **once** and verify it:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/pack_bundle.py --prepare
uv run --project training python training/phase3/copy-fidelity-v1/pack_bundle.py --verify
```

`--prepare` refuses an existing destination and aliases of its protected inputs, checks the 200 GiB storage cap, stages to a fresh temporary directory, verifies reconstruction before publication and cleans temporary files on failure. Do not remove, rewrite, or regenerate existing files to retry a failed run; inspect the failure first.

**Checkpoint provenance distinction:** the original P3-02 checkpoint's `tokenizerRecord.bundleManifestSha256` identifies its inherited **P2-48 tokenizer bundle** (`71fec0882692f5eb1b47b72f4a9f539e53e77882d003649beacd5f341b5a4ea7`), while `datasetRecord.bundleManifestSha256` identifies the **P3-02 training dataset** (`0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354`). They are deliberately different; both must remain pinned. The first staging implementation incorrectly expected the P3-02 data hash in the tokenizer field and was corrected after the owner's local bundle successfully packed but source staging refused. This correction does not modify original checkpoint bytes or training permission.

The zero-stage tool verifies the original **330,899,611-byte** P3-02 step-100 checkpoint at `$HOME\PlexArtifacts\file-edits\p3-02-first-run\checkpoints\step-0100.pt` against SHA-256 `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`, frozen trainer provenance and P3-02 tokenizer/dataset identities. It also inspects the newly prepared bundle. The default stage command is **read-only**:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/zero_stage.py
```

Only after source and bundle inspection have passed, an explicit **zero-update stage** (not training) can be created once:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/zero_stage.py --authorize-zero-stage
uv run --project training python training/phase3/copy-fidelity-v1/zero_stage.py --verify-stage
```

The stage creates a new `$HOME\PlexArtifacts\file-edits\p3-copy-fidelity-v1-stage0\stage-checkpoint.pt`. It initializes a **fresh, empty** AdamW optimizer for serialization only: **never calls optimizer.step, backward, training or evaluation**. The new checkpoint is at step zero with zero tokens processed, no optimizer moments, no scheduler state, and explicit non-authorization for training. The verifier reloads both files on CPU and compares **every tensor weight** to the original checkpoint, refusing any mismatch. A stage failure removes only the newly created stage directory. The original step-100 weights, older diagnostic artifacts and final holdout remain untouched.

These commands are for the owner's private PC because GitHub CI does not have the original 331 MB checkpoint/tokenizer. CI runs **mocked storage and tokenizer fixtures** to verify the control logic and refusal cases; it does not assert that real local staging already occurred.

Even after a successful bundle and zero-stage, **no additional training is authorized**. The original proposal for at most 100 optimizer updates / 600 seconds requires a separately approved training implementation, objective/sampler audit, fresh authorization contract and provenance. This work does not authorize another generation run or a final-project holdout.

## First-run runner and step-zero validation baseline — implementation only

The next bounded experiment has a [first-run implementation](first_run.py) and a [**unapproved** first-run contract](../../pretraining/p3-copy-fidelity-first-run-authorization-draft.json). **No training occurred in this PR.** The historical P3-02 0/36 development gate and new original 42-task 0/42 report remain unchanged.

This entry point has three distinct behaviors:

1. **Default, read-only source preflight.** Independently verifies the original P3-02 checkpoint, the SHA-pinned new step-zero stage `3524620d830d363585d501285a9cfdea09d1c2448be08239e2516d78291b3ca9`, the existing frozen bundle `8eceed468fdcb7202484eba82c8ebb8159a0d9e05787538f2c81f06f62719e2d`, exactly 288 train and 96 validation records, masks, and the full 1,600-draw deterministic sampler. The draw-index SHA-256 must equal `6d21424636cdb6b5912dacd1564f51b63cc3f9897ec13cb5235e99c17930e9f4`, cover all 288 training records, and have exactly **40,477** answer/EOS target positions. No weights are updated.
2. **`--baseline`, read-only teacher-forced validation.** After full preflight, loads the fixed step-zero stage once more, performs a complete 96-example validation pass against precisely **2,420 supervised answer+EOS targets**, and returns mean cross-entropy and per-language cross-entropy. No optimizer, backward pass, output file, development evaluation, generation, or final holdout. Results are model-specific and **cannot be calculated on GitHub** because the private 331 MB checkpoint is on the owner's machine.
3. **Future `--authorize-first-run` path is deliberately SEALED.** The implementation contains a max-100-update/600-second CUDA run with 16-example gradient accumulation, the previously frozen 1,600-record shuffled epoch sampler, teacher-forced validation steps 25/50/75/100, fixed checkpoints and non-overwriting artifact destination, but it **refuses before touching CUDA, data, or model files** because both reviewed SHA constants `APPROVED_CONTRACT_SHA256` and `APPROVED_BASELINE_SHA256` are `None`. CLI flags, draft JSON edits, or the old P3-02 training authorization cannot unlock this experiment. A separate explicit owner authorization and subsequent reviewed code change must pin the exact new approval contract and actual measured step-zero baseline SHA.

Run the following commands in PowerShell from the repository root, **after pulling the merge**:

```powershell
# First review the input identities, unpacking and deterministic sampler, read-only.
uv run --project training python training/phase3/copy-fidelity-v1/first_run.py

# Independently measure the full step-zero validation loss using CUDA.
# This does NOT generate edited files or train. CPU is available via --device cpu.
uv run --project training python training/phase3/copy-fidelity-v1/first_run.py --baseline --device cuda
```

The owner supplied a successful CUDA step-zero baseline and its exact results are frozen in [the owner-reported baseline evidence](../../pretraining/p3-copy-fidelity-step-zero-baseline-evidence.json) (SHA-256 `5183726a621fb82539137b009c1514b5ed59d750e3ef4ad4bcd71453e644ff01`). Mean validation answer/EOS loss is **7.116717401614859** over **2,420** supervised targets and **96** rows (HTML **6.3204158690510965** across 980 targets; CSS **7.546227005124092** across 800; JavaScript **7.799167118966579** across 640). Those are owner-reported teacher-forced numbers, not independent GitHub checkpoint inference. The draft contract now references the evidence SHA but retains `modelTrainingAuthorized: false`; both trusted authorization SHA constants in the runner remain `None`. This fixes a measurement baseline only: model copying and editing are still unproven. **Do not run `--authorize-first-run`: it is intentionally blocked until an approved research contract and baseline are frozen.**

A successful low step-zero loss would only establish the original checkpoint's cross-entropy against the *new* validation corpus, not complete-file copying. No post-training capability claim can be made without the already frozen actual output-generation gates (24/24 exact validation copies, ≥60/72 exact validation edits, ≥18/24 complete edit groups, unchanged 42-task development controls and fresh independent confirmation).

**Research boundary:** 27,566,080 parameters, original 16,384-tokenizer and 512 context remain fixed. No checkpoint choice, optimizer updates, changed held-out names, generated test responses or final-project holdout access is made in this preparation.

## Owner-approved single bounded training run · 2026-10-17

**Authorization:** The owner explicitly approved one P3 copy-fidelity training run on October 17, 2026, limited to **100 optimizer updates or 600 seconds, whichever happens first**, from the unchanged verified stage. Training is still **not performed** merely by merging this approval. It begins only after the owner explicitly runs the CLI command on the Windows PC.

- [Approved one-run contract](../../pretraining/p3-copy-fidelity-first-run-authorization-draft.json) SHA-256: `a7432a3bfa7942b6086bc8e64fa3b24290351acfd9fb700aac2af613a8c0911b` (filename retains `-draft` for stable versioned compatibility; contents now show `status: owner-approved-one-run` and `modelTrainingAuthorized: true`).
- Independently frozen **owner-reported CUDA** baseline SHA-256: `5183726a621fb82539137b009c1514b5ed59d750e3ef4ad4bcd71453e644ff01`; step-zero mean teacher-forced loss **7.116717401614859**, 96 examples, 2,420 targets.
- Starting zero-update checkpoint SHA-256: `3524620d830d363585d501285a9cfdea09d1c2448be08239e2516d78291b3ca9`; source step-100 SHA-256: `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`.
- Frozen bundle SHA-256: `8eceed468fdcb7202484eba82c8ebb8159a0d9e05787538f2c81f06f62719e2d`. The exact 1,600-record sampler has SHA-256 `6d21424636cdb6b5912dacd1564f51b63cc3f9897ec13cb5235e99c17930e9f4`.
- The code verifies both compiled trusted hashes and the original stage, dataset, sampler, and baseline; **uses CUDA for the owner-approved baseline comparison** (to avoid comparing CPU and CUDA numeric drift); requires an unused dedicated output folder; does not resume, reselect checkpoints, or open the final holdout. Checkpoints only at fixed 25/50/75/100 steps. It trains only the 288 train rows; the 96 validation rows are for teacher-forced loss, not gradients.
- The training-clock cap applies to the bounded update loop. Source-verification and baseline-check work occurs beforehand. If the bounded loop expires mid-update, remaining gradients are discarded; no optimizer step occurs for that incomplete update. An interrupted or partial run is recorded as incomplete and **is not automatically resumed**.

From the repository root after pulling the authorized merge, verify the unchanged inputs first:

```powershell
git pull origin master
uv run --project training python training/phase3/copy-fidelity-v1/first_run.py
```

Then explicitly initiate the **one** authorized run using:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/first_run.py --authorize-first-run
```

The command writes only into the new `$HOME\PlexArtifacts\file-edits\p3-copy-fidelity-v1-first-run` folder (metrics, result JSON and fixed checkpoint files); it **refuses to overwrite an existing folder**. Preserve all historical checkpoints and the frozen baseline. If there is an error, do **not** delete the output and rerun: inspect the evidence before deciding whether any new authorization is necessary. The nonfatal missing-NumPy warning observed in earlier PyTorch verification has not prevented past runs, but does not guarantee this GPU run will pass.

**After execution** report `result.json` fields `completedMaximumSteps`, `optimizerUpdates`, `answerAndEosTargetsProcessed`, `validation`, `fixedEndpointSha256`, `stageCheckpointSha256`, `bundleManifestSha256`, `finalHoldoutOpened` and checkpoint hashes. Do not report successful copying/editing solely from lower validation loss. Evaluation of exact-file generations remains a **separate, frozen, post-run step**.

The older paragraphs in this document describing the runner as “SEALED” refer to the pre-authorization design and historical point in time. This later authorization supersedes the runner's disabled state, not its safety requirements or the earlier failed P3-02 evaluation.

## Fixed step-100 generation evaluation — next capability gate

Following the owner's completed one-run step-100 training (100 updates; fixed checkpoint SHA-256 `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`), the only meaningful next improvement check is whether **generated complete files** are correct. Validation loss dropped from **7.1167174 to 2.7984009**, but a previous P3-02 experiment also improved loss while failing all generated exact-file edits. We therefore do **not** infer that copy fidelity has improved.

The [read-only generation evaluator](evaluate_step100.py) reuses only the frozen **96 never-trained validation** cases from the `copy-fidelity-candidate-v1.json` SHA and **42 already-inspected developer controls** from the original copy/edit manifest SHA `17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a`. Both input sets are checked byte-for-byte; original control files and their historical checkpoint references remain unchanged. The **final project holdout stays closed**.

It loads only the **fixed, hash-verified step-100 checkpoint**, validates checkpoint training provenance and saved report, uses greedy temperature 0 / seed 1337 / maximum **192** output tokens with full prompt visibility, and reports exact full-file agreement, EOS, source preservation, extractable requested literals, per-language edit/copy results and contrast-group accuracy. Missing source boundaries produce **unknown**, not a fabricated requested-value score. Full correctness requires EOS, successful decode, and full prompt visibility. No optimizer, gradient, training, checkpoint selection, modification of inputs, or automatic retry.

**Prespecified validation success targets:** 24/24 unchanged copies, ≥60/72 exact edits, ≥20/24 exact edits in **each language**, and ≥18/24 fully correct four-case groups. **Historical developer regression:** 6/6 exact copies and ≥30/36 exact edits on the original 42 controls. Developer controls have been inspected before and are not pristine final holdout. In both cases exact copying means matching every output character, including whitespace.

Run from the repository root on the owner's PC, in order. The first command is **read-only, metadata-only preflight**; the second performs **model generation** but no training, and saves the only new evaluation report under `$HOME\PlexArtifacts\file-edits\p3-copy-fidelity-v1-step100-evaluation\evaluation-result.json`.

```powershell
git pull origin master
uv run --project training python training/phase3/copy-fidelity-v1/evaluate_step100.py
uv run --project training python training/phase3/copy-fidelity-v1/evaluate_step100.py --evaluate --device cuda
```

**Do not rerun `--evaluate` if a report directory already exists**, and do not delete it to retry after a failure. Save the complete report and provide its score summary for research review. Checkpoint `step-0075.pt` had lower teacher-forced loss but **must not** be used for a retrospective alternative endpoint. A failed generation result is a legitimate research outcome, not grounds for automatic training or checkpoint reselection.

CI exercises scorer semantics and the synthetic evaluation path only: real GPU checkpoint inference requires the owner's private Windows files. This is an evaluation gate, not a new authorized optimizer run.

## Research limitations

The 384 examples are a small hypothesis-driven synthetic intervention, not evidence that 27M parameters suffice for exact copying. They intentionally isolate unseen identifiers and literal/source preservation but do not evaluate paraphrase robustness. Historical numeric results and manifests are immutable. No final holdout access, checkpoint reselection, model-size change or autonomous repository-agent pivot.

Related: [original 42-task preparation](../diagnostics/p3-copy-edit-controls-v1/PREPARATION.md), [decision issue #177](https://github.com/keepithandy/plex-nano/issues/177), [recovery issue #181](https://github.com/keepithandy/plex-nano/issues/181).

**Repository implementation verification:** The Linux and Windows CI paths cover pack/inspect integrity, split-exclusion, exact next-token masks, deterministic sampling, storage rollback, and step-zero authorization/weight preservation using synthetic fixtures. Passing CI proves the safety logic is exercised, not that the owner's local bundle or checkpoint stage was created.
