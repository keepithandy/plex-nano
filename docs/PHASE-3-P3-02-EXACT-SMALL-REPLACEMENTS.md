# P3-02 — Exact Small Replacements

## October 13 owner approval — one bounded P3-02 first run

The owner explicitly authorized the first P3-02 training run on **2026-10-13**, limited to **100 optimizer updates or 600 seconds**, whichever comes first. The committed `training/pretraining/p3-02-first-run-contract.json` now records `status: owner-approved-first-run`, `modelTrainingAuthorized: true`, approver `keepithandy`, and approval date `2026-10-13`. The exact baseline, stage checkpoint, bundle, tokenizer, sampler, loss objective, hyperparameters, and evaluation boundaries remain unchanged. The validator checks the exact one-run authorization scope and rejects tampering with any training bound or provenance field.

This **authorizes** the run, but does not imply that it was **executed**. No P3-02 optimizer updates have yet been observed or reported; the separately required `--authorize-first-run` CLI flag must be supplied by the owner. The read-only training preflight should be rerun after pulling the approval commit.

Run from the repository root on the approved owner's NVIDIA/CUDA machine using the already verified external artifact root (not the original OneDrive artifacts root):

```powershell
git switch master
git pull --ff-only origin master
$artifactRoot = Join-Path $HOME "PlexArtifacts"
$bundleDir = Join-Path $artifactRoot "p3-02-training-bundle"
$stageCheckpoint = Join-Path $artifactRoot "p3-02-stage0/stage-checkpoint.pt"
$outputDir = Join-Path $artifactRoot "file-edits/p3-02-first-run"

uv run --project training python -m plex_training.cli file-edit-replacements-training-preflight --bundle-dir "$bundleDir" --stage-checkpoint "$stageCheckpoint"
uv run --project training python -m plex_training.cli file-edit-replacements-run --authorize-first-run --bundle-dir "$bundleDir" --stage-checkpoint "$stageCheckpoint" --artifact-root "$artifactRoot" --output-dir "$outputDir"
```

Do **not** delete or reuse an existing output directory. The run is one-shot: no overwrite, resume, automatic continuation, checkpoint reselection, or final-holdout access. The result, if completed, is recorded under `$outputDir/result.json` and metrics under `$outputDir/metrics.jsonl`; do not claim successful model learning until comparing the measured losses and subsequent exact-edit evaluations. Earlier preparation-only notes below describe historical states, not the current training authorization.

**Status: candidate preparation and frozen-tokenizer review passed. Model training and the learning milestone are not complete.**

This experiment will measure whether Plex Nano can follow a request to change one small region of a supplied HTML, CSS, or JavaScript file and return the whole edited file while preserving unrelated text. The owner authorized continuing preparation on October 12, 2026. No research checkpoint was loaded, optimizer created, checkpoint staged, or model updated during this preparation.

## Task and candidate

Use the unchanged P3-01 prompt and raw edited-file target, followed by EOS:

```text
Edit the supplied file to satisfy the request.
Language: html
Request: For the button with id cart-save, change its text to Confirm.
File:
<section class="cart">
  <p>Keep this note.</p>
  <button id="cart-save">Cancel</button>
</section>
Edited file:
```

The correct output changes only `Cancel` to `Confirm`. Two more requests for this exact file ask for `Save` and `Continue`, with correspondingly different correct outputs. These contrast variants test request binding rather than copying a single target for a familiar file.

The candidate has **216 project-authored records in 72 three-request groups**:

| Split | Records | Groups | Records per language |
|---|---:|---:|---:|
| Train | 144 | 48 | 48 |
| Validation | 36 | 12 | 12 |
| Development | 36 | 12 | 12 |

Each language has four edit families:

- HTML: button text, link destination, input placeholder, and navigation label.
- CSS: gap, padding, color, and border radius.
- JavaScript: retry constant, message string, query selector, and default argument.

All three requests for one supplied file stay in the same split. Training uses four identifiers/file contexts per family; validation and development each use one different context. Literal replacement values and authored templates are deliberately familiar across splits. This is a narrow first learning pilot, not a claim of transfer to unseen syntax, larger files, repeated ambiguous literals, variable renaming, or multiple edits. Those tasks remain later milestones.

Exact contrast-group, request, input-file, expected-file, and full-prompt overlap is **zero across every split pair**. Exact request/input/target overlap with the 18 P3-01 contract fixtures is also zero. The 36 development examples are a new authored development gate, not the closed final project holdout. Their requests and targets are visible for review and are never training records.

## Frozen preparation evidence

- Candidate: `training/phase3/drafts/p3-02-exact-replacements-v1.jsonl`.
- Candidate SHA-256: `98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403`.
- Candidate bytes: **128,214**, UTF-8 with canonical LF JSONL.
- Preparation contract: `training/pretraining/p3-02-exact-replacements-preparation-contract.json`.
- Review result: `training/pretraining/p3-02-exact-replacements-review-result.json`.
- Frozen tokenizer SHA-256: `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`.
- P2-48 derived tokenizer-bundle manifest SHA-256: `71fec0882692f5eb1b47b72f4a9f539e53e77882d003649beacd5f341b5a4ea7`.

The read-only tokenizer review passed exact prompt and target roundtrips. Encoding is `encode(prompt) + encode(target) + [EOS=3]`, with no automatic BOS or padding and no control/unknown tokens inside ordinary text. Record lengths are **87–138 tokens including EOS**. The largest prompt is **90 tokens**, and the largest target including EOS is **48 tokens**.

The proposed 192-token decoding cap plus every prompt fits wholly inside the 512-token context; no prompt eviction is needed within that cap. Training, validation, and development record-token totals are **16,644**, **4,074**, and **4,074**. These are prepared token counts, not optimizer target positions or observed learning results.

All **72 JavaScript targets** passed Node syntax checks without executing their code. HTML/CSS targets remain constrained authored examples; no general HTML/CSS parser or nonexact-output syntax claim is introduced. Scoring reports `known-valid-target` only for an exact authored target and `not-checked` otherwise.

## Proposed bounded experiment — not training authorization

The preparation contract records this proposal for later implementation and explicit approval:

- Keep the 27,566,080-parameter model, 512-token context, and 16,384-token vocabulary.
- Start from the unchanged fixed P2-48 step-100 endpoint, SHA `fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7`. Do not retrospectively select its better step-25 validation checkpoint.
- Use complete-record sampling and an answer/EOS-only objective. Supplied prompt tokens remain visible but receive zero loss weight; padding receives no loss.
- Proposal: seed 1337, micro-batch 1, gradient accumulation 16, learning rate 0.0001, at most 100 updates and 600 seconds.
- Measure validation at steps 0/25/50/75/100; retain the fixed step-100 endpoint for development evaluation. No checkpoint reselection or automatic continuation.
- Greedy development decoding, seed 1337, 192 new tokens maximum, and explicit full-prompt visibility required.

The fresh P3 stage, optimizer behavior, deterministic sampling totals, checkpoint/run ownership, read-only baseline evaluation, and signed training packet still need implementation and verification before this proposal can run. No automatic optimizer or run command is added by this preparation.

## Read-only answer-only supervision preflight

The next preparation increment adds `file-edit-replacements-supervision-preflight`. This command revalidates the frozen 216-record candidate, its draft metadata, and the committed frozen-tokenizer review, then previews **train/validation only** as `encode(prompt) + encode(expectedFile) + [EOS=3]`. It does not create a packed bundle, read a model checkpoint, initialize an optimizer, or run training.

For every complete record, the next-token loss mask sets all prompt targets to zero and weights the **first edited-file token through EOS** as one; padding remains zero. The preflight also computes reproducible little-endian packed-token SHA-256 previews, per-split/per-language supervised target counts, and a seed-1337, 1,600-example uniform-with-replacement sampler replay, without sampling the 36 development examples.

Run from the repository root after restoring the locally held P2-48 tokenizer bundle:

```powershell
uv run --project training python -m plex_training.cli file-edit-replacements-supervision-preflight
```

The command fails closed if the locally recomputed token lengths differ from the committed 87–138-token review or the exact frozen tokenizer identity. Passing this preflight will provide *read-only accounting evidence*, not a training authorization or empirical model-editing result. Reviewed bundle writing and pinned P2-48 checkpoint source inspection remain separate follow-on tasks.

## P3-02 train/validation bundle and checkpoint-source preflight

Implemented bundle commands prepare **only** the 144 train and 36 validation records, with no 36-record development split in the output. The bundle carries a JSONL text stream, packed uint16 little-endian tokens, a separate one-byte-per-next-token mask, and an index whose prompt/answer boundaries are reconstructed from the authored rows. Each target mask is zero for the supplied request/file prompt, and one for the edited-file answer and EOS. The parser verifies every byte against regenerated frozen records and manifests, rejecting mutation of tokens, JSONL, indices, loss weights, and split metadata. Bundle writing uses a fresh temporary artifact directory, the 200 GiB allocation, rollback on failure, and atomic publication without overwriting existing outputs.

The source preflight requires the **unchanged P2-48 step-100 checkpoint** SHA-256 `fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7` and its original P2-48 result record, then safely loads it on CPU in read-only mode to verify 27,566,080 parameters, step 100, 429,375 recorded target positions, scratch provenance, P2-48 stage/training identities, and frozen tokenizer compatibility. It **does not** create an optimizer or checkpoint.

From the repository root, using the owner's **local frozen tokenizer and checkpoint artifacts**:

```powershell
uv run --project training python -m plex_training.cli file-edit-replacements-supervision-preflight
uv run --project training python -m plex_training.cli file-edit-replacements-bundle-prepare
uv run --project training python -m plex_training.cli file-edit-replacements-bundle-verify
uv run --project training python -m plex_training.cli file-edit-replacements-source-preflight
```

These commands require local P2-48 artifacts excluded from GitHub. CI verifies portable synthetic data and failure paths but **cannot attest** that the real frozen tokenizer, bundle, or checkpoint-source preflight has passed on the owner's machine. Until those exact local preflights pass, stage creation, baseline loss measurement, and training authorization remain pending. No trainer or owner-approved P3 training contract is provided here.

## October 13 verified-local evidence and guarded stage-zero baseline

The owner reported successful **Windows local** execution of the frozen supervision review, packed-bundle integrity check, and exact P2-48 source-checkpoint preflight. This evidence is recorded (not reproduced in cloud CI) in `training/pretraining/p3-02-local-verification-evidence.json`:

- Verified train/validation bundle manifest SHA-256: `0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354`.
- Source P2-48 step-100 checkpoint SHA-256: `fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7`.
- Frozen tokenizer SHA-256: `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`.
- Train: 144 records, 16,644 record tokens, 5,424 answer/EOS targets. Validation: 36 records, 4,074 record tokens, 1,320 answer/EOS targets. Proposed seed-1337 sampler: 1,600 records, 60,445 supervised positions. No model updates were performed.

The implementation adds a **separately opted-in zero-stage-only serializer**, a read-only stage verifier, and a **full 36-record validation-only answer/EOS baseline evaluator**. Staging verifies the frozen source first, then preserves every model tensor exactly, records step zero and zero processed tokens, and serializes a fresh **empty** optimizer state only for checkpoint compatibility. Staging cannot run without the explicit `--authorize-zero-stage` CLI flag. It refuses existing output and removes partial stage outputs on failure. The stage verifier reopens the stage, confirms strict tensor equality against the original checkpoint and an empty optimizer state, and writes nothing.

The baseline evaluator reads the full frozen validation split only, weights answer tokens and EOS at one and prompt tokens at zero, restricts output loss to the frozen 16,384-token vocabulary, and reports the target-weighted mean loss and per-language/per-record results. It neither writes metrics nor trains, accesses development examples, or selects alternate checkpoints. Baseline outputs must be reviewed before separately authorizing a bounded run; **no P3-02 model-training command or approval contract is provided**.

Since the old OneDrive artifacts tree has an inaccessible unrelated historical dataset, the owner placed the verified P3-02 bundle under a separate artifact root. On the feature branch (or after merge), from the repository root:

```powershell
$artifactRoot = Join-Path $HOME "PlexArtifacts"
$bundleDir = Join-Path $artifactRoot "p3-02-training-bundle"
$stageDir = Join-Path $artifactRoot "p3-02-stage0"

uv run --project training python -m plex_training.cli file-edit-replacements-stage --authorize-zero-stage --bundle-dir "$bundleDir" --artifact-root "$artifactRoot" --output-dir "$stageDir"
uv run --project training python -m plex_training.cli file-edit-replacements-stage-preflight --bundle-dir "$bundleDir" --stage-checkpoint (Join-Path $stageDir "stage-checkpoint.pt")
uv run --project training python -m plex_training.cli file-edit-replacements-baseline-preflight --bundle-dir "$bundleDir" --stage-checkpoint (Join-Path $stageDir "stage-checkpoint.pt") --device cpu
```

The original P2-48 source checkpoint must remain readable at its verified default location. Do not repeat staging into an existing output directory; verification and baseline operations are read-only. No local stage or baseline results are claimed until the owner actually runs the commands.

## October 13 measured stage-zero baseline and first-run authorization packet

The owner reported a successful full **36-record, 1,320 answer/EOS target** CPU validation baseline on the unchanged zero-stage model. The checkpoint SHA-256 is `50f18e463d6a51bac5a21c0101b13d4b33c7e0d3eb41904600f6fce8bc622246`. The frozen training bundle manifest remains `0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354`. The teacher-forced, supervised-target-weighted **mean validation cross-entropy is 10.566839536031088**, with CSS 10.045777009682096 (521 targets), HTML 10.963397183261074 (455), and JavaScript 10.831490253293238 (344). These are **losses, not exact-edit success rates**. We did not run or independently reproduce the owner's local baseline through GitHub Actions. The result is pinned in `training/pretraining/p3-02-step-zero-baseline-result.json`.

The **unapproved** proposed first-run packet is `training/pretraining/p3-02-first-run-contract.json` and starts with `status: owner-approval-required`, `modelTrainingAuthorized: false`, null approver/date, and no continuation. A bounded trainer implementation and read-only training preflight are now available for source review and portable safety tests. Neither the user-pasted baseline nor successful CI implies model-training authorization. The real trainer requires **both** a separately owner-approved exact packet and explicit CLI `--authorize-first-run`; a draft and any altered bounds fail before CUDA/model access.

The proposed immutable training settings are seed 1337, AdamW learning rate 0.0001, betas (0.9, 0.95), weight decay 0.1, epsilon 1e-8, gradient clipping 1, micro-batch 1, accumulation 16, **uniform complete-record sampling with replacement**, and answer/EOS-only per-record normalized loss. The exact train order must reproduce 1,600 selections over 100 steps and **60,445 supervised answer/EOS targets**. The runner requires CUDA and the frozen stage checksum; all 36 validation examples remain excluded from gradients. It validates at 0/25/50/75/100, saves checkpoints at 25/50/75/100, and enforces no more than 100 optimizer steps or 600 seconds from run start. It refuses an existing output, alternate checkpoint selection, automatic resume, and development/final-holdout access. Interrupted or time-capped partial runs cannot masquerade as the fixed 100-step endpoint; continuation would need another authorization.

From the repository root, after installing this change through a checked-out feature branch or a merge, with the existing external bundle and stage under `$HOME\PlexArtifacts`:

```powershell
$artifactRoot = Join-Path $HOME "PlexArtifacts"
$bundleDir = Join-Path $artifactRoot "p3-02-training-bundle"
$stageCheckpoint = Join-Path $artifactRoot "p3-02-stage0/stage-checkpoint.pt"
uv run --project training python -m plex_training.cli file-edit-replacements-training-preflight --bundle-dir "$bundleDir" --stage-checkpoint "$stageCheckpoint"
```

The training preflight is **read-only**. It revalidates the exact source/stage/bundle, recomputes the frozen CPU baseline within 0.02 cross-entropy of the recorded result, and replays the deterministic sampler. It reports the draft's authorization status and **runs zero updates**. Do **not** run `file-edit-replacements-run` until after explicit owner approval and review of the authorization packet; it is intentionally blocked. The true 100-step endpoint and exact-edited-file development results are not yet known.

## Predeclared development gate

Assess all 36 development requests at the fixed endpoint. Require all of:

- **30/36 exact requested edits** or better.
- **8/12 exact edits per language** or better.
- **34/36 preserving unrelated text** or better.
- **10/12 contrast groups** with all three requests answered exactly.
- The whole supplied prompt visible on every model evaluation, including EOS, with no missing/truncated evaluation records.

Normalize CRLF/CR to LF only; retain all other whitespace. Exact edited-file equality measures correctness. Preservation checks the reviewed prefix and suffix around the single replacement. Record baseline and endpoint scores, per-family errors, and group consistency; preserve failures without changing this gate. Preparation success alone does not pass it.

## Commands and verification

```text
uv run --project training python -m plex_training.cli file-edit-replacements-generate
uv run --project training python -m plex_training.cli file-edit-replacements-review
```

Defaults are absolute repository paths. Generation uses the existing staged candidate/review publication and path protections. Review requires the frozen tokenizer, rejects changed candidate/draft identities, and refuses report paths that alias protected inputs. Optional report creation is exclusive and never overwrites an existing report.

Local regression verification: **50 passed, 2 deselected** across the new preparation tests, existing P3 contract, CLI, and CI-inventory checks. The two deselections are local P3-01 frozen-tokenizer tests; P3-02 received its own real read-only frozen-tokenizer review. The new test module is assigned to the automatic portable suite; the inventory is now **64 modules = 43 automatic + 21 manual experiment modules**.

Next development step: implement reviewed bundle packing and read-only baseline/stage-source preflight, then produce a concrete bounded run packet for explicit training authorization. The final project holdout remains closed.
