# P3-02 — Exact Small Replacements

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
