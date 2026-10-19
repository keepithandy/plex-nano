# Plex-Nano SB-03 V2 — Historical training-source audit

**Evaluation-name review and narrowed claim:** the [separate six-set review](PHASE-3-R2-FROZEN-EVALUATION-NAMESPACE-REVIEW-2026-10-18.md) found five evaluation spellings in ancestor input text. No matches for `ulvar`, `nyxen`, `granite`, `saffron`, `spruce` means only absence under the stated rule in the listed history. Original proposals and evaluation fixtures are preserved; current V2 is preparation with mixed historical spelling exposure, not a globally unseen-name intervention. No training gate is waived.

**Executed October 18:** the [ten-source ancestor screen](PHASE-3-R2-ANCESTOR-NAME-COLLISIONS-2026-10-18.md) verified all file hashes and found **8/48 proposed train names in five ancestor corpora**: `bonus`, `hover`, `inser`, `layer`, `media`, `proxy`, `track`, `users`. All eight were confirmed in parsed training-text fields. The original V2 fails the source-exclusion gate and remains unchanged. The five earlier P3 checks remain a partial exclusion claim; no global novelty or model capability pass follows.

**Status:** Research preparation only. [Issue #189](https://github.com/keepithandy/plex-nano/issues/189) remains open. [The V2 tokenizer feasibility](PHASE-3-SB03-V2-TOKENIZER-FEASIBILITY.md) found **48/48 matched names, 384/384 matched signatures** under the frozen real tokenizer (owner-reported); the resulting **owner-local** unapproved proposal SHA-256 was independently checked by PowerShell as `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21`.

**This audit does not authorize model training, inference, checkpoint staging, bundle creation, or final project holdout access.** Existing model capability remains **0/138 exact complete files**. Our immediate objective is to establish whether the 48 proposed train-only source names collide with *any actual earlier training corpus* in Plex's checkpoint ancestry.

## 1. Why one more provenance audit matters

The successful V2 search screened **five SHA-frozen P3 repository datasets**, but the fixed 27,566,080-parameter checkpoint also has earlier Phase 2 training inputs and owner-local source artifacts. The [October 18 trace](PHASE-3-R2-TRAINING-ANCESTRY-TRACE-2026-10-18.md) establishes a fresh initialization followed by P2-29 through the recorded Phase 2/3 chain; earlier pre-reset Phase 1/2 weights and the P2-28 trained pilot are outside this weight ancestry. Names such as `users`, `media`, `proxy`, and `track` are plausible prior training text; being a single tokenizer vocabulary entry does **not** prove the model has never seen the spelling in training. The new test must measure actual historical training-source overlap instead of assuming any selected name is unseen.

An earlier V1 proposal also failed the real tokenizer parity gate: **240/384 unequal records**, with 40,439 versus 44,925 draw-weighted supervised answer/EOS positions. V1 and the original finished step-100 model checkpoints stay unchanged.

## 2. Implemented model-free audit

[Streaming SHA + collision auditor](../training/phase3/copy-fidelity-v1/audit_v2_ancestor_history.py)

The script:
- Verifies the **actual local 330,673-byte V2 proposal** against its independently reported SHA-256, checkpoint identity, frozen tokenizer identity, original data identity, 1,600-draw sampler identity, 288/96 split and 48 distinct selected names. It does not create or edit a V2 candidate.
- Accepts an **explicit JSON inventory** of already identified non-final-holdout training ancestor *text files*, each with its actual absolute path, exact source SHA-256 and declared role.
- Streams each source in **1 MiB chunks**, computing full SHA-256 and finding all candidate names as **case-insensitive ASCII substrings**, including matches that cross chunk boundaries. It does not load a large corpus into memory.
- Rejects wrong hashes, absent files, unsupported binary formats, duplicate paths, symbolic links, files/directories named *holdout*, and file sizes over 200 GiB. It never enumerates directories.
- Reports `collidingNames`, source hashes and per-file collision identities; **not** private source content. A matching hash validates **only the listed file's bytes**.
- Always returns `historicalTrainingLineageComplete=false`, `readyForTraining=false`, `trainingAuthorized=false`, `optimizerUpdates=0`, and `finalHoldoutOpened=false`. Even a zero-collision report does not prove the inventory is complete.

The report reveals selected-name collisions only in user-identified training sources; it is not an evaluation of Plex's ability to code.

## 3. Prepare an explicit inventory on Windows

**Reviewed recorded-input inventory available:** [ten exact training-text paths and SHA-256 values](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json), supported by the [checkpoint and bundle verification receipt](../training/pretraining/p3-sb03-v2-training-ancestry-trace.json) and subsequent [coverage review](PHASE-3-R2-TRAINING-INVENTORY-COVERAGE-REVIEW-2026-10-18.md). All ten listed texts and corresponding training-token files matched their recorded hashes on October 18; all 4,163 rows account for the full packed streams. Eight texts are repository-local Phase 2 artifacts; two are private owner-local Phase 3 artifacts. No known missing training text or unresolved recorded source identity was found. Collision screening is now recorded with eight confirmed names; external independent certification has not been performed. This inventory uses absolute paths for this Windows checkout; any relocation must preserve the recorded hashes.

Only include actual **non-final-holdout** text sources you can independently trace to the training run ancestry (e.g., source `.jsonl` files *actually used* to produce the frozen token bundles). Inspect documented Phase 1/2 dataset manifests, training contracts and the owner-local training bundle manifests first. Do **not** guess dataset paths or supply binary model weights or token-only binary arrays as source text. No sealed final benchmark or unseen final holdout belongs here.

Use the following PowerShell as a template, replacing the example `C:\\ABSOLUTE\\PATH\\TO\\ACTUAL\\TRAINING\\train.jsonl` with absolute paths identified from your training lineage. Add one entry per source file (do not use wildcards).

```powershell
$sourceFiles = @(
    'C:\ABSOLUTE\PATH\TO\ACTUAL\TRAINING\train.jsonl'
    # Add all further actual source TEXT files from the model's ancestry
)
$inventory = @{
    schemaVersion = 1
    scope = 'p3-sb03-v2-non-holdout-ancestor-training-text-inventory-v1'
    files = @(
        foreach ($source in $sourceFiles) {
            $file = Get-Item -LiteralPath $source -ErrorAction Stop
            if ($file.PSIsContainer) { throw "Expected a file: $source" }
            @{
                path = $file.FullName
                sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                role = 'ancestor-training-text'
            }
        }
    )
}
$destination = "$HOME\PlexArtifacts\file-edits\p3-sb03-v2-feasibility\ancestor-training-text-inventory.json"
$inventory | ConvertTo-Json -Depth 7 | Out-File -LiteralPath $destination -Encoding utf8 -NoClobber
```

**Important:** the path example is not a claim such a file exists. Replace it with actual files found in the model's source manifests. This inventory is not complete merely because the script is able to read it.

## 4. Run the read-only streamed review

From the repository root, after reviewing the inventory for completeness:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/audit_v2_ancestor_history.py --inventory "training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json"
```

The command prints a JSON report. **No model is loaded, no GPU is used, no optimizer updates happen, and neither the inventory nor the original training files are modified.** Do not report the full file contents; hashes, file count and collision summary are sufficient.

## 5. Acceptance gate and safe branches

| Outcome | Meaning | Next controlled action |
|---|---|---|
| Source SHA mismatch/missing file/holdout path | Provenance cannot be verified | Correct inventory; do not waive source integrity |
| Any 48-name collision | The new name may have appeared in actual ancestor text | Record collisions; design a **separate V2 revision** reselecting those names with tokenizer parity and updated SHA; preserve original V2 |
| Zero collisions in listed files | Only the **listed** source texts are clean | Cross-check manifest ancestry completeness; do not certify global novelty yet |
| All historical, non-holdout ancestor texts independently inventoried, byte verified and collision-free | Evidence for a full source-exclusion gate | Independent review and distinct bounded comparator/intervention authorization proposals; **still no automatic training** |

**Next model improvement experiment (once legitimately authorized):** run matched comparator and intervention from the same frozen starting checkpoint, using identical full-coverage 1,600 draws, identical tokenizer and exact token-count parity, followed by prespecified **generated complete-file** copy/edit benchmarks. Compare actual ability to preserve unfamiliar HTML/CSS/JavaScript identifiers, preserve unrelated code, and select the requested value, not just validation loss. Train neither arm on the sealed final project holdout or on previously inspected diagnostics.

## 6. Review discipline

The [original P3 source-binding roadmap](PHASE-3-SOURCE-BINDING-RECOVERY-ROADMAP.md) remains the active research sequence. Keep independent checkpoints, candidate hashes, optimizer authorization, model inference authorization and final holdout access as distinct gates. The readiness of code and tests does **not** establish improvement in the model's coding capability.
