# P3-SB03 — Approved intervention choice and matched namespace-diversity preparation

**Research decision:** Hypothesis **A — train-namespace diversity** is selected for technical development following owner approval. This document green-lights *model-free candidate preparation, frozen comparison and read-only parity checks*. It does **not** authorize a training run, checkpoint stage, model generation, new final holdout access, or a matched comparator run.

**Repository tracking:** [source-binding issue #189](https://github.com/keepithandy/plex-nano/issues/189) · [technical failure review](PHASE-3-SOURCE-BINDING-TECHNICAL-REVIEW.md) · [six-stage roadmap](PHASE-3-SOURCE-BINDING-RECOVERY-ROADMAP.md).

## 1. Why this intervention

The owner-reported frozen P3 step-100 model generated **0/138** exact full files. Subsequent model-free audit of the original 138 saved completions found **0/138** occurrences of the supplied source namespace and **138/138** occurrences of at least one prior training namespace. The 138 examples all reached EOS and kept the full supplied prompt visible. This is strong evidence of failed source-name binding, but **not proof that identifier diversity causes recovery**.

The intervention changes **only the source-identifier distribution in new training records** while preserving the complete-file task, target replacement relation, prompt template, four-case contrast composition, training example count, copy/edit ratio, frozen validation examples, fixed starting weights and optimizer budget. The causal comparison requires a separately authorized re-run of the unchanged comparator from the **same** starting checkpoint; historical step-100 loss and old 0/138 scores are not that matched comparator.

## 2. Files prepared (no model execution)

| File | Purpose |
|---|---|
| [`prepare_namespace_diversity_candidate.py`](../training/phase3/copy-fidelity-v1/prepare_namespace_diversity_candidate.py) | Reconstructs SHA-verified intervention from frozen old 288/96 candidate, keeping source byte lengths and all other authored fields fixed |
| [`sb03-namespace-diversity-proposal-v1.json`](../training/phase3/copy-fidelity-v1/sb03-namespace-diversity-proposal-v1.json) | **Exactly 288 train + 96 unchanged validation rows**, complete sampler, 48-name pool and explicit unauthorized statuses |
| [`audit_namespace_diversity_token_parity.py`](../training/phase3/copy-fidelity-v1/audit_namespace_diversity_token_parity.py) | Owner-local **tokenizer-only** read-only audit of baseline/intervention row-token matching; no model, training or output writes |
| [`test_p3_source_binding_diversity_candidate.py`](../training/tests/test_p3_source_binding_diversity_candidate.py) | Synthetic CI identity, split, refused-corruption, sampler and tokenizer-mismatch tests |

Proposed canonical SHA-256 of `sb03-namespace-diversity-proposal-v1.json` (ASCII, sorted keys, compact separators, final LF):

```text
4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea
```

Frozen sampler canonical array SHA-256:

```text
9ddf29a3a2972cee0c8c2f79db97d7ef3f823065a2db6ee0840e89bb01760c13
```

Frozen existing real model checkpoint remains **step 100**:

```text
c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230
```

Historical model/task results stay unchanged. No output claim above is a new model measurement.

## 3. Candidate design and integrity

| Dimension | Matched comparator | Intervention proposal |
|---|---|---|
| Train examples | 288 existing frozen rows | 288 same ordered requests/targets; train names replaced |
| Validation | Original 96 | Original **identical** 96 rows |
| Languages | HTML/CSS/JavaScript | Unchanged |
| Task composition | 72 exact copy, 216 single edit | Unchanged |
| Context | Minimal and unrelated surrounding code | Unchanged |
| Four-case train groups | 72 | Same 72 groups, names replaced consistently within each |
| Train source names | 6 original names | **48 deterministic SHA-derived ASCII names** |
| Name length | 5–6 ASCII letters | Each replacement matches its old name's byte length |
| Source and target relation | Complete-file exact | Unchanged except source namespace |
| Source/checkpoint | Fixed step-100 first-run endpoint | Same endpoint if later authorized |
| Sampler | Requires new same-start comparator | Exactly the same frozen 1,600 indices if authorized |
| Optimizer updates | **0 authorized** | **0 authorized** |

The 48 pool entries are computed from SHA-256(`p3-sb03|old-name|slot|attempt`), constrained to alphabetic ASCII, same original name length, no pool duplicates, no collision with known prior/microprobe names, and no substring appearances in the five SHA-verified historical P3 candidate/control files. This excludes **five repository corpora only**: it does not establish globally unseen names across all phase-2 corpora and privately held training sources.

Every original 12-group namespace family rotates through eight new names, producing 48 distinct train names over 72 groups; all four contrasts of one group retain the same name. Source/expected-file overlap against the five frozen corpora is zero. Exact validation rows, including their identifiers, prompts and targets, are byte-identical to the comparator. Character-offset and source preservation requirements are checked.

## 4. Sampler defect corrected before freezing

The initial technical proposal specified 1,600 repeated `random.Random(1337).randrange(288)` draws. Direct calculation shows it hits **287/288** distinct training rows, not 288. That is a completeness defect if the experimental design promises full record coverage.

**Corrected proposal:** six deterministic SHA-256-permuted epochs, taking a total of **1,600** positions from complete 288-row permutations. This produces:

- **288/288** distinct training rows drawn at least once;
- every row drawn **5 or 6** times;
- a fully specified 1,600-element draw-index array frozen with the digest above;
- platform-independent replay, independent of Python's PRNG implementation;
- identical candidate indexing for each arm.

Both arms must start from identical weights and fresh optimizer state and share identical draw indices and the same validation cases. The comparator and intervention are two distinct future optimizer runs, **each requiring separate owner approval**.

## Actual owner-local tokenizer finding (October 18, 2026)

**BLOCKED.** The owner validated both candidate and sampler checksums but the real frozen Plex tokenizer reported **240/384 prompt/answer+EOS length mismatches**, with **40,439 comparator** versus **44,925 intervention** draw-weighted supervised targets. Equal ASCII length was not enough. **Do not train from this V1 proposal.** Its fixed SHA, original 96 validation records, sampler and failure evidence remain unchanged. [Separate V2 read-only tokenizer-aware feasibility roadmap](PHASE-3-SB03-V2-TOKENIZER-FEASIBILITY.md) provides the next safe research preparation; a matched V2 candidate does not yet exist.

## 5. Blocking audits before any later training decision

The proposed candidate is intentionally frozen with `tokenizerParityVerified=false` and `entireHistoricalTrainingInventoryVerified=false`. It must remain **not ready for training**, even when GitHub CI is green, until these gates are satisfied:

1. **Tokenizer parity.** On the owner's PC, verify the actual SHA-pinned 16,384-vocabulary tokenizer byte identity; compare all 384 paired rows' exact prompt token count and answer-plus-EOS count, with EOS and causal target-mask alignment. Changing ASCII name lengths alone is insufficient for BPE parity. Audit 1,600 draw-weighted answer/EOS positions for both arms. **Any mismatch blocks calling this a strict single-variable matched experiment** and requires a new reviewed preparation revision; do not silently proceed or rebalance after seeing model output.
2. **Historical data inventory.** Enumerate and SHA-pin all additional training corpora contributing to prior checkpoint ancestry; check all 48 proposed names against training text/sources. Missing privately held sources cannot be declared absent. The final project holdout remains unopened and cannot be scanned to screen names.
3. **Candidate and sampler identity.** Recompute candidate bytes and the exact frozen draw-array digest from the generator and verify that original comparator, 96 validation examples, entire 42-task historical control suite and 48-case fresh microprobe are unmodified.
4. **Runtime safety and source lineage.** Before any new zero-update stage or bundle, separately review fixed tensor/config/tokenizer identity, reproducible masks, fresh destination, disk headroom within 200 GB, non-overwrite, interruption handling, no resume, no retrospective checkpoint selection. No model loader or optimizer is included in this proposal.
5. **Approval packet.** Require new SHA-pinned baseline, exact run wall-clock and update ceilings, settings, training source, prepared bundle/stage and explicit owner approval for **each** arm. The prior October 17 authorization is exhausted and cannot be reused.
6. **Evaluation plan.** Freeze unchanged full-file success thresholds and score both arms on the *same* future permitted evaluation, retaining the 48-case SB-01 proposed microprobe and original 96/42 controls with correct historical provenance. Do not convert inspected developer cases into independent final holdout evidence.

A tokenizer mismatch does **not** mean the model failed a capability test. It means the experimental comparison requires a revised, reviewed design before either arm can train.

## 6. Reproducibility commands on owner's Windows PC

Run from the repository root **only after the associated PR is merged and both Linux/Windows CI are green**. Neither command loads weights or changes previous results:

```powershell
git pull origin master

# Checks canonical committed dataset and complete sampler; no writing.
uv run --project training python training/phase3/copy-fidelity-v1/prepare_namespace_diversity_candidate.py

# Read-only tokenizer identity and per-record token-parity report.
uv run --project training python training/phase3/copy-fidelity-v1/audit_namespace_diversity_token_parity.py
```

The second command prints `tokenParityExact`, `tokenParityMismatchCount`, first mismatches, full draw coverage and both arms' target totals. It returns `readyForTraining=false` regardless of parity because full-history provenance and separate training authorizations are not satisfied.

**Do not run `--prepare` locally** against a committed manifest; it is intended for first-time creation only and will refuse an existing file. Do not delete old files to bypass refusals.

## 7. Model-wide next steps and stopping rule

**Immediate objective:** learn to copy unfamiliar supplied names *and* keep every unrelated byte intact when generating an entire requested HTML/CSS/JavaScript file. This is prerequisite infrastructure for all subsequent edit abilities, not an isolated benchmark trick.

Once a properly matched, separately approved experiment completes, measure on exact full generated code rather than loss alone. If the namespace diversity arm improves copying but not request-conditioned value selection, pursue the already documented **SB-02 C** value-binding hypothesis as the next controlled intervention. If source copying still fails, revisit the objective (copy-heavy supervision) or deterministic patch/span output as independent proposals. Do not advance P3-03 Insert/Delete or the later logical-edit/bug-fix phases on fabricated success.

**Handoff disposition:** SB-02 intervention *choice* approved; SB-03 **repository-side model-free preparation completed** when CI passes. Remaining owner-local token/history audits and separate future model authorizations are explicitly blocked, not silently completed.
