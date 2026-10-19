# R2 - Frozen evaluation-name exclusion review

> **Completed October 18, 2026. Claim narrowing chosen; frozen artifacts preserved.** Ten evaluation spellings from six visible, non-final-holdout sets were screened separately against the ten-file training-history inventory. Five have ancestor-text substring matches. This is separate from the eight proposed train-name collisions.

## Evaluation-name findings

| Frozen set | Cases | Names | Result against listed ancestor inputs |
|---|---:|---|---|
| SB-01 proposed microprobe | 48 | `nyxen`, `ulvar` | No match: `nyxen`, `ulvar` |
| Copy-fidelity / V1 / V2 validation | 96 | `granite`, `saffron` | No match: `granite`, `saffron` |
| Inspected copy/edit developer controls | 42 | `cobalt`, `spruce` | Matches: `cobalt`; No match: `spruce` |
| P3-03 source-preservation diagnostic | 72 | `harbor`, `orbit` | Matches: `harbor`, `orbit` |
| Older P3-02 validation context | 36 | `gallery` | Matches: `gallery` |
| Older P3-02 development context | 36 | `account` | Matches: `account` |

The 96 validation rows were checked to be identical across the original copy-fidelity candidate, V1 proposal and saved unapproved V2. They are counted once above. P3-02 context names were grounded in the generator's explicit `CONTEXT_NAMES`/`SPLITS` mapping and confirmed in each selected frozen input/expected file; they were not inferred from folder names.

### Matches in actual training text

| Spelling | Stage: number of training rows containing substring | Interpretation |
|---|---|---|
| `account` | P2-29: 13; P2-30: 1; P2-35: 6; P2-38: 1; P2-41: 6; P2-44: 12; P3-02: 4 | Includes standalone ASCII-name matches; this is not proof of full identifier/file reuse. |
| `cobalt` | P2-29: 1 | Includes standalone ASCII-name matches; this is not proof of full identifier/file reuse. |
| `gallery` | P2-29: 17 | Includes standalone ASCII-name matches; this is not proof of full identifier/file reuse. |
| `harbor` | P2-29: 2 | Embedded-string matches only; no standalone ASCII-name match. |
| `orbit` | P2-29: 1 | Embedded-string matches only; no standalone ASCII-name match. |

All raw-byte collision identities were confirmed in parsed model-input `text` fields. `harbor` and `orbit` occur only within longer strings under the diagnostic ASCII boundary definition; they still fail the fixed **case-insensitive ASCII substring** exclusion rule. The rule has not been relaxed after seeing results.

## Relevant history and proposed arms

- **Current copy-fidelity endpoint:** `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`; all ten inventoried training stages are relevant to current/future reuse of these fixtures.
- **Historical P3-02 endpoint:** `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`; only the first nine stages are its ancestors. Older P3-02/P3-03/control claims were checked against that prefix separately. All five matched evaluation spellings already occur within that earlier prefix; no later copy-fidelity stage was used to manufacture a historical exposure claim.
- The original comparator, V1 and unapproved V2 **288-row train splits** were separately checked for all ten evaluation spellings in their `prompt`/`expectedFile` fields: **zero matches in each split**. This prospective split separation does not erase exposure in earlier ancestor corpora.
- Training-corpus presence is not proof that a spelling entered a particular sampled gradient, was memorized, or caused an output failure. This review does not establish full-identifier, full-file or semantic contamination.

## Explicitly narrowed research claim

**Disposition: preserve the existing proposals and evaluation fixtures; create no replacement dataset in this task.**

The current V2 is a **tokenizer-matched preparation artifact with mixed historical spelling exposure**. Eight proposed train names already occur in the recorded ancestor inputs. It is not a clean history-excluded new-name intervention and remains blocked under the original source-exclusion gate.

For `ulvar`, `nyxen`, `granite`, `saffron` and `spruce`, the supported statement is: **no case-insensitive ASCII substring was found in the ten listed, hash-verified ancestor training texts**. These names are not certified globally unseen. The earlier four/five P3 exclusion checks remain claims about those specific frozen artifacts.

The older `cobalt`, `gallery`, `account`, `harbor` and `orbit` cases remain **fixed diagnostics with documented lexical exposure limits**. Do not cite them as evidence of globally unseen identifier generalization. The 96 validation cases were loss-monitored, and developer diagnostics were previously inspected; neither becomes an untouched final benchmark through this review.

The small coding model's measured copying, requested-value selection, unrelated-byte preservation and exact-file scores retain their original meanings and numbers. Narrowing wording does not turn the historical **0/138** capability result into success and does not change scoring thresholds.

**No gate is waived.** A future revision that aims for stronger train-name exclusion needs a distinct frozen identity, renewed ancestor screening, real-tokenizer parity and separate design review before any execution authorization. Current artifacts remain unapproved for new model execution.

## Evidence and limits

- [Evaluation-name review receipt](../training/pretraining/p3-sb03-frozen-evaluation-namespace-history-review.json): all seven frozen fixture/proposal identities, six name-set definitions, original/current history cutoffs, ten per-file paths/hashes, text-row match counts and prospective train-split checks.
- [Explicit training inventory](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json), SHA-256 `111784c169f9d180b8ec5329de7acda9f0259214847136a7fab3684ce24b58d1`; [recorded coverage review](PHASE-3-R2-TRAINING-INVENTORY-COVERAGE-REVIEW-2026-10-18.md). All ten source hashes verified again, covering **7,134,286 bytes**.
- [Separate eight-name proposed-train collision report](PHASE-3-R2-ANCESTOR-NAME-COLLISIONS-2026-10-18.md). Its original result and interpretation receipts remain unchanged.
- Scope is limited to recorded, explicitly inventoried training text and ASCII substring matching. No external independent completeness certification, Unicode-equivalence analysis, complete-file duplication search or new capability evaluation is claimed.
- Original V1, unapproved V2, frozen evaluation fixtures, checkpoints, historical results and thresholds remain unchanged. No model loading, inference, training, checkpoint staging or final project holdout access occurred.

[Return to the active R2/R3 roadmap](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md#r2)
