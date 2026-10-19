# R2 - Ancestor training-name collision results

**Subsequent disposition:** the [separate frozen evaluation-name review](PHASE-3-R2-FROZEN-EVALUATION-NAMESPACE-REVIEW-2026-10-18.md) is now recorded. The permitted **claim-narrowing path** was chosen: preserve the current V2 and describe mixed historical spelling exposure; do not certify globally unseen names. The original train-name scan and execution receipts below remain unchanged. Any future revised comparison still needs distinct review; no exclusion gate is waived.

> **Checker completed on October 18, 2026. The current unapproved V2 proposal fails the selected-name exclusion gate:** 8 of 48 proposed train names occur in the listed ancestor inputs. All 10 file hashes verified.

**Colliding names:** `bonus`, `hover`, `inser`, `layer`, `media`, `proxy`, `track`, `users`.

The scan covered **7,134,286 bytes** across ten explicitly inventoried training-text files. Five files have matches. All eight names were also confirmed inside parsed model-input `text` fields; none is explained solely by JSON metadata. The remaining forty names have no match under this scan rule in these listed sources; that is not a global novelty claim.

## Exact run identities

| Input or output | SHA-256 |
|---|---|
| Original unapproved V2 proposal | `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21` |
| Explicit ten-file inventory | `111784c169f9d180b8ec5329de7acda9f0259214847136a7fab3684ce24b58d1` |
| Checker source used | `a2985d9ba8227cf9d459071a0d8627dfa3ad5e808ec9e4c2a142de85c5a98a61` |
| Unmodified checker output | `f3b6702c4e43926d69679cbabf92081882f3363edaa647dfdae70c46b0c26114` |

- [Raw checker result](../training/pretraining/p3-sb03-v2-ancestor-screen-result.json)
- [Execution and interpretation receipt](../training/pretraining/p3-sb03-v2-ancestor-screen-execution.json): exact command/paths, input hashes, inventory-index mapping, per-file matches and parsed-text confirmations
- [Reviewed inventory](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json) and [coverage review](PHASE-3-R2-TRAINING-INVENTORY-COVERAGE-REVIEW-2026-10-18.md)

## Per-file hashes and collisions

The index is the one-based position in the unchanged inventory. Exact absolute file paths are recorded there and in the execution receipt.

| Index | Training stage | Bytes | Verified SHA-256 | Proposed names found |
|---:|---|---:|---|---|
| 1 | P2-29 | 6,256,500 | `8d621a488f3554a976ea4ee0e9581fae2c07aff58ad7912b2f69f2cc0233aaa6` | `bonus`, `hover`, `inser`, `layer`, `media`, `proxy`, `track`, `users` |
| 2 | P2-30 | 128,630 | `05fb9b35277b979ef45473af5b2a2c98bef31ee2c1c3133f0af0c68d7cfba554` | `track` |
| 3 | P2-32 | 96,858 | `239eaf48f6ad5701e0abe7fcb1e98a2ea379a7377d7e8c523f267e5b9dbd4e71` | `media` |
| 4 | P2-35 | 75,717 | `b921dd54657e77619835c5d9bce7a92c12c3a8f388d8ebe8ab8f6a9550ad69f3` | `media` |
| 5 | P2-38 | 87,003 | `ecff52a775a2f876b2a243b3254c37ac05bd75c72b372d2ecaddc06ae82b6ecf` | `layer`, `media` |
| 6 | P2-41 | 90,318 | `458de009e4ef227361e2813c7380bfa2d42739f36335ed333ea3b6a250800db3` | None in listed file |
| 7 | P2-44 | 135,646 | `0141dbd0209d10e2138a0a90ff688e97b4b1de39f91c28ca367ed7267b309789` | None in listed file |
| 8 | P2-48 | 73,594 | `39879c1199e975096ba962fcad023b3b112c2c8b6444c0a3a9f35971a945d530` | None in listed file |
| 9 | P3-02 | 68,592 | `45d42817582f6b909d08695dfc93336414762b198cae7a4e4d760c09bf7249de` | None in listed file |
| 10 | P3-copy-fidelity-v1 | 121,428 | `682e12d1516243d94ef7601d13f7bf788472ad81db9c3f75a9e85cdeaafbfdaf` | None in listed file |

## Actual training-text confirmation

The checker uses **case-insensitive ASCII substrings**, including names embedded within longer words. The following counts are training-row memberships with a substring match, not occurrence counts or a reconstruction of gradient exposure.

| Name | Training-text rows by stage |
|---|---|
| `bonus` | P2-29: 72 |
| `hover` | P2-29: 78 |
| `inser` | P2-29: 27 |
| `layer` | P2-29: 52; P2-38: 1 |
| `media` | P2-29: 226; P2-32: 8; P2-35: 6; P2-38: 3 |
| `proxy` | P2-29: 12 |
| `track` | P2-29: 39; P2-30: 1 |
| `users` | P2-29: 10 |

`inser` has zero standalone ASCII-name matches in these texts; it still fails the predeclared substring rule because it appears inside longer strings. Some other matches are also embedded. The supplemental standalone-name diagnostic in the receipt is informational and does not replace the fixed matching rule.

All eight colliding names occupy the eight proposed replacements for the original `prism` namespace. They appear in **48 proposed training rows**. No proposal rows were edited during this audit.

## Coverage limits and decision

- **The five earlier P3 checks remain a partial exclusion claim.** This new scan adds the ten documented ancestor inputs; it does not turn the earlier result or the forty unmatched names into a universal unseen-name claim.
- Coverage is limited to the explicit inventory and the recorded ancestry reviewed by the same assistant. `historicalTrainingLineageComplete=false` and `historyCoverageOnlyListedFiles=true` remain unchanged.
- Matches establish that a spelling is present in an eligible ancestor training corpus. They do not prove every matching byte was sampled into gradients, retained by the model, or used as a standalone identifier.
- The original checker scans JSONL bytes. The supplemental pass confirmed the same per-file identities in parsed `text` fields, checking all 48 names; no additional name identity was found there. No Unicode-equivalence or semantic-name analysis is claimed.
- Validation/development/final project holdout contents were not opened or included. Frozen evaluation-namespace exclusion claims still require their own controlled review; checking proposed train names does not cover them automatically.
- Two private Phase 3 bundles and ignored repository artifacts must be supplied with identical hashes to reproduce this run elsewhere.

**Decision:** preserve the original V1 and unapproved V2 proposal. The current V2 fails the selected-name source-exclusion gate. A separately reviewed revision must reselect the colliding names, repeat ancestry screening and frozen-tokenizer parity checks, and receive a distinct frozen identity before later authorization. Narrowing the claim would require an explicit research-design decision; the eight collisions cannot be relabeled globally unseen.

**R2's original exclusion gate remains unmet.** The checker run, separate evaluation-name review and explicit claim narrowing are now recorded. All frozen proposals/fixtures are preserved; a future revised comparison requires separate review. `readyForTraining=false`. No model was loaded; no training, inference, checkpoint staging or final-holdout access occurred.

Reproduce from the repository root with the original owner-local proposal and exact inventory sources available:

```powershell
training/.venv/Scripts/python.exe training/phase3/copy-fidelity-v1/audit_v2_ancestor_history.py --inventory (Join-Path (Get-Location) "training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json")
```

[Return to R2](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md#r2)
