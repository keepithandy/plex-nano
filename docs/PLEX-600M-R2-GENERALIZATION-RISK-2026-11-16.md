# Plex 600M R2: prospective synthetic curriculum overlap audit

**Date:** November 16, 2026. **Scope:** read-only inspection of the exact repository R2 candidate JSONL and frozen 36-case M8 development benchmark. **Training performed: none.** R2 candidate approval status remains unchanged.

## What the repository bytes show

| Audit measure | Observed |
| --- | ---: |
| R2 proposed candidate rows | 1,152 |
| Languages | 384 HTML / 384 CSS / 384 JavaScript |
| Template families | 18 (6 per language; 64 variations per family) |
| Distinct exact (oldText, newText) edit operations | 843 |
| Template families whose 64 rows all use one identical edit operation | 4 |
| Rows in those four fixed-edit families | 256 |
| M8 development cases with an exact edit-operation match in R2 | 4 of 36 |
| R2 rows with such an exact M8 operation match | 256 |
| R2 template families implicated in those matches | 4 |
| Rows outside those four implicated families | 896, **still unapproved** |
| Exact source/expected/request string matches with benchmark | 0 |
| Newly approved training rows | **0** |

The four overlapping frozen evaluation case IDs are `m8-html-02`, `m8-javascript-03`, `m8-javascript-05`, and `m8-javascript-08`. The four affected R2 template groups are:

- `r2-html-button-pressed`
- `r2-javascript-limit-compare`
- `r2-javascript-feature-flag`
- `r2-javascript-array-operation`

The concrete operations include `false → true`, `> → >=`, and `map → filter`. A single edit operation can appear in multiple R2 families; for example, both the HTML button and JavaScript feature flag share `false → true`.

**Important distinction:** R2 is **quarantined and was not used in the owner-reported M7 four-update training experiment**. This overlap is a *prospective risk* if R2 were approved and trained on later. It is **not proof that the existing M7 checkpoint saw M8 test data**, and it is **not proof of semantic contamination** beyond the reported string matches.

All R2 records come from 18 templates with repeated surface variations. Unique request strings, file contents and IDs do not imply 1,152 independent programming problems. A random row-level train/validation split would distribute variations of the same 18 templates to both sides and substantially overstate generalization. Any future approved experiment must split at least by the originating `templateGroup` and review near-duplicates and shared edit behavior before evaluating.

## Reproduce the read-only audit

From the repository root, using the already configured Python environment:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_research candidate-risk-audit
```

This validates all 1,152 records and the 36-case benchmark first, checks immutable source bytes after the audit, and reports the affected case and family IDs. It **does not** create a training split, write output files, start a model, touch the local checkpoint, read sealed final holdout, approve data, or authorize optimizer updates. The new unit tests lock the current metrics and fail if the input identities or expected warning counts change.

## Highest-value next experiment design

1. Keep R2 quarantined. Do not train on this collection or adjust existing M8 benchmarks retroactively based on these results.
2. Collect truly diverse, owned or individually rights-reviewed before/after file edits. Each example should contain an exact supplied file, a natural-language request, and the minimally changed expected file. Include unchanged-text and identifier-preservation controls.
3. Partition prospective training and internal validation by source/project **and** template/semantic family where applicable. Do not use row-level shuffling of template variations as an independence claim.
4. Require provenance and source-rights review, exact/near-duplicate screening, secret scanning, token budget assessment, and a new explicit training contract with compute and stopping limits.
5. Use the frozen, independently authored M8 development set to characterize the *existing M7 checkpoint first*. Once a future experiment trains on overlapping edit operations, report that evaluation as *development performance on partly overlapping operations*, not a new independent generalization claim. An independent, properly isolated evaluation protocol must be separately defined before product capability claims.

**Inference/training status:** Neither run here. The original 27M controlled baseline, 600M M7 owner checkpoint, tokenizer, M8 benchmark, R2 rows, source approvals and final holdout are unchanged.
