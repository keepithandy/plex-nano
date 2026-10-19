# R2 - Training inventory coverage review

> **Review completed: October 18, 2026.** The ten-file inventory covers the frozen training inputs identified by the recorded checkpoint chain. **No known training-text source is unavailable, and no recorded source identity remains unresolved.** Name-collision screening is next.

**Subsequent result:** [the checker has now run](PHASE-3-R2-ANCESTOR-NAME-COLLISIONS-2026-10-18.md). It verified all ten hashes and found eight proposed names in five ancestor texts. This coverage-review receipt preserves its earlier pre-screening state; the original V2 remains unchanged and fails the selected-name exclusion gate.

This was a second review pass by the same assistant, not external independent certification. It does not establish global name novelty or authorize training. The final project holdout remained sealed and outside the inventory.

## Coverage checked

| Evidence | Result |
|---|---|
| Checkpoints | Rehashed all 11 files, including fresh initialization; endpoint and parent identities agree with the earlier trace. |
| Training text | All 10 files available; all SHA-256 values unchanged; 4,163 rows accounted for. |
| Packed training input | All 10 token files match their recorded hashes. Every indexed record ends in EOS, decodes exactly to its JSONL `text`, and collectively covers the complete token file without gaps or leftover ranges. |
| Tokenizer | All 10 actual `tokenizer.json` files match the frozen SHA `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`. |
| Source manifests | All 8 Phase 2 copied source manifests are available and match their recorded hashes. The 2 Phase 3 formats instead bind candidate and training-split identities directly in their bundle manifests. |
| Completed-run evidence | All 10 run receipts are available and agree with endpoint identities; the 9 later receipts also agree with bundle and authorization identities. |
| Training loaders | Reviewed the recorded input paths against loader code. The ten listed train streams cover the documented input route; tokenizer-source pointers do not add another training stream. |
| Inventory changes | None. No additional corpus was justified by the evidence. |

The text/token check includes the **formatted prompts and answers** actually packed into training inputs. The inventory therefore covers more than raw source-code snippets. P3 prompt context remains relevant exposure even where its next-token loss weights are zero.

## Unavailable and ambiguous sources

**Known unavailable training text: none.** Both private owner-local Phase 3 bundles are present. Their paths and full hashes remain in the [ten-file inventory](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json).

| Apparent ambiguity | Review finding | Effect on inventory |
|---|---|---|
| P2-41 contract still says training was not executed | The actual local `training/artifacts/structured-plan/p2-41-first-run/run-result.json` exists. It reports the matching step-100 endpoint, bundle and authorization identities, plus all 72 training records selected. A separately named tracked summary is unnecessary to identify its corpus. | Resolved; retain its existing text file. |
| P2-48 corpus carries P2-47 authorship | P2-47 prepared the request-grounded curriculum; P2-48 trained it. The copied source manifest, candidate identity, train-text hash and endpoint record agree. | One corpus; no duplicate entry. |
| Older tokenizer-bundle pointers skip some weight stages | Repack provenance and weight ancestry are different links. Every actual tokenizer file has the same frozen hash, and all training-token ranges are accounted for. | No extra training source inferred. |
| Private/owner-local sources | The two external Phase 3 bundles are fully represented. P2-30 source metadata separately identifies owner-approved Codex-authored request-following material; its actual rendered training split is already present. Storage outside Git does not establish another unknown upstream source. | Retain all three existing inputs in their respective stages. |
| Earlier Phase 1/2 artifacts and the P2-28 pilot | P2-29's explicit fresh-step-zero replay and saved initialization records exclude those earlier trained weights. | Excluded from this fixed checkpoint's weight ancestry. |
| Upstream catalog contains sources absent from training | The P2-29 training rows identify five upstream sources; Bulma templates have no rows in that train split. | Catalog membership alone does not add a training ancestor. |

## Remaining evidence limits

| Limit | What remains unproven | Required handling |
|---|---|---|
| Exact sampling exposure | P2-29 used random windows, so inclusion in its eligible corpus does not prove every byte was sampled. The seven later Phase 2 completed-run receipts report every train record selected; this pass did not reconstruct P3 draw histories. | Keep the full eligible splits as a conservative exclusion inventory. Do not claim every byte affected gradients. |
| Availability on another machine | Repository-local training artifacts are ignored by Git, and the two Phase 3 bundles live outside the checkout. Their bytes were verified on this machine. | Preserve or supply the exact hash-matching artifacts for another reviewer; the JSON inventory alone is insufficient. |
| Original-source rebuilds | The normalized/rendered training text is available and byte-verified. Current availability of original raw clones and full candidate files was not audited. | This does not block screening the exact recorded training text. Treat a full upstream rebuild audit as a separate scope. |
| Recorded-history boundary | This pass verifies the saved chain and its documented inputs. It is not external independent review or proof against undocumented changes. | Retain the conservative `historicalTrainingLineageComplete=false` guard; report the narrower `recordedInputCoverageComplete=true`. |
| Collision interpretation | The existing checker scans raw JSONL bytes, including metadata and JSON escaping. A reported match alone does not prove occurrence in model-input text; a zero result is a claim about those bytes under that matching rule. | Preserve per-file results and review their meaning before making an unseen-name claim. No collision screen ran in this review. |
| Evaluation separation | Validation, development and final project holdout contents were not opened or inventoried. Their disjointness was not newly tested by comparing contents. | Keep them outside this ancestry screen. The final project holdout remains sealed. |

## Evidence and next gate

- [Original trace: exact paths and full hashes](PHASE-3-R2-TRAINING-ANCESTRY-TRACE-2026-10-18.md)
- [Unchanged inventory](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json), SHA-256 `111784c169f9d180b8ec5329de7acda9f0259214847136a7fab3684ce24b58d1`
- [Coverage-review receipt](../training/pretraining/p3-sb03-v2-training-inventory-coverage-review.json), including local run receipts, source-manifest hashes, token coverage, loader evidence and limitations

**Next:** run the existing read-only collision checker against this explicit inventory, then assess its source coverage and matching limits. The inventory review is complete for the recorded inputs; the overall R2 unseen-name gate remains open until screening and interpretation are complete. No checkpoint staging, model inference, training or final-holdout access occurred.
