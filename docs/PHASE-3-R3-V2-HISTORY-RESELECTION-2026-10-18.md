# R3 · V2 reselection against reviewed history

> **Result: blocked.** Both bounded searches stopped at `prism.0`. No complete replacement proposal or reviewed V2 manifest was created. V1 and the original unreviewed V2 retain their original hashes.

## What ran

The owner requested a repeated V2 selection using the reviewed history inventory, followed by freezing a distinct reviewed manifest if a complete candidate could be established. This was model-free preparation using the existing frozen tokenizer and deterministic V2 candidate stream.

All **10 explicit ancestor training-text files**, totaling **7,134,286 bytes**, were available and matched the [reviewed inventory](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json). The inventory SHA remains `111784c169f9d180b8ec5329de7acda9f0259214847136a7fab3684ce24b58d1`.

The inventory uses the ancestry auditor's schema rather than the selection tool's optional-inventory schema. The execution driver therefore used the auditor's path validation and checked every file hash, then passed the verified bytes directly to the existing search function. It lowercased those bytes before searching, preserving the earlier **case-insensitive ASCII substring** exclusion rule. Candidate names are already lowercase. The five frozen P3 fixture checks also remained active. No directories or final project holdout were searched.

## Frozen results

| Search bound per slot | Total candidates examined across attempted slots | Selected slots | First failed slot |
|---|---:|---:|---|
| 2,500 | 16,623 | 16 of 48 | `prism.0` |
| 20,000, the tool's maximum | 34,123 | 16 of 48 | `prism.0` |

Each search stopped after its first failed slot, as required by the existing fail-closed selection procedure. The remaining **31 slots were not attempted**. The counts describe overlapping deterministic searches, not independent samples or unique candidates across both runs.

- [2,500-bound raw result](../training/pretraining/p3-sb03-v2-reviewed-history-reselection-2500-result.json): `12d6efbc0941b8995a0b78f53f2bb8c5924f4dc44dea59520726f30ab0437b3a`.
- [20,000-bound raw result](../training/pretraining/p3-sb03-v2-reviewed-history-reselection-20000-result.json): `1df0290a7eaf147eadacd14f7384da971468145f8b92c599dc8e497f34220c10`.
- [Second-pass review and execution record](../training/pretraining/p3-sb03-v2-reviewed-history-reselection-review.json): `572cfd0b81919dedc7cc9c995fcb49f31735fd444cd77bf5b715ce5141d7c9c2`.

The execution record contains the maximum-bound driver source, its hash and arguments, source-code identities, inventory identity, and preserved artifact hashes. The raw receipts retain the original tool's generic five-P3 labels and conservative global-history flags; the execution record specifies the actual additional ten-source scope and case handling.

## Why selection stopped

The frozen tokenizer encodes `prism` as **one token**, ID `11068`. In the first assigned HTML copy row it also occupies one token between the identifier delimiters. All **880 five-letter lowercase alphabetic vocabulary words** examined by the vocabulary-first selection stage occur in the listed ancestor bytes. Thus those vocabulary candidates fail the conservative history exclusion rule. The subsequent generated candidates did not produce a valid replacement within the stated bound.

This is evidence of a conflict between exact token matching and historical spelling exclusion under the current selection method. It is **not an exhaustive proof that every possible replacement is impossible**. There is no new capability result.

## Review and preservation

A second pass by the same assistant rechecked all ten ancestor hashes, all 16 partial selected-name exclusions, and prompt/answer token lengths and roundtrips for their **96 assigned training rows**. This is not external independent review. A complete 384-row revision did not exist, so complete-revision parity and validation equality were not certified. No partial candidate was promoted to a manifest.

| Preserved input | SHA-256 |
|---|---|
| V1 proposal | `4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea` |
| Original unreviewed V2 | `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21` |
| Base curriculum, including 96 validation rows | `9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a` |
| Frozen tokenizer | `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697` |

Coverage remains the ten recorded ancestor inputs, not certification against undocumented training history. The separate [evaluation-namespace review](PHASE-3-R2-FROZEN-EVALUATION-NAMESPACE-REVIEW-2026-10-18.md) and its exposure limits still apply. No name is relabeled globally unseen.

## Roadmap decision

- [x] Repeat deterministic selection against the reviewed inventory at the default and maximum supported bounds.
- [x] Freeze the unsuccessful search receipts and second-pass review with distinct SHA identities.
- [ ] Establish a complete replacement dataset satisfying history exclusion and exact per-row parity.
- [ ] Freeze a separately reviewed V2 manifest after all required checks pass.

**R3's dataset gate remains blocked.** Its next design review should choose whether to investigate a different bounded selection method, or define a distinct experiment that explicitly changes token-exposure matching or narrows the historical-exposure claim. Either change needs a new comparison contract; neither is silently adopted here. Repeating larger random searches without a revised rationale is not the immediate next step.

No model was loaded; no training, inference, optimizer updates, staging, bundle creation, checkpoint changes, or final project holdout access occurred.
