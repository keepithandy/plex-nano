# Plex 600M · historical source reuse review

**Completed:** October 31, 2026. **M3 remains open.** This local evidence review
and source-reuse recommendation supports traceable inputs for learning from
requests and supplied code. It is not a 600M corpus approval.

## Verified local evidence

| Check | Observed result |
|---|---|
| Historical manifests and source catalogs | Both manifest hashes matched prior records; each catalog matched its manifest's catalog hash. |
| Corpus split bytes | All four train/validation JSONL hashes and counts matched their manifests. |
| Accepted source records | **3,228/3,228** local raw files reproduced corpus text after the declared BOM/line-ending/NFC normalization, with matching content hashes. Source/group identities and accepted-file counts matched. |
| Packaged notices | **8/8** matched their manifest SHA-256 and the local source notice bytes. This verifies retained evidence, not a new rights determination. |
| Starter upstream file lock | **90/90** selected files, including notices, matched locked byte counts and Git blob identities. |
| Exact normalized-content duplicates | **0 duplicate groups** across all 3,228 records, including across the two corpora. This does not establish near-duplicate or semantic separation. |
| Exact accepted subsets | Sorted paths for each source are retained in the receipt, plus an ordered record-identity digest per corpus. |

The earlier M3 packet checked manifests only. This review additionally checked
accepted source content and notice evidence. It did not build data, fit a
tokenizer, execute source programs, run a model or open final-holdout material.
Revision fields agree between historical catalogs and manifests; no upstream
fetch or new authentication of the P2-26 revision/file trees was performed.

## Source-by-source recommendation

All recommendations are **conditional candidates for the M3 decision**. They
do not promote material or select a final 600M training mixture.

| Source | Historical subset | Recommendation and limitation |
|---|---|---|
| Microsoft Web Dev | 34 train records: 23 Markdown, 4 HTML, 3 CSS, 4 JavaScript; 35 locked files including LICENSE matched. | Consider the exact teaching/code subset as a small explanation-and-code input. Choose any new code/explanation balance explicitly; the historical mix is heavily Markdown and may include intentionally partial snippets. |
| Traversy projects | 54 validation records: 18 HTML, 18 CSS, 18 JavaScript; 55 locked files including LICENSE matched. | Retain as **historical validation/regression** in the proposed route. Prior use prevents an untouched-benchmark claim. |
| MDN CSS examples | 645 train records: 550 HTML, 68 CSS, 27 JavaScript. | Consider the exact subset as part of the single MDN family; keep all four MDN repositories together. |
| MDN DOM examples | 438 train records: 192 HTML, 86 CSS, 160 JavaScript. | Same conditional MDN-family role; reference overlap and behavioral completeness still need review. |
| MDN interactive examples | 1,665 train records: 623 HTML, 399 CSS, 643 JavaScript. | Same conditional MDN-family role. This source dominates the historical record count; review the resulting mixture before adoption. |
| MDN learning area | 233 train records: 151 HTML, 34 CSS, 48 JavaScript. | Same conditional MDN-family role. Teaching snippets may depend on other context; standalone correctness was not established. |
| JavaScript30 | 90 train records: 74 HTML, 12 CSS, 4 JavaScript. | Consider the accepted browser-example subset. File extensions are not token-language proportions; inline JavaScript can appear in HTML. |
| Bulma Templates | 69 validation records: 35 HTML, 25 CSS, 9 JavaScript. | Retain as **historical validation/regression**; previously measured material is not fresh evaluation. |

The eight repositories remain five related-source groups, with all four MDN
repositories together. No exact duplicate was found, but shared templates,
tutorial lineage, short snippets and near duplicates remain possible. Recorded
MIT/CC0 notices and old approval records do not settle new scope automatically;
retain attribution and inspect relevant file-level exceptions before adopting
a concrete source decision. No adequate 600M training scale follows from this
historical pool.

## Evidence and reproducibility

- [Read-only audit driver](../training/pretraining/plex600-m3-source-subset-review.py),
  SHA-256 `41e07801c889b08a29e5fb0cf70871889f1a7c60438ac0bf64bb68d413093f3e`.
- [Receipt with exact accepted paths](../training/pretraining/plex600-m3-source-subset-review-2026-10-31.json),
  SHA-256 `057f6aeb94df2cd8cbb9667964d6c007e0de5386d2be9382e8428905e9da2566`.
- [Starter subset and historical rights scope](DATASET-SOURCE-REVIEW.md).
- [P2-26 source batch](PHASE-2-P2-26-SOURCE-BATCH-2.md).
- [M3 preparation packet](PLEX-600M-M3-REVIEW-PACKET.md).

The driver requires exact historical manifest identities and a new receipt
destination. Reproduction needs the same ignored local corpora and raw
snapshots; the committed receipt does not replace them. It reads accepted
source records plus the explicitly locked starter files and notices. This is
a file-identity audit, not a fresh syntax, secret, near-duplicate, behavioral or
evaluation-contamination check; earlier results retain their historical scope.

Validation also checked the saved driver/receipt identities and confirmed that
in-memory changes to a manifest, packaged notice and locked source file are
rejected. No original artifact was modified by those checks.

## Remaining M3 work

The historical subsets and notices are now concrete inputs to a reuse decision.
The proposed route remains candidate training from Microsoft, MDN and
JavaScript30, with Traversy/Bulma retained as historical validation. Its data
volume and mix are still undecided.

Next, review permitted source scope, file-level exceptions and cross-family
similarity risks, then record whether to adopt, narrow or expand this pool.
Evaluation design, final source/split assignments and preparation budget also
remain open. Do not mark M3 complete or start M4 preparation from this evidence
alone.
