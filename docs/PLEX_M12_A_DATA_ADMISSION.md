# Plex M12-A — Source-Copy Data Admission

**Status:** audit implementation available; **no owner dataset collected, approved or trained**.

M12-A is a research gate for the original scratch-trained Plex 600M. It must establish that source and target data can be trusted and that training and development groups do not leak into one another. Nothing in this tool starts training, approves data rights, loads model weights, or accesses the sealed final holdout.

## What is implemented

The read-only standard-library tool scripts/plex_m12_data_audit.py accepts an explicitly supplied manifest.json referencing exactly two sibling files: train.jsonl (256 records) and development.jsonl (64 records). It rejects changes in file digests or counts, malformed records, model-family mismatches, missing SHA-256 source/target identities, unequal copy targets, claimed training authorization, unexpected file paths, repeated IDs, and train/development leakage.

Each sample supplies provenance claims and one of three fixed prompt templates:

- **copy-space:** ends with Copy followed by colon and a space (the original M11 delimiter).
- **copy-newline:** ends with Copy followed by colon and a newline.
- **console-general:** an ordinary request with source file and edited file delimiters.

The tool enforces disjoint train/development identifiers, identifier-family groups, source-template groups, prompt-template groups, and exact source hashes. Development slice tags must claim at least 32 unseen-identifier cases and 16 prompt-transfer cases, but a tag is **not evidence** of achieved generalization.

An optional audit uses the **frozen M11 tokenizer** to verify every source, target and computed prompt roundtrips exactly and the combined input+target+EOS stays within 1024 token positions. It does not fit or change the tokenizer.

## Required manifest structure

The caller provides a UTF-8 JSON document with these exact fields. Replace all SHA placeholders with real SHA-256 values before running:

    {
      "schema": "plex-m12-a-source-copy-review-v1",
      "modelFamily": "plex-600m-v1",
      "status": "research-only-not-training-authorized",
      "trainingAuthorized": false,
      "rightsVerification": "author-claims-unverified",
      "splits": {
        "train": {
          "file": "train.jsonl",
          "count": 256,
          "sha256": "SHA256_OF_EXACT_TRAIN_JSONL_BYTES"
        },
        "development": {
          "file": "development.jsonl",
          "count": 64,
          "sha256": "SHA256_OF_EXACT_DEV_JSONL_BYTES"
        }
      },
      "sealedHoldout": {
        "count": 64,
        "commitmentSha256": "COMMITMENT_ONLY_NO_HOLDOUT_CONTENT",
        "contentNotIncluded": true
      }
    }

The holdout commitment is metadata only. The auditor cannot establish that a locked holdout exists or independently validate its content, and never opens it. A protected holdout requires a separate authorized process.

## Required record — one JSON object per line

Fields:

- id, split, source, expected, instruction, promptTemplate
- identifierFamily, sourceTemplate, provenanceReference
- sourceOrigin, rightsClaim, sourceSha256, expectedSha256
- trainingAuthorized, developmentSlices

For this first copy task, source and expected must be identical strings. Source and expected SHA-256 must hash their exact UTF-8 bytes. sourceOrigin must say owner-authored, rightsClaim must say owner-claims-rights, and trainingAuthorized must be false. This enforces a provenance **declaration**, not independent legal verification of the author's rights.

The developmentSlices list is empty in train. Development records can carry unseen-identifier and/or prompt-transfer annotations. A group ID such as identifierFamily must never occur in both splits. This deliberately tests generalization to unseen families.

## Read-only commands

For a manually supplied, owner-authored dataset at an actual local directory:

    python3 -I -S scripts/plex_m12_data_audit.py --manifest /path/to/owner-data/manifest.json

With the installed WSL training environment and existing frozen tokenizer:

    /home/quali/.local/bin/uv run --project training --no-sync python scripts/plex_m12_data_audit.py \
      --manifest /path/to/owner-data/manifest.json \
      --tokenizer-bundle training/artifacts/plex600-fresh-tokenizer

The output includes counts, digests, slice claims and trainingAuthorized: false. Without the tokenizer flag, the report says tokenizerAudit: NOT RUN. Successful data structure review is **not** training approval, proof of model accuracy, or verification of true source rights.

GitHub CI tests only **temporary synthetic records** to validate the auditor. No such synthetic data is admitted to actual training, and no owner-authored M12 corpus is asserted to exist.

## Existing owner-prepared research material

A separate [historical source inventory](PLEX_M12_A_HISTORICAL_SOURCE_SURVEY.md) verifies a prior preparation-only 600M corpus: 120 train and 24 internal validation records, including 36 and 8 JavaScript examples respectively. It is request/answer data, not the proposed exact-copy M12 corpus; historical preparation rights do not constitute M12 training permission.

## Read-only model-learning diagnostics

The [M12-A teacher-forced source-conditioning probe](PLEX_M12_A_SOURCE_CONDITIONING.md) provides a separate existing-M11 evaluation of target-token ranks, gold-prefix NLL and whether the score/value preference follows changes in supplied code. It does not require or approve an M12 dataset, and its GPU results remain unmeasured until run locally.

## What remains for M12-A

The following work is **not complete**: owner-authored corpus collection and rights review, actual train/dev group audit, real tokenizer roundtrip audit against such a corpus, read-only starting checkpoint comparison, source-conditioned gold-prefix evaluation, and GPU memory/admission assessment.

M12-B training admission remains separate and requires explicit approval of exact executable, permitted dataset hashes, starting checkpoint and optimizer budget.

[Research design and measurable M12 gates](PLEX_M11_FINAL_INVESTIGATION_M12_PLAN.md) · [Current Plex roadmap](../PLEX-MASTER-ROADMAP.md)
