# Plex 600M — first owner-authored dataset (local pilot)

**Date:** 2026-10-31
**Scope:** local Plex 600M dataset preparation and tokenizer feasibility only.
**Format:** plex600-owner-authored-pilot-v1, deliberately distinct from external-source M4/M5.
**Status:** prepared for tokenizer; **NOT approved for training execution**.

## Why this route

The six proposed external source repositories remain on hold from the
[license and source audit](PLEX-600M-SIX-SOURCE-RIGHTS-AUDIT-2026-10-31.md).
No open source-license root declaration justifies asserting blanket clearance
for vendored, copied or linked file content. To prepare an actual first pilot
without false six-source approvals, the owner-requested local preparation
instead uses existing original Codex-authored source examples.

The repository contains two earlier owner approvals. Those approved local P2
training only, NOT automatic 600M reuse or public licensing. On October 31
the owner explicitly requested preparation of the first approved Plex Nano
600M training dataset while narrowing/excluding questionable sources. The
present report records that owner-directed local 600M **preparation** scope,
not an invented new signature or an approval of optimizer execution.

Original immutable sources and owner approvals:

- [Code pairs source](../training/phase2/drafts/p2-02-code-pairs-v2.jsonl)
  and [owner P2 approval](../training/phase2/data/authored/p2-02-code-pairs-v2/approval.json).
  Original candidate SHA-256: 88c35a023b0093cb177552c4ddc45533e92d63e0d006f26dbcdad2bd0a2ef72a.
- [Request-following source](../training/phase2/drafts/p2-02-request-following-v2/candidate.jsonl)
  and [owner P2 approval](../training/phase2/data/authored/p2-02-request-following-v2/approval.json).
  Original candidate SHA-256: 9f6ad8eff96298f45880f9bfe831648364e2a7ed11996e7a7d91d5e984eb3d0b.

## Actual prepared dataset

[Dataset folder](../training/pretraining/plex600-owner-authored-pilot-v1),
[manifest](../training/pretraining/plex600-owner-authored-pilot-v1/manifest.json),
[verification receipt](../training/pretraining/plex600-owner-authored-pilot-v1/receipt.json).

| Evidence | Count |
| --- | ---: |
| Original approved P2 owner-authored examples in collections | 216 |
| Prior P2 validation examples excluded | **72** |
| Original P2 train-only examples eligible for resplit | **144** |
| New train examples | **120** |
| New internal validation examples | **24** |
| Source families (HTML, CSS, JavaScript across two collections) | **6** |
| External historical records | **0** |
| New six-source external rights approvals | **0** |
| Optimizer updates in this preparation | **0** |

Each row keeps an original source pointer, source/content SHA-256, group,
language, and request/answer text. A seeded SHA-256 per-family holdout
puts one complete group from each source family into internal validation.
No original P2-validation record was allowed into either split.

This validation is **not** an independent coding benchmark. All 144 rows
were formerly P2 train examples and the 600M has not trained on them yet.
The dataset is approximately 117 KiB JSONL, far too small to suggest
scratch 600M coding competence. These examples emphasize simple requested
code outputs, not complex file editing.

The [source-bound read-only verifier](../training/src/plex_training/plex600_owner_pilot.py)
reconstructs each row from the original candidate source, verifies P2
approval digest, refuses any former P2 validation record, recomputes
record and split hashes and verifies group isolation and manifest receipt.

## Use on the local Windows checkout

After pulling latest master, verify without touching CUDA:

~~~powershell
git pull --ff-only origin master
uv run --project training --no-sync python -m plex_training.plex600_owner_pilot
~~~

The first tokenizer fit is permitted as dataset-preparation work but does not
authorize training and does not modify the frozen 27M baseline:

~~~powershell
uv run --project training --no-sync python -m plex_training.plex600_setup tokenizer-train --dataset-dir training/pretraining/plex600-owner-authored-pilot-v1 --output-dir training/artifacts/plex600-owner-authored-pilot-tokenizer-v1 --vocab-size 32768

uv run --project training --no-sync python -m plex_training.plex600_setup tokenizer-check --bundle-dir training/artifacts/plex600-owner-authored-pilot-tokenizer-v1
~~~

The 32,768 vocabulary is a *maximum*, not a measured size. The saved
tokenizer fits only the train split and encodes both splits with hashes.
A separate source-check additionally requires explicit independently noted
manifest hashes. Choose a new output path if a tokenizer directory exists.

## Open gates

1. Real tokenizer fit measurement and packed source-binding verification.
2. New independent HTML/CSS/JavaScript editing development evaluation.
3. Strictly bounded M7 optimizer-training contract and fresh execution
   approval. The dataset manifest explicitly records trainingAuthorized=false.
4. Continued six-source review if expanding beyond the owner-authored pilot.

**Conclusion:** we now have a small actual owner-authorized local dataset
for preparation and tokenizer testing, without waiving any third-party,
evaluation or training-execution safeguards.
