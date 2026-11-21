# M12-A — Historical Owner Dataset Feasibility Survey

**Status:** read-only research inventory; **not M12 data approval**.

The repository already contains a source-bound, locally prepared Plex 600M pilot at training/pretraining/plex600-owner-authored-pilot-v1. Its existing verifier reconstructs the records from two earlier P2 owner-approved candidate collections, checks original source approvals, validates SHA-256 and split-group isolation, and excludes the **72 old P2 validation records**.

The original permission scope covers *Plex 600M local dataset preparation and tokenizer testing*, not automatic optimizer execution or unrestricted reuse for a new M12 curriculum.

## Verified historical inventory

| Split | JavaScript | HTML | CSS | Total |
| --- | ---: | ---: | ---: | ---: |
| Earlier 600M train | 36 | 42 | 42 | 120 |
| Earlier 600M internal validation | 8 | 8 | 8 | 24 |
| Combined | **44** | **50** | **50** | **144** |

These records consist of a natural-language request followed by a code answer. They are **not** the task M12-A proposes (exactly reproducing unfamiliar code supplied in the input), and they are not a ready-to-use 256/64 M12 dataset. The previous internal validation records must never be silently promoted to training samples.

The existing source manifest explicitly indicates:

- 0 external historical-source records used.
- 72 old P2 validation/evaluation rows excluded.
- 0 M12 training permissions granted.
- No independent new M12 development split created.
- No sealed final holdout opened.

## Reproduce without GPU

From the repository checkout:

    python3 -I -S scripts/plex_m12_history_inventory.py

The script calls the original historical verifier first and refuses altered manifest, approval, source or split data. It prints only aggregate counts and rights-scope indicators, not record contents, model scores or training permissions. CI runs the same verifier on Windows and Ubuntu and deliberately corrupts a temporary copy to test rejection.

## How this can help M12

This historical material supplies provenance patterns, source-format examples and a known-good audit implementation. It does **not** establish M12 training approval or independently measured generalization. New M12 examples must meet the exact-copy task, include independently reviewed rights and grouped train/development separation, and preserve any prior evaluation restrictions.

The next useful progress is to audit actual owner-authored copy-task candidates against [the M12-A data gate](PLEX_M12_A_DATA_ADMISSION.md), not to rename or reuse prior validation examples to reach an arbitrary count. A later eight-update model experiment still requires **separate explicit owner approval** tied to dataset bytes, starting checkpoint and schedule.

[Original historical owner pilot](PLEX-600M-OWNER-AUTHORED-PILOT.md) · [M12 research design](PLEX_M11_FINAL_INVESTIGATION_M12_PLAN.md)
