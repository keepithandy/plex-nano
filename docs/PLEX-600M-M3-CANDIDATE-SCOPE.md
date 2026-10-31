# Plex 600M — first bounded M3 candidate scope

**October 31, 2026. Status: review-only; M3 is not closed.**

The untrained 600M scratch model now has a machine-checkable **candidate
source/split policy**, rather than only a prose proposal. The decision is
conservative: the exact historical Microsoft, MDN (all four related repositories
grouped together), and JavaScript30 training subsets are conditional candidates.
Traversy and Bulma remain **historical regression only**, not independent
600M development evaluation.

## Candidate counts from the existing hash-pinned review

| Measure | Count |
| --- | ---: |
| Historical accepted records (train and validation) | 3,228 |
| Historical training records | 3,105 |
| Records proposed for embedded-library omission | 69 |
| Notice-flagged records, all quarantined pending classification | 101 |
| Records in both quarantine sets | 25 |
| Distinct quarantined records (all historical roles) | 145 |
| Distinct quarantined historical training records | 123 |
| **Remaining conditional training-review candidates** | **2,982** |

Quarantine is deliberately broader than the earlier 69-record omission
proposal. It is **not** a finding that any excluded material is illegal, or
that remaining material has been cleared for reuse. The zero flagged pairs
from the bounded 12-token shingle scan do not establish split independence.

The [review-only candidate policy](../training/pretraining/plex600-m3-candidate-scope-v1.json)
caps the proposed first materialization at **3,000 records and 16 MiB**.
Actual source bytes, quality and token mix have not yet been measured.
This is a stop budget, not an adequate data scale for a 600M model.

The [verifier](../training/src/plex_training/plex600_m3_scope.py) pins the
historical review and hold-record hashes, checks accepted paths, record identities,
source groups, historical train/validation roles, all notice quarantine and
the record budget. It accesses **metadata only** and produces no dataset.

Run from repository root:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m3_scope --check
uv run --project training --no-sync python -m pytest training/tests/test_plex600_m3_scope.py -q
```

## Open gates

Source-rights review, within-family/semantic overlap, actual bytes and
quality, independent authored development tasks (36-case design), and
explicit M4 materialization authorization are **not complete**.
No tokenizer fitting, checkpoint, optimizer, inference, or training has been
authorized or performed here. The final project holdout remains sealed.

The next engineering step is a bounded candidate materializer that independently
rehashes source inputs, applies the quarantine policy, records every skipped
record, and produces immutable split manifests. Implement and test it before
approving any model training. The fresh development evaluation must be
prepared and frozen separately, with no evaluation data fed to the tokenizer.
