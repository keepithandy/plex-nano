# Plex 600M — M4 bounded corpus preparation pipeline

**Status:** implemented as an offline candidate-data builder, **not** a prepared or
approved 600M training corpus. The M3 rights/source and independent evaluation
decisions are still open. No tokenizer fit, optimizer update, or model training.

## Pipeline and stop rules

The new [builder](../training/src/plex_training/plex600_m4.py) reuses the
hash-pinned M3 source review, quarantine list, and candidate policy. It:

1. Requires a separately recorded rights/preparation decision **before**
   reading historical JSONL source text. M3's review-only policy cannot
   authorize source reuse merely because the checks pass.
2. Rechecks each historical manifest SHA-256, split SHA-256, record content hash,
   ordered record-identity digest, accepted path set, family/split mapping and
   packaged license-notice hash. No upstream network requests occur.
3. Holds every M3 proposed omission and notice-flagged record, with additional
   exclusions for oversized (128 KiB) or invalid text, recognized secret patterns,
   and exact overlap with historical validation text. Emits reasons and identities
   for every skipped historical training record.
4. Sorts candidate training records deterministically, checks the **3,000-record**
   and **16 MiB complete-output** budgets and refuses to publish partial output.
5. Stages a new create-only output directory and publishes it as one directory:
   `train.jsonl`, `manifest.json`, `skipped-audit.json`,
   and pinned retained `licenses/*.txt`. Existing destinations cannot be overwritten.
6. Offers a read-only published-output verifier. Neither this verifier nor its
   original data source hashes establish legal clearance or independence.

**Important:** the historical Traversy/Bulma validation corpora are rechecked
but are **not** exported as a fresh validation set, nor offered for tokenizer
fitting. The M4 artifact deliberately uses a distinct
`plex600-m4-candidate-v1` manifest format; existing M5 tokenizer
commands reject it until a new independently reviewed train/validation
preparation and M5 compatibility step are completed. The final holdout
remains closed.

### Read-only candidate inspection

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m4 inspect
uv run --project training --no-sync python -m pytest training/tests/test_plex600_m4.py -q
```

### Controlled materialization (only after rights review)

Provide an independently reviewed local JSON file under your own path, with
kind `plex600-m4-rights-and-preparation-v1` and:

- `sourceReviewSha256`: SHA-256 from M3's pinned historical review
- `holdsSha256`: SHA-256 from M3's pinned proposed holds
- `scopeSha256`: SHA-256 of the committed M3 candidate policy
- `m4PreparationApproved`: true, explicitly approved for this scope
- `trainingAuthorized`: false
- `finalHoldoutAccessed`: false
- `sources`: one record for **each of six candidate source IDs**, with
  `sourceId`, `rightsStatus` set to `approved`,
  and nonempty `reviewer`, `evidence`,
  `approvedScope`. This is a signed-off source-use decision in
  concept; the tool does not independently verify reviewer authority.

No rights-approval file has been created here. Do not fabricate these values
to bypass the gate. The user request authorizes implementing M4 code, **not**
automatically declaring all historical source files rights-approved.

When that review exists, the command is:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m4 build --historical-datasets-root training/artifacts/datasets --reviewed-decision PATH_TO_REVIEWED_DECISION.json --output-dir training/artifacts/plex600-m4-candidate-v1
uv run --project training --no-sync python -m plex_training.plex600_m4 verify --dataset-dir training/artifacts/plex600-m4-candidate-v1
```

**Note:** choose an output directory **outside** the historical datasets root.
For example, use `training/artifacts/plex600-m4-candidate-v1`
instead of the example argument above. An output path inside the
historical inputs is refused. Both the input corpus and the rights decision
must be present locally.

## Verification, current limits, and remaining M4 gates

Synthetic tests cover deterministic bytes, publication, quarantine and audit,
changed historical hashes, approval refusal before historical data access,
secret filtering, output tampering and output-budget failure. CI runs them on
Linux and Windows; no real source texts are committed to the repository.

Open limits: the historical source audit is not an independent fresh upstream
verification; zero overlapping shingle flags do not prove split independence;
non-flagged text may still contain rights or secret issues; all six source-rights
decisions remain pending. Large-scale training data adequacy has **not** been
established by this modest candidate pool.

Next: finish and independently review M3 source rights, materialize and
independently inspect the bounded candidate output, freeze a genuinely distinct
development evaluation and suitable validation split, then adapt the approved
dataset format for a dedicated M5 tokenizer. Optimizer and sustained training
remain explicitly separate work.
