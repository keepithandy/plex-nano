# Plex 600M input integrity — implementation plan

**Status: implemented; 31 focused tests passed locally on Windows.** Remote
Linux/Windows CI verification remains pending. Tests use synthetic temporary
datasets, not research datasets or checkpoints.

Before publication, the full portable required suite passed **713 tests** on
Windows using local temporary storage. Earlier attempts in the synced workspace
hit sandbox hardlink restrictions and intermittent directory-rename denials;
the unrestricted run with fixtures outside OneDrive passed. The existing
optional NumPy and tiny CUDA-test context warnings were reported.

The former sampler reopened verified token paths as memory maps. It now reads
an immutable byte snapshot and verifies its hash against the checked bundle's
split hash before sampling. Replacement, truncation or modification before
capture causes rejection; later file changes cannot change the private bytes
sampled. Each selected split is limited to **256 MiB**; larger splits fail
before reading. This trades memory for a portable guarantee and is not yet a
large-corpus streaming solution. Temporary reads and source-check buffers add
memory overhead beyond the retained snapshot.

The new read-only `source-check` command requires an explicitly supplied source
dataset and independently supplied bundle/source manifest SHA-256 values. It
compares ordered record IDs and exact encoded text plus EOS against the verified
index and immutable packed-token snapshot for train and validation. It checks
source row content hashes, source/group references and the complete split
hashes/counts. It does not fit a tokenizer or certify rights, quality, split
independence or globally unseen content. Hashes obtained from compromised inputs
cannot independently establish authenticity; retain the expected identities in
an independently reviewed record.

Synthetic regression cases cover path replacement, same-size in-place changes,
truncation, post-capture modification/deletion, snapshot size limits, rehashed
index-ID and token alterations, source-order changes, mismatched external hashes,
and a correctly paired bundle and CLI command. Closing the sampler releases its
snapshot; no file mappings or handles are retained. Nano's corpus behavior and
recorded experimental artifacts were not changed.

From the repository root (use reviewed expected hashes, not placeholder text):

```powershell
& .\training\.venv\Scripts\python.exe -m plex_training.plex600_setup source-check --bundle-dir PATH_TO_BUNDLE --dataset-dir PATH_TO_SOURCE --bundle-manifest-sha256 EXPECTED_BUNDLE_SHA --source-manifest-sha256 EXPECTED_SOURCE_SHA
```

No real dataset source-check or model experiment was run for this change. The
new source check is explicit; constructing a sampler verifies bundle integrity
but does not automatically certify its relationship to an external source.
