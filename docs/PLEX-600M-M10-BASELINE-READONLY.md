# M10 — read-only M9 exact-copy baseline (stacked after PR #205)

**No training, approval or checkpoint writes.** This stage adds a distinct
read-only evaluator for M9 before the proposed M10 optimizer experiment. It is
stacked on PR #205 and must not be merged ahead of that review.

## Why

M7 generated 96 periods and M9 generated 96 spaces on the 36-case edit
development evaluation; neither solved an edit. M10 proposes 12 train
candidates, six heldout candidates and a 10/25/50-copy experiment.
**Before using any more training updates**, gather M9's autonomous copying
baseline on exactly those proposed prompts and the two original decoding
policies. This is a source-conditioned copying diagnostic, *not* a coding
benchmark, not proof of an improved model and not authorization to train.

## Safety/provenance

- Default CLI is a plan: no files, PyTorch, GPU, or model loaded.
- `--audit-inputs` is CPU-only and requires the independent copied M9
  `report.json` + `weights.pt` pair; the original M9 output is rejected.
  Its bytes, report and 8-update source/approval history are SHA verified
  using PR #205's backup audit. The original M7 source-bound tokenizer is
  independently verified, exactly round-tripped, and all selected prompts
  reserve the 96-token generation capacity in a 1,024-position context.
- `--run-read-only-baseline` is an *explicit opt-in to GPU inference*.
  It checks input identities and ensures the selected output directory does
  not exist **before** model loading, then validates the clean, pinned model
  source and RTX 4080 SUPER/native BF16 availability. Weights are loaded
  with `weights_only=True`, accepting native `dict`/`OrderedDict`,
  then applied with strict `load_state_dict`. Only `eval()` and
  `inference_mode()` are used. There is no AdamW, backward pass, optimizer,
  training dataset loader, or checkpoint save operation in this runner.
- Read-only output is an atomic, create-only directory under
  `training/artifacts` containing a response JSONL and a report JSON.
  Every response carries the **M9 checkpoint SHA**, input-prompt SHA,
  case ID, split, decoder policy, raw output and EOS/stop reason.
- Default selects **three cases**, one per language including
  `m10-js-counter` (the proposed single-example warmup). A full, explicitly
  requested 18-case run includes 12 train candidates and six heldout
  candidates, each under both `unmasked-v1` and `fitted-vocab-v1`
  policies (36 prompt/policy generations). No other benchmark/final
  holdout is opened.
- Generates up to 96 tokens per prompt/policy and checks elapsed time
  between inference cases; the 30-minute guard is **cooperative, not a hard
  wall-clock kill for a stuck individual forward pass**. It does not claim
  hard real-time enforcement.
- Reports exact matches, exact + EOS, character similarity, repetitions,
  whitespace/period collapses and split summaries. Partial runs never claim
  18-case completeness or an early-stopping condition.
- The pure CPU scorer accepts *caller-supplied predictions*, so its
  independent result correctly says **not verified model generation**;
  only the explicit GPU runner may set `readOnlyGenerationExecuted: true`.

## One-command Windows helper

The already-installed `Plex-Control.ps1` defines the default verified M7
tokenizer bundle and the backup destination used by `plex backup m9`:
`/mnt/c/Users/quali/Plex600M-Backups/m9`. The standalone helper reuses
these known paths without copying a backup or altering the installed
launcher. It does not guess missing files. It can be invoked once the
`feature/plex600-m10-readonly-baseline` branch is present in the local
checkout; do not discard local changes just to switch branches.

Audit only (**default**, CPU only, never starts GPU):

~~~powershell
.\scripts\Invoke-PlexM10Baseline.ps1
~~~

Explicit **read-only** three-case GPU baseline (both decoders):

~~~powershell
.\scripts\Invoke-PlexM10Baseline.ps1 -Run -Cases 3
~~~

Explicit full 18-case read-only baseline:

~~~powershell
.\scripts\Invoke-PlexM10Baseline.ps1 -Run -Cases 18 -Decoder both
~~~

Each `-Run` invocation first performs the CPU audit, then creates a
new timestamped destination under `training/artifacts` and prints the
result's paths. Failures leave the historical M7/M9 checkpoints unchanged.
If the backup is absent, the helper fails closed; the existing separate
`plex backup m9` command is the supported create-only backup workflow.
`-BackupDirectory` can override the default with an explicitly verified
independent WSL backup path. No branch switching, training authorization,
optimizer step, or backup creation is implicit.

## Usage after reviewing the branch locally

The following commands are **examples**. Replace the three path placeholders
with *verified locations* of the existing frozen tokenizer bundle and the
separately backed-up M9 checkpoint/report. These paths are intentionally
not guessed.

Read-only plan (does not access the project files):

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_baseline
~~~

CPU-only identity and tokenizer/preparation check:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_baseline --audit-inputs --bundle-dir "/path/to/verified/frozen-m7-bundle" --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55 --m9-report "/path/to/independent-m9-backup/report.json" --m9-weights "/path/to/independent-m9-backup/weights.pt"
~~~

Explicit **read-only** GPU baseline, three cases across both decoders:

~~~powershell
wsl.exe -d Ubuntu --cd /home/quali/plex-nano -- /home/quali/.local/bin/uv run --project training --no-sync python -m plex_training.plex600_m10_baseline --run-read-only-baseline --bundle-dir "/path/to/verified/frozen-m7-bundle" --bundle-manifest-sha256 14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55 --m9-report "/path/to/independent-m9-backup/report.json" --m9-weights "/path/to/independent-m9-backup/weights.pt" --max-cases 3 --decoding both --output-dir "training/artifacts/plex600-m10-baseline-three-review"
~~~

To evaluate all 18 proposed cases, specify `--max-cases 18` and a **different,
nonexistent** output directory. Do not rerun into an existing output path.

## What this does not establish

An M9 copying baseline cannot show that M10 training worked: no M10 update
is performed. It also cannot certify the model will improve at file editing,
even if it copies perfect short snippets. The training proposal and owner
review requirements in PR #205 remain unchanged.

The later, separately approved experiment still needs a bounded training
worker, authorization tied to the exact M9 backup/candidate SHA, checked
mechanical training measurements, stop gates and create-only intermediate
checkpoints. **Do not treat this PR or successful read-only inference as
training authorization.**
