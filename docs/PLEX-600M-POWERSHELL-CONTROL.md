# Plex 600M from Windows PowerShell (WSL2 CUDA engine)

This is a convenience wrapper for the existing **bounded, guarded** M7/M9 Python tools.
It does not install CUDA, relax source approvals, change training math, or authorize
more optimization steps. It exists because Windows Application Control blocked
the owner's Windows PyTorch CUDA DLLs, while WSL2 Ubuntu can train with the
RTX 4080 SUPER. Run the launcher in **PowerShell**, not the Ubuntu shell.

## Current local installation assumed

- Ubuntu distribution: `Ubuntu`, Linux project: `/home/quali/plex-nano`
- uv executable: `/home/quali/.local/bin/uv`
- Exact tokenizer bundle: `training/artifacts/plex600-fresh-tokenizer`
- Verified fresh four-step M7 checkpoint: `training/artifacts/plex600-m7-fresh-bootstrap`
- Proposed eight-step M9 output: `training/artifacts/plex600-m9-r3-first-eight`
- Windows backup root via WSL: `/mnt/c/Users/quali/Plex600M-Backups`

The WSL distribution/project/uv executable can be customized using
`PLEX_WSL_DISTRO`, `PLEX_WSL_PROJECT`, and `PLEX_WSL_UV` Windows environment
variables. This first version targets the known owner experiment paths; it is
not a generic trainer for arbitrary datasets, models, or checkpoints.

## One-time PowerShell setup

After the PR is merged, from **Ubuntu** run `cd ~/plex-nano && git pull --ff-only origin master`.
Then from **Windows PowerShell**, copy the launcher to a local Windows file
and register a convenience function in the current session:

~~~powershell
$source = '\\wsl.localhost\Ubuntu\home\quali\plex-nano\scripts\Plex-Control.ps1'
$target = Join-Path $env:LOCALAPPDATA 'Plex600M\Plex-Control.ps1'
New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
Copy-Item -LiteralPath $source -Destination $target
function global:plex { & $target @args }
plex help
plex status
~~~

The function exists only in the current PowerShell session unless added to
your PowerShell profile. To make it persistent, add this line to the profile
referencing the installed local `$env:LOCALAPPDATA` file:

~~~powershell
function global:plex { & "$env:LOCALAPPDATA\Plex600M\Plex-Control.ps1" @args }
~~~

PowerShell execution policy or organizational App Control may still restrict
script execution; the launcher must not be used to bypass that policy. If
blocked, follow your system's authorized script execution procedure.

## Day-to-day commands

~~~powershell
plex status
plex review m9
plex approve m9
plex train m9
plex backup m9
plex evaluate m7
~~~

`plex review m9` displays all 8 selected training edits and 3 validation
edits verbatim. `plex approve m9` requires the owner to personally affirm
review with `APPROVE M9 8`, then independently verifies the existing M7
checkpoint SHA and generates a **create-only** exact-source approval document.
If an approved file already exists, the command refuses to overwrite it.
It never treats preflight success or candidate validity as approval.

`plex train m9` calls the existing `plex600_m9 --execute-approved-run` with
the approved local file, pinned tokenizer bundle, and **original M7**
weights/report; all of the runner's authorization, SHA, GPU capacity,
checkpoint destination, and update-limit checks remain unchanged.
There is no option for automatic or unbounded training.

`plex backup m9` creates a new destination in the Windows backup directory,
hashes source and copied weights/report, and refuses existing destinations.
`plex backup m7` does the same in a separate `m7` directory. It does
not replace the owner's existing legacy backup files.

`plex evaluate m7` runs the read-only M8 36-case developer generation on
the original M7 checkpoint with the fitted-vocabulary 96-token decoder.
Evaluation output is create-only. **There is no `plex evaluate m9` yet:**
the current M8 inference runner binds its weights SHA to the M7 report.
A verified M9-versus-M7 generated-code comparison requires a separately
reviewed evaluator; the launcher explicitly refuses to misrepresent the
M7-only evaluation as an M9 result.

The launcher invokes `wsl.exe`, never Windows Python. NumPy warnings from
PyTorch are non-fatal for the existing experiments.

## Important limits

- The original M7 checkpoint is weights-only: M9 creates a new AdamW state.
- Training targets only full edited-file/EOS positions; source/request prompt
  positions are masked out. M9 uses at most 8 updates/224 input positions.
- M9 candidates remain unapproved until exact owner review of the selected
  rows. The final project holdout remains sealed.
- The launcher shows useful status from report files but does not claim to
  rehash a checkpoint when printing `plex status`. Training and `plex approve`
  perform their own stronger checks.
- New output directories refuse overwrite. To run a new experiment version,
  change the documented and reviewed protocol rather than recycling an
  existing M9 destination.
