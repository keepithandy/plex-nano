# Plex Nano — Windows launcher (U1.1 + U1.2)

The original U1.1 launcher offered one saved-model completion. U1.2 now
opens a terminal where you can send multiple independent coding requests
without reloading the weights. This is **inference only**; no training or
file modification occurs.

## First-time setup

In PowerShell at the repository root, install the environment once:

~~~powershell
uv sync --project training
~~~

The trained checkpoints and tokenizer bundles are stored under the
gitignored `training/artifacts` directory on your own computer. GitHub does
not include or download the trained weights.

## Start Plex without copying any paths

Double-click **plex.cmd**, or run this from the repository terminal:

~~~powershell
.\plex.cmd
~~~

The launcher automatically selects **the previously verified P2-48 step-100
checkpoint** and its matching tokenizer, relative to the repository:

- `training/artifacts/request-grounded/p2-48-first-run/checkpoints/step-0100.pt`
- `training/artifacts/request-grounded/p2-48-training-bundle`

The default checkpoint's SHA-256 must match the recorded
`fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7`.
This prevents silently loading an unexpected model at that path.
A missing or different checkpoint produces a clear error; the launcher
does not search for, create, or download a substitute model.

When the files are present and verified, the terminal will reach
`You >` without asking for either path.

## Optional: run a different model or one request

The PowerShell entry point supports explicit overrides:

~~~powershell
.\plex.cmd
~~~

For another checkpoint and its matching tokenizer, use the PowerShell
script with the paths (if PowerShell's execution policy permits it):

~~~powershell
.\plex.ps1 -Checkpoint "PATH\TO\OTHER_TRAINED.pt" -BundleDir "PATH\TO\MATCHING_TOKENIZER"
~~~

If PowerShell blocks .ps1 files, use a process-scoped invocation:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\plex.ps1" -Checkpoint "PATH\TO\OTHER_TRAINED.pt" -BundleDir "PATH\TO\MATCHING_TOKENIZER"
~~~

To keep the older one-shot JSON completion behavior:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\plex.ps1" -Once
~~~

You can also supply `-Prompt "your request"` for a single completion,
or `-Device cpu` if CUDA is not available.

See [U1.2 interactive guide](USABILITY-U1-INTERACTIVE.md) for commands
such as `/help`, `/file`, `/clear`, `/new` and `/exit`.

## Troubleshooting

- uv unavailable: install uv, then run `uv sync --project training`.
- Environment missing: run `uv sync --project training`.
- Default checkpoint missing: verify the file exists in
  `training/artifacts/request-grounded/p2-48-first-run/checkpoints`.
- Default checkpoint SHA mismatch: stop and inspect it; do not pretend
  the weights are the recorded P2-48 trained model.
- Tokenizer bundle missing or incompatible: supply the exact matching
  P2-48 bundle.
- CUDA unavailable: use `-Device cpu`.
- Poor code output: this is a limitation of the trained model, not the
  launcher. The UI does not correct or modify generated text.

The previous U1.1 single-prompt inference path has already worked on the
owner's Windows system. This no-path U1.2 launch change still requires
a local Windows smoke test.
