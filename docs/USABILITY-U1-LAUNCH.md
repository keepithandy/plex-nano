# U1.1 - Launch Plex on Windows

This is **step 1 only**: a simpler way to run an existing Plex Nano
checkpoint. It wraps the existing `plex_training.cli complete` command.
It does not implement a chat session, add capabilities, modify a model, or
launch training.

## First-time setup

Open PowerShell in your local Plex Nano repository and install the locked
training environment (if you have not already):

~~~powershell
uv sync --project training
~~~

Plex's training checkpoints and matching tokenizer bundles live in
gitignored local artifact folders. The repo **does not** ship trained weights.
If you do not have a checkpoint, this launcher cannot generate output yet.
Use the trained checkpoint associated with your experiments, not a step-zero
initialization or staging checkpoint. The P2-48 final checkpoint recorded
SHA-256 `fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7`;
this is an identity reference, not a download link or auto-selection rule.

## Easy launch

Double-click **plex.cmd** in the repository folder.

**U1.2 update:** the launcher now starts an interactive session by default.
See [U1.2 terminal guide](USABILITY-U1-INTERACTIVE.md). For one request,
use `.\plex.ps1 -Once` or supply the `-Prompt` argument.

The launcher asks you for:
1. The path to your **trained** `.pt` checkpoint.
2. The matching tokenizer **folder** (unless a folder named `tokenizer`
   exists directly beside the checkpoint).
3. One short HTML, CSS, or JavaScript coding request.

It uses the existing trained model to produce **one real completion** and
prints the existing command's JSON result, including the `completion` field.
It does not save or edit any source files.

To skip the prompts, launch from PowerShell (replace placeholders with real
local paths):

~~~powershell
.\plex.ps1 -Checkpoint "PATH\TO\TRAINED_CHECKPOINT.pt" -BundleDir "PATH\TO\MATCHING_TOKENIZER_BUNDLE" -Prompt "Write a JavaScript function that adds two numbers."
~~~

Add `-Device cpu` if CUDA is unavailable. The default `-Device auto`
chooses CUDA when accessible or otherwise CPU.

## Troubleshooting

- **uv missing**: install uv and run `uv sync --project training`.
- **Environment missing**: run `uv sync --project training`.
- **Checkpoint not found**: verify its exact local path; the launcher never
  invents or downloads a model.
- **Tokenizer mismatch**: select the exact bundle that matches the checkpoint.
- **Step zero / not trained**: supply trained weights, not initialization.
- **CUDA error**: retry with `-Device cpu`.
- **Prompt context error**: shorten the request.

**U1.2 update:** interactive terminal support is now implemented. This
single-request mode remains available with -Prompt or -Once. Model quality,
training and checkpoint weights are unchanged. A real Windows smoke test
is still required for U1.2.
