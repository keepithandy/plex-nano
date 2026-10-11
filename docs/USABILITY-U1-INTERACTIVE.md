# U1.2 — Plex Nano interactive terminal (Windows)

Plex stays open for multiple **independent coding requests** using one
loaded checkpoint and tokenizer. This is inference only, not further training
or a general-purpose chatbot. Replies are raw model output; invalid code can
still occur.

## 1. Get the U1.2 branch

From your local Plex Nano repository in PowerShell:

~~~powershell
git fetch origin
git switch feature/u1-plex-interactive
~~~

This branch builds on the working U1.1 launcher.

## 2. Launch

Your uv sync has already succeeded. You can double-click **plex.cmd**, which
will ask for the existing trained checkpoint path and matching tokenizer.

Or run the known local P2-48 paths directly from the repository root:

~~~powershell
.\plex.ps1 -Checkpoint ".\training\artifacts\request-grounded\p2-48-first-run\checkpoints\step-0100.pt" -BundleDir ".\training\artifacts\request-grounded\p2-48-training-bundle"
~~~

Plex will show a prompt similar to:

~~~text
Plex Nano | interactive terminal (U1.2)
Loaded trained step 100 on cuda.
Type /help for commands or /exit to quit.

You >
~~~

This is an illustrative prompt, not a verified U1.2 Windows test result.

## 3. Send multiple prompts

~~~text
You > Write a JavaScript function that adds two numbers.
Plex >
[raw output produced by your checkpoint]

You > Write CSS that makes a button navy.
Plex >
[raw output produced by your checkpoint]
~~~

The original model is unchanged. Do not assume the answer will be correct.
Each prompt is **independent**; responses are not carried into the next
request. Commands do not add conversational memory.

## 4. Optional: provide one file

Create a small .html, .htm, .css or .js file and try:

~~~text
You > /file C:\path\to\button.html
You > Change the button text to Save Changes.
You > /clear
You > /new
You > /exit
~~~

Plex reads the specified local file only when /file is entered, and uses
the existing P3-01 request + file format for the model prompt. The model
**never edits or saves the original file**. If you change the file outside
Plex, run /file again to refresh it.

Files must be up to 4096 bytes, the complete prompt no more than 4096
UTF-8 bytes, and the model input no longer than 512 tokens. Shorten the
file or request if the context is too large.

## Commands

| Command | Meaning |
|---|---|
| /help | Show commands |
| /file PATH | Supply one read-only HTML/CSS/JS file |
| /clear | Stop using the selected file |
| /new | Start fresh and clear any selected file |
| /exit | Quit Plex |

The original single-prompt JSON interface is still available:

~~~powershell
.\plex.ps1 -Once
~~~

Or use the -Prompt "your request" parameter. Add -Device cpu if CUDA
is unavailable.

## Limits and privacy

- No new model, retraining, automatic checkpoint choice, or optimization.
- No automatic input/output logging or feedback capture.
- No chat memory or repository discovery.
- Model output may be wrong, partial, invalid or repetitive.
- There is no automatic source-file editing.
- Diagnostics, guided practice and feedback logs belong to U1.3-U1.5.

A **real Windows smoke test is still needed** with the already tested
P2-48 checkpoint. Automated tests use a mocked generation backend, not
real trained weights.
