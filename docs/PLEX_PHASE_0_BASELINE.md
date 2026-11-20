# Phase 0 · Local assistant baseline

## Reproduce the console example

From PowerShell with Git and Python 3.12, use a fresh directory. The pinned revision contains both the console and smoke test.

```powershell
git clone https://github.com/keepithandy/plex-nano.git plex-phase0
cd plex-phase0
git switch --detach 74628dbf70f78ababc6b4af6cfa33682f9ae7ae6
py -3.12 -I -S scripts/plex_phase0_smoke.py
py -3.12 scripts/PlexLocal.py
```

Enter the following lines at the console:

```text
paste
let value = 1;
console.log("value", value);
.end
rename value to count
diff
quit
```

The built-in editor changes the declaration and the reference in the log call, preserving the string `"value"`. The smoke test drives the console through standard input, checks the unified diff and compares the saved copy byte for byte. It passed on Windows with isolated Python 3.12.10. The final smoke also enters `Return the supplied JavaScript line exactly as written.` for each of the four M11 lines and checks the displayed result, built-in attribution, and saved bytes. All five editor cases passed from a clean installation of the two committed scripts under isolated Windows Python 3.12.10.

## Raw checkpoint evidence

This is a transcription of the owner's completed repair1 run. Local weight and tokenizer bytes were unavailable for independent hashing during this packaging pass.

| Artifact | Recorded identity |
| --- | --- |
| Worker commit | `89998f5f6301d8bec1401047e0a2904fc372dd47` |
| M11 step-200 weights SHA-256 | `d7a7776de86230320f6e0e48b07dc76761c3d42ac80e4c330839e47c9ffc93dd` |
| Tokenizer | `plex600-fresh-tokenizer`, reported tokenizerSha256 `9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2` |

| Step | Bidirectional digit preference | Development exact | Training exact |
| ---: | --- | --- | --- |
| 25 | Fail | Fail | Fail |
| 50 | Fail | Fail | Fail |
| 100 | Pass | Fail | Pass |
| 200 | Pass | Fail | Pass |

The two decoder columns freeze eight final generations. Each code cell represents one line followed by LF. Each generation stopped at EOS.

| Split | Supplied line and expected copy | unmasked-v1 | fitted-vocab-v1 |
| --- | --- | --- | --- |
| Training | `let score = 0;` | `let score = 0;` | `let score = 0;` |
| Training | `let score = 1;` | `let score = 1;` | `let score = 1;` |
| Development | `let value = 0;` | `let score = 0;` | `let score = 0;` |
| Development | `let value = 1;` | `let score = 1;` | `let score = 1;` |

## Windows console inference

Before the exact-copy routing fix, the owner ran the console from Windows PowerShell with Ubuntu WSL and supplied the following transcript.

```text
plex> paste
Paste your code. End with .end on its own line.
let value = 1;
.end
let value = 1;
plex> Return the supplied JavaScript line exactly as written.
Generating locally with Plex 600M M11 step 200...
Plex 600M returned no code (stop: eos).
```

The console completed the model call and identified the checkpoint. The copy case failed because the console received no non-whitespace code. The console transcript does not include token counts or raw output bytes.

The console sent the request through the general `Request:` / `Source file:` / `Edited file:` prompt. The historical M11 probes used a fixed copy instruction, a `Code:` section, and a final `Copy: ` delimiter with a trailing space. Whether that prompt difference explains the result remains untested on this installation.

## Model path and release status

On the owner's Windows setup, a request outside the built-in editor invokes the Ubuntu WSL checkout at `/home/quali/plex-nano` through its existing uv training environment. It requires CUDA, the tokenizer in `training/artifacts/plex600-fresh-tokenizer` and the existing weights in `training/artifacts/plex600-m11-eos1-f68483d42b-repair1/step-200/weights.pt`. The console names M11 step 200 and checks the weights digest before loading. Its active decoder is `fitted-vocab-v1` with a 96-token limit.

Worker repair [PR #218](https://github.com/keepithandy/plex-nano/pull/218) passed the Linux and Windows training test workflows and merged to master. The recorded worker commit identifies the implementation used for the original measurements.

## Phase 0 closeout

The clean-install smoke reproduced the documented edit. The owner's Windows-to-WSL transcript identifies M11 step 200 and records the model's EOS response. The pinned console also prints the verified weights SHA-256 when inference returns. These checks satisfy the roadmap's Phase 0 exit gate.

For an inference-only comparison on the owner's installation, run `py -3.12 scripts/PlexLocal.py --compare-copy-prompts` from the pinned checkout. This loads the checkpoint once and runs each frozen source line with both prompt formats using the fitted-vocabulary decoder. Each JSON line includes the decoded text, generated token count, stop reason, checkpoint digest, and exact-copy result. The historical generations and console transcript above are the measured GPU evidence. A fresh paired GPU run is still outstanding.
