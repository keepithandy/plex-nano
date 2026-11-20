# Phase 0 · Local assistant baseline

## Reproduce the console example

From PowerShell with Git and Python 3.12, use a fresh directory. The pinned revision contains both the console and smoke test.

```powershell
git clone https://github.com/keepithandy/plex-nano.git plex-phase0
cd plex-phase0
git switch --detach b685937f87770ba4cb051b8f3de6502af28c8f61
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

The built-in editor changes the declaration and the reference in the log call, preserving the string `"value"`. The smoke test drives the console through standard input, checks the unified diff and compares the saved copy byte for byte. It passed on Windows with isolated Python 3.12.10. This case scores the Python editor.

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

The owner ran the console from Windows PowerShell with Ubuntu WSL and supplied the following transcript.

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

The repaired worker remains in [draft PR #218](https://github.com/keepithandy/plex-nano/pull/218). Its Git blob differs from the worker on master at the time of this check. Refer to the pinned worker commit for historical reproduction. The console's WSL inference path has now run. Release integration of the repaired worker remains open.
