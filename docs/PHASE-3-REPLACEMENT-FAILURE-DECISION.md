# Phase 3 — replacement failure decision after the 72-task diagnostic

**Status: fixed-endpoint evaluation complete; next controlled probe proposed, not authorized.** Plex Nano remains the 27,566,080-parameter coding model for requests with supplied HTML, CSS, or JavaScript. The prior P3-02 development gate remains failed at 0/36 exact edits. No model training or final-project-holdout access occurred in this diagnostic.

## Frozen evidence

The owner authorized continuation on October 17, 2026, immediately after the 72-task evaluation authorization decision. The checked-in [response envelope](../training/phase3/diagnostics/p3-03-source-preservation-v1/P3-03-FROZEN-RESPONSES.json) has SHA-256 `07c89354fafec181d337cd3b35bc8b471d865ef25bbc29c329e83db2583323a5`. It records runner revision `c33ab6a9be90884f417bf75ddd5d608f3c91963b`, fixed step-100 checkpoint SHA-256 `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`, tokenizer SHA-256 `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`, and P3-02 bundle manifest SHA-256 `0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354`.

The run used all 72 frozen tasks, greedy generation at seed 1337 and temperature 0 with a 192-token cap. Every response ended at EOS, after 16–62 generated tokens; all complete prompts remained visible. The [formal scorer report](../training/phase3/diagnostics/p3-03-source-preservation-v1/P3-03-SOURCE-PRESERVATION-DIAGNOSTIC.md) and [per-task JSON audit](../training/phase3/diagnostics/p3-03-source-preservation-v1/P3-03-SOURCE-PRESERVATION-DIAGNOSTIC-AUDIT.json) preserve the full results. The frozen manifest SHA-256 remained `119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578`.

| Measured result | Overall | HTML | CSS | JavaScript |
|---|---:|---:|---:|---:|
| Source-name roles all preserved | 0/72 | 0/24 | 0/24 | 0/24 |
| Requested target value correct | 0/72 | 0/24 | 0/24 | 0/24 |
| Unrelated bytes preserved | 0/72 | 0/24 | 0/24 | 0/24 |
| Exact edited file | 0/72 | 0/24 | 0/24 | 0/24 |
| Target boundary unavailable for unrelated-byte scoring | 68/72 | 20/24 | 24/24 | 24/24 |

No three-request contrast group was fully correct (0/24). Eleven groups returned one complete output for all three requests, four returned two distinct outputs, and nine returned three distinct outputs; none of that diversity yielded a correct edit. The outputs contained 30 distinct complete strings across 72 tasks. None exactly matched a P3-02 training target; this difference from the earlier 25/36 overlap is not evidence of improved editing because the task inputs changed.

**Interpretation limit:** The scorer counts missing or malformed target structures as value-selection failures. In 68/72 outputs it could not extract a target span; those records do not independently show whether Plex would have selected the requested value inside a valid target. Four HTML aria-label outputs had extractable targets with the wrong value, and those four also changed unrelated bytes. All 72 failed at least one supplied-name role, so source preservation failure is directly established even where byte-level preservation was unscorable. The result follows the predeclared **both fail** branch without establishing the model's internal cause.

## Selected next proposal: isolate copying from applying the change

Prepare a **new, smaller, read-only diagnostic** using the same fixed 27M checkpoint and supplied-file prompt contract. Freeze fresh files, names, requested values, tokenizer checks, complete-file overlap checks, scoring, and pass thresholds *before* generating any responses. Do not reuse the scored 36-task development files or the 72-task diagnostic as training examples.

Use one simple replacement family per language, two fresh source-name variants per family, and three distinct requested values. Compare three conditions:

1. **Copy control (6 tasks):** return each supplied file unchanged. Score exact byte copying and source-name retention. A failed copy control makes the preservation weakness visible without requiring value selection.
2. **Minimal edit (18 tasks):** replace one target literal in a minimal supplied file with no unrelated named content. Score target structure, requested value, and exact file. This lowers the copying burden while retaining the request-to-code task.
3. **Context edit (18 tasks):** make the same requested replacements in files with an unrelated identifier, comment, or declaration that must survive. Score target structure, value, source names, and every byte outside the target span.

Interpret these controls in order: copy failure selects a source-copying proposal; copy success with minimal-edit failure selects a target-formation/request-value proposal; success on those controls with context-edit failure selects an unrelated-content preservation proposal. If all pass, compare against the failed P3-02 and 72-task results and evaluate on fresh independent replacements before advancing. Report every language and family separately. A no-change request may itself be unfamiliar to Plex, so copy-control failure is evidence about this output contract, not proof of an internal copy mechanism defect.

This is a proposal for review. It does not authorize another evaluation, optimizer updates, checkpoint selection, a larger model, or opening the final project holdout. The formal roadmap's P3-03 Insert and Delete milestone remains behind the replacement decision.
