# Plex M11 → M12 Research Gate

**November 20, 2026 | Proposal and evidence review only | Training PAUSED**

## What we know

Plex 600M M11 step-200 is the scratch-trained reference checkpoint. The owner's local inference reports show:

| Experiment | Raw M11 model result |
| --- | --- |
| Original M11 prompt, familiar training pair | **2/2** exact with EOS |
| Original M11 prompt, unseen value pair | **0/2** exact with EOS |
| Original M11 prompt with ending space changed to newline | **0/4** exact; **4/4 immediate EOS** |
| General console prompt | **0/4** exact; **4/4 immediate EOS** |
| Phase 2 baseline | **0/20** edits, **0/13** refusals, **33/33 empty EOS** |
| Phase 2 alternative | **0/20** edits, **0/13** refusals, **33/33 empty EOS** |
| Built-in Python editor | **20/20** edits and **13/13** correct refusals on development cases |

These are owner-reported real GPU measurements, not GitHub CI model results. The built-in score is NOT a neural model score. The 33-case Phase 2 set is a development benchmark, not a final holdout.

Original local evidence under training/artifacts:

- plex600-phase2-m11-20261120-230232-36012
- plex600-phase2-eos-20261121-021456-18716

The EOS run recorded first-token logits. The console summary alone does **not** reveal their numeric magnitudes. Avoid claims about the precise EOS margins until inspecting the saved rows.

## Final M11 investigation — short and bounded

1. **Inspect existing reports only.** Run scripts/plex_m11_final_evidence.py to verify response SHA-256, experiment identity, all required cases, EOS output flags, and receipt-level checkpoint/tokenizer identity.
2. **Review measured first-token values.** Inspect EOS-first rankings, EOS-minus-best-code margins, and gold-first-token-minus-EOS margins for familiar, changed-delimiter, and console prompts. Keep original raw output alongside summaries.
3. **Classify the capability precisely.** Familiar training-copy success is established; unseen-copy, prompt-transfer, and natural-language edit success are NOT established.
4. **Stop M11 research scope.** Preserve weights, tokenizer and logs. Do not train, overwrite evidence, mask EOS to inflate apparent results, or access a final holdout.

### Read-only local command

From WSL after synchronizing the Git checkout:

    python3 -I -S scripts/plex_m11_final_evidence.py \
      --phase2-dir training/artifacts/plex600-phase2-m11-20261120-230232-36012 \
      --eos-dir training/artifacts/plex600-phase2-eos-20261121-021456-18716

This script reads receipts and response bytes. It **does not load or rehash the model's checkpoint weights**; weights were checked during original inference. If local files are missing or altered, report evidence unavailable; do not substitute assumptions.

## M12 — Proposed, NOT authorized

### Narrow learning goal

**Copy unfamiliar user-supplied JavaScript exactly, preserve all identifiers/literals, and stop at the correct EOS.** Only after that is demonstrated should a separate edit-instruction task be trained. Reducing training loss or making the model emit more tokens is not sufficient.

### Starting model, compute and data

Retain the original 600M architecture and M11 checkpoint as an immutable reference. Do not import pretrained weights, expand model size, or resume M11 training automatically. Compare available owner-verified M9 backup and M11 step-200 **read-only on the same development cases** before selecting a starting point. If exact starting checkpoint identity cannot be verified, refuse training admission.

Proposed candidate pilot: **256 distinct owner-authored training cases, 64 development cases, plus 64 separately locked future holdout cases.** This is a requested data specification, not a claim that those examples exist or are authorized.

All sources require owner-approved rights/provenance, exact source/expected-file hashes, deduplication, and grouped splitting by identifier family, source template and instruction format. Development identifiers must be unseen in training. Cover integers, declarations, identifiers, comments, strings, spacing, and LF/CRLF using initially simple, valid JavaScript. Assistant-authored M10/M11 proposals must never silently become approved training data. The protected holdout stays inaccessible during exploratory work.

Keep the existing verified M11 tokenizer initially, and check encode/decode roundtrips, identifier tokenization and exact source copying. Changing tokenizer is a separate experiment, not an implicit fix.

A pilot with these modest data counts would measure whether the pipeline learns **narrow transfer**. It would not establish general coding competence.

### Predeclared development metrics — proposed research gates

| Metric | Proposed minimum |
| --- | --- |
| Exact full-file output **and EOS** on 64 unseen development cases | **48/64** |
| Exact copied unseen identifier, on subset of at least 32 cases | **24/32** |
| Changed-delimiter/paraphrase subset, at least 16 cases | **12/16** |
| Premature first-token EOS on the nonempty development targets | At most **3/64** |
| Unrelated modifications | Zero accepted collateral changes, measured later on editing tasks |

Record training and development performance **separately**, all denominators, token-level source-conditioning preference, prompt bytes, checkpoint SHA, tokenizer SHA, failures, and latency. These are pilot gates, not release criteria. The final holdout requires separate evaluation admission.

### Future phases (all require separate approval)

**M12-A: Design and data admission.** Validate owner-authored data rights, grouped splits, source hashes, tokenizer roundtrips, read-only baseline outputs, gold-prefix token diagnostics, memory feasibility on the 16 GB RTX 4080 SUPER, and a bounded optimizer update/token budget. No training entry point is needed for this stage.

**Implementation progress:** [M12-A dataset structure auditor](PLEX_M12_A_DATA_ADMISSION.md) checks two owner-supplied dataset splits, copy-target SHA-256, disjoint identifier/source/prompt groups, future holdout isolation, and optional frozen M11 tokenizer roundtrips. GitHub CI uses only generated synthetic test records. No approved real M12 corpus has been ingested or created. Rights and actual data provenance remain owner-side review work; model baselines and GPU admission remain pending.

**Source-conditioning measurement:** The [M12-A gold-prefix source-contrast probe](PLEX_M12_A_SOURCE_CONDITIONING.md) is implemented for the four frozen M11 development examples. It measures target-token ranks, premature EOS under correct prefixes and whether score/value preferences switch when the source changes. It has **not been run against local M11 weights** in this research change. CI validates only synthetic mathematics and safety behavior. No model improvement is claimed.

**M12-B: Eight-update safety pilot.** Only after specific owner approval tied to exact executable revision, dataset hash, checkpoint identity, optimizer configuration and schedule. Measure finite gradients, weight movement, checkpoint reload parity and no protected holdout access.

**M12-C: Fifty-update learning pilot.** New approval required; evaluate at updates 10, 25 and 50; stop on divergence, bad EOS behavior, provenance failure or unseen-source regression. No automatic extension.

**M12-D: Longer curriculum proposal.** Consider only if unseen copying genuinely improves on verified M12-C development cases. Do not award progress for memorizing training lines.

### Approval boundary

**This research document does NOT authorize training.** No optimizer, training run, checkpoint write, changed tokenizer, or final holdout read is approved. Each new training stage requires an explicit owner decision after the data, starting weights and measured evidence are reviewed.

## Related research

- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022): balance model capacity with enough training data, instead of increasing parameters alone. https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html
- Eldan and Li, *TinyStories* (2023): narrow, coherent task distributions can help probe small-model learning; domain-specific success is not general coding ability. https://arxiv.org/abs/2305.07759
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2020): decoding can change model output, but output changes alone cannot demonstrate learned source conditioning. https://arxiv.org/abs/1904.09751

**Next decision:** review the existing M11 first-token evidence and the M12 data/evaluation proposal. Training stays paused until that review and separate authorization are complete.
