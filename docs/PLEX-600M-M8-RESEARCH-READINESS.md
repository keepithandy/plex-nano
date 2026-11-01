# Plex 600M: M8 research readiness and further training gates

**Date:** October 31, 2026. **Verdict:** prepared for evaluation and corpus review; **not authorized for another training run**.

| Deliverable | Current evidence | Restriction |
| --- | --- | --- |
| Measured M7 baseline | [Owner-reported result and local verification instructions](PLEX-600M-M7-OWNER-RESULT-2026-10-31.md) | Full local report/weights not supplied; checkpoint hashes unknown here |
| Independent M8 development evaluation | [36 manually authored edit tasks](../training/phase3/evaluation/plex600-m8-independent-v1.jsonl), 12 HTML / 12 CSS / 12 JavaScript | Development-only, never a training input or protected final holdout |
| M8 scoring and audit | [Read-only scorer](../training/src/plex_training/plex600_research.py) | Exact-file, target-string and unchanged-prefix/suffix metrics only; no runtime/syntax tests and no observed model answers yet |
| Larger R2 synthetic curriculum | [1,152 candidate code-edit examples](../training/phase3/candidates/plex600-r2-synthetic-edit-v1.jsonl), 384 per language, 18 template families | **Quarantined pending owner approval and deeper QA; 0 newly approved rows** |
| External sources | Original six-source M3 rights holds | No external-source rights approval, downloads or training expansion |

**Model inference is now implemented but not executed:** [M8 read-only generation runner](../training/src/plex_training/plex600_m8_eval.py) verifies the local checkpoint report and SHA-256 weights, the exact tokenizer bundle and source binding before any CUDA access. Its separate explicit read-only flag can produce a create-only raw JSONL response transcript plus provenance. It never updates weights, initializes an optimizer, opens the final holdout, or trains on evaluation records. It is deliberately limited to 1–36 tasks and 1–64 generated tokens per task; the default plan covers six tasks and 32 generated tokens.

The R2 rows originate from **18 project-authored edit templates**, 64 synthetic variations per family. These are not 1,152 independent source projects. The increase in record count is meaningful for pipeline testing, but does NOT supply a realistic 600M scratch-model pretraining corpus. Template redundancy, shortcuts and overfitting remain risks. The auditor checks uniqueness, exact replacement integrity, flags, distributions, and exact text overlap with M8 development cases; it **does not prove semantic non-contamination or legal clearance**.

Use the read-only tools from the repository root:

~~~powershell
git pull --ff-only origin master
uv run --project training --no-sync python -m plex_training.plex600_research candidate-audit
uv run --project training --no-sync python -m plex_training.plex600_research benchmark-inspect
~~~

The auditor must return trainingAuthorized false and approvedTrainingRows zero. A scoreable model output file is JSONL with one row containing an ID and complete rewritten file content, e.g.:

~~~json
{"id":"m8-html-01","output":"<nav>...complete revised source...</nav>"}
~~~

Use the scorer after actual model-generated outputs are collected:

~~~powershell
uv run --project training --no-sync python -m plex_training.plex600_research score --responses training/artifacts/m8-responses.jsonl
~~~

It reports exact-file correctness, requested target presence, unchanged-context preservation, unscored cases, and language-level breakdowns. Missing model generations are NEVER counted as passes; simulated ideal responses in CI merely test the scorer.

### Read-only M8 local model examination (not training)

First inspect the plan:

~~~powershell
uv run --project training --no-sync python -m plex_training.plex600_m8_eval
~~~

Only when choosing to evaluate the saved checkpoint locally, use the explicit read-only operation. It verifies both the M7 report and the full checkpoint hash (a substantial disk read) before loading weights:

~~~powershell
$bundle = "training/artifacts/plex600-owner-authored-pilot-tokenizer-v1"
$bundleSha = "14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55"

uv run --project training --no-sync python -m plex_training.plex600_m8_eval --run-read-only-eval --bundle-dir $bundle --bundle-manifest-sha256 $bundleSha --m7-report training/artifacts/plex600-m7-first-four-update/report.json --m7-weights training/artifacts/plex600-m7-first-four-update/weights.pt --output-dir training/artifacts/m8-read-only-six-case-run --max-cases 6 --max-new-tokens 32

uv run --project training --no-sync python -m plex_training.plex600_research score --responses training/artifacts/m8-read-only-six-case-run/responses.jsonl
~~~

The example output directory must not exist. This examination can be slow because the 600M model recomputes context without a KV cache; it is not a performance benchmark and no successful output is claimed here. The full 36-case development evaluation can be chosen separately, with the same checkpoint and explicit invocation, after reviewing this bounded six-case result. The generation transcript is kept separate from the scored report. No new optimizer training authorization is implied.

## Milestones remaining before more training

1. Locally authenticate the M7 report and checkpoint bytes and preserve their SHA-256 receipts. Weights-only checkpoint does not permit exact AdamW resumption.
2. **Run the implemented read-only M8 inference harness locally** against the original M7 checkpoint, preserving every raw result on the fixed development benchmark; then score and analyze failure modes. The default plan does not load weights. Actual model outputs have NOT been collected in this GitHub work.
3. Freeze the evaluation set and prohibit training leakage. Distinguish human benchmark quality review from model-measured performance.
4. Review all 1,152 synthetic candidates for invalid edits, near duplicates, stylistic artifacts, and template-family contamination. Keep them quarantined until source-scoped **new owner approval**; then create grouped train/validation splits and retrain a tokenizer on the approved train split only.
5. Expand well beyond synthetic edits with genuinely diverse, rights-reviewed code and language material. Existing 6,970 train tokens and R2 candidate rows cannot establish a useful scratch 600M model.
6. Predeclare the next compute hypothesis, VRAM/time/storage limits, optimizer-state/checkpoint policy, validation protocol, stop conditions and fresh explicit training authorization. Keep the original 27M baseline frozen.

## Research basis (context, not new source authorization)

Hoffmann et al., *Training Compute-Optimal Large Language Models*, NeurIPS 2022, https://arxiv.org/abs/2203.15556, finds that compute-efficient language-model training depends heavily on the balance between model parameters and data tokens. That study does **not** imply that these tiny edit datasets suffice.

BigCode *The Stack v2* official dataset card, https://huggingface.co/datasets/bigcode/the-stack-v2, describes individual source-license, attribution, provenance and opt-out obligations, including additional bulk-access requirements. Its mention is **research only**: no Stack material has been downloaded or approved for Plex.

**Research status:** the project has measured a four-update learning pipeline and can now curate independent development tests and larger potential training material. It is technically prepared to plan more research, **not ready for sustained effective 600M training** on the present data, and no new optimizer updates are authorized.
