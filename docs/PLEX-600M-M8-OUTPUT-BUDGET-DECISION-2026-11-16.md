# Plex 600M M8: full-file output budget and training decision

**Date:** November 16, 2026. **Status:** implementation and tests proposed in a pull request; *no owner-local token-count observation, inference run, or new training performed by this work*.

## The measurement problem

M8 asks the model for an entire rewritten HTML/CSS/JavaScript file, not an edit command or a changed substring. The existing greedy runner allows **1–64 generated steps** per case and defaults to **32**. It scores exact full-file output, target replacement, and preservation of surrounding context.

A valid full-file answer may need more than 64 BPE tokens. A run truncated by a known output cap is not a fair test of whether Plex can complete that edit. The precise counts depend on the **owner's source-bound, 1,191-token BPE bundle**, which is local and not present in GitHub. Do **not** estimate them from character counts or substitute a new tokenizer.

The new, CPU-only audit counts frozen expected-file tokens under the original verified tokenizer, reserves one additional step for EOS, and checks the runner's current 1,024-position context reservation. It prints case-by-case and language-level coverage and the minimum cap that would fit the selected reference files. It **never** supplies expected code to the model, executes inference, or changes weights. The normal M8 execution path now refuses an under-budget full-file evaluation **before CUDA/model initialization**, rather than producing a misleading capability score.

## Run the audit (read only, no checkpoint required)

From the repository root on the owner workstation, after reviewing/pulling this PR:

```powershell
$bundle = "training/artifacts/plex600-owner-authored-pilot-tokenizer-v1"
$bundleSha = "14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55"

uv run --project training --no-sync python -m plex_training.plex600_m8_eval `
  --audit-token-budget `
  --bundle-dir $bundle `
  --bundle-manifest-sha256 $bundleSha `
  --max-cases 36 `
  --max-new-tokens 64
```

Review `completeWithinBudget`, `blockedByBudget`, `requiredMaxNewTokensIncludingEos`, `requiredMaxTotalContext` and the 36 case records. **These fields are feasibility observations, not a model success score.** The audit verifies frozen tokenizer/source binding, never touches the local weights or optimizer, makes no file artifacts and does not access final holdout. A missing/mismatched bundle fails closed.

The original default `--run-read-only-eval --max-cases 6 --max-new-tokens 32` remains guarded: it will be refused if any selected expected full file cannot fit. Do not bypass the guard and report truncated generations as failed full-file capability.

## Decision after the real tokenizer audit

1. **If the required cap fits the existing bounded runner and context:** choose explicit case count and budget, run the existing, separately opted-in read-only generation against the SHA-verified M7 checkpoint, preserve its raw response transcript, then score the unchanged benchmark. No optimizer training is needed for this first capability baseline.
2. **If gold answers exceed 64 tokens:** separately design and test an extended-output inference path (KV caching or another measured bounded approach) with a declared wall-clock and memory budget. The current full-recomputation generator is not a performance baseline. Do not silently increase the cap and launch costly inference.
3. **If full-file continuation is not viable on the current dataset/model:** explicitly version a small edit-span or structured-patch output contract and deterministic, source-preserving reconstruction, then establish a *new* evaluation baseline. Do not compare that new protocol's scores directly with historical full-file scores.
4. **Only after valid baseline generations:** review error categories (wrong target, missing source identifiers, syntax, truncation, unrelated-code damage), then decide whether the next bounded, rights-approved experiment should target data diversity, copying/source binding, or output serialization. A lower teacher-forced loss alone is not a deployment gate.

## The actual data/compute bottleneck

The **M7 four-update pilot** trained on 120 narrowly owner-authored records, totaling **6,970 training tokens** in its prepared token split, and sampled **220 supervised positions across four optimizer updates**. The reported internal validation loss moved **10.5988 → 8.9729**, but there are **no actual M8 development generations** in GitHub. Its weights-only local checkpoint does **not** preserve AdamW state for exact continuation.

The **1,152 R2 synthetic rows** comprise just **18 edit-template families** with **64 variations per family**. All remain **quarantined**, with **zero approved for training**, and six other historical web sources remain on legal/provenance hold. Treat the varied rows as correlated, not 1,152 independent files. Any future train/validation split must group by originating template/source family and remove both exact and meaningful near-overlap with development/holdout tasks.

The evidence therefore favors **a valid output/evaluation protocol and genuinely broader, licensed data** ahead of an extended 600M scratch training run. Compute-optimal language-model research shows that training data must scale with model capacity, but it is **not** a promise that a particular token-to-parameter heuristic solves this small, domain-specific task. Research on contamination likewise motivates stronger overlap checks; a zero exact string match is not proof of an independent evaluation.

## External research (reading only; no source authorization)

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html
- Choi et al., *How Contaminated Is Your Benchmark?*, ICML 2025, https://proceedings.mlr.press/v267/choi25b.html
- SWE-bench Verified, human-reviewed benchmark design (a comparative research example, **not** a matching benchmark for Plex's supplied-single-file contract), https://www.swebench.com/verified

**Safety and evidence:** the 27M controlled baseline, M7 checkpoint, dataset approvals, original 36 benchmark records, R2 candidate bytes, hash identities and sealed final holdout are unchanged. No GPU training/inference or dataset approval is implicit.
