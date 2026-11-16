# Plex 600M M8: independent reconstruction and complete-file budget measurement

**Date:** November 16, 2026. **Scope:** CPU tokenizer fitting and development-benchmark token-count research, *not* model optimizer training or checkpoint evaluation.

## Verified independent experiments

The project includes the original, repository-committed owner-authored M7 preparation-approved training split (120 records) and internal validation split (24 records). Its original tokenizer was created locally on Windows. We independently rebuilt **only the tokenizer**, using the same frozen source and pinned `tokenizers==0.23.2` dependency, from GitHub Actions on Linux and Windows. The standalone implementation runs only with the explicit `--reconstruct-read-only` flag; it uses a self-cleaning temporary directory and does not download unapproved data or access weights, CUDA, optimizer, or final holdout.

| Reconstructed property | Independent Linux | Independent Windows |
| --- | --- | --- |
| Frozen M7 tokenizer SHA-256 matches | Yes | Yes |
| Frozen M7 tokenizer bundle manifest SHA-256 matches | Yes | Yes |
| Actual tokenizer vocabulary | 1,191 | 1,191 |
| Complete source binding verifies | Yes | Yes |
| M8 benchmark cases | 36 | 36 |
| Reference complete files fitting 32 tokens | **1 / 36** | **1 / 36** |
| Reference complete files fitting 64 tokens | **26 / 36** | **26 / 36** |
| Reference complete files requiring more than 64 tokens | **10 / 36** | **10 / 36** |
| Longest reference output including EOS | **90 tokens** | **90 tokens** |
| Largest prompt plus complete reference output | **216 tokens** | **216 tokens** |

Pinned tokenizer SHA-256:
`9feef87bdf5fc28b3315df589e4e0b8ffcf5f40b18a4c357c91759182340a2e2`

Pinned bundle manifest SHA-256:
`14b5465ae03550f1757dd82c7df1f35710db4173d6b70e5585624b1585a4ad55`

Independent runs used `research/m8-output-budget-gate` commit `556f615dbacc7b03720ff095790e61d29f4384d2`:
- [Linux GitHub Actions run 43878950348](https://github.com/keepithandy/plex-nano/actions/runs/43878950348) — 790 tests passed, 9 skipped; reconstruction verified
- [Windows GitHub Actions run 43878950320](https://github.com/keepithandy/plex-nano/actions/runs/43878950320) — Windows portability tests and reconstruction verified

## Original M7 vs. M8 literal prompt-format audit

The same read-only reconstruction program now independently checks the
**original 120-row M7 *training split*** against the frozen 36-case M8
development prompt contract, without opening new sources or the final
holdout. The original owner-authored rows are already source/hash-verified.

| Literal prompt-form observation | Frozen source count |
| --- | ---: |
| M7 owner-authored training rows with `Request: … Answer:\\n` format | **120 / 120** |
| M7 training rows containing `Source file:` marker | **0 / 120** |
| M7 training rows containing `Edited file:` marker | **0 / 120** |
| M7 training rows containing both file-edit prompt markers | **0 / 120** |
| M8 development cases using source-file/edited-file prompt contract | **36 / 36** |

This establishes a **literal prompt-format distribution shift**. It is
*not* proof the existing model cannot generalize to file editing; the
actual checkpoint hasn't been evaluated. It is nevertheless strong
evidence to require source-conditioned, preservation-sensitive curriculum
experiments and to measure gold-conditioned likelihood and generated-file
success separately. The added
[teacher-forced reference diagnostic](PLEX-600M-M8-TEACHER-FORCED-REFERENCE-DIAGNOSTIC-2026-11-16.md)
provides the latter independent research pathway when original weights
become available. All R2 synthetic candidates remain unapproved.

## Per-language limits using the frozen tokenizer

| Language | 32-token complete-file capacity | 64-token complete-file capacity | Min–max reference output (including EOS) |
| --- | ---: | ---: | ---: |
| HTML | 0 / 12 | 7 / 12 | 48–74 |
| CSS | 1 / 12 | 11 / 12 | 29–70 |
| JavaScript | 0 / 12 | 8 / 12 | 34–90 |

**Interpretation:** the existing M8 default of 32 generated tokens renders 35 of 36 correct full-file answers infeasible regardless of model quality, and the old hard cap of 64 renders 10 of 36 infeasible. A **96-token explicitly opted-in budget** covers all **36 frozen reference files** (maximum requirement 90, including EOS) and remains under the 1,024-context ceiling. The latest branch implements that bounded option; **no 96-token GPU generation, inference timing, model success, or checkpoint improvement has been measured**.

The cap check measures reference-text token capacity only. It cannot prove a model will generate a correct file, reach EOS, preserve source, or decode in reasonable wall time. The M7 checkpoint remains owner-local and is **not present on GitHub**. Its optimizer state was not saved in the weights-only M7 checkpoint.

## Reproducible command and safety boundaries

From the repository root after checking out the branch:

```powershell
uv run --project training --no-sync python -m plex_training.plex600_m8_reproduce --reconstruct-read-only --max-cases 36
```

Default invocation without the `--reconstruct-read-only` flag is *plan-only* and fits nothing. The explicit operation verifies committed source approvals and SHA identities; fits a fresh tokenizer **only on committed original train text**, not the frozen development evaluation; produces no lasting artifacts; and returns case-level evidence for cap32, cap64, and cap96. It refuses to label any alternative tokenizer as an M7-comparable result.

The code and evidence **do not** authorize any 600M optimizer training, change source permissions, promote quarantined synthetic data, open the final holdout, or claim improved coding ability. M8 generation remains a separate, explicit, SHA-verified checkpoint operation.

## Resulting research decision

Use the *measured 96-token* option for a **small first read-only, fixed-checkpoint M8 evaluation** after accessing the actual M7 weights and timing that run. Inspect raw generations, termination reason, and exact-file/preservation scores before scaling to 36 cases. Do not select new checkpoints, change model size, reuse M8 as training data, or approve a larger training corpus based solely on token-capacity results.

The independent tokenizer reconstruction has already resolved the previously stated requirement for user-local tokenizer verification. The remaining blocker is **checkpoint availability**, not tokenizer identity.
