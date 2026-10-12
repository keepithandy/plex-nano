# Plex Nano test suites

This file documents the CI partition introduced for verification gap G03.

## Scope

The repository has two Python test suites:

1. **Portable required suite** — runs automatically for relevant pull requests and pushes to `master`.
2. **Historical experiment-contract suite** — runs only by explicit `workflow_dispatch`.

The manual experiment workflow is authorization to run those **tests only**. It is not authorization to execute Plex model training, promote a corpus, modify a research checkpoint, or open the final holdout.

## Inventory baseline

The G03 audit snapshot described 62 top-level `test_*.py` files with 26 selected and 36 omitted.

By the time G03 was verified on current `master`, earlier CI fixes had already changed the inventory to:

- 62 top-level test modules
- 29 modules in the automatic workflow
- 33 omitted modules

G03 reviews those 33 omissions individually.

This change adds `test_ci_test_inventory.py`, so the post-change inventory is:

- **63 total top-level test modules**
- **42 portable required modules**
- **21 manual experiment-contract modules**
- **0 unclassified modules**

These are filename counts, not executed test-case counts and not a coverage percentage.

## Previously omitted modules promoted to the portable required suite

| Module | Reason |
|---|---|
| `test_answer_focused_records.py` | Deterministic answer/EOS supervision and bounded-record validation; no research training. |
| `test_benchmark.py` | Static benchmark/parser behavior; subprocess failures are mocked and the one real Node syntax check is skipped when Node is unavailable. |
| `test_complete_records.py` | Core complete-record/resume integrity. Uses tiny CPU fixtures and temporary artifacts rather than the controlled research model. |
| `test_config.py` | Pure configuration and parameter-count validation. |
| `test_dataset.py` | Dataset construction, license/review boundaries, reproducibility, and filesystem behavior using temporary local fixtures. |
| `test_model.py` | Tiny-model forward/backward, causal masking, loss weighting, and tied-projection behavior. |
| `test_record_sampling.py` | Core sampler/resume integrity with bounded tiny CPU training in temporary directories. |
| `test_source_fetch.py` | Pinned-source fetch validation with network calls mocked; no live network dependency. |
| `test_tokenizer_review.py` | Fits small fresh tokenizer candidates to temporary reviewed data; explicitly performs no model training. |
| `test_web_dataset.py` | Web-corpus builder filtering with local temporary fixtures. |
| `test_web_materialize.py` | Reviewed-source materialization with local temporary fixtures and mocks. |
| `test_web_sources.py` | Rights/language/artifact filtering with local temporary fixtures. |

The new `test_ci_test_inventory.py` is also part of the portable required suite. It enforces that every top-level Python test module belongs to exactly one suite and that the experiment suite remains manual-only.

## Previously omitted modules retained in the manual experiment-contract suite

These modules are useful regression evidence, but they are coupled to historical Phase-2 candidates, approvals, frozen hashes, experiment contracts, or phase-specific diagnostics. They are therefore not required on every current-product pull request.

- `test_binding_candidate.py`
- `test_binding_experiment.py`
- `test_code_pair_approval.py`
- `test_code_pair_candidate.py`
- `test_compositional_binding_probe.py`
- `test_curated_request_approval.py`
- `test_curated_request_candidate.py`
- `test_failure_gap_approval.py`
- `test_failure_gap_candidate.py`
- `test_p2_04_compositional_run.py`
- `test_p2_05_counterbalanced_composition.py`
- `test_p2_06_binding_representation_probe.py`
- `test_p2_06_binding_representation_run.py`
- `test_p2_07_instruction_invariance_probe.py`
- `test_p2_07_instruction_invariance_run.py`
- `test_p2_08_selector_format_experiment.py`
- `test_p2_08_selector_format_probe.py`
- `test_p2_10_explicit_answer_start_candidate.py`
- `test_request_following_candidate.py`
- `test_structured_plan_request_binding_curriculum.py`
- `test_transfer_diagnostic.py`

The manual suite exists so those frozen research contracts can still be checked deliberately without making every historical Phase-2 artifact a merge gate for current Phase-3 work.

## Workflow files

- Portable required: `.github/workflows/training-tests.yml`
- Manual experiment-contract: `.github/workflows/training-experiment-tests.yml`

Changes to either workflow trigger the portable required suite on pull requests and on relevant pushes to `master`, which also validates the inventory partition.
