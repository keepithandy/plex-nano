# Plex 600M — scratch-model development record

**Started:** October 22, 2026  
**Current purpose:** produce an independently trained, small, useful language-and-coding model, initialized **entirely from random weights** and developed with repeatable human testing. The owner explicitly prefers genuine scratch research even when training takes a long time.

## Goal we will not lose

Build **Plex 600M** as a general language + practical coding model that can receive instructions, understand supplied HTML/CSS/JavaScript, generate coherent answers, debug and make precise edits. Demonstrate these skills on frozen development examples, not merely declining training loss. Human-observed failures feed reviewed future training data, not silent modifications to evaluation sets.

- **Never replace this goal with an agent/repository navigator, pretrained-model adaptation or licensing/product roadmap.**
- Use **random initialization**. Keep vocabulary, data lineage, optimizer history and checkpoints reproducible.
- Preserve **Plex Nano 27,566,080** as a separately frozen baseline. Never load Nano checkpoints as Plex 600M.
- Keep the sealed final evaluation holdout sealed.
- **Preparation, tests, training authorization and actual optimizer updates are distinct actions.** New full-scale training needs a reviewed experiment and approval.

## Hardware and target

RTX 4080 SUPER with **16,376 MiB** reported video RAM; CUDA available. A full-size CUDA forward/backward/optimizer fit is **not yet measured**. No guarantee that 600M training fits before a bounded hardware test.

| Setting | Initial scratch-model target |
|---|---:|
| Trainable parameters | 600,372,480 |
| Decoder blocks | 24 |
| Width / query heads / KV heads | 1,280 / 20 / 5 |
| SwiGLU hidden width | 4,992 |
| Context | 1,024 tokens |
| Tokenizer model capacity | 32,768 tokens |
| Positional encoding | RoPE |
| Model construction | Random initialized only |

## Current implementation and evidence

| Track | State | Evidence |
|---|---|---|
| Architecture and exact size | **Implemented** | [Plex600Model](../training/src/plex_training/plex600.py) and [design review](PLEX-600M-ARCHITECTURE-REVIEW.md) |
| Lightweight model forward/backward, causal mask, weighted loss | **Tested in CI** | [tests](../training/tests/test_plex600.py), Linux and Windows workflow |
| PR #197 | **Merged to master** | [Merge](https://github.com/keepithandy/plex-nano/pull/197), commit `156ef1de959e7c991d16e28a7a8caf80ac17fa20` |
| Dedicated 600M tokenizer preparation | **Implemented; Linux and Windows CI passed** | [600M setup CLI](../training/src/plex_training/plex600_setup.py), [guard tests](../training/tests/test_plex600_setup.py) |
| Input data and independent split/provenance verification | **Bundle verification implemented; large-scale dataset absent** | Requires reviewed rights, disjoint splits, and SHA-pinned data |
| Record-bounded next-token batches | **Implemented; Linux and Windows CI passed** | [packed sampler](../training/src/plex_training/plex600_batches.py) and tests (no optimizer) |\n| Full 600M GPU feasibility | **Not run** | Memory and throughput unknown |
| Full 600M optimizer updates | **Not authorized / not run** | No 600M trained weights exist |
| Real-world coding ability | **Not measured** | Random weights are not a working assistant |\n| Regression status for current foundation | **Green** | Linux 678 passed / 9 skipped; Windows 329 passed (see links below) |

## First engineering deliverable

A safely testable, isolated foundation on `master`: exact architecture, simple read-only model inspection, dedicated tokenizer preparation without fitting on validation examples, explicit batch/label conventions and tiny CPU tests. Keep 600M training execution gated and **never** run optimizer updates as an incidental result of tests or inspection.

## What qualifies as progress

- Checks must run on Linux **and** Windows; test changes actually exercise new code.
- Record exact observed outcomes and failures without hiding them.
- A passing tiny CPU test does **not** establish a full-model GPU fit.
- A tokenizer vocabulary size is a capacity; a small demonstration corpus will not necessarily populate all 32,768 entries.
- Training data must be approved, hash-pinned and split before fitting the tokenizer.
- Once there are genuine saved weights, measure instruction following, syntax, source-preservation and complete-file accuracy alongside loss.

## Log

- **2026-10-22:** Confirmed the scratch-model goal and RTX 4080 SUPER hardware; merged architecture PR #197 after green Linux/Windows checks. Continued work directly on `master`, without new pull requests.

## Safe local inspection commands

From the root of the repository, these do not construct the full 600M model:

```powershell
uv run --project training python -m plex_training.plex600_setup inspect

# After you have a reviewed, rights-approved, hashed dataset with disjoint
# train/validation JSONL and its manifest:
uv run --project training python -m plex_training.plex600_setup tokenizer-train --dataset-dir PATH_TO_REVIEWED_DATASET --output-dir PATH_TO_NEW_600M_BUNDLE --vocab-size 32768

# Read-only verification of that freshly produced bundle:
uv run --project training python -m plex_training.plex600_setup tokenizer-check --bundle-dir PATH_TO_NEW_600M_BUNDLE

uv run --project training python -m pytest training/tests/test_plex600.py training/tests/test_plex600_setup.py -q
```

A tiny training dataset may yield **fewer** than 32,768 learned tokens; that does not invalidate the model's vocabulary *capacity* and is not evidence of model training. The batch sampler pads short records and ignores its padding in the loss; windows never cross an EOS/document boundary. **No optimizer update is executed by these commands.**

## Verified first-foundation result

- **Linux:** [training-tests run 39886569785](https://github.com/keepithandy/plex-nano/actions/runs/39886569785): **678 passed, 9 skipped**, PyTorch emitted an optional NumPy initialization warning.
- **Windows:** [training-windows-tests run 39886569715](https://github.com/keepithandy/plex-nano/actions/runs/39886569715): **329 passed**, same optional NumPy initialization warning.
- **Tested code:** 600M architecture shape and causal/gradient behavior using small configurations; fresh scratch tokenizer identity and approved-data fit; SHA and split corruption refusal; token/window sampling that stays inside individual records; small-model next-token backward pass.
- **Evidence limit:** CI did **not** instantiate or train all 600M parameters, run on the owner's GPU, acquire substantial training data, or certify language skill. No 600M model checkpoint or weights exist.

**First milestone outcome: PASS for repository-side foundation, NOT YET VERIFIED for full-size GPU training.** The next independent gate is a reviewed, bounded 600M CUDA memory/forward/backward check, followed by dataset/tokenizer source review and separately authorized optimizer execution.
