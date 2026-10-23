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
| Dedicated 600M tokenizer preparation | **Pending verification** | Must not relabel an existing frozen 16K tokenizer as 32K |
| Input data and independent split/provenance verification | **Not yet prepared for 600M** | Require reviewed rights, no validation leakage |
| Full 600M GPU feasibility | **Not run** | Memory and throughput unknown |
| Full 600M optimizer updates | **Not authorized / not run** | No 600M trained weights exist |
| Real-world coding ability | **Not measured** | Random weights are not a working assistant |

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
