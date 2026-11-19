# M10 warmup diagnostics on saved weights

The approved 600M copying experiment used the same newline token (ID 202) immediately before two different supervised targets on its repeated JavaScript example. After `Copy:\n`, the target was `l`. After the complete `let score = 0;\n`, the target was EOS. The model needed to use the preceding code to distinguish those positions. These diagnostics test how well it made that distinction and whether it could copy once given a correct start.

The evidence comes from the M9 backup and updates 10, 25 and 50 of `plex600-m10-approved-20261118-182055`. All experiments used forward passes on frozen checkpoints. The 18 cases comprise the existing 12 training and six evaluation-only copying prompts. Input substitutions existed only in memory. The final project holdout stayed closed.

## The final newline changes the stopping decision

Each replacement changed only the final token, preserving all preceding token IDs and the sequence length. Probabilities are full-vocabulary medians across prompts. The counts use the native BF16 greedy decision.

| Checkpoint | EOS after original newline | EOS after space | Immediate EOS, newline / space | EOS after the complete reference and newline |
|---|---:|---:|---:|---:|
| M9 | 0.0100% | 0.0098% | 0 / 0 | 0.0094% |
| M10 update 10 | 14.89% | 1.94% | 2 / 0 | 16.68% |
| M10 update 25 | 72.14% | 4.13% | 18 / 2 | 74.55% |
| M10 update 50 | 49.75% | 0.0134% | 18 / 0 | 56.52% |

At every M10 checkpoint, every prompt had higher EOS probability with the original newline. At update 50, all 18 space-ending prompts chose `=` first. Removing the newline altogether gave a median EOS probability of 0.0021% there. The replacement is a diagnostic input outside the original prompt format. Its effect on a saved model establishes sensitivity to the delimiter. The source of that sensitivity during optimization remains open.

## The warmup shift is distributed across the transformer

For each original prompt, the two checkpoints independently computed the hidden state at all 25 boundaries, from the input embeddings through block 24. At each boundary, the recipient finished its normal forward pass from the donor's full sequence of hidden states, using its own remaining blocks, final normalization and output weights. At six boundaries the probe also exchanged the last position alone or all preceding positions alone. Same-checkpoint controls reproduced the native boundary logits exactly at those six cuts in both directions.

| Update-10 states supplied to M9 | Median prompt EOS |
|---|---:|
| Input embeddings only | 0.066% |
| Through block 4 | 0.56% |
| Through block 8 | 4.12% |
| Through block 12 | 11.16% |
| Through block 16 | 13.79% |
| Through block 24 | 11.48% |

At block 12, exchanging only the last position gave 9.38%. Exchanging only the preceding positions gave 0.043%. By that depth, much of the EOS preference could be transferred in the final position. Its state already includes information from earlier positions. The reverse exchange also mattered: the update-10 recipient gave 8.09% with M9 states through block 12 and 0.0168% with M9 states through block 24. These are interacting components in mixed-checkpoint forward passes. Block depth does not give the chronology of optimizer updates.

## A correct start still leaves most of the copying task unsolved

The first intervention forced the highest-scoring fitted content token. The second supplied the first reference token. Each then used normal fitted-vocabulary greedy generation for the remaining budget, with EOS available from the second decision onward. Exactness includes all whitespace. The gold-prefix measurement separately gives the model each correct preceding answer token and scores its next full-vocabulary greedy choice.

| Checkpoint | Exact copies with best content first | Exact copies with reference first | Correct content predictions with gold prefixes | Correct terminal EOS with gold prefixes |
|---|---:|---:|---:|---:|
| M9 | 0/18 | 0/18 | 1/286 | 0/18 |
| M10 update 10 | 0/18 | 0/18 | 27/286 | 12/18 |
| M10 update 25 | 0/18 | 0/18 | 24/286 | 18/18 |
| M10 update 50 | 0/18 | 0/18 | 35/286 | 18/18 |

At update 25, all 18 best-content-first runs emitted exactly a period and a newline, then EOS. At update 50, 15 of 18 best-content-first runs and 10 of 18 reference-first runs reached the 96-token cap. With a gold prefix of one token, the second content token was correct on 0/18 prompts at both updates. Removing an empty output from the score would therefore obscure a second failure in content generation.

## Does changing the supplied code change the requested answer?

For each case, two versions of the supplied snippet differed in exactly one token. The model received the same correct answer prefix on both sides, stopping immediately before the changed token. For example, the counter contained either `let score = 0;` or `let score = 1;`, and both answers were cued through the same token prefix. Sixteen pairs used a digit suffix on an existing name or text, and two used a numeric literal. The tokenizer checks both prompt lengths, the unique changed position, and the identity of the two compared answer tokens.

For alternatives A and B, the score is the B-minus-A logit. A model tracks the substitution on both sides when this score is negative with source A and positive with source B. A positive paired change is a weaker result and can occur when the same alternative still wins twice.

| Checkpoint | FP32 changes toward the substituted token | Median FP32 paired logit change | Correct preference on both sides, BF16 / FP32 |
|---|---:|---:|---:|
| M9 | 6/18 | -0.01657 | 0 / 0 |
| M10 update 10 | 15/18 | +0.00415 | 0 / 0 |
| M10 update 25 | 16/18 | +0.00275 | 0 / 0 |
| M10 update 50 | 15/18 | +0.00596 | 0 / 0 |

At update 10, the counter's original zero token beat the one token by about 10.6 logits for **both** supplied values. It was the greedy choice on both sides in both precisions. This is evidence of a strong preference for the memorized continuation on the repeated example. The paired inputs cover a narrow set of numeric substitutions, and the 18 related templates do not support a population-wide estimate of copying ability.

## How to reproduce

Use a checkout containing the diagnostic module and the existing WSL environment with CUDA and native BF16. From the repository root, set the four paths to the saved inputs. The module verifies the approved report digest, model source, tokenizer, candidate set, M9 backup, all snapshot weights and saved response receipts. Without the explicit inference flag it prints a plan. During inference it writes JSON to stdout and progress to stderr.

```bash
export PYTHONPATH=training/src
scope=(
  --bundle training/artifacts/plex600-fresh-tokenizer
  --m9-backup /mnt/c/Users/quali/Plex600M-Backups/m9
  --baseline training/artifacts/plex600-m10-m9-full-baseline-20261117-164621
  --run-dir training/artifacts/plex600-m10-approved-20261118-182055
)
set -C
training/.venv/bin/python -m plex_training.plex600_m10_warmup_probes \
  --run-inference --study warmup "${scope[@]}" > warmup-evidence.json
training/.venv/bin/python -m plex_training.plex600_m10_warmup_probes \
  --run-inference --study source-copy "${scope[@]}" > source-copy-evidence.json
```

The warmup study records 432 context measurements, 1,332 activation exchanges, 144 first-token intervention outcomes and 72 gold-prefix traces. The source-copy study records 144 paired observations, each with both input variants, across the two precisions. In the context records, `correctFirstProbability` always refers to the snippet's original first token, including when the full gold answer has been supplied. At the gold endpoint EOS is the correct next token.

## Decision before another training run

The owner subsequently authorized the two-arm replay described below. The reference run reproduced every saved update-10 weight. [Matched replay results](PLEX-600M-M10-MATCHED-REPLAY.md).

The endpoint evidence supports testing a shared newline cue and terminal EOS supervision. It also gives a way to reject an apparent improvement that merely makes the model emit a nonempty string. A proposed separately authorized replay would start both arms from the same verified M9 weights and fresh optimizer. One arm would reproduce the original ten updates on the counter. The other would zero only the terminal EOS loss term, retaining the original denominator of eleven supervised positions, optimizer, precision, seed, example order and prompt format.

Record step zero and every subsequent update. Compare the paired prompt EOS-minus-correct-token margin, original newline versus space probability, terminal EOS after the complete answer, and source-pair margins. Retain exact copies, correct content tokens and the 96-token cap counts alongside stopping metrics. Check the reference arm against the historical loss ledger and the saved update-10 boundary results before interpreting differences between arms. If a terminal-loss ablation reduces early stopping while the source preference and copying failures persist, it explains part of the stopping behavior while leaving the content-learning question open.
