# Ten-update prompt-delimiter replay

One new ten-update arm started from the verified M9 weights. It repeated the JavaScript counter with the original seed (41), fresh AdamW state and learning rate (0.00005), zero weight decay, and the original mean loss over ten content tokens and terminal EOS. Only input token 202 at the final prompt position changed, to space token 224. The complete target tensor matched the original, and the final answer newline remained token 202. The control was the completed original-loss replay whose update-10 weights matched the saved historical checkpoint exactly.

Each entry below is a count out of 18, reported as original newline control / new space-delimited replay. Each arm is evaluated in its own training format. Terminal EOS is the greedy decision given the complete correct answer. Exact copy is autonomous fitted-vocabulary greedy decoding with a 96-token limit.

| Update | Stops immediately | Stops after correct answer | Exact copies | Hits length cap |
|---:|---:|---:|---:|---:|
| 0 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 1 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 2 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 3 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 4 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 5 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 6 | 0 / 0 | 0 / 0 | 0 / 0 | 18 / 18 |
| 7 | 16 / 0 | 18 / 16 | 0 / 0 | 2 / 18 |
| 8 | 18 / 0 | 18 / 18 | 0 / 0 | 0 / 18 |
| 9 | 18 / 0 | 18 / 18 | 0 / 0 | 0 / 0 |
| 10 | 2 / 0 | 12 / 18 | 0 / 0 | 0 / 0 |

## Common-format comparison

The new weights were also evaluated on the original newline prompts at every update. The table below compares the two training runs with exactly the same evaluation tokens, then shows the effect of switching the new model back to its space format.

| Update 10 | Immediate EOS | Correct terminal EOS | Exact copies | Correct first token | Correct content tokens with gold prefixes |
|---|---:|---:|---:|---:|---:|
| Original training, newline evaluation | 2/18 | 12/18 | 0/18 | 1/18 | 27/286 |
| Space training, newline evaluation | 18/18 | 18/18 | 0/18 | 0/18 | 27/286 |
| Space training, space evaluation | 0/18 | 18/18 | 0/18 | 0/18 | 27/286 |

Both input formats were evaluated live after every new update. The initial M9 boundary scores reproduced the archived control scores for both formats on all 18 cases. Only the counter was optimized. The model weights stayed in memory, and the run wrote no checkpoints. These results apply to this seed, example, and ten-update warmup.

## What changed

At updates 7–9, both training arms showed the same immediate-stop counts when evaluated on the same delimiter. With newline these were 16, 18 and 18. With space they were zero at each update. The completed control supplied the original-trained model's space-boundary decisions. Thus the reduction in empty responses during that interval was already available by switching the evaluation delimiter alone.

At update 10 on space prompts, the earlier frozen control chose EOS after the full correct answer on 11/18 cases. The space-trained model did so on 18/18. Both chose the wrong first token on every space prompt. The space-trained model produced exactly the same five-character response on all 18 prompts: an equals sign, space, zero, semicolon and newline. Even the repeated training example failed. This separates improved stopping in the space format from the unresolved need to condition generated content on the supplied code.
