# M10 matched warmup replay

Removing only the terminal EOS contribution to the training loss changed the ten-update trajectory. With the original loss, immediate EOS first appeared on 16 of 18 prompts at update 7 and on all 18 at updates 8 and 9. The update-10 snapshot caught a partial recovery to 2 of 18. With the terminal term removed, the maximum immediate-EOS count across the ten measured updates was 0 of 18.

## Experiment and fidelity

Both arms started from M9 weights `e913be27855548a8f7dea766a2d367c7eaea357014ca82c80dec088958dfb123`, used seed 41, a fresh AdamW optimizer (learning rate 0.00005, zero weight decay), and the same ten presentations of `m10-js-counter`. Each used the original prompt, tokenizer, target sequence, CUDA BF16 autocast and FP32 parameters. The original objective averaged ten content-token losses and one terminal EOS loss. The ablation summed the same ten content losses and divided by eleven. A synthetic gradient check established that the loss change preserved every content-position logit gradient for identical logits and zeroed the terminal-position gradient.

The reference replay's maximum historical pre-update loss error was 0. A direct comparison with the saved update-10 state found zero differing parameter elements. The two arms gave identical step-zero probe records. Only the counter was optimized. The 12/6 split and final project holdout policy were preserved, and the experiment wrote no model checkpoints. Results are specific to these 18 related prompts, one seed and one warmup example.

## When the stopping behavior changed

Both arms initially increased median prompt EOS probability from 0.010026%. After one update it was 0.13745% with the original loss and 0.082997% with the terminal term removed. Other supervised positions can therefore produce an initial increase in this setup. By update 10 the original-loss arm had higher prompt EOS probability on 18 of 18 matched cases.

| Update | Median prompt EOS, original | Median prompt EOS, terminal loss removed | Immediate EOS, original | Immediate EOS, terminal loss removed |
|---:|---:|---:|---:|---:|
| 0 | 0.010026% | 0.010026% | 0/18 | 0/18 |
| 1 | 0.13745% | 0.082997% | 0/18 | 0/18 |
| 2 | 0.41004% | 0.031104% | 0/18 | 0/18 |
| 3 | 0.94519% | 0.0084336% | 0/18 | 0/18 |
| 4 | 1.7292% | 0.0027485% | 0/18 | 0/18 |
| 5 | 2.8382% | 0.0010562% | 0/18 | 0/18 |
| 6 | 4.7945% | 0.00051689% | 0/18 | 0/18 |
| 7 | 7.8725% | 0.00031556% | 16/18 | 0/18 |
| 8 | 10.858% | 0.0002376% | 18/18 | 0/18 |
| 9 | 12.21% | 0.00018174% | 18/18 | 0/18 |
| 10 | 14.893% | 0.00014886% | 2/18 | 0/18 |

Counts use the full-vocabulary greedy first decision. Medians use the unmasked softmax distribution. The M9 and update-10 reference decisions were checked against the earlier frozen-checkpoint evaluation. Median EOS probability continued rising at update 10, but sixteen prompts gave the warmup's first token `l` a higher score and emitted it.

## Stopping versus copying at update 10

| Metric across 18 prompts | Original loss | Terminal loss removed |
|---|---:|---:|
| Median EOS probability after the complete correct answer | 16.681% | 0.00014999% |
| Correct greedy terminal EOS after the complete correct answer | 12/18 | 0/18 |
| Correct gold-prefix content tokens | 27/286 | 14/286 |
| Exact copies in fitted-vocabulary greedy generation | 0/18 | 0/18 |
| Generations reaching the 96-token cap | 0/18 | 18/18 |
| Source substitutions preferring the matching token in both directions | 0/18 | 0/18 |

The counterfactual identifies the effect of the terminal loss term under the original training setup. The shared newline preceding both the first content target and the terminal target is a plausible route for transfer. The replay held the delimiter fixed, so it cannot isolate the delimiter's contribution during learning. The frozen-checkpoint delimiter controls address how the saved model responds at inference.

## Measurements and reproduction

At step zero and after each update, both arms were probed on each original prompt, the same-length final-space control, each full reference answer, and the matched one-token source substitutions. Fitted-vocabulary greedy generation used the original 96-token limit. Step-zero generations used the hash-verified existing M9 responses. All subsequent generations were computed live. The raw JSONL contains per-case measurements, generated text, update loss and gradient receipts, the reference-weight comparison, and the final update count.
