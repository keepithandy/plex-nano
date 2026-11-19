# Does the prompt newline cue EOS?

All 18 supplied prompts were re-encoded from text using the frozen tokenizer. The original final newline was ID 202 and the replacement ASCII space was ID 224. For each case, the lengths matched, and the sole changed token index was the last prompt position. The expected first token, answer tokens, trailing answer newline and terminal EOS target were identical under both versions. This was an evaluation of saved weights with zero optimizer updates.

## Checkpoint comparison

Each entry gives the count out of 18 with **newline / space**. EOS above correct means the EOS logit strictly exceeded the logit of the expected first answer token. Immediate EOS and correct first use the actual greedy decision. Terminal EOS is the greedy decision after the complete reference answer under the indicated prompt. Exact copy uses native unmasked greedy generation with the original 96-token cap.

| Checkpoint | EOS above correct first | Immediate EOS | Correct first | Correct terminal EOS | Exact copies |
|---|---:|---:|---:|---:|---:|
| M9 | 10 / 11 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| M10-10 | 17 / 17 | 2 / 0 | 1 / 0 | 12 / 11 | 0 / 0 |
| M10-25 | 18 / 18 | 18 / 2 | 0 / 0 | 18 / 18 | 0 / 0 |
| M10-50 | 18 / 1 | 18 / 0 | 0 / 0 | 18 / 18 | 0 / 0 |

For each prompt, the delimiter effect is the EOS-minus-correct-first logit margin with newline minus the corresponding margin with space. Positive values mean that the newline favored EOS relative to the correct first token. The median effect was −0.265 logits at M9 and +1.690 at update 10. It grew on all 18 matched prompts. The median within-prompt increase over M9 was +2.186 logits. At update 10 the space lowered the median EOS rank from second to tenth, yet EOS still outranked the correct token in 17 cases and all 18 greedy starts were `=`. At update 50, space put the correct token above EOS in 17 cases, yet all 18 still began with `=`.

## The interval the saved checkpoints missed

The already-completed original-loss replay recorded first-position probes for both delimiters at every update. At updates 7, 8 and 9, the original newline gave 16, 18 and 18 immediate stops. Space gave zero immediate stops and zero correct first choices at each update. EOS still outranked the correct first token on every one of those prompts under both delimiters. The median within-prompt newline effect on the EOS-minus-correct margin grew from +0.419 to +0.638 to +0.981 logits across those updates.

## Interpretation

The prompt newline became a stronger stopping cue during warmup. Changing it at inference did not recover autonomous copying. These saved weights cannot establish how warmup would have unfolded with a different training delimiter.

A separately authorized test would repeat only the ten counter updates from M9 with a space at the prompt boundary. The existing original-loss replay is the control. Keep the answer newline, terminal EOS loss, denominator, tokenizer, seed, optimizer and evaluation set fixed. Evaluate both prompt formats on both arms at every update to distinguish a training-delimiter effect from an evaluation-format effect. For the three outcome measures, compare immediate stops, correct terminal EOS given the complete answer, and exact autonomous copies. Report correct first choices and 96-token caps alongside them.

The fresh forward passes reproduced the prior unmasked greedy IDs and EOS-versus-correct margins on all 144 checkpoint/prompt/format combinations. Newline completions came from the verified original response files. Space completions and terminal decisions after each prompt format were evaluated from the saved weights. The per-case CSV includes both ranks, probabilities, token IDs, decoded completions and the hash of the common prompt-token prefix.
