# M11 proposal — source-conditioned copying before scale

Status: **review only, not owner-approved for training**. This is a proposal
and a CPU-only scoring/tokenizer contract, not an M11 model checkpoint or run.
No GPU execution, optimizer, authorization generator, or checkpoint writer is
included. The M7/M9/M10 artifacts and the final project holdout remain unchanged.

## Reason for the experiment

M10 PR #209 established that a final space instead of newline can prevent
premature EOS, but at update 10 the space-trained model emitted \`= 0;\n\`
on all 18 prompts and produced 0/18 exact copies. Terminal EOS after a
teacher-forced correct answer improved to 18/18. Neither metric proves that
the model actually reads the source. The M10 substitution probe also found
0/18 instances of bidirectionally correct changed-token preference.

The next test should ask a single, falsifiable question: **if the input
changes from \`let score = 0;\n\` to \`let score = 1;\n\`, does the output
change correctly as well, including its exact final newline and EOS?**

## Proposed data and boundary

Two assistant-proposed *training candidates*, not yet approved:

- \`let score = 0;\n\`
- \`let score = 1;\n\`

Two separate *development probes*, explicitly excluded from training:

- \`let value = 0;\n\`
- \`let value = 1;\n\`

This is a small, visible development test, **not an untouched generalization
holdout**. Do not use M10's six evaluation-only cases or the project's final
holdout as training examples. Do not silently enlarge the dataset.

Use the original M10 copying instruction, changing **only the final prompt
newline to a literal space**. Preserve the answer's newline, frozen M7
tokenizer, source text, and EOS identity. Verify exactly one changed source
token and one changed answer token in each matched pair. The CPU-only
auditor checks these properties and token budgets before any proposed run.

## Independent measurements (do not blend them into one score)

1. Autonomous generation must be byte-for-byte correct for **both** values,
   with EOS reached after the complete output, not immediate EOS or cap.
2. At the changed output-token position after the correct gold prefix,
   logit(1)-logit(0) must be negative with source 0 and positive with source 1.
   The **unmasked greedy** token must also match in each direction. A
   single fixed winner fails.
3. Report first-token accuracy, content-token accuracy with teacher-forced
   prefixes, immediate EOS, terminal EOS given the full correct answer,
   length-cap counts, and both existing decoder policies separately.
4. Measure training-pair and development-probe-pair performance separately.
   Perfect training memorization without probe improvement is limited progress.

The pure helper \`score_caller_supplied\` deliberately labels scores
unverified. It does not assert model generation, provenance, or checkpoint
identity. A future evaluator must bind actual inference receipts to hashes
and score them independently before any results can claim model behavior.

## Proposed next gated training study (NOT approved or implemented)

Start from the separately hash-verified backed-up M9 **model** state, with
fresh optimizer state. Compare normal terminal EOS weighting against a
predeclared reduced-weight condition (not zeroed without further rationale).
Hold architecture, tokenizer, seed, precision, prompt and sample schedule
fixed; alternate the two source variants fairly, one example per update.
The provisional milestones are **0, 25, 50, 100, and 200 updates**. These
are **planning ceilings**, not an authorization or established GPU budget.
Review source rights, approval scope, memory/disk/time feasibility, exact
run schedule and independent milestone receipt validation first.

Do not select a winning arm using the visible development pair and then
describe it as an untouched holdout success. Preserve raw results from
both arms; reject apparent improvement if generations are constant, truncated,
stop incorrectly, or merely achieve teacher-forced terminal EOS.

## Read-only commands

\`\`\`bash
PYTHONPATH=training/src python -m plex_training.plex600_m11_source_conditioning
PYTHONPATH=training/src python -m plex_training.plex600_m11_source_conditioning \
  --audit-frozen-tokenizer --bundle training/artifacts/plex600-fresh-tokenizer
\`\`\`

The default command prints a no-training plan without loading any model
or examining external data. The explicit audit verifies the frozen bundle
and candidate tokenization but performs no training. There is intentionally
no \`--train\` or \`--execute-approved-run\` option. Never treat this PR as
consent for M11 training or as a replacement for the M10 experiment record.
