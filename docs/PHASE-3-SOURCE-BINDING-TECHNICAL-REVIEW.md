# P3-SB technical review and proposed intervention

**Date:** October 17, 2026. **Tracking:** [#189](https://github.com/keepithandy/plex-nano/issues/189), remains open. This work improves the small coding model's preservation of **user-supplied** HTML, CSS and JavaScript. Model invariants remain 27,566,080 parameters, 16,384-token vocabulary and 512-token context.

**Disposition:** SB-00 measured from the existing local report; SB-01 fixture/tokenizer verified; SB-02–04 specifications proposed for independent owner review; SB-05 pure scorer implemented and tested with synthetic strings. No new model responses, checkpoint load, stage, bundle or optimizer updates. No final project holdout access. The predecessor remains a failed **0/138** exact-file experiment.

## Owner decision / follow-on preparation

The owner approved **hypothesis A: deterministic train-namespace diversity** for the next preparatory step after this independent SB-00/SB-01 technical review. The [follow-on SB-03 candidate](PHASE-3-SB03-APPROVED-NAMESPACE-DIVERSITY.md) has now been drafted with **48 new source names, unchanged 96-row validation, SHA-pinned original comparator, 1,600 index-equivalent training draws and full 288-record coverage**. It is **not token-count parity certified** or globally training-history screened. The technical-review hypothesis discussion below is preserved as the decision's original historical basis. Neither fresh live model generation nor training has been authorized.

## Review findings, by severity

| Severity | Finding | Resolution / remaining gate |
|---|---|---|
| P2 | SB-00 checked selected counts but trusted top-level pass flags and omitted per-language/group/gate reconciliation. Altering both top-level flags could report a pass with all 138 files wrong. | Reconstruct every original scope aggregate and pass flag at [diagnose_saved_outputs.py:226–275](../training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py#L226). Test forged top-level flags, nested gates, language totals, group counts and perfect synthetic responses. |
| P2 | SB-00 did not check the saved tokenizer, stage and authorization-contract identities; generation policy numeric comparisons accepted Python type aliases. Non-boolean completion flags could also be treated as failed generation rather than malformed evidence. | Require original SHA pins and exact policy types at [diagnose_saved_outputs.py:68–101](../training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py#L68), strict flags/token counts at lines 148–153. Synthetic refusal tests cover these gates. |
| P2 readiness gap | SB-01 had conservative ASCII byte bounds, but no actual frozen-tokenizer roundtrip/mask audit for this proposal. | Add read-only SHA-verified audit at [review_source_binding_microprobe.py:24–77](../training/phase3/copy-fidelity-v1/review_source_binding_microprobe.py#L24). All 48 records roundtrip; masks supervise exactly answer plus EOS. The original proposal is unchanged. |
| P2 readiness gap | SB-05 lacked an executable pure scorer; proposed pass criteria could not yet be regression-tested independently of model execution. | Add [score_source_binding_microprobe.py:41–139](../training/phase3/copy-fidelity-v1/score_source_binding_microprobe.py#L41) with all 48 ordered IDs, frozen provenance declarations, completion flags, anchored values and four capability gates. Synthetic thresholds and refusal tests added. |
| Execution blocker | No owner-approved intervention, new dataset/stage/baseline/authorization SHA packet or permission for new generation exists. | Specifications below are reviewable proposals. They do not enable a runner or reuse the previous training authorization. Independent review and fresh experimental permissions remain required. |

No P0/P1 defect was established in this scoped review. Source locations and regression tests are in [the diagnostic](../training/phase3/copy-fidelity-v1/diagnose_saved_outputs.py), [tokenizer audit](../training/phase3/copy-fidelity-v1/review_source_binding_microprobe.py), [pure scorer](../training/phase3/copy-fidelity-v1/score_source_binding_microprobe.py), [diagnostic tests](../training/tests/test_p3_source_binding_diagnostic.py) and [scoring tests](../training/tests/test_p3_source_binding_scoring.py). Linux and Windows required CI both include these tests. Final commit/CI evidence belongs in the PR; a local test pass does not establish remote Linux CI success.

**Local validation:** 51 source-binding tests passed; the prescribed Windows portability subset passed all 238 tests. Running the broader Linux-required test list on Windows gave 594 passes and two existing path-separator assertions in `test_request_grounded_coding_training.py:184` and `test_structured_plan_evidence_composition_training.py:182` (expected `/`, actual `\\`). Those tests and their source implementations are unchanged. Remote Linux CI remains the required evidence for that suite. An earlier sandboxed attempt also encountered temporary-folder ACL errors; the normal-access run removed those environment failures.

## SB-00: observed morphology and evidence

The saved local `evaluation-result.json` was read without modification, with all 138 ordered outputs reconciled against original candidate/control manifests and scorer aggregates. Its byte SHA-256 is:

`971a78971387200df9523c6ca177006f60f2834be0664c2b590e6ec72f06a197`

The new [technical-review evidence](../training/pretraining/p3-source-binding-technical-review-evidence.json) records the input SHA, complete aggregate diagnosis, examples, tokenizer audit and hashes of the three reviewed tools. Evidence file SHA-256:

`064c52150951c2510f2f87f6406b57634a0e9369525dd0f603e5ffafceb2e43b`

| Observation | Validation | Previously inspected development |
|---|---:|---:|
| Cases | 96 | 42 |
| Exact files / unrelated source preserved | 0 / 0 | 0 / 0 |
| Supplied namespace occurs anywhere | 0 | 0 |
| At least one known training namespace occurs | 96 | 42 |
| EOS / decode / full prompt visible | 96 / 96 / 96 | 42 / 42 / 42 |
| Empty / 192-token cap reached | 0 / 0 | 0 / 0 |
| Requested literal mentioned (edits only) | 14 of 72 | 3 of 36 |
| Requested value extractable with original boundaries | 0 of 72 | 0 of 36 |

There are **107 records in 37 repeated-output groups**, with **68 distinct outputs when keyed by scope, language and copy/edit condition**. These counts do not describe global unique files; repeats across keys are deliberately separate. Outputs range from 23–132 characters and 9–46 generated text tokens. The code-shape heuristic passes 106/138 responses; it is not a parser or a syntax-validity claim.

All six scope/language combinations have zero supplied-namespace presence and training-name intrusion in every response. Across validation, training-name hits are meadow 53, amber 24, prism 20 and cedar 13; development hits are prism 25, amber 15, meadow 9 and zephyr 3. Hits can overlap within one response and must not be summed as distinct cases.

Failure families, in priority order: (1) supplied source namespace lost in every case, including all unchanged copies; (2) training-name text appears instead, with repeated response templates; (3) request changes frequently produce reused outputs; (4) unrelated source structure/bytes change. EOS, missing prompts and output-budget exhaustion are unsupported explanations for this saved failure. Literal mentions in 17 edits provide **no independently extractable correct requested values** under the original contract. The observations support a namespace-focused intervention proposal, but do not prove memorization or isolate its neural cause.

## SB-01: integrity and contamination scope

The fixture remains SHA `0a30df999f8e1c0f12cf0f6e4a71e0eab814bdd99b3bd83d0e2184637ad31fda`: 48 records, 12 copies, 36 edits, 12 complete groups; three languages, two contexts and names `ulvar`/`nyxen`. All source-to-target relations and deterministic bytes pass existing tests. The four SHA-pinned historical P3 candidate/control files exclude both new names and have zero new source/target overlap. This is a claim about **those four files**, not every earlier pretraining corpus. Before treating names as globally unseen in a future training study, a separately approved candidate review must enumerate and SHA-pin all historical training corpora; missing history must block that claim. Do not read the final holdout to screen names.

The actual tokenizer SHA remains `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`. Read-only measurement gives maximum 95 prompt tokens, 50 answer/EOS tokens and 145 combined tokens; 1,296 total answer/EOS positions. Prompts also retain room for the proposed 192-token generation ceiling. This audit neither materializes a training bundle nor performs teacher-forced model evaluation.

```powershell
training/.venv/Scripts/python.exe training/phase3/copy-fidelity-v1/prepare_source_binding_microprobe.py
training/.venv/Scripts/python.exe training/phase3/copy-fidelity-v1/review_source_binding_microprobe.py
```

## SB-02: falsifiable hypothesis tests

All tests below require later experimental authorization. Match starting checkpoint, examples, optimizer budget, request wording, context, scoring and sampling policy except for the named independent variable. Report exact full files by copy/edit, language and group. A matched training comparator is a separate proposed arm requiring its own approval; historical loss or 0/138 alone cannot establish a causal comparison.

| Hypothesis | Single independent variable / matched comparison | Negative control | Disconfirmation |
|---|---|---|---|
| A: training-name templates dominate supplied names | Train-source namespace diversity: six fixed names versus a frozen diverse pool, keeping row count, templates, copy ratio and literal distribution identical. Match per-record token lengths or block before training. | Same-source, same-request cases; also inspect performance on familiar names to detect general code degradation. | On a separately frozen, approved unseen-name diagnostic, unfamiliar-name preservation and exact files do not improve over matched comparator despite successful optimization. Low loss alone cannot support A. |
| B: copy supervision is insufficient | Copy proportion only: 25% versus 50% at equal update/draw budget and namespace policy. Report differing answer/EOS exposure explicitly; this tests the allocation intervention, not a mechanism independent of exposure. | Unchanged inputs must still be copied when edit wording/value distributions are fixed. | Copies remain wrong or only familiar names improve; edit accuracy regresses below its predeclared floor. |
| C: requested values are weakly bound to instructions | Request value only, using intact four-case same-source groups and all other prompt bytes identical. Compare exact outputs and anchored values across requests, without changing weights. | Unchanged-copy request, and identical request repeated under identical greedy policy. | All edit groups track their requested values correctly and only identifiers/unrelated bytes fail; value binding then does not explain the primary failure. Unextractable values remain inconclusive. |
| D: complete-file generation creates most of the burden | Output contract only: full-file text versus bounded replacement text with a frozen deterministic reconstruction rule, same input and requested change. | No-op replacement and wrong-offset reconstruction controls. | Both contracts fail exact reconstructed files on matched cases. A span-contract pass never changes the historical full-file score. Any training for a new contract needs separate approval. |

**Proposed first intervention: A, deterministic train-namespace diversity.** Evidence: all 30 copy cases lose their supplied names; every saved response contains a training name; EOS/context/budget pass. This is stronger direct support for addressing namespace reliance than for immediately changing output contracts or enlarging the model. A remains a hypothesis. B is secondary; C's existing anchored-value evidence is unextractable; D remains plausible even on short files and needs its own controlled comparison. No live SB-01 result has been observed, so ranking may change after separately approved probes. Owner selection and independent review remain pending.

## SB-03: candidate reproducibility specification (not constructed)

Proposed candidate keeps the earlier curriculum topology: 288 train and 96 validation records, two source shapes, two contexts, three languages and four-case copy/edit groups; 25% copies. The intervention changes train namespace diversity only. Preserve a comparator with six fixed names and the same ordered row/template/request topology. Predeclare how 48 deterministic train names map onto the 72 train groups; use each name only inside its assigned language/shape/context group(s). Hold validation names separate. This is an authored finite augmentation, not runtime randomization.

Before construction, approve a complete name-pool algorithm and mapping table, including the deterministic collision-rejection rule. Then freeze canonical ASCII JSON with sorted keys and a final LF; freeze exact row ordering, IDs, character offsets, requests, expected complete files, source hashes and every record's token counts. Reject nonunique target literals, malformed copy/no-op targets, source/target mismatches and broken four-case groups. Derive any token-matched comparator before inspecting new model outputs; if exact token matching cannot be achieved, revise the proposal for review instead of claiming a single-variable comparison.

Keep `ulvar`, `nyxen`, all historical validation/developer namespaces and every source/target/prompt/group overlap out of gradients. Candidate names must be disjoint across train/validation/development and checked against an enumerated SHA-pinned historical training inventory. Report exact-file cross-split overlap and namespace/subsequence overlap separately. The final holdout must remain unopened; provenance and a prohibition on loading it replace content scanning of that set. No new split or name pool has been created by this review.

Pack each record independently as prompt + answer + EOS with no truncation or cross-record attention. For prompt length P and answer length A, next-token mask is P−1 zeros followed by A+1 ones; masks and token arrays both have P+A positions. Verify IDs are ordinary 4–16383 except final EOS=3, complete roundtrips, P+A+1≤512, P+192+1≤512 and A+1≤192. Pin tokenizer bytes above and the original model config SHA before building. Enforce equal example weighting with token-normalized answer/EOS loss, preserving the predecessor's accumulation semantics rather than averaging prompt loss.

Sampler proposal: canonical train-row order, `random.Random(1337).randrange(288)` for each of exactly 1,600 draws, grouped into 100 updates of 16 microbatches. Freeze the resulting complete draw-index array, its canonical JSON SHA, Python/runtime identity and supervised-target totals. Future execution reads the frozen array instead of depending on cross-runtime PRNG behavior. Validation rows never enter draws. Pin both the actual ordered data and the complete sampler, not only a seed.

Starting checkpoint proposal: historical fixed step-100 SHA `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`. Independently verify its full lineage and tensor/config identity before any separately approved stage. Fresh optimizer state; no resume from inherited moments. A fresh destination, 200 GB aggregate artifact budget, bounded serialization, interruption-safe publication and refusal to overwrite any checkpoint/report/source are required. Freeze dataset, tokenizer/config, sampler, bundle, stage, baseline and authorization SHAs before execution. **Those future SHAs do not exist yet: their absence is a hard stop, not permission to accept placeholders.** No bundle or checkpoint stage was built.

## SB-04: bounded training authorization proposal (unapproved)

| Setting | Proposed limit |
|---|---|
| Starting weights | Fixed step-100 SHA above; no step-75 selection |
| Architecture | Same 27,566,080 / 16,384 / 512 |
| Device / seed | CUDA, seed 1337; record exact runtime and determinism settings |
| Optimizer | Fresh AdamW; LR 0.0001; betas (0.9, 0.95); eps 1e-8; weight decay 0.1 |
| Batch / objective | Microbatch 1; accumulation 16; answer/EOS only; global gradient clip 1.0 |
| Updates / wall time | At most 100 updates and 600 seconds including validation and saves; stop at whichever limit is reached first |
| Validation / saves | Validation 0/25/50/75/100; saves 25/50/75/100; only a successfully completed step-100 endpoint eligible for evaluation |
| Artifact budget | 200 GB total; fresh destination with declared disk headroom and no overwrite |
| Continuation | No resume, automatic extension, second run or checkpoint reselection |

Before a future runner imports a model loader, opens a checkpoint or creates an optimizer, require fresh explicit owner permission for the named intervention, an independently reviewed SHA-pinned authorization packet, exact candidate/sampler/stage/baseline pins and both CI platforms passing on its final PR commit. The previous authorization is inapplicable. This change adds **no training flag or runner**. Approval must separately identify any matched comparator arm; one intervention approval does not authorize two runs.

Abort on any identity/config/count/mask discrepancy, overlap, missing history, insufficient disk, nonfinite loss/logits/gradients, invalid sampler draw, partial publication or time cap. Check time before each update, validation and save and immediately afterward; an operation that crosses 600 s makes the run timed out and its endpoint ineligible, even if the operation cannot be interrupted exactly at the deadline. Preserve failures and partial artifacts with explicit status, never resume or promote an earlier checkpoint because loss was better. An interrupted or incomplete step-100 run has no official capability endpoint.

## SB-05: frozen full-file scoring contract

[Pure scorer](../training/phase3/copy-fidelity-v1/score_source_binding_microprobe.py) accepts 48 supplied response dictionaries in the fixture's exact order. It verifies the unchanged fixture SHA, explicit declared checkpoint/tokenizer identities, string text, ordinary token-count bounds and strict boolean EOS/decode/full-prompt flags. Exact success requires all three flags and character-exact output (equivalently UTF-8 byte-exact for these valid strings). No whitespace normalization, fence stripping or substring acceptance. Copy success means the entire source is unchanged. Edit preservation requires nonoverlapping exact prefix/suffix boundaries, with the requested value extracted only between them. If those boundaries or completion flags fail, requested correctness is null, counted as unknown rather than a known wrong value.

Report per-case raw text/flags/count, per-language and total copy/edit/exact/preservation scores, wrong versus unknown requested values and fully correct four-case groups. Source namespace lexeme multisets provide an additional lexical diagnostic including comments/attribute values. They do **not** establish syntax, source locations or complete-file preservation; all capability gates use exact files. Synthetic tests demonstrate that name-only output can pass the lexical metric while failing capability.

All four unchanged proposal gates must pass: **12/12 copies, ≥30/36 edits, ≥10/12 edits in each language, ≥9/12 complete groups**. Preservation and identifiers are therefore exact in every counted successful file; no extra 100%-of-all-edits preservation gate is silently added. Publish preservation across failures too. These are the original proposed criteria, not proof that the owner has approved model execution.

The scorer's identity arguments are caller declarations; synthetic scores cannot attest checkpoint execution. It labels execution provenance unverified and live evaluation unauthorized. No new model-output reader or generation entrypoint exists. A future approved evaluator must independently hash weights/config/tokenizer, verify the separately authorized checkpoint against a **new** contract if different from the historical endpoint, freeze greedy policy (seed 1337, temperature 0, 192-token ceiling), store raw responses once in a fresh destination and freeze report SHA before scoring. It must score the original 96/42 controls separately under their existing identities/criteria. Do not pool them with 48 probes or label inspected controls a final holdout. No evaluation contract may change after outputs are observed.

## Readiness and remaining decisions

Technical implementation is reviewable with recorded SB-00 evidence and verified SB-01 fixture. SB-02 intervention selection, fully pinned candidate/sampler design, independent proposal review, fresh staging/training/generation approvals and genuine exact-file capability remain outstanding. Keep #189 open. The final project holdout remains sealed.
