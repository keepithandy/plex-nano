# R3 · Comparison design review

> **Recommendation: prepare a distinct six-versus-48 comparison in which both arms use history-screened spellings.** A model-free witness satisfies exact token matching between these new arms. This is a proposed replacement design, not an approved experiment or a successful V2 revision.

**Current question:** can we test training-name diversity without accepting the original V2's eight historical collisions or losing token matching?

[Options](#options) · [Measured feasibility](#feasibility) · [Proposed comparison](#comparison) · [Claims and outcomes](#claims) · [Remaining work](#remaining)

<a id="options"></a>

## 1. Options reviewed

| Option | What it preserves | What it costs or changes | Disposition |
|---|---|---|---|
| Keep the original comparator and continue strict V2 selection | Original six familiar training names, exact original token lengths, history exclusion for replacements | The existing deterministic search failed at `prism.0`, including the 20,000-candidate maximum. Feasibility beyond that method is unproven. | Preserve the failure; no further blind expansion of the same search. |
| Use the original unreviewed V2 with a narrower claim | Exact original token lengths and six-versus-48 composition | Eight of the 48 replacement spellings occur in ancestor text. Diversity and historical spelling exposure differ together. | Valid as a documented mixed-exposure proposal; does not meet the original exclusion gate. |
| Keep the original comparator and allow token-length differences | Original comparator and screened replacement spellings | Equal draws no longer imply equal supervised tokens. Token-normalized sampling would change example exposures, order or update accounting. | A different protocol; lower priority while an exactly paired alternative is feasible. |
| **Rebuild both arms: six screened spellings versus 48 screened spellings** | **History screening, exact per-row token matching between arms, original tasks and validation rows, common draw sequence** | **Changes the comparator too. Tests diversity among screened spellings rather than replacing the six familiar spellings.** | **Recommended for a separate preparation and design review. Actual tokenizer feasibility demonstrated below.** |

The last option removes the requirement that every replacement match the original `prism` tokenization. Both new arms can use equally segmented names while keeping the frozen tokenizer. It does not weaken the history matching rule.

BPE can represent unfamiliar words using sequences of subword units; a new full spelling need not be a new token. That is background for the design, not evidence that Plex can copy it correctly. [Sennrich, Haddow and Birch, 2016](https://aclanthology.org/P16-1162/).

<a id="feasibility"></a>

## 2. What the model-free check established

A deterministic, bounded witness was prepared solely to test whether the recommended alternative is concrete. No model output or loss was used to choose names. The search tag, first-eligible selection rule, 2,500-attempt limit per selection, name maps, rows and source identities are recorded in the artifacts below.

| Check | Actual result |
|---|---|
| Lower-diversity arm | 6 new training spellings, one per original 48-row namespace family |
| Higher-diversity arm | 48 distinct training spellings, eight slots per original family |
| Combined spelling pool | 54 unique names; the two pools are disjoint |
| Ancestor history | All 10 hashes verified; 7,134,286 raw bytes and 4,163 parsed training-text rows screened |
| Proposed-name history matches | **0**, using case-insensitive ASCII substring matching in both raw bytes and parsed `text` |
| Additional screening | All five frozen P3 fixtures checked; no full training-file overlap with their source/target files |
| Train rows per arm | 288: 72 copies and 216 edits |
| Validation per arm | The **same original 96 rows**, unchanged |
| Four-case groups per arm | 72 train and 24 validation, all intact |
| Paired prompt and answer-plus-EOS lengths | **384/384 equal between the new arms** |
| Logical prompt/answer supervision spans | 384/384 equal; actual packed training masks remain unbuilt and unverified |
| Draw sequence | Original SB03 SHA-permuted 1,600 indices; all 288 train rows drawn 5–6 times |
| Draw-weighted prompt tokens | **111,950 in each arm** |
| Draw-weighted answer-plus-EOS targets | **43,841 in each arm** |
| Maximum prompt / answer-plus-EOS | 98 / 54 tokens, within existing context and response budgets |
| Prospective evaluation-name matches in either train split | 0 for all ten previously reviewed evaluation spellings |

The paired rows preserve requests, task language, source/target edit relation, target offsets, source literals, requested literals, byte lengths, ordering and group composition. Namespace fields and namespace-bearing row/group IDs change in each train arm. Validation rows do not change.

**Difference from the original design:** 248/384 paired token signatures change relative to the original curriculum, although the new arms match each other exactly. Their 43,841 draw-weighted targets exceed the original SB03 comparator's 40,439 by 3,402, or about 8.41%. Therefore this witness cannot be compared with the historical training run as if it were the original matched comparator. Both proposed arms would need fresh runs from the same fixed checkpoint.

### Frozen evidence

| Artifact | SHA-256 |
|---|---|
| [Design witness, including both arm row arrays](../training/pretraining/p3-sb03-r3-paired-new-name-design-witness-v1.json) | `58a57d6754852c09a1b933be733132ebf364ee163f25f09e02e3f4a1c4a229e6` |
| [Feasibility receipt and 384 paired signatures](../training/pretraining/p3-sb03-r3-paired-new-name-feasibility-review-v1.json) | `23020ec2d47c3e4901d9ccce10a34dd86dd21544b090d604da7ce990569e2236` |
| [Execution and second-pass review, including driver sources](../training/pretraining/p3-sb03-r3-comparison-design-review-execution-v1.json) | `ef186851469ffc345513b69bd49ab64e579b5ba4978ab3b35e0a0cbf7fc7afbb` |

These are design-review artifacts. They are not reviewed training manifests, packed bundles, stages or execution permissions. A second pass by the same assistant separately reconstructed row changes, checked all 384 logical supervision spans, recomputed draw totals, and rechecked history matches and preserved hashes. External independent review remains outstanding.

<a id="comparison"></a>

## 3. Proposed comparison to review next

Use a distinct protocol identity, provisionally **P3-SB03-R3-PAIRED-NEW-NAMES-V1**, with arms **L6** and **D48**. Do not mark the blocked V2 gate complete or relabel this witness as an approved V2 manifest.

| Dimension | Proposed common condition or declared difference |
|---|---|
| Research question | At a fixed starting checkpoint, does this 48-name training curriculum improve supplied-code copying/editing relative to this paired six-name curriculum? |
| Starting checkpoint | Same fixed copy-fidelity step-100 endpoint, SHA `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230` |
| Model and tokenizer | 27,566,080 parameters, 512-token context, frozen 16,384-token tokenizer SHA `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697` |
| Comparator | L6: six spellings screened against the recorded pre-run history |
| Intervention | D48: 48 different spellings screened against that same history |
| Tasks | Same 288 ordered train tasks and original 96 validation tasks |
| Token matching | Equal prompt and answer/EOS lengths for every paired row, with identical logical loss spans and draw totals |
| Sampling | Same 1,600-index sequence, canonical SHA `9ddf29a3a2972cee0c8c2f79db97d7ef3f823065a2db6ee0840e89bb01760c13` |
| Future execution conditions | Same optimizer initialization, declared seed, learning rate, accumulation, update ceiling, validation schedule, endpoint rule and device policy; exact values and wall-time limits remain for the separate execution packet |
| Output contract | Complete supplied HTML/CSS/JavaScript file with exactly the requested change |

Matching token counts does not match token identities, subword frequencies, spelling difficulty, or learned representations. These particular name pools and their allocations are part of the intervention. One pool pair and one training seed support only a bounded pilot conclusion; confirming a general diversity effect requires further independently chosen pools/seeds.

<a id="claims"></a>

## 4. Defensible claims and proposed outcome rules

**Preparation claim already supported:** a six-versus-48 design witness exists with exact token-count matching between arms, unchanged validation rows, and no name matches in the ten listed hash-verified ancestor training texts under the stated raw/parsed ASCII rule.

This does not establish globally unseen names, unseen constituent tokens, complete undocumented-history exclusion, coding improvement, or a benefit caused solely by the number of names. The original V2 still has eight collisions and failed its exclusion gate.

**Proposed pilot effect measure:** `deltaExact = D48 exact files - L6 exact files` on the same frozen 48-case SB-01 microprobe, with both raw counts and the difference out of 48 reported. Publish paired per-case outcomes, 12 whole four-case groups, copy/edit counts, language breakdowns, source preservation, request-value selection and unrelated-byte preservation. Do not treat the four related cases in a group as independent replications.

The directional pilot hypothesis is `deltaExact > 0` for this fixed setup. A positive difference is local movement, not a statistical significance claim. Zero or a negative difference does not support that directional hypothesis in this setup; retain it as the result. It does not prove that name diversity cannot help any other model or curriculum.

**Capability and comparison effect are separate axes.** Retain the existing SB-01 capability gates:

| Metric | Unchanged requirement |
|---|---:|
| Exact copies | 12/12 |
| Exact requested edits | At least 30/36 |
| Exact edits per language | At least 10/12 |
| Fully correct four-case groups | At least 9/12 |

A positive difference while D48 fails a capability gate is limited improvement. D48 passing all gates supports recovery on this probe; if L6 performs equally well or better, the result does not support the proposed diversity advantage. Provenance failures or incomplete runs make the comparison inconclusive rather than a model-capability pass.

The frozen 96 validation rows remain loss-monitoring material. Previously inspected controls remain regression diagnostics with the [recorded evaluation-name exposure limits](PHASE-3-R2-FROZEN-EVALUATION-NAMESPACE-REVIEW-2026-10-18.md). SB-01 has only two source namespaces and familiar task structures; it does not by itself establish broad structural transfer. A later fresh confirmation set belongs to R6; the final project holdout remains sealed.

Keep requested values unscorable where target extraction is unavailable, rather than turning missing evidence into a proven wrong choice. Require exact complete-file or valid anchored evidence for source binding. The current scorer's historical checkpoint restriction must be revised and reviewed before it can certify L6 or D48 output provenance.

<a id="remaining"></a>

## 5. Roadmap from this review

- [x] Compare the available design options and document the changed research question.
- [x] Demonstrate a concrete paired six-versus-48 witness with the actual frozen tokenizer.
- [x] Verify the listed-source exclusions, paired lengths, original validation equality, group integrity and draw totals; preserve prior artifacts and failures.
- [ ] Adopt or revise the proposed distinct comparison protocol. The recommendation is not yet an adopted experiment.
- [ ] Independently review the witness, source inventory scope and exact outcome rules; freeze separately identified L6/D48 manifests after review.
- [ ] Version and validate arm-specific scoring/provenance against the new manifest identities.
- [ ] Under separate R4 decisions, prepare actual packed bundles and zero-update stages, verify their masks and starting-weight equality, and freeze per-arm execution packets.
- [ ] Obtain separate training and evaluation authorization before any model execution.

**Preservation verified:** V1 SHA remains `4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea`; the original unreviewed V2 remains `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21`; the [failed reselection evidence](PHASE-3-R3-V2-HISTORY-RESELECTION-2026-10-18.md) remains unchanged. Historical 0/138 exact files remains a failed capability result.

No checkpoint was loaded or changed. No training, inference, optimizer update, training bundle, checkpoint stage or final project holdout access occurred.

[Return to the active R3 roadmap](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md#r3)
