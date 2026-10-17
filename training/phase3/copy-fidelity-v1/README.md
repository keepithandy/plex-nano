# Phase 3 — Copy-fidelity recovery curriculum v1

**Status: deterministic candidate frozen; neither training nor inference is authorized by this preparation.** This is replacement recovery, separate from formal roadmap P3-03 Insert and Delete.

## Evidence and diagnosis

The owner reported the October 17, 2026 fixed step-100 42-task evaluation: **0/42 exact full files, 0/6 exact unchanged copies, 0/18 minimal edits, 0/18 contextual edits, 0/12 complete three-request contrast groups**. All 42 responses terminated at EOS. The edited target values were **36/36 unscorable independently** because source boundaries were not retained. The full response envelope is on the owner's Windows PC, not GitHub.

All six inspected copies replaced unseen cobalt/spruce identifiers with prior-style cart/settings identifiers, sometimes producing malformed code. In the earlier P3-02 authored training split, cart occurs in 36/144 records and settings in 36/144. **Template fallback is an evidence-based hypothesis, not a proven internal mechanism.**

The P3-02 generator creates 216 authored edit examples (144 train, 36 validation, 36 development) from six repeating namespaces, with **zero exact-copy requests**. Its answer_only_target_mask supervises complete edited answers and EOS while masking the prompt. This is valid causal language-model training, but not enough to establish source-copy generalization; earlier decreased validation loss did not pass P3-02's 0/36 exact-edit development gate.

## Frozen candidate and data split

Manifest: [copy-fidelity-candidate-v1.json](copy-fidelity-candidate-v1.json). Generator: [prepare_copy_fidelity.py](prepare_copy_fidelity.py).

**SHA-256, canonical UTF-8 LF:** 9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a

| Example condition | Train | Validation |
|---|---:|---:|
| Return supplied file unchanged | 72 | 24 |
| Replace exactly one source literal | 216 | 72 |
| **Total** | **288** | **96** |

There are 96 train + 32 validation examples per language (HTML, CSS, JavaScript). Each source group contains one copy request and three different literal-edit requests on the *same source file*, retained within a single split (96 groups). Half the examples use short minimal code, half have unrelated context; two source shapes per language vary selector/identifier syntax. Complete-file outputs and EOS are the intended training target, not short diff fragments.

Train-only namespaces: **cedar, lunar, prism, amber, zephyr, meadow**. Validation-only namespaces: **granite, saffron**. Previously evaluated controls use **cobalt, spruce**, and remain excluded from train and validation. No complete-file inputs or outputs overlap those in the prior P3-02 authored candidate, frozen 72-task diagnostic or frozen 42-task control. Train and validation also have **zero group, prompt, full-input and full-output overlap**; train/validation share broad natural-language request patterns intentionally, so this is an *identifier-transfer* experiment, not a held-out request-paraphrase test.

All candidate text is ASCII. Maximum prompt length is **313 bytes**, maximum answer is **127 bytes**. Even a conservative byte-BPE upper bound gives 313 prompt + 192 generation + 1 step = **506/512**. Actual tokenizer SHA, encode/decode round-trip and packed-token checks still require an independent local review before model work. Unit tests do not exercise the model.

Verify immutable data from the repository root using these PowerShell commands:

    python training/phase3/copy-fidelity-v1/prepare_copy_fidelity.py --verify
    uv run --project training python -m pytest training/tests/test_p3_copy_fidelity_curriculum.py -q

The builder refuses to overwrite an existing file.

## Proposed measurements and decision gates

**Primary validation copy requirement:** 24/24 complete byte-exact files (8/8 each language), identifiers unchanged, EOS present.

**Validation edit requirement:** at least 60/72 complete exact edits and at least 20/24 per language. Unrelated bytes must remain exact on every credited edit; requested value must be called **unscorable** when its source boundaries were altered.

**Contrast sensitivity:** at least 18/24 validation three-request edit groups must all be correct. Record syntax-check status independently; simple structural validation of these restricted HTML/CSS/JavaScript shapes is not the same as general language parser verification.

**Historical controlled development comparison:** the old 42 tasks remain unchanged, with baseline 0/6 copy and 0/42 exact. For proposed advancement, require 6/6 copy and >=30/36 edits, all supplied names preserved. These controls were seen by the curriculum designer and are **not a pristine final held-out test**. Require genuinely fresh, independently reserved confirmation before claiming broadly improved model capability; leave the final project holdout closed.

Record exact files, EOS, full-source preservation, per-language and per-condition counts, complete edit contrasts, syntactic completeness, token loss and generated outputs. A lower training or validation loss alone is insufficient.

## Proposed one-run training ceiling — NOT AUTHORIZED

| Property | Fixed proposal |
|---|---|
| Initial model weights | P3-02 step-100; SHA-256 020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d |
| Model/tokenizer | 27,566,080 parameters, 16,384 byte-BPE; frozen SHA 2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697 |
| Context / output ceiling | 512 input-context tokens / max 192 newly generated |
| Loss | Supervise complete answer + EOS, mask prompt and padding, as P3-02 |
| Optimizer | Fresh AdamW; do not silently reuse previous optimizer state |
| LR / seed | 0.0001 / 1337 (review before authorizing) |
| Batch / accumulation | 1 / 16 |
| Max optimizer updates / wall time | 100 / 600 seconds |
| Validation steps | 0, 25, 50, 75, 100 |
| Proposed fixed endpoint | Step 100 only if the bounded run completes, no post-hoc checkpoint selection |
| Development inference | Greedy, temperature 0, seed 1337, <=192 output tokens |
| Auto resume / retries | Forbidden |
| Training authorized / performed | **NO / NO** |
| Evaluation authorized / performed for this candidate | **NO / NO** |
| Project final holdout opened | **NO** |

**Blocking prerequisites:** verify the frozen tokenizer and whole-record fit locally; validate next-token target mask and train sampler/weighting; verify actual original checkpoint identity and storage budget; review independent validation and inference contract; implement safe stage and training commands; obtain fresh explicit owner approval for a bounded optimizer run. This PR deliberately adds **no trainer or optimizer entry point** and does not reuse P3-02 training authorization.

## Research limitations

The 384 examples are a small hypothesis-driven synthetic intervention, not evidence that 27M parameters suffice for exact copying. They intentionally isolate unseen identifiers and literal/source preservation but do not evaluate paraphrase robustness. Historical numeric results and manifests are immutable. No final holdout access, checkpoint reselection, model-size change or autonomous repository-agent pivot.

Related: [original 42-task preparation](../diagnostics/p3-copy-edit-controls-v1/PREPARATION.md), [decision issue #177](https://github.com/keepithandy/plex-nano/issues/177), [recovery issue #181](https://github.com/keepithandy/plex-nano/issues/181).
