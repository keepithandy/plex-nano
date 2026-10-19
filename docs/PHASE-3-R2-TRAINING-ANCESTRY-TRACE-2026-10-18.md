# R2 - Fixed-checkpoint training ancestry

> **Tracing complete.** The later [recorded-input coverage review](PHASE-3-R2-TRAINING-INVENTORY-COVERAGE-REVIEW-2026-10-18.md) found no missing training text or unresolved source identity. The subsequent [collision screen](PHASE-3-R2-ANCESTOR-NAME-COLLISIONS-2026-10-18.md) found eight proposed names in ancestor text. The original trace receipt preserves its pre-review state.

**Verified locally: October 18, 2026.** Ten training-text files, ten training-token files, and eleven checkpoint files (including the fresh initialization) were hashed. Every listed training text/token hash matched its provenance record. The ten training splits contain **4,163 records** in total; this is a sum of split memberships, not a claim of unique records across stages.

## Fixed endpoint and recorded chain

**Fixed checkpoint:**

`C:\Users\quali\PlexArtifacts\file-edits\p3-copy-fidelity-v1-first-run\checkpoints\step-0100.pt`

**SHA-256:** `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`

```text
Fresh random initialization (P2-28 step 0)
  -> P2-29, 500 updates
  -> P2-30 -> P2-32 -> P2-35 -> P2-38 -> P2-41
  -> P2-44 -> P2-48 -> P3-02 -> P3 copy-fidelity v1
     (each later endpoint is fixed step 100)
```

Each later endpoint records the preceding endpoint SHA in its saved `stageTransitionRecord.baseCheckpointSha256`. All nine such links matched the actual parent checkpoint bytes. P2-29 has no stage-transition record; its result pins the initialization checkpoint and initial weights, and its saved initialization record agrees. Its training contract explicitly describes replay from step zero rather than continuation of P2-28 step 100.

**The P2-28 trained pilot is a sibling experiment.** Earlier pre-reset Phase 1/2 checkpoints are outside this weight ancestry. The initialization record declares fresh random initialization, seed 1337, and no pretrained checkpoint/model weights. Its recorded initial-weights SHA is `98785d70ee7f68fcfde35ad6136bb3be05ff562374f0bdede2faae93cb80a193`; this audit verified the initialization checkpoint file, without reconstructing tensors to recompute that internal weights hash.

## Training-text inventory

| Stage | Records | Location | SHA-256 prefix |
|---|---:|---|---|
| P2-29 | 3071 | Repository-local artifact | `8d621a488f35` |
| P2-30 | 156 | Repository-local artifact | `05fb9b35277b` |
| P2-32 | 72 | Repository-local artifact | `239eaf48f6ad` |
| P2-35 | 108 | Repository-local artifact | `b921dd54657e` |
| P2-38 | 72 | Repository-local artifact | `ecff52a775a2` |
| P2-41 | 72 | Repository-local artifact | `458de009e4ef` |
| P2-44 | 108 | Repository-local artifact | `0141dbd0209d` |
| P2-48 | 72 | Repository-local artifact | `39879c1199e9` |
| P3-02 | 144 | Private owner-local | `45d42817582f` |
| P3-copy-fidelity-v1 | 288 | Private owner-local | `682e12d15162` |

The [explicit inventory](../training/pretraining/p3-sb03-v2-ancestor-training-text-inventory.json) contains full absolute paths and full hashes in the existing auditor format. Its SHA-256 is `111784c169f9d180b8ec5329de7acda9f0259214847136a7fab3684ce24b58d1`. The [machine-readable receipt](../training/pretraining/p3-sb03-v2-training-ancestry-trace.json) also records checkpoint paths/hashes, parent links, dataset records, bundle manifests, training-token paths/hashes, and supporting provenance-file hashes.

### Exact paths and full hashes

Expand a stage to inspect its text, binary training input and checkpoint identity. Each text path identifies the frozen training split, rather than a complete candidate containing other splits.

<details>
<summary><strong>P2-29 - 3071 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\datasets\p2-26-web-batch-2-v2\train.jsonl`

**Text SHA-256:** `8d621a488f3554a976ea4ee0e9581fae2c07aff58ad7912b2f69f2cc0233aaa6`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\tokenizer-reviews\p2-27-web-v1\candidates\vocab-16384\train.tokens.u16le`

**Token SHA-256:** `a8ce0d446eb2edc7fbf1d93718a0696f24ef9d869300d64ec7b8aa87bb80278d`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\tokenizer-reviews\p2-27-web-v1\candidates\vocab-16384\manifest.json`

**Manifest SHA-256:** `fb650f91bbc7f53e788a7f7c22af4950b079423229c1ef85bdd865025e56e880`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\pilot\p2-29-web-v1-500step\pilot-checkpoint.pt`

**Checkpoint SHA-256:** `3b8303f8a6f56527329774b51278b93588b8383205e85bb2f594a58349acca8e`

**Path provenance:** `training/pretraining/p2-28-training-contract.json`.

</details>

<details>
<summary><strong>P2-30 - 156 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\task-finetune\p2-30-request-v3-16k\train.jsonl`

**Text SHA-256:** `05fb9b35277b979ef45473af5b2a2c98bef31ee2c1c3133f0af0c68d7cfba554`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\task-finetune\p2-30-request-v3-16k\train.tokens.u16le`

**Token SHA-256:** `27d6682349c4917aba43c6e9eadd95f468b692eb718ea330582302b9e2bacf58`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\task-finetune\p2-30-request-v3-16k\manifest.json`

**Manifest SHA-256:** `7753b1518b737e7f6b8e0b64f4f829b027ed3aad5b33a7613b3715481448eec5`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\task-finetune\p2-30-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `28064a22f322d6b9cde04c2424f3c257de8c0803245c1db83072ab29c67f6d6e`

**Path provenance:** `training/pretraining/p2-30-preparation-result.json`.

</details>

<details>
<summary><strong>P2-32 - 72 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-32-training-bundle\train.jsonl`

**Text SHA-256:** `239eaf48f6ad5701e0abe7fcb1e98a2ea379a7377d7e8c523f267e5b9dbd4e71`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-32-training-bundle\train.tokens.u16le`

**Token SHA-256:** `2048a0a623efaa5471b33077066524c679f5ce591ad0c0d4a0b6917e2bc47935`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-32-training-bundle\manifest.json`

**Manifest SHA-256:** `a0d3663bf3ba9a9653ceb52e35594bf3271e4e56cc483dd66f9d149ca38bc6fd`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-32-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `707e46f9e3e87cdd9beec705e2bd55701b37a40e79aad7ab93858fa63f8ebcf4`

**Path provenance:** `training/pretraining/p2-32-first-run-contract.json`.

</details>

<details>
<summary><strong>P2-35 - 108 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-35-training-bundle\train.jsonl`

**Text SHA-256:** `b921dd54657e77619835c5d9bce7a92c12c3a8f388d8ebe8ab8f6a9550ad69f3`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-35-training-bundle\train.tokens.u16le`

**Token SHA-256:** `1780c059901706a1568e37729c8fd9697c135a3dfa3a798fb6d2affb3715cec9`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-35-training-bundle\manifest.json`

**Manifest SHA-256:** `17f02f7f0b786be770f964b445854684fee0a010793672149e7fcc6c611aae5d`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-35-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `1fafce16260ab8910465517c7f571b34dccf7f21ec1ad9d281dad53ab87fbcd0`

**Path provenance:** `training/pretraining/p2-35-first-run-contract.json`.

</details>

<details>
<summary><strong>P2-38 - 72 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-38-training-bundle\train.jsonl`

**Text SHA-256:** `ecff52a775a2f876b2a243b3254c37ac05bd75c72b372d2ecaddc06ae82b6ecf`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-38-training-bundle\train.tokens.u16le`

**Token SHA-256:** `8a27ad57643e08e4471b22bdd1bab623455cfe11fe3066f75dfae9d6f1cc9894`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-38-training-bundle\manifest.json`

**Manifest SHA-256:** `9a69a848cdf26f427a83d30590e40e447075c6a2397629084c824f20d6b81158`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-38-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `9117e34433d6faa404117f557a48d12e840355ed5c7580d5b60f8e565564dbf6`

**Path provenance:** `training/pretraining/p2-38-first-run-contract.json`.

</details>

<details>
<summary><strong>P2-41 - 72 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-41-training-bundle\train.jsonl`

**Text SHA-256:** `458de009e4ef227361e2813c7380bfa2d42739f36335ed333ea3b6a250800db3`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-41-training-bundle\train.tokens.u16le`

**Token SHA-256:** `6026967d77319d9f4aaea205a87204f51c73f624a1603bb7f0991f24732a961e`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-41-training-bundle\manifest.json`

**Manifest SHA-256:** `a05e09493778ca772d583a17e01fbd3a5efa84c3897f7865f89ab9679518b22a`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-41-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `adec7fa31e40a52caa89aa7bb7150983ce7c4f1c5df899285cec3e3f24bf79cc`

**Path provenance:** `training/pretraining/p2-41-first-run-contract.json`.

</details>

<details>
<summary><strong>P2-44 - 108 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-44-training-bundle\train.jsonl`

**Text SHA-256:** `0141dbd0209d10e2138a0a90ff688e97b4b1de39f91c28ca367ed7267b309789`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-44-training-bundle\train.tokens.u16le`

**Token SHA-256:** `cf73a877ae67ad60abf51566417fe76a0df2672df476f6bb0929d4442888d6e3`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-44-training-bundle\manifest.json`

**Manifest SHA-256:** `46d6e4501d56c45196e604fb78dafaaada49e6c386badaae07ffbd4d6794f5d6`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\structured-plan\p2-44-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `69c357db13aae930374f013754a4b89c9bc071bd299c34cfa1c1ab362db0842e`

**Path provenance:** `training/pretraining/p2-44-stage-source-preparation-contract.json`.

</details>

<details>
<summary><strong>P2-48 - 72 training records</strong></summary>

**Training text:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\request-grounded\p2-48-training-bundle\train.jsonl`

**Text SHA-256:** `39879c1199e975096ba962fcad023b3b112c2c8b6444c0a3a9f35971a945d530`

**Training tokens:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\request-grounded\p2-48-training-bundle\train.tokens.u16le`

**Token SHA-256:** `2aa8cb0d7970ae8ac329f71fd5d2df00df2345181ae9e8e79534f44409b46785`

**Bundle manifest:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\request-grounded\p2-48-training-bundle\manifest.json`

**Manifest SHA-256:** `71fec0882692f5eb1b47b72f4a9f539e53e77882d003649beacd5f341b5a4ea7`

**Endpoint checkpoint:** `C:\Users\quali\OneDrive\Documents\ChatGPT\plex-0.1\training\artifacts\request-grounded\p2-48-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `fd86d11375e547f05b2fab7a36188ca30a7ecd0b0f38198de62bcddbfba489b7`

**Path provenance:** `training/pretraining/p2-48-stage-source-preparation-contract.json`.

</details>

<details>
<summary><strong>P3-02 - 144 training records</strong></summary>

**Training text:** `C:\Users\quali\PlexArtifacts\p3-02-training-bundle\train.jsonl`

**Text SHA-256:** `45d42817582f6b909d08695dfc93336414762b198cae7a4e4d760c09bf7249de`

**Training tokens:** `C:\Users\quali\PlexArtifacts\p3-02-training-bundle\train.tokens.u16le`

**Token SHA-256:** `050a978831d60f88ccbb8508c989ce58f18f4025c554b9214e28014505c84f23`

**Bundle manifest:** `C:\Users\quali\PlexArtifacts\p3-02-training-bundle\manifest.json`

**Manifest SHA-256:** `0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354`

**Endpoint checkpoint:** `C:\Users\quali\PlexArtifacts\file-edits\p3-02-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`

**Path provenance:** `training/pretraining/p3-02-first-run-contract.json`.

</details>

<details>
<summary><strong>P3-copy-fidelity-v1 - 288 training records</strong></summary>

**Training text:** `C:\Users\quali\PlexArtifacts\file-edits\p3-copy-fidelity-v1-bundle\train.jsonl`

**Text SHA-256:** `682e12d1516243d94ef7601d13f7bf788472ad81db9c3f75a9e85cdeaafbfdaf`

**Training tokens:** `C:\Users\quali\PlexArtifacts\file-edits\p3-copy-fidelity-v1-bundle\train.tokens.u16le`

**Token SHA-256:** `780029e643a5a81b938629e6192e11121bd74d2cbcd6437b7eb388549a563972`

**Bundle manifest:** `C:\Users\quali\PlexArtifacts\file-edits\p3-copy-fidelity-v1-bundle\manifest.json`

**Manifest SHA-256:** `8eceed468fdcb7202484eba82c8ebb8159a0d9e05787538f2c81f06f62719e2d`

**Endpoint checkpoint:** `C:\Users\quali\PlexArtifacts\file-edits\p3-copy-fidelity-v1-first-run\checkpoints\step-0100.pt`

**Checkpoint SHA-256:** `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`

**Path provenance:** `training/pretraining/p3-copy-fidelity-first-run-authorization-draft.json`.

</details>

## Where the older web training text came from

P2-29 consumed the normalized P2-26 web training JSONL through the P2-27 token bundle. Its 3,071 training rows identify these five upstream sources. Counts were read from the hash-verified training split; each row's stored content SHA also matched its UTF-8 text.

| Source ID | Training records | Recorded revision |
|---|---:|---|
| mdn-css-examples | 645 | `8cfcea69111e8bbacc9c92e3b04ead512693b937` |
| mdn-dom-examples | 438 | `f49fdb84422425b24c716d1eeaf8d408b9d7f710` |
| mdn-interactive-examples | 1665 | `13024db581b08fae11a0ca229497198b9b0534e7` |
| mdn-learning-area | 233 | `d55e1c2e3ff401519811b64e5b2a7d3643b43d0f` |
| soyaine-javascript30 | 90 | `a934c4adecbf8cc8ff2650a05a434892ad8e58d2` |

The source manifest also lists Bulma templates, but none of the verified training rows has that source ID. A source-catalog entry or a nearby source checkout does not establish gradient ancestry. Raw repository clones, entire candidate files, development sets and validation splits are therefore not added to this training-text inventory.

## How paths were established

- **P2-29:** the P2-28 contract identifies the P2-26 dataset directory and P2-27 token directory. The P2-29 result and saved dataset record pin their source-manifest, training-text and training-token hashes. The P2-27 selection record independently supplies the tokenizer-review directory and selected relative bundle path.
- **P2-30:** the preparation result records the repack/stage commands and exact `p2-30-request-v3-16k` bundle location. The saved endpoint dataset record agrees with its training split and token hashes.
- **P2-32/35/38/41:** each first-run contract records the explicit `--bundle-dir`; its pinned manifest and the endpoint dataset record agree with the local training text/tokens.
- **P2-44/48:** the stage-source preparation contracts record explicit bundle paths, manifests and parent checkpoint identities.
- **Private P3-02:** the [P3-02 run documentation](PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md) explicitly places the bundle at `$HOME/PlexArtifacts/p3-02-training-bundle`. The local manifest matches the SHA stored inside the actual endpoint checkpoint and run contract.
- **Private copy-fidelity:** `training/phase3/copy-fidelity-v1/first_run.py` and `zero_stage.py` explicitly select `$HOME/PlexArtifacts/file-edits/p3-copy-fidelity-v1-bundle`. Its manifest matches the actual fixed endpoint and authorization record.

These are location records followed by byte-identity checks. No corpus was included because it happened to share a folder with a checkpoint. For manifests that record a training JSONL hash rather than a text filename, the canonical `train.jsonl` file was accepted only after matching that exact hash and recorded row count.

The private entries are locally stored training artifacts. Their location alone does not imply an additional third-party/private upstream source. No unavailable training text or mismatching hash was found among the ten linked inputs.

## Tokenizer exposure and audit limits

The frozen tokenizer SHA is `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`. P2-27 records train-only fitting on the same P2-26 web training text used by P2-29. This exposure does not add a different corpus to the inventory. Later bundle-manifest hashes identify repacks; they are distinct from the tokenizer identity and the weight-parent identity.

- This establishes the **recorded checkpoint chain and listed training inputs**. Independent coverage review remains open; it is not a proof against unrecorded modifications or omitted provenance.
- Training split membership identifies eligible training text. It does not establish that every byte or every row was sampled into gradients. The records include complete-record and random-window samplers.
- The P2-41 approved first-run contract retains its pre-run `trainingExecuted=false` state. Its actual saved step-100 checkpoint, matching downstream parent SHA and dataset record establish the ancestry; a stale preparation flag is not used as execution evidence.
- Saved stage-transition records retain `trainingPerformed=false` or equivalent flags from staging. Those flags describe stage creation; the completed endpoint step and pinned endpoint identities describe the subsequent run.
- Checkpoint ZIP `data.pkl` structure was interpreted symbolically using `pickletools`; no pickle callable or tensor constructor executed, no tensor storage was read, and no model was loaded. Full-file checkpoint hashing did read the checkpoint bytes.
- Validation, development and final project holdout contents were not read or inventoried. No training, inference, checkpoint staging or collision screen ran during this trace. No capability claim changes.

## Next roadmap gates

1. **Completed for recorded inputs:** [review the inventory against linked records and account for ambiguities](PHASE-3-R2-TRAINING-INVENTORY-COVERAGE-REVIEW-2026-10-18.md). External independent certification has not been performed.
2. **Completed:** [run the streamed name-collision auditor](PHASE-3-R2-ANCESTOR-NAME-COLLISIONS-2026-10-18.md) against this exact inventory and preserve its per-file hash/collision receipt. Eight names collide; the current V2 exclusion gate fails.
3. Review coverage and collision results before freezing a revised V2 proposal. Training authorization remains separate.

[Return to R2 in the merge-audit roadmap](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md#r2)
