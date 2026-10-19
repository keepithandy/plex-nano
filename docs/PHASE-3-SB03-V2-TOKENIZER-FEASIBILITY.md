# P3-SB03 V2 — Tokenizer-Aware Source-Identifier Feasibility

**Evaluation-name review and claim disposition:** the [separate history review](PHASE-3-R2-FROZEN-EVALUATION-NAMESPACE-REVIEW-2026-10-18.md) is complete. Five older evaluation spellings match ancestor text; SB-01 `ulvar`/`nyxen` and validation `granite`/`saffron` have no matches in the ten listed inputs under the ASCII substring rule. Original proposals/fixtures are preserved; the explicit claim is preparation and fixed diagnostics with documented exposure limits, not globally unseen names. No execution gate is waived.

**Earlier October 18 audit receipt (before the owner saved V2):** the default read-only search selected **48/48 names**, matched **384/384** paired token-count signatures and preserved all 96 validation rows against the five screened P3 corpora. No extra historical files were screened and no V2 file was saved. Training-ready remains false. The [audit recovery roadmap](PLEX-NANO-MERGE-AUDIT-ROADMAP-2026-10-18.md) records the in-memory proposal checksum and the remaining history/review gates.

**Decision: V1 is blocked; V2 search is MODEL-FREE FEASIBILITY ONLY.** No model loading, training, inference, optimizer stage, candidate bundle, checkpoint selection or final-holdout access is authorized here. [Tracking issue #189](https://github.com/keepithandy/plex-nano/issues/189).

## 1. Preserved V1 result (October 18, 2026)

The owner ran the read-only V1 builder and real SHA-pinned tokenizer-parity preflight. V1 and its full-coverage sampler verified byte-for-byte, but **240/384 paired outputs had unequal token-count signatures**.

| Frozen measure | Comparator | V1 intervention |
|---|---:|---:|
| All 384 records, supervised answer/EOS positions | 9,720 | 10,528 |
| Exact 1,600 frozen training draws, answer/EOS positions | 40,439 | 44,925 |
| Maximum prompt tokens | 95 | 98 |
| Maximum answer/EOS tokens | 51 | 54 |
| Training authorization | **None** | **None** |

The V1 intervention has **4,486 additional supervised positions across the exact same training draws** (about 11.1%), confounding any single-variable causal claim. The V1 candidate SHA `4bbdc55782d9447271323378367831656dc1f05cb5bb174138738a6a7e5e7dea` and sampler SHA `9ddf29a3a2972cee0c8c2f79db97d7ef3f823065a2db6ee0840e89bb01760c13` are unchanged. Do not mutate either original file, merge a revised row set into V1, or use V1 for training.

## 2. What the V2 feasibility tool does

[Read-only analyzer](../training/phase3/copy-fidelity-v1/review_namespace_diversity_v2.py) identifies possible source names **using the owner's real frozen Plex BPE tokenizer** (SHA-256 `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`).

For each of the original **six training source names**, it retains the V1 plan's **eight stable slots**, which cover the original **12 four-case contrast groups per name**. Thus it needs **48 distinct new train names**. Each proposed name is checked against **every prompt and expected complete output** in all four edit/copy contrasts across all original groups assigned to its slot.

A candidate is accepted **only if**:
- it is lowercase ASCII alphabetic, has exactly the old name's byte length, and is unique across the entire new 48-name pool;
- it is absent as a byte substring from the original five SHA-verified repository P3 candidate/control files and the excluded old train/validation/microprobe namespaces;
- it is absent from all **explicit, SHA-verified, non-final-holdout** extra historical files supplied through the optional inventory (when provided);
- each assigned case produces the **same prompt token count** and **same answer-plus-EOS token count** as its original comparator case, using the **same frozen 16,384-token vocabulary**;
- after a complete 48-name pool is assembled, **all 384 paired rows** match their exact tokenizer signatures, the original 96 validation rows stay unchanged, all 96 contrast groups remain complete, and the previously frozen 1,600 draw indices remain identical;
- tokenizer encode/decode roundtrips and full-context / maximum-new-token bounds remain valid.

The deterministic search examines eligible 5/6-letter ordinary vocabulary words first in SHA-sorted order, then additional SHA-derived lowercase ASCII candidates. Per-slot work is strictly bounded, default **2,500 checked names**, maximum **20,000**. The procedure does **not** assume that a token count matches merely because source character lengths match.

**Crucial limitation:** Tokenizer-vocabulary membership can indicate previously learned subwords. Neither this search nor the five screened P3 corpora proves a name never appeared in **all historical training text**. Complete historical non-holdout data inventory remains required. The final project holdout must **never** be scanned for name exclusion.

If a slot exhausts its bounded search, the tool reports `token-parity-infeasible-within-search-budget` and **creates no V2 artifact**. This means only that the configured search failed, not that mathematical matching is impossible. Do not relax signatures or claim an effect from a different token exposure.

## 3. Owner-local read-only command

After merging the reviewed PR and pulling `master`, from your existing Windows repo:

```powershell
git pull origin master

# Read only: real frozen tokenizer, frozen V1/candidate/control files.
uv run --project training python training/phase3/copy-fidelity-v1/review_namespace_diversity_v2.py
```

The command prints one JSON receipt containing each selected name or the first slot that exhausted its attempt limit, candidate and tokenizer provenance, optional-history coverage, and explicit **`trainingReady=false` / `trainingAuthorized=false`**.

To conduct a deeper **bounded** search without modifying any files:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/review_namespace_diversity_v2.py --attempts-per-slot 10000
```

Even if 48/48 names match, **do not start training**. A successful tokenizer-only feasibility pass establishes a provisional candidate only, not independence from all past training history.

## 4. Optional SHA-verified historical training-source screening

Only explicitly named **non-final-holdout** historical files may be screened. You can supply a JSON inventory at your own path with this structure:

```json
{
  "scope": "non-final-holdout-training-corpora",
  "files": [
    {
      "path": "C:\\path\\to\\known-training-data.jsonl",
      "sha256": "the-actual-64-character-sha256-of-that-file"
    }
  ]
}
```

Use `--history-inventory C:\path\to\inventory.json`. The tool validates the content hash of each named source before searching. Paths containing *holdout* are refused, symbolic links are refused, no directories are crawled, and individual files over **128 MiB** and **more than 128 MiB total historical source bytes** are refused to keep the memory footprint bounded. The inventory itself must be no more than 1 MiB and contain at most 100 file records. Large historical source files require a separately reviewed streaming inventory mechanism before they can be counted as screened.

A provided inventory **does not automatically prove completeness**. New source names remain provisional until the owner and independent reviewer can attest that every ancestor training source was enumerated. Private data should remain local; report only relevant hashes/collision statistics to GitHub.

## 5. Optional, explicitly requested LOCAL proposal export

If all 48 slots find matches and after inspecting the default receipt, you may create a **new local, unapproved** V2 JSON file once:

```powershell
uv run --project training python training/phase3/copy-fidelity-v1/review_namespace_diversity_v2.py --prepare-local
```

The tool uses create-only `xb` at:

```text
$HOME\PlexArtifacts\file-edits\p3-sb03-v2-feasibility\v2-proposal-unreviewed.json
```

It refuses overwrites, incomplete selections, attempts to write outside `PlexArtifacts`, or checkpoint/holdout destinations. **The default command never writes files.** This local V2 record includes all 384 rows, row/signature evidence, new names, frozen sampler identity, strict authorization refusals, and an independently calculable SHA-256. It does **not** create a bundle, training stage, authorized dataset, or checkpoint.

For owner review, supply the concise JSON receipt and the resulting draft hash (not private training-corpus contents). Only then can the proposal be independently audited and, if appropriate, given a distinct reviewed V2 version with frozen bytes.

## 6. Hard stop and follow-on experiments

- No new tokenizer may be trained or swapped in to make names look matched. The model and tokenizer stay unchanged.
- No changes to V1, historical outputs, checkpoint `c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230`, or final project holdout.
- Synthetic Linux/Windows CI can verify the search algorithm, SHA refusals, bad inventory handling and pure fake-tokenizer successes/failures, but **cannot attest actual Plex BPE feasibility** without the owner's private tokenizer.
- No live model evaluation on the new 48-case microprobe is authorized; no matched comparator or intervention optimizer arm is authorized. Each future arm would require **its own fresh approval**, independently pinned stage/bundle, and full intervention comparison design.
- If exact parity cannot be established after reasonable reviewed search bounds, do **not** fabricate a token-matched experiment. Report that result, then review a controlled, clearly different design (e.g., token-exposure-normalized training). Its causal claim would differ and require a separate contract.

**Actual owner-reported outcome (October 18, 2026):** V2 real-tokenizer search succeeded for **48/48 names and 384/384 paired records**. The resulting unapproved V2 proposal was written locally with independent PowerShell SHA-256 `5f957cc30b1c58e7e22a8e2565efbc143303431526a61ecc2b65ef0e741ebb21`, matching the tool's expected digest. The subsequent [ten-source ancestor screen](PHASE-3-R2-ANCESTOR-NAME-COLLISIONS-2026-10-18.md) confirmed eight proposed-name collisions after all ten hashes verified. The original V2 remains unchanged and fails the source-exclusion gate. Its five P3 checks remain a partial exclusion claim; a separate revision requires renewed history screening and tokenizer parity. Neither run is authorized.

**Reviewed-history reselection (October 18):** the [R3 replay](PHASE-3-R3-V2-HISTORY-RESELECTION-2026-10-18.md) verified all ten inventoried sources and repeated selection with case-insensitive history exclusion at 2,500 and 20,000 attempts per slot. Both stopped at `prism.0` with 16/48 slots selected. No complete revised proposal or reviewed manifest was created. The original V1/V2 artifacts remain unchanged; bounded failure receipts and a second-pass review are frozen separately.

**Next acceptance gate:** resolve the documented token-parity/history-exclusion conflict under a reviewed comparison design, establish a complete reviewed candidate, then obtain separate bundle/training/evaluation decisions. A generated-file score is required before claiming capability improvement.
