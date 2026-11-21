# Plex M12-A — Teacher-Forced Source Conditioning

**Status:** inference-only diagnostic implemented; **no new M12 GPU measurements**, dataset approvals, optimizer updates, or training.

The M11 evidence audit established immediate EOS for unfamiliar prompt formats and incorrect copying of unseen identifier value. The EOS-first-token margin explains why generation often stops but not the full source-to-output copying failure.

This probe measures **next-token logits at the exact target positions**, conditioned on each correct (gold) target prefix. It does not generate, edit, accept or save code.

## Fixed comparison

The diagnostic evaluates only the four already-exposed M11 development microprobes under two prompt formats:

- Original M11 Copy-colon-space prompt: two familiar score cases and two probe-only value cases.
- General console prompt: same four cases using the ordinary Request / Source file / Edited file format.

For each prompt it scores every gold code token and terminal EOS under the **existing fitted-vocabulary decoding policy**, without changing its token mask, target, model weights or decoder. It saves a SHA-256-bound, create-only evidence report.

### Metrics and what they mean

- **Gold token rank and fitted-vocabulary NLL:** does the checkpoint give high probability to the correct next token when earlier output is supplied correctly? This is **not** an autonomous copy success.
- **Premature EOS under teacher forcing:** how many code-token positions prefer EOS even with a correct prefix?
- **EOS rank at the correct ending:** is EOS preferred when full expected code was supplied?
- **Source-token contrast:** for two source lines differing only in score versus value but sharing digit 0 or 1, compare the model's value-token logit minus score-token logit at the first target token position where the answers differ. A tokenizer may merge the shared character prefix, making this position token zero. A **preference switch** requires the familiar score source to favor its token and unseen value source to favor its token. This is a deliberately narrow diagnostic, not a general attention explanation.
- **Training/development split:** keep the familiar two training probes distinct from the two previously exposed development probes.

The script labels its outputs as teacher-forced. The correct target is supplied to the forward pass. Its numeric NLL is **over fitted vocabulary IDs 3..V-1** and may differ from the original full-vocabulary training loss; do not equate them.

## Run on the existing local WSL GPU — optional

After updating the Windows and WSL Plex checkouts, run from PowerShell:

    .\scripts\Invoke-PlexM12Conditioning.ps1 -Run

The PowerShell launcher refuses to load the model without the explicit Run switch. It accepts the same optional WSL distro/path overrides as earlier diagnostics. The output is a new directory under training/artifacts with rows.jsonl and report.json.

The Python entrypoint is also explicitly gated:

    /home/quali/.local/bin/uv run --project training --no-sync python scripts/plex_m12_source_conditioning.py \
      --execute-read-only --output-dir training/artifacts/plex600-m12-conditioning-your-unique-label

Do not reuse an output path. The model loader checks the known M11 step-200 SHA, and the tokenizer is checked against the frozen M11 identity. The diagnostic uses CUDA inference mode and BF16 autocast without gradients, optimizer, weight writes, dataset creation or any final holdout access. GitHub CI tests only synthetic scoring and output behavior, **not** the local 600M model.

## Interpretation and M13 readiness

If M11 prefers the score token for both input lines, it reinforces the observed source-identifier overfitting. If it switches preference with the supplied source under teacher forcing while autonomous generation still fails, that suggests prompt-specific EOS behavior or subsequent-token errors should be investigated.

Neither outcome is proof of successful M12 learning, because this uses **unchanged M11 weights**. M13 should not be declared ready until separately approved M12 experiments demonstrate unseen copying improvement and then controlled editing.

M12-A still needs: an owner-reviewed provenance-verified M12 corpus, actual tokenizer/rights audit on those records, read-only M9-versus-M11 starting-checkpoint comparison if M9 bytes remain available, and a GPU training admission plan. M12-B and later optimizer updates require explicit owner authorization.

[Current M12-A data admission](PLEX_M12_A_DATA_ADMISSION.md) · [M12 roadmap and gates](PLEX_M11_FINAL_INVESTIGATION_M12_PLAN.md)
