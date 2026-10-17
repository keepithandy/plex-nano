# Copy-control failure audit — 2026-10-17

**Scope:** six owner-supplied original step-100 copy-control responses, not new inference. Source: user-pasted PowerShell output for the frozen 42-task diagnostic (manifest SHA-256 17834efb801ac963a4672ae616ca47e9eea40b2568f01951c3891c7e4997f01a). Full 42-response envelope remains on the owner's Windows machine. The current repository did not read that private local file.

## Observed copy responses

| Frozen task | Supplied source identifiers | Generated replacement pattern | Byte-exact / name preserved |
|---|---|---|---|
| HTML cobalt | article section owner cobalt, cobaltButton, cobalt-note | settings wrapper, settings-save, Keep this note | No / no |
| HTML spruce | owner spruce, spruceButton, spruce-note | cart wrapper, cart-save, Keep this note; malformed closing button/section | No / no |
| CSS cobalt | cobaltGrid, cobaltNote, 8px gap | settings-gap and settings-aside, block and opacity declarations | No / no |
| CSS spruce | spruceGrid, spruceNote, 8px gap | settings-padding and settings-aside; malformed brace structure | No / no |
| JavaScript cobalt | cobaltRetries, cobaltEnabled | settingsButton and settingsEnabled, unterminated quote/declaration | No / no |
| JavaScript spruce | spruceRetries, spruceEnabled | two settingsEnabled declarations with different values | No / no |

All six outputs came from a request to return the **unchanged supplied file**, and 6/6 were nonidentical. All 42 evaluation responses terminated at EOS; completion of a response does not imply correct or syntactically valid code. The copied examples failed even though local preflight verified the original step-100 checkpoint and tokenizer.

## Training-pattern comparison (read-only repository evidence)

The authored original P3-02 training records contain 144 rows; in these, cart occurs in **36** records and settings in **36** (in input or expected file). Across 144 train records, the repeated string Keep this note. occurs in **48**. Its HTML training examples include a cart section wrapper and cart-save button, CSS includes settings-prefixed selectors, and JavaScript includes settingsEnabled.

The observed new outputs reuse this same family of names and structural fragments, with material syntax corruption. The repeated names strongly support **training-pattern fallback** over a trivial whitespace-only error. They do not show whether the original model memorized complete examples, failed to bind the source context, suffered from insufficient task diversity, or all three. Nor can the 36 edit tasks' unscorable target boundaries be counted as separately incorrect request-value choices.

## Curriculum-design inference

- Original P3-02 generated only **single-literal edits**, no unchanged-file copy requests. The answer+EOS objective was masked over the prompt; the model received no direct per-token *prompt reconstruction* loss.
- New [copy-fidelity v1](README.md) pairs copy and three literal edits for each identical source within one split, to teach that changes depend on the requested action, without dropping source identity.
- Fresh identifiers are separated by split; previously measured cobalt/spruce names are excluded from this training candidate.
- Evaluate unchanged-copy exactness *before* claiming generalized edit improvements. Do not train until distinct local tokenizer, packing, optimizer and owner authorization gates are reviewed.

Historical 0/36 P3-02, 0/72 source/value and 0/42 copy-control scores remain immutable.
