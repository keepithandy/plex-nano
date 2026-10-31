# Plex 600M · overlap and notice triage

**Reviewed:** October 31, 2026. **Disposition:** narrow the proposed source
scope before a 600M reuse decision; M3 remains open. This review informs the
choice of traceable supplied-code training examples and independent evaluation
material. It does not prepare training data or establish coding capability.

## Bounded overlap check

The two historical corpora and their record identities were checked against the
[previous source audit](PLEX-600M-M3-SOURCE-REUSE-REVIEW.md). The comparison used
case-sensitive Unicode word/punctuation tokens, ignored whitespace, and formed
sets of 12-token shingles. A pair was flagged at Jaccard similarity at least
4/5 or at least 9/10 containment of the smaller file's shingle set. Each file
needed at least 24 unique shingles. These thresholds were fixed in the driver
before the corpus scan; they are review heuristics, not a capability gate.

| Observation | Result |
|---|---:|
| Historical records considered | 3,228 |
| Cross-family pairs considered | 758,345 |
| Cross-family pairs spanning historical train/validation roles | 381,915 |
| Pairs flagged at the declared thresholds | **0** |
| Records below the shingle minimum | 218 |
| Records with notice/attribution keyword flags | 101 |

The zero flag count does **not** establish contamination-free or independent
splits. Same-family pairs were deliberately excluded; short files were not
eligible for a similarity flag. The method can miss renamed code, small copied
fragments, paraphrases and semantic similarity. It does not compare with any
external evaluation or the final project holdout. Do not tune these thresholds
after the result and present a later scan as the same check.

## File-level notices change the reuse recommendation

The scan flagged copyright, license, terms and attribution markers. Selected
headers and flagged lines were inspected locally. Some hits are instructional
prose, demonstration footers or references to external media; a marker is not
automatically a license exception or a reason to reject a file.

The following embedded libraries/examples have concrete file-level notices
beyond a repository-level CC0 label. The labels below are observations of the
local file contents, not a legal determination of permitted use.

| Historical source/path scope | Observed evidence | Proposed treatment for an initial new subset |
|---|---|---|
| `mdn-css-examples/editable-samples/codemirror/` | 61 accepted files in the embedded editor tree; inspected CodeMirror headers name MIT, with additional theme authors/notices. | Omit the complete accepted tree pending a separate provenance decision; examples and vendor code need distinct scope. |
| `mdn-css-examples/editable-samples-2/js/prettify.js` | Header names Google and Apache License 2.0. | Omit pending separate review. |
| `mdn-dom-examples/web-storage/jscolor/jscolor.js` | Header names GNU Lesser General Public License. | Omit pending separate review. |
| `mdn-dom-examples/webxr/gl-matrix.js` | Third-party authors and a permission notice in the header. | Omit pending separate review. |
| `mdn-learning-area/css/css-layout/legacy/normalize.css` and `skeleton.css` | Embedded library names and MIT labels. | Omit pending separate review. |
| `mdn-learning-area/css/css-layout/practical-positioning-examples/tabs-manual.js` | Header names the W3C Software License. | Omit pending separate review. |
| `mdn-learning-area/tools-testing/cross-browser-testing/javascript/es6-promise.js` and `polyfills.js` | Embedded Promise implementation with third-party authors and MIT labels. | Omit pending separate review. |

This proposal identifies **69 accepted records**: 62 from MDN CSS, two from MDN
DOM and five from MDN Learning Area. It does not remove them from the historical
corpora, alter past scores or decide that their use is prohibited. A future
decision could instead retain a particular component with separately reviewed
scope and notice evidence.

The 101 notice-flagged records overlap these proposed omissions in 25 cases.
That leaves **76 additional flagged records** for classification if the 69
omissions are adopted. Examples include external-font/media references, example
copyright footers, attribution strings and Bulma footer text distinguishing
source code and website content. Referenced external assets were not downloaded
or assessed. Absence of a keyword flag is not rights clearance.

## Evidence and verification

- [Read-only driver](../training/pretraining/plex600-m3-overlap-notice-review.py),
  SHA-256 `545c21c8bdb988e95ae721b2e6f31047e80065478b3e437b59904ebdf3fd3d83`.
- [Full comparison/notice receipt](../training/pretraining/plex600-m3-overlap-notice-review-2026-10-31.json),
  SHA-256 `5684e1c5cd61977aa87ac37fdb4950a2437b9a5c9e46842cc102bcea19ba2145`.
- [Proposed omission and notice-triage records](../training/pretraining/plex600-m3-proposed-source-holds-v1.json),
  SHA-256 `020c2f331c3a1a9982e536b9cf7b3524616f96247bb8467a5429f5c2b9d115e5`.

The disposition record selects the exact accepted CodeMirror prefix and the
eight other paths listed above from the comparison receipt. Its notice queue
contains every record flagged by that receipt; it is a proposal, not a dataset
filter run. Synthetic checks verified token/whitespace handling, exact threshold
boundaries, the short-file rule and marker line numbers before the scan.

The driver bounds corpus bytes, record count, total shingles, comparisons,
output pairs and elapsed time, and only writes a new receipt. Only the explicit
historical corpus files and prior review receipt were read by the scan. Selected
local source headers were then inspected for triage. No network, model,
tokenizer fit, source execution or final-holdout access occurred. The committed
historical manifests, corpora and numeric results remain unchanged.

## Next M3 action

Adopt or revise the narrower candidate scope, then classify the remaining
notice queue and record the actual source-selection decision. Keep all MDN
repositories grouped. Previously used Traversy/Bulma validation remains
historical validation; the zero similarity count does not make it a fresh
benchmark. The fresh evaluation design, final split and preparation budget
remain separate open decisions.
