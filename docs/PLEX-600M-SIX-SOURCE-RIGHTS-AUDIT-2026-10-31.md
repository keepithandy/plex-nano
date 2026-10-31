# Plex 600M — six-source rights and data audit

**October 31, 2026 · Disposition: HOLD ALL SIX FOR FURTHER REVIEW**

**Scope:** an evidence-based, read-only review of six already proposed 600M *training* sources at exact Git revisions. This is not a new source-use authorization, not a legal opinion, and not a new raw-dataset scan. No M4 output, tokenizer, optimizer update, model checkpoint or final holdout was accessed during this review. The six training sources are distinct from the Traversy and Bulma *historical validation* sources.

**Decision:** **0/6 sources newly approved**, **0 records newly authorized for preparation**. Preserve the existing 69 proposed embedded-file omissions, 101 total notice flags, and all currently enforced quarantines. Do not set sourceRightsApproved, preparationAuthorized or trainingAuthorized to true. Previously recorded license reviews were for historical experiments, not automatically a new 600M training approval.

Machine-readable metadata: [six-source evidence receipt](../training/pretraining/plex600-m3-six-source-rights-evidence-2026-10-31.json).

## What was verified in this GitHub review

- Read the [historical source/identity review](../training/pretraining/plex600-m3-source-subset-review-2026-10-31.json) and the [proposed holds receipt](../training/pretraining/plex600-m3-proposed-source-holds-v1.json). The prior M3 scan reports **3,228/3,228** accepted historical source records and **8/8** packaged license notices matched against its local originals. Those checks are **historical evidence**; this audit did not rerun them on the owner's filesystem.
- Fetched **six upstream root LICENSE texts at their exact pinned revisions** via GitHub. Two state MIT; four state CC0-1.0. These are root-level texts, not attestations that every nested file shares the license. No fresh byte-for-byte SHA-256 match between upstream fetched text and the saved local notices is claimed: transport/line-ending representation can differ.
- Inspected selected **file headers at pinned revisions**, confirming separately credited CodeMirror, Google Code Prettify, jscolor, gl-matrix, normalize.css, W3C tab example and es6-promise files. This is targeted evidence, **not** exhaustive inspection of every candidate.
- Read the pinned [Soyaine JavaScript30 README](https://github.com/soyaine/JavaScript30/blob/a934c4adecbf8cc8ff2650a05a434892ad8e58d2/README.md), which distinguishes the author's Chinese-language adaptations from original Wes Bos challenge START/FINISHED material. Confirm lineage and original license before reuse.
- Retained the [M3 overlap limitations](PLEX-600M-M3-OVERLAP-NOTICE-REVIEW.md): zero pairs exceeded its predeclared *cross-family* shingle thresholds, but **218 records** were too short for that test. Zero flags is not proof against near duplicates, train/evaluation contamination or third-party material.

## Exact candidate counts

These are from the saved M3 accepted-path and quarantine receipts, not a fresh M4 materialization. The candidate count is **conditional** and precedes further filtering and tokenizer measurement.

| Source (historical train only) | Root license observed | Accepted train | Embedded holds | Notice flags | Overlap of the two | Quarantined union | Conditional remainder | New rights decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Microsoft Web Dev | MIT | 34 | 0 | 14 | 0 | 14 | 20 | **HOLD** |
| MDN CSS examples | CC0-1.0 | 645 | 62 | 29 | 18 | 73 | 572 | **HOLD** |
| MDN DOM examples | CC0-1.0 | 438 | 2 | 15 | 2 | 15 | 423 | **HOLD** |
| MDN interactive examples | CC0-1.0 | 1,665 | 0 | 2 | 0 | 2 | 1,663 | **HOLD** |
| MDN learning area | CC0-1.0 | 233 | 5 | 19 | 5 | 19 | 214 | **HOLD** |
| Soyaine JavaScript30 | MIT | 90 | 0 | 0 | 0 | 0 | 90 | **HOLD** |
| **Total candidate train** | | **3,105** | **69** | **79** | **25** | **123** | **2,982** | **0 approved** |

The wider historical receipts contain 123 more records formerly used as validation (54 Traversy and 69 Bulma). Bulma has 22 additional notice flags; both validation sources remain outside the proposed fresh training scope. Therefore **145 historical records** are conservatively quarantined across both roles. The 101 total keyword flags include 79 in historical training and 22 in historical validation. Of the 79 training flags, 25 overlap the 69 embedded-library omissions, so **54 additional training records** are held for notice classification.

## Source-by-source evidence and decisions

### 1. Microsoft Web Dev — HOLD

- **Pinned source:** [microsoft/Web-Dev-For-Beginners at 7808539](https://github.com/microsoft/Web-Dev-For-Beginners/tree/78085398be0dea7abaaf5eda914817f46f6ddbbc). Its [root LICENSE](https://github.com/microsoft/Web-Dev-For-Beginners/blob/78085398be0dea7abaaf5eda914817f46f6ddbbc/LICENSE) states MIT and names Microsoft.
- **Scope:** 34 previously accepted train files; 14 notice/attribution-keyword flags, 20 conditional candidates. The 20 comprise 11 Markdown, three HTML, three CSS and three JS files. This is largely instruction prose, not a balanced coding corpus.
- **Unresolved:** classify the 14 notices at the file level; distinguish original tutorial content from embedded examples, external media, linked assets and attribution. Verify applicable notice preservation and specific rights for the intended source use. No claim that all keyword hits are violations.

### 2. MDN CSS examples — HOLD, embedded vendor subtree explicitly excluded

- **Pinned source:** [mdn/css-examples at 8cfcea6](https://github.com/mdn/css-examples/tree/8cfcea69111e8bbacc9c92e3b04ead512693b937). Its [root LICENSE](https://github.com/mdn/css-examples/blob/8cfcea69111e8bbacc9c92e3b04ead512693b937/LICENSE) states CC0-1.0.
- **Observed exceptions:** [CodeMirror library](https://github.com/mdn/css-examples/blob/8cfcea69111e8bbacc9c92e3b04ead512693b937/editable-samples/codemirror/lib/codemirror.js) names separate authors and MIT; [Google Code Prettify](https://github.com/mdn/css-examples/blob/8cfcea69111e8bbacc9c92e3b04ead512693b937/editable-samples-2/js/prettify.js) names Google and Apache-2.0.
- **Scope:** 645 accepted, 62 exact third-party omissions, 29 notice flags with 18 overlapping, so 73 held and 572 conditional. Remaining examples still need asset and notice inspection; CC0 at repository root cannot override third-party headers. Most conditional records are HTML.

### 3. MDN DOM examples — HOLD, third-party components excluded

- **Pinned source:** [mdn/dom-examples at f49fdb8](https://github.com/mdn/dom-examples/tree/f49fdb84422425b24c716d1eeaf8d408b9d7f710) and its [CC0 root license](https://github.com/mdn/dom-examples/blob/f49fdb84422425b24c716d1eeaf8d408b9d7f710/LICENSE).
- **Observed exceptions:** [jscolor](https://github.com/mdn/dom-examples/blob/f49fdb84422425b24c716d1eeaf8d408b9d7f710/web-storage/jscolor/jscolor.js) carries an LGPL header; [gl-matrix](https://github.com/mdn/dom-examples/blob/f49fdb84422425b24c716d1eeaf8d408b9d7f710/webxr/gl-matrix.js) contains separate author and copyright/permission notices.
- **Scope:** 438 accepted, two proposed embedded omissions, 15 notice flags including both omissions, 423 conditional. Continue per-file attribution and linked-asset review; the previous source notices alone are not blanket clearance.

### 4. MDN interactive examples — HOLD, dominance and provenance review

- **Pinned source:** [mdn/interactive-examples at 13024db](https://github.com/mdn/interactive-examples/tree/13024db581b08fae11a0ca229497198b9b0534e7), [CC0 root license](https://github.com/mdn/interactive-examples/blob/13024db581b08fae11a0ca229497198b9b0534e7/LICENSE) and [README](https://github.com/mdn/interactive-examples/blob/13024db581b08fae11a0ca229497198b9b0534e7/README.md).
- **Scope:** 1,665 accepted, two notice flags and 1,663 conditional. This source contributes most candidate records, potentially dominating HTML/CSS/JS snippets. The README describes linked media and example templates; not every referenced asset or short example has been independently reviewed.
- **Unresolved:** per-file notice triage, third-party asset/dependency provenance, sample validity and tokenizer/corpus mixture. The README notes the repository was archived in October 2025; archival status does not revoke or expand its license.

### 5. MDN learning area — HOLD, third-party components excluded

- **Pinned source:** [mdn/learning-area at d55e1c2](https://github.com/mdn/learning-area/tree/d55e1c2e3ff401519811b64e5b2a7d3643b43d0f) and [CC0 root license](https://github.com/mdn/learning-area/blob/d55e1c2e3ff401519811b64e5b2a7d3643b43d0f/LICENSE).
- **Observed exceptions:** [normalize.css](https://github.com/mdn/learning-area/blob/d55e1c2e3ff401519811b64e5b2a7d3643b43d0f/css/css-layout/legacy/normalize.css) names MIT; [ARIA tabs-manual.js](https://github.com/mdn/learning-area/blob/d55e1c2e3ff401519811b64e5b2a7d3643b43d0f/css/css-layout/practical-positioning-examples/tabs-manual.js) names the W3C Software License; [es6-promise.js](https://github.com/mdn/learning-area/blob/d55e1c2e3ff401519811b64e5b2a7d3643b43d0f/tools-testing/cross-browser-testing/javascript/es6-promise.js) credits separate authors under MIT.
- **Scope:** 233 accepted, five proposed omissions, 19 notice flags including those five, 214 conditional. Confirm rights and provenance for the remaining flagged examples and referenced materials.

### 6. Soyaine JavaScript30 — HOLD, original-author lineage unresolved

- **Pinned source:** [soyaine/JavaScript30 at a934c4a](https://github.com/soyaine/JavaScript30/tree/a934c4adecbf8cc8ff2650a05a434892ad8e58d2), [MIT root license](https://github.com/soyaine/JavaScript30/blob/a934c4adecbf8cc8ff2650a05a434892ad8e58d2/LICENSE) and [README](https://github.com/soyaine/JavaScript30/blob/a934c4adecbf8cc8ff2650a05a434892ad8e58d2/README.md).
- **Material difference:** the README credits Wes Bos as creator of the original 30-day challenge and distinguishes original **START/FINISHED** files from the adaptation's **SOYAINE** implementations. Some accepted HTML files use these original naming conventions. The adaptation's MIT declaration does not, by itself, resolve original-author license and derivative/embedded asset scope.
- **Scope:** 90 accepted files, no keyword notice flags and no M3 omissions. **Zero flags does not mean cleared.** Establish provenance and the appropriate licenses for START/FINISHED and translated/adapted examples separately. Until then, do not authorize any of the 90.

## Safeguards retained and gaps

- Existing [M3 candidate policy](../training/pretraining/plex600-m3-candidate-scope-v1.json) remains **review-only**, with rightsApproved=false, preparationAuthorized=false, trainingAuthorized=false and the union quarantine policy unchanged.
- The existing [M4 materializer](PLEX-600M-M4-CORPUS-PIPELINE.md) still requires a separate rights-and-preparation decision naming all proposed training sources and the exact M3 hashes. **This report is deliberately not that approval-shaped file.**
- Previous 27M training/evaluation artifacts, historical validation source roles, protected project holdout and existing numeric results are untouched.
- The M3 matched historical source files against local corpora but **did not authenticate pinned upstream revision trees anew**. This audit fetched pinned root licenses and selected headers, but did not rerun a comprehensive local source-commit, license byte, per-file provenance, secrets, data quality or overlap scan.
- No token counts, full training text-byte mix, M4 filtered output or 600M tokenizer metrics exist yet. A pool of 2,982 conditional examples does **not** establish adequate data scale for a useful scratch 600M coding model.

## Explicit next review actions

1. Reconcile **source scope and file provenance** before any source-level approval: start with JavaScript30 original/translated lineage, then MDN third-party dependencies and candidate-linked media/assets.
2. Classify the **54 extra notice-flagged historical training files** not already in the 69 embedded omissions; retain all quarantines until each file is decided. The other 22 flagged files belong to historical validation and stay excluded.
3. Check the remaining clean-path candidates for third-party and derivative material beyond keyword headers, and preserve required license notices and provenance.
4. Record an actual *scoped decision* per source with reviewer, specific eligible paths, evidence and remaining exclusions. A rejection/narrowed pool may require a revised M3/M4 scope and fresh validation, **not** an invented approval file.
5. Only then propose a signed-off M4 materialization and M5 tokenizer run, plus a separately designed independent coding development evaluation. Training needs another fixed run contract and authorization.

**Conclusion:** root-level MIT/CC0 evidence exists, but file-level rights and training-use scope are not yet adequately established for any of the six sources. **No new authorization is granted.**
