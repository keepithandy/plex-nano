# P3-03 — Source Preservation and Request-Value Diagnostic

## 1. Baseline and artifact verification

Plex Nano remains the 27,566,080-parameter coding model. The frozen P3-02 development gate remains failed: 0/36 exact files, 0/36 preservation passes, and 0/12 fully correct contrast groups. The October 17 saved-response review also recorded 36/36 outputs losing `account`, 25/36 outputs matching a P3-02 training target, and 8/12 groups with byte-identical outputs for all three requests. Those historical measurements remain unchanged. This report adds separate diagnostic measurements and does not replace that gate.

- Fixed checkpoint: step 100, SHA-256 `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d` (execution provenance reports the hash was verified).
- Tokenizer SHA-256: `2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697`.
- P3-02 bundle manifest SHA-256: `0cd6df3a6d8eec1d7da8c7a02af067b15299393d8b0e1756a071be17b9aca354`.
- Inference: `greedy-unreselected-fixed-endpoint-v1`, seed 1337, temperature 0, maximum 192 new tokens.
- Inference code revision: `c33ab6a9be90884f417bf75ddd5d608f3c91963b`.
- Read-only analysis implementation SHA-256: `d36a6b00e7a38c3abbb194bd9cc3ab3843da308fb98d310867b7f6f96715c6a9`.
- Separate evaluation authorization reference: `Owner message on 2026-10-17: "continue the work from here now" following the stated decision whether to authorize the bounded 72-task evaluation`.
- Final project holdout opened: no.

## 2. Frozen task manifest and SHA-256

- Manifest: `119eaa829825e3daa6c9156bd0ba84dfa32726dc2ebfa3d7cfd9ccf7612df578`.
- Frozen tasks: 72; contrast groups: 24.
- P3-02 candidate hash: `98d7ff276cf28d2509ad775c95c49ae0ffdb975de7e47f11e54430f3ca426403`; only the 144 authored training records were used for overlap checks.
- Complete-file overlap preflight counts: `{"prospectiveInputsMatchingTrainingInputs": 0, "prospectiveInputsMatchingTrainingTargets": 0, "prospectiveTargetsMatchingTrainingInputs": 0, "prospectiveTargetsMatchingTrainingTargets": 0}`.

## 3. Name-preservation results

0/72 tasks passed all source-name roles.

| Language | Role-complete passes | Tasks |
|---|---:|---:|
| html | 0 | 24 |
| css | 0 | 24 |
| javascript | 0 | 24 |

Role-level expected and observed values for every task are in the accompanying JSON audit.

Name-preservation failures:

- `p303-html-button-text-harbor-0` (html/button-text): failed roles button id attribute.
- `p303-html-button-text-harbor-1` (html/button-text): failed roles button id attribute.
- `p303-html-button-text-harbor-2` (html/button-text): failed roles button id attribute.
- `p303-html-button-text-orbit-0` (html/button-text): failed roles button id attribute.
- `p303-html-button-text-orbit-1` (html/button-text): failed roles button id attribute.
- `p303-html-button-text-orbit-2` (html/button-text): failed roles button id attribute.
- `p303-html-link-destination-harbor-0` (html/link-destination): failed roles anchor class name.
- `p303-html-link-destination-harbor-1` (html/link-destination): failed roles anchor class name.
- `p303-html-link-destination-harbor-2` (html/link-destination): failed roles anchor class name.
- `p303-html-link-destination-orbit-0` (html/link-destination): failed roles anchor class name.
- `p303-html-link-destination-orbit-1` (html/link-destination): failed roles anchor class name.
- `p303-html-link-destination-orbit-2` (html/link-destination): failed roles anchor class name.
- `p303-html-placeholder-harbor-0` (html/placeholder): failed roles input id attribute.
- `p303-html-placeholder-harbor-1` (html/placeholder): failed roles input id attribute.
- `p303-html-placeholder-harbor-2` (html/placeholder): failed roles input id attribute.
- `p303-html-placeholder-orbit-0` (html/placeholder): failed roles input id attribute.
- `p303-html-placeholder-orbit-1` (html/placeholder): failed roles input id attribute.
- `p303-html-placeholder-orbit-2` (html/placeholder): failed roles input id attribute.
- `p303-html-aria-label-harbor-0` (html/aria-label): failed roles navigation id attribute.
- `p303-html-aria-label-harbor-1` (html/aria-label): failed roles navigation id attribute.
- `p303-html-aria-label-harbor-2` (html/aria-label): failed roles navigation id attribute.
- `p303-html-aria-label-orbit-0` (html/aria-label): failed roles navigation id attribute.
- `p303-html-aria-label-orbit-1` (html/aria-label): failed roles navigation id attribute.
- `p303-html-aria-label-orbit-2` (html/aria-label): failed roles navigation id attribute.
- `p303-css-gap-harbor-0` (css/gap): failed roles first rule selector, preserved namespace comment.
- `p303-css-gap-harbor-1` (css/gap): failed roles first rule selector, preserved namespace comment.
- `p303-css-gap-harbor-2` (css/gap): failed roles first rule selector, preserved namespace comment.
- `p303-css-gap-orbit-0` (css/gap): failed roles first rule selector, preserved namespace comment.
- `p303-css-gap-orbit-1` (css/gap): failed roles first rule selector, preserved namespace comment.
- `p303-css-gap-orbit-2` (css/gap): failed roles first rule selector, preserved namespace comment.
- `p303-css-padding-harbor-0` (css/padding): failed roles first rule selector, preserved namespace comment.
- `p303-css-padding-harbor-1` (css/padding): failed roles first rule selector, preserved namespace comment.
- `p303-css-padding-harbor-2` (css/padding): failed roles first rule selector, preserved namespace comment.
- `p303-css-padding-orbit-0` (css/padding): failed roles first rule selector, preserved namespace comment.
- `p303-css-padding-orbit-1` (css/padding): failed roles first rule selector, preserved namespace comment.
- `p303-css-padding-orbit-2` (css/padding): failed roles first rule selector, preserved namespace comment.
- `p303-css-color-harbor-0` (css/color): failed roles first rule selector, preserved namespace comment.
- `p303-css-color-harbor-1` (css/color): failed roles first rule selector, preserved namespace comment.
- `p303-css-color-harbor-2` (css/color): failed roles first rule selector, preserved namespace comment.
- `p303-css-color-orbit-0` (css/color): failed roles first rule selector, preserved namespace comment.
- `p303-css-color-orbit-1` (css/color): failed roles first rule selector, preserved namespace comment.
- `p303-css-color-orbit-2` (css/color): failed roles first rule selector, preserved namespace comment.
- `p303-css-radius-harbor-0` (css/radius): failed roles first rule selector, preserved namespace comment.
- `p303-css-radius-harbor-1` (css/radius): failed roles first rule selector, preserved namespace comment.
- `p303-css-radius-harbor-2` (css/radius): failed roles first rule selector, preserved namespace comment.
- `p303-css-radius-orbit-0` (css/radius): failed roles first rule selector, preserved namespace comment.
- `p303-css-radius-orbit-1` (css/radius): failed roles first rule selector, preserved namespace comment.
- `p303-css-radius-orbit-2` (css/radius): failed roles first rule selector, preserved namespace comment.
- `p303-javascript-retry-limit-harbor-0` (javascript/retry-limit): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-retry-limit-harbor-1` (javascript/retry-limit): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-retry-limit-harbor-2` (javascript/retry-limit): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-retry-limit-orbit-0` (javascript/retry-limit): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-retry-limit-orbit-1` (javascript/retry-limit): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-retry-limit-orbit-2` (javascript/retry-limit): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-message-harbor-0` (javascript/message): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-message-harbor-1` (javascript/message): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-message-harbor-2` (javascript/message): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-message-orbit-0` (javascript/message): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-message-orbit-1` (javascript/message): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-message-orbit-2` (javascript/message): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-selector-harbor-0` (javascript/selector): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-selector-harbor-1` (javascript/selector): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-selector-harbor-2` (javascript/selector): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-selector-orbit-0` (javascript/selector): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-selector-orbit-1` (javascript/selector): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-selector-orbit-2` (javascript/selector): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-default-size-harbor-0` (javascript/default-size): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-default-size-harbor-1` (javascript/default-size): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-default-size-harbor-2` (javascript/default-size): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-default-size-orbit-0` (javascript/default-size): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-default-size-orbit-1` (javascript/default-size): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.
- `p303-javascript-default-size-orbit-2` (javascript/default-size): failed roles preserved namespace comment, editable declaration identifier, unrelated declaration identifier.

## 4. Requested-value selection results

0/72 requested values were selected correctly. Missing, malformed, duplicated, or ambiguous target structures count as failures.

| Language | Correct selected values | Tasks |
|---|---:|---:|
| html | 0 | 24 |
| css | 0 | 24 |
| javascript | 0 | 24 |

Each extracted literal, structural-validity result, and failure reason is in the accompanying JSON audit.

Requested-value failures:

- `p303-html-button-text-harbor-0` requested `Launch`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-button-text-harbor-1` requested `Inspect`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-button-text-harbor-2` requested `Reset`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-button-text-orbit-0` requested `Launch`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-button-text-orbit-1` requested `Inspect`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-button-text-orbit-2` requested `Reset`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-link-destination-harbor-0` requested `/alpha`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-link-destination-harbor-1` requested `/beta`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-link-destination-harbor-2` requested `/gamma`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-link-destination-orbit-0` requested `/alpha`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-link-destination-orbit-1` requested `/beta`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-link-destination-orbit-2` requested `/gamma`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-placeholder-harbor-0` requested `Find files`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-placeholder-harbor-1` requested `Find pages`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-placeholder-harbor-2` requested `Find people`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-placeholder-orbit-0` requested `Find files`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-placeholder-orbit-1` requested `Find pages`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-placeholder-orbit-2` requested `Find people`; extracted `None`; structural validity `False`; reason `missing-or-ambiguous-element`; source-name preservation `False`.
- `p303-html-aria-label-harbor-0` requested `Quick links`; extracted `Main menu`; structural validity `True`; reason `None`; source-name preservation `False`.
- `p303-html-aria-label-harbor-1` requested `Site links`; extracted `None`; structural validity `False`; reason `missing-duplicate-or-nonliteral-attribute`; source-name preservation `False`.
- `p303-html-aria-label-harbor-2` requested `Help links`; extracted `Account menu`; structural validity `True`; reason `None`; source-name preservation `False`.
- `p303-html-aria-label-orbit-0` requested `Quick links`; extracted `Main menu`; structural validity `True`; reason `None`; source-name preservation `False`.
- `p303-html-aria-label-orbit-1` requested `Site links`; extracted `None`; structural validity `False`; reason `missing-duplicate-or-nonliteral-attribute`; source-name preservation `False`.
- `p303-html-aria-label-orbit-2` requested `Help links`; extracted `Account menu`; structural validity `True`; reason `None`; source-name preservation `False`.
- `p303-css-gap-harbor-0` requested `4px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-gap-harbor-1` requested `12px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-gap-harbor-2` requested `20px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-gap-orbit-0` requested `4px`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-gap-orbit-1` requested `12px`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-gap-orbit-2` requested `20px`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-padding-harbor-0` requested `6px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-padding-harbor-1` requested `14px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-padding-harbor-2` requested `22px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-padding-orbit-0` requested `6px`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-padding-orbit-1` requested `14px`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-padding-orbit-2` requested `22px`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-color-harbor-0` requested `#224466`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-color-harbor-1` requested `#663322`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-color-harbor-2` requested `#226633`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-color-orbit-0` requested `#224466`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-color-orbit-1` requested `#663322`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-color-orbit-2` requested `#226633`; extracted `None`; structural validity `False`; reason `unclosed comment`; source-name preservation `False`.
- `p303-css-radius-harbor-0` requested `2px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-radius-harbor-1` requested `10px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-radius-harbor-2` requested `18px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-radius-orbit-0` requested `2px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-radius-orbit-1` requested `10px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-css-radius-orbit-2` requested `18px`; extracted `None`; structural validity `False`; reason `target-property-missing`; source-name preservation `False`.
- `p303-javascript-retry-limit-harbor-0` requested `3`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-retry-limit-harbor-1` requested `6`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-retry-limit-harbor-2` requested `9`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-retry-limit-orbit-0` requested `3`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-retry-limit-orbit-1` requested `6`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-retry-limit-orbit-2` requested `9`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-message-harbor-0` requested `"Queued"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-message-harbor-1` requested `"Paused"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-message-harbor-2` requested `"Done"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-message-orbit-0` requested `"Queued"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-message-orbit-1` requested `"Paused"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-message-orbit-2` requested `"Done"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-selector-harbor-0` requested `"#one"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-selector-harbor-1` requested `"#two"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-selector-harbor-2` requested `"#three"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-selector-orbit-0` requested `"#one"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-selector-orbit-1` requested `"#two"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-selector-orbit-2` requested `"#three"`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-default-size-harbor-0` requested `12`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-default-size-harbor-1` requested `16`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-default-size-harbor-2` requested `24`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-default-size-orbit-0` requested `12`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-default-size-orbit-1` requested `16`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.
- `p303-javascript-default-size-orbit-2` requested `24`; extracted `None`; structural validity `False`; reason `target-declaration-missing`; source-name preservation `False`.

## 5. Cross-category failure matrix

| Source names | Requested value | Tasks |
|---|---|---:|
| Pass | Pass | 0 |
| Fail | Pass | 0 |
| Pass | Fail | 0 |
| Fail | Fail | 72 |

## 6. Unrelated-file modifications

0/72 tasks preserved every byte outside the authorized target span.

| Modification category | Tasks |
|---|---:|
| after-target-delete | 4 |
| after-target-insert | 4 |
| authorized-target-span-unavailable | 68 |
| before-target-insert | 4 |
| before-target-replace | 4 |

Affected task IDs:

- `after-target-delete`: `p303-html-aria-label-harbor-0`, `p303-html-aria-label-harbor-2`, `p303-html-aria-label-orbit-0`, `p303-html-aria-label-orbit-2`
- `after-target-insert`: `p303-html-aria-label-harbor-0`, `p303-html-aria-label-harbor-2`, `p303-html-aria-label-orbit-0`, `p303-html-aria-label-orbit-2`
- `authorized-target-span-unavailable`: `p303-css-color-harbor-0`, `p303-css-color-harbor-1`, `p303-css-color-harbor-2`, `p303-css-color-orbit-0`, `p303-css-color-orbit-1`, `p303-css-color-orbit-2`, `p303-css-gap-harbor-0`, `p303-css-gap-harbor-1`, `p303-css-gap-harbor-2`, `p303-css-gap-orbit-0`, `p303-css-gap-orbit-1`, `p303-css-gap-orbit-2`, `p303-css-padding-harbor-0`, `p303-css-padding-harbor-1`, `p303-css-padding-harbor-2`, `p303-css-padding-orbit-0`, `p303-css-padding-orbit-1`, `p303-css-padding-orbit-2`, `p303-css-radius-harbor-0`, `p303-css-radius-harbor-1`, `p303-css-radius-harbor-2`, `p303-css-radius-orbit-0`, `p303-css-radius-orbit-1`, `p303-css-radius-orbit-2`, `p303-html-aria-label-harbor-1`, `p303-html-aria-label-orbit-1`, `p303-html-button-text-harbor-0`, `p303-html-button-text-harbor-1`, `p303-html-button-text-harbor-2`, `p303-html-button-text-orbit-0`, `p303-html-button-text-orbit-1`, `p303-html-button-text-orbit-2`, `p303-html-link-destination-harbor-0`, `p303-html-link-destination-harbor-1`, `p303-html-link-destination-harbor-2`, `p303-html-link-destination-orbit-0`, `p303-html-link-destination-orbit-1`, `p303-html-link-destination-orbit-2`, `p303-html-placeholder-harbor-0`, `p303-html-placeholder-harbor-1`, `p303-html-placeholder-harbor-2`, `p303-html-placeholder-orbit-0`, `p303-html-placeholder-orbit-1`, `p303-html-placeholder-orbit-2`, `p303-javascript-default-size-harbor-0`, `p303-javascript-default-size-harbor-1`, `p303-javascript-default-size-harbor-2`, `p303-javascript-default-size-orbit-0`, `p303-javascript-default-size-orbit-1`, `p303-javascript-default-size-orbit-2`, `p303-javascript-message-harbor-0`, `p303-javascript-message-harbor-1`, `p303-javascript-message-harbor-2`, `p303-javascript-message-orbit-0`, `p303-javascript-message-orbit-1`, `p303-javascript-message-orbit-2`, `p303-javascript-retry-limit-harbor-0`, `p303-javascript-retry-limit-harbor-1`, `p303-javascript-retry-limit-harbor-2`, `p303-javascript-retry-limit-orbit-0`, `p303-javascript-retry-limit-orbit-1`, `p303-javascript-retry-limit-orbit-2`, `p303-javascript-selector-harbor-0`, `p303-javascript-selector-harbor-1`, `p303-javascript-selector-harbor-2`, `p303-javascript-selector-orbit-0`, `p303-javascript-selector-orbit-1`, `p303-javascript-selector-orbit-2`
- `before-target-insert`: `p303-html-aria-label-harbor-0`, `p303-html-aria-label-harbor-2`, `p303-html-aria-label-orbit-0`, `p303-html-aria-label-orbit-2`
- `before-target-replace`: `p303-html-aria-label-harbor-0`, `p303-html-aria-label-harbor-2`, `p303-html-aria-label-orbit-0`, `p303-html-aria-label-orbit-2`
The JSON audit includes every differing prefix or suffix hunk, categorized as insertion, deletion, or replacement, with byte ranges and expected/observed text.

## 7. Exact-file correctness

0/72 decoded responses exactly matched their frozen expected files as UTF-8 bytes.

## 8. Contrast-group results

Fully correct groups: 0/24. Per-group output diversity, extracted-value diversity, correct values, and exact correctness are in the audit record.

| Language | Name roles | Requested values | Unrelated bytes | Exact files | Unscorable boundaries | Tasks |
|---|---:|---:|---:|---:|---:|---:|
| html | 0 | 0 | 0 | 0 | 20 | 24 |
| css | 0 | 0 | 0 | 0 | 24 | 24 |
| javascript | 0 | 0 | 0 | 0 | 24 | 24 |

| Language / family | Name roles | Requested values | Unrelated bytes | Exact files | Unscorable boundaries | Tasks |
|---|---:|---:|---:|---:|---:|---:|
| css / color | 0 | 0 | 0 | 0 | 6 | 6 |
| css / gap | 0 | 0 | 0 | 0 | 6 | 6 |
| css / padding | 0 | 0 | 0 | 0 | 6 | 6 |
| css / radius | 0 | 0 | 0 | 0 | 6 | 6 |
| html / aria-label | 0 | 0 | 0 | 0 | 2 | 6 |
| html / button-text | 0 | 0 | 0 | 0 | 6 | 6 |
| html / link-destination | 0 | 0 | 0 | 0 | 6 | 6 |
| html / placeholder | 0 | 0 | 0 | 0 | 6 | 6 |
| javascript / default-size | 0 | 0 | 0 | 0 | 6 | 6 |
| javascript / message | 0 | 0 | 0 | 0 | 6 | 6 |
| javascript / retry-limit | 0 | 0 | 0 | 0 | 6 | 6 |
| javascript / selector | 0 | 0 | 0 | 0 | 6 | 6 |

## 9. Training-target overlap observations

0/72 outputs exactly matched an expected file in the frozen P3-02 authored training split. This is output overlap evidence only; it does not prove memorization or an internal cause.

## 10. Limitations and proposed next investigation

These measurements classify observable output behavior. They do not establish an internal architectural or training cause and do not establish a general model-quality claim. The full-pass result, if any, supports only a proposal for further independent evaluation. The P3-02 gate remains failed and no training is authorized by this diagnostic.

## Preregistered decision

- Full diagnostic pass: **no**.
- Preservation failure observed: **yes**.
- Requested-value selection failure observed: **yes**.
- Tasks with an unscorable unrelated-content boundary: **68**.
- Decision outcome: **both-fail**.
- Next proposal: Propose a simpler controlled task that isolates copying supplied code from applying the requested change.
- Internal cause inferred: **no**.
