# Plex 600M · M3 preparation packet

**State:** prepared for source/split review; M3 remains open.<br>
**Prepared:** October 31, 2026<br>
**Roadmap:** [M3 — source and evaluation-split decision](PLEX-600M-ROADMAP.md#m3--source-and-evaluation-split-decision)

This packet organizes the next decisions for the separate scratch 600M model.
Its coding target remains a correct change to caller-supplied HTML, CSS or
JavaScript. It does not change the controlled 27M Nano baseline. The inventory
below describes historical inputs, not a selected or approved 600M dataset.

## Work queue

| Step | State | Required result |
|---|---|---|
| M3.1 — Inventory existing source evidence | Prepared below | Exact identities, historical roles, counts and evidence limits. |
| M3.2 — Review scope and select candidate families | Local subset/notice review complete; selection open | [3,228 records and eight notices verified](PLEX-600M-M3-SOURCE-REUSE-REVIEW.md); source decisions, file-level exceptions and similarity risks remain. |
| M3.3 — Review source-family split | Proposed below | Fixed family assignments and documented prior exposure; investigate overlaps before claiming independence. |
| M3.4 — Review development evaluation | Proposed below | Task coverage, scoring rules and exposure policy decided before authoring/freezing the actual set. |
| M3.5 — Record preparation decision | Open | Approved corpus scope and storage budget, unresolved issues and the precise M4 work authorized. |

## Existing-source inventory

Counts are **historical accepted records**, read from the existing manifests.
They are not newly accepted 600M records, and the two corpora have not been
combined or deduplicated against each other in this review. Licenses below are
the declarations recorded by the earlier source reviews, not new legal findings.

| Source ID | Immutable revision | Recorded license | Historical records | Related-source group |
|---|---|---|---:|---|
| `microsoft-web-dev` | `78085398be0dea7abaaf5eda914817f46f6ddbbc` | MIT | 34 | `microsoft-web-dev-for-beginners` |
| `traversy-web-projects` | `c92b0bc3a87fc95d44b89552b7b6a2eeb36ca481` | MIT | 54 | `bradtraversy-50projects50days` |
| `mdn-dom-examples` | `f49fdb84422425b24c716d1eeaf8d408b9d7f710` | CC0-1.0 | 438 | `source-family-mdn-web-examples` |
| `mdn-css-examples` | `8cfcea69111e8bbacc9c92e3b04ead512693b937` | CC0-1.0 | 645 | `source-family-mdn-web-examples` |
| `mdn-interactive-examples` | `13024db581b08fae11a0ca229497198b9b0534e7` | CC0-1.0 | 1,665 | `source-family-mdn-web-examples` |
| `mdn-learning-area` | `d55e1c2e3ff401519811b64e5b2a7d3643b43d0f` | CC0-1.0 | 233 | `source-family-mdn-web-examples` |
| `bulma-templates` | `188672e66bccc5e2e946440ebc6214d600860969` | MIT | 69 | `github-bulmatemplates-bulma-templates` |
| `soyaine-javascript30` | `a934c4adecbf8cc8ff2650a05a434892ad8e58d2` | MIT | 90 | `github-soyaine-javascript30` |

The [starter source review](DATASET-SOURCE-REVIEW.md) and
[locked subset](../training/dataset-source-lock.json) define Microsoft's English
lessons and selected completed projects, and Traversy's 18 selected projects.
Their [historical catalog](../training/dataset-sources.starter-v1.json) records
source URLs, groups and the original rights-review scope.

The [P2-26 batch record](PHASE-2-P2-26-SOURCE-BATCH-2.md) and
[candidate registry](../training/pretraining/p2-26-source-batch-2.json) identify
the four MDN repositories, Bulma Templates and JavaScript30. HTML/CSS/JavaScript
and related JavaScript suffixes were screened by that historical pipeline.
The exact accepted paths and notices still need checking against their frozen
build artifacts before any proposed reuse. A repository revision alone does
not specify the accepted file subset.

### Evidence checked for this packet

| Existing local manifest | SHA-256 | Historical train / validation |
|---|---|---:|
| `training/artifacts/datasets/p1-14-starter-v1/manifest.json` | `58845cbc9251c0c33d5f2980d8b47d28dccb05638d68ac572a5841cdf96c8fcc` | 34 / 54 |
| `training/artifacts/datasets/p2-26-web-batch-2-v2/manifest.json` | `2f02f199d101050c9939df6d389851e46bbd858157cb68d7cb8cc75863178a91` | 3,071 / 69 |

These local manifest bytes were rehashed and matched the earlier reports.
Source catalogs and historical reports were read. No split JSONL, raw source
notice or content was revalidated for this packet. No corpus-size adequacy,
near-duplicate exclusion or new rights approval follows from these checks.

## Proposed treatment of sources and splits

The following is a **reuse discussion proposal**, not a final corpus split.
Retaining the historical roles avoids silently moving previously monitored
validation into training, but does not prove independence between corpora.

| Family | Proposed discussion role | Reason and remaining work |
|---|---|---|
| Microsoft Web Dev | Candidate training/explanation source | English teaching material and code; check exact subset, partial snippets, notices and links to related examples. |
| MDN family — all four repositories together | Candidate training source | Broad Web examples in the historical pool; treat the entire family together and assess overlap with other tutorials. |
| JavaScript30 | Candidate training source | Browser interaction examples; confirm exact accepted scope and related-source overlap. |
| Traversy projects | Historical validation/regression only | Previously used validation; retain exposure label. |
| Bulma Templates | Historical validation/regression only | P2-26 validation was subsequently measured during tokenizer/model research; retain exposure label. |
| Fresh development tasks | Separate proposed evaluation | Newly authored supplied-code tasks after design review; not derived by merely renaming historical examples. |

Before adopting any assignment, investigate cross-family shared snippets,
template lineage, copied lessons and near duplicates. Merge related families
or exclude affected material as warranted; do not present separate repository
names as proof of independence. Record resulting per-family counts, languages,
code versus explanation mix and source coverage. Percentages and final corpus
volume remain undecided until those quantities and the intended pilot scope
are reviewed. Additional broad language or coding sources are not selected here.

The historical pool is a starting point for a review, not sufficient evidence
for a useful 600M training mixture. Fresh development tasks must be kept out of
tokenizer fitting and training. The sealed final project holdout is outside
this work and remains unopened.

## Proposed development-evaluation design

**For review:** 36 cases, with 12 per language. The numbers are a planning
proposal, not a frozen benchmark, authored dataset or execution authorization.

| Task family | Cases per language | What the supplied-code task measures |
|---|---:|---|
| Exact copy control | 2 | Return supplied code without losing identifiers, whitespace or unrelated context. |
| One literal/attribute/property replacement | 4 | Bind the request to the correct location and value. |
| One bounded addition or removal | 4 | Apply the requested change and preserve code outside its declared span. |
| One small repair | 2 | Correct a concrete defect while retaining surrounding behavior. |

Author distinct source fixtures with explicit expected results and checks.
Record shared templates or other dependencies as groups; avoid treating related
cases as independent replications. Do not select fixtures, endpoints or outcome
rules using the model's completions. Review reference outputs and the scorer
with controlled correct and incorrect responses before model evaluation.

Proposed per-case reporting:

- Exact complete-file match under the declared LF fixture format; preserve raw
  responses, with no post-hoc cleanup that turns an incorrect answer into a pass.
- Syntax and requested-behavior checks appropriate to the language, reported
  separately. A parse pass alone does not establish correct behavior.
- Preservation of all unrelated bytes outside declared edit spans, unnecessary
  changes, and source-name retention. Diagnose missing targets as unscorable
  when the requested-value judgment cannot be supported.
- Prompt visibility, response termination, truncation and decoding status.
  After M5, verify that the full prompt and predeclared response budget fit
  within context; do not silently truncate supplied code.
- Counts by language, task family and any shared-fixture group, alongside raw
  per-case results. Separate exact-copy controls from edit success.

An aggregate capability threshold, exact tasks, scorer, decoding policy,
response budgets and endpoint must be reviewed and frozen before execution.
This design alone makes no capability claim and does not authorize inference.

## Decisions still needed to close M3

1. Select the source families and exact reuse scope, or replace/expand this
   candidate pool based on the intended pilot. Record the rights decisions and
   unresolved source relationships.
2. Adopt or revise the source-group assignments and the proposed 36-case
   evaluation coverage before authoring the actual set.
3. Specify the initial corpus/storage limits and required filtering/overlap
   evidence. The current packed-input path caps each selected split at 256 MiB;
   larger inputs require a reviewed scalable design before use.
4. Record the precise preparation work permitted for M4. Keep tokenizer fitting,
   model evaluation, optimizer checks and training as separate later decisions.

**Subsequent evidence:** the [source reuse review](PLEX-600M-M3-SOURCE-REUSE-REVIEW.md)
now verifies the exact local subsets and notices and records source-by-source
recommendations. Next, review source-use scope, file-level exceptions and
cross-family similarity risks before a source-selection decision. M3 remains open.
