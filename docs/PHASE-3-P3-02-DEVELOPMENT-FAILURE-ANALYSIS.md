# P3-02 — Saved development-response review

Reviewed October 17, 2026. Plex Nano remains the controlled **27,566,080-parameter coding model**, taking a request and one supplied HTML/CSS/JavaScript file.

The saved step-100 outputs show systematic failure to preserve supplied names and distinguish requested values. The official development result remains **0/36 exact edits, 0/36 preservation, 0/12 fully correct contrast groups**. This review performs no generation, model loading, checkpoint selection, or training.

## Evidence and verification

Read the existing local `PlexArtifacts/file-edits/p3-02-development-evaluation/development-result.json` under the owner's home directory. Its **38,761 bytes** have SHA-256:

```text
aa5d302baf7199a65401da36cb9f5b6672cc8254f00932fdfa819b26b1940643
```

The result records fixed step-100 checkpoint SHA-256 `020d821cedd9a181be3a993cf9a306ab4d8fed6761daf962433797829afecd0d`. The checkpoint itself was not reopened or rehashed during this review. Every record's request, input, expected file, language, family, and group matched the authored P3-02 candidate. Replaying `score_development_responses` on all 36 saved responses reproduced **every returned score and per-record audit field**, including the failed gate. The frozen candidate and historical result files were left unchanged; raw responses remain local.

## Observed failure patterns

- **36/36 outputs lose the supplied `account` name entirely** and contain `settings` or `search`. These are names used in the training split. Changes affect identifiers, selectors, wrapper classes, and, for CSS/JavaScript, the supposedly preserved comments or companion declarations.
- **8/12 contrast groups return byte-identical output for all three different requests.** This includes every JavaScript family, CSS gap/padding/radius, and HTML input-placeholder. The other four groups each produce only two distinct files for three requests.
- **25/36 outputs exactly equal a complete edited-file target in the 144-record training split.** This is direct output overlap, consistent with reproducing familiar targets instead of grounding the edit in the supplied file. It does not establish the internal cause or prove that every failure is memorization.
- **0/36 outputs simply copy the unchanged input.** All responses decoded, ended at EOS, and retained full prompt visibility according to the saved audit. Generated lengths are 25–44 tokens, below the 192-token cap. Length exhaustion and prompt eviction do not explain this recorded failure.
- Additional damage is visible: `p302-html-aria-label-05-1` emits `<nav id="settings-label="settings-label="Quick links"></nav>`; `p302-css-color-05-2` omits the requested `color` declaration entirely. No syntax-parser pass rate is inferred from visual inspection.

| Language / family | Distinct outputs across 3 requests | Outputs equal to a training target |
|---|---:|---:|
| HTML / button-text | 2 | 3/3 |
| HTML / link-href | 2 | 3/3 |
| HTML / input-placeholder | 1 | 3/3 |
| HTML / aria-label | 2 | 2/3 |
| CSS / gap | 1 | 0/3 |
| CSS / padding | 1 | 0/3 |
| CSS / color | 2 | 2/3 |
| CSS / radius | 1 | 0/3 |
| JavaScript / retry-limit | 1 | 3/3 |
| JavaScript / message | 1 | 3/3 |
| JavaScript / selector | 1 | 3/3 |
| JavaScript / default-size | 1 | 3/3 |

For example, the supplied `accountRetries` requests ask for **4**, **5**, and **7**. All three saved outputs are:

```javascript
// Keep this note for settings.
const settingsRetries = 5;
const settingsEnabled = true;
```

Even the request for 5 fails because the supplied identifier and unrelated declaration/comment changed. Recognizing a familiar code shape or emitting the requested literal occasionally is insufficient for the complete-file editing contract.

## Reproduce the review without generation

From the repository root, run this Python in the installed training environment. It reads the single existing development result and the frozen authored candidate; it does not access checkpoints or the final project holdout. An assertion failure means the inputs no longer support this review.

```python
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from plex_training.file_edit_replacements import build_replacement_rows
from plex_training.file_edit_replacements_development import score_development_responses

path = Path.home() / "PlexArtifacts/file-edits/p3-02-development-evaluation/development-result.json"
raw = path.read_bytes()
assert len(raw) == 38761
assert hashlib.sha256(raw).hexdigest() == "aa5d302baf7199a65401da36cb9f5b6672cc8254f00932fdfa819b26b1940643"
result = json.loads(raw)
scores = score_development_responses(result["perRecord"])
assert all(result[key] == value for key, value in scores.items())
rows = {row["id"]: row for row in build_replacement_rows()}
train_targets = {row["expectedFile"] for row in rows.values() if row["candidateSplit"] == "train"}
groups = defaultdict(list)
for record in result["perRecord"]:
    for key in ("request", "inputFile", "expectedFile", "language", "editFamily", "contrastGroupId"):
        assert record[key] == rows[record["recordId"]][key]
    assert "account" not in record["actual"]
    assert "settings" in record["actual"] or "search" in record["actual"]
    groups[record["contrastGroupId"]].append(record["actual"])
assert sum(len(set(outputs)) == 1 for outputs in groups.values()) == 8
assert sum(record["actual"] in train_targets for record in result["perRecord"]) == 25
assert all(record["actual"] != record["inputFile"] for record in result["perRecord"])
assert min(record["generatedTokenCount"] for record in result["perRecord"]) == 25
assert max(record["generatedTokenCount"] for record in result["perRecord"]) == 44
print("Saved-response review reproduced; official gate remains failed.")
```

## Implication and next bounded work

The observable gap involves both **copying/preserving supplied source context** and **binding the requested value**, despite lower teacher-forced validation loss. This review identifies output behavior; it does not isolate a causal training or architecture defect, establish a model-size limit, or measure new generalization.

Next, draft a diagnostic proposal that measures source-name preservation separately from requested-value selection, with fixed tasks and decision rules before any new run. Keep that proposal separate from training authorization. Do not tune against these 36 development responses, change the scorer, reselect checkpoints, or open the final project holdout. No further optimizer updates are authorized.

See the [original P3-02 report](PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md) and [frozen failed-gate summary](../training/pretraining/p3-02-development-gate-result-summary.json).
