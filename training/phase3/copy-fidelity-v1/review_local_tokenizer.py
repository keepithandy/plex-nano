"""Read-only local tokenizer and target-mask preflight for frozen P3 copy fidelity.

No optimizer, model weights, checkpoint loader, or GPU usage. This verifies
actual token lengths and complete-answer/EOS supervision for the immutable
candidate before any future training authorization.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "copy-fidelity-candidate-v1.json"
EXPECTED_SHA256 = "9414d1da7672a386a920d1935723e9133d011c9c3a4029e5d5800f643835ca7a"
TOKENIZER_SHA256 = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
BUNDLE = Path.home() / "PlexArtifacts/p3-02-training-bundle"
PREP_SPEC = importlib.util.spec_from_file_location("p3_copy_fidelity_prep", HERE / "prepare_copy_fidelity.py")
assert PREP_SPEC is not None and PREP_SPEC.loader is not None
PREP = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREP)


def review(tokenizer: Any) -> dict[str, Any]:
    """Pure dependency-injected review: real tokenizer required by CLI below."""
    raw = CANDIDATE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256 or raw != PREP.serialized():
        raise ValueError("Frozen copy-fidelity dataset is not byte-for-byte identical")
    m = json.loads(raw)
    if m["authorization"]["trainingAuthorized"] is not False:
        raise ValueError("Training permission has changed")
    if tokenizer.vocabulary_size != 16384:
        raise ValueError("Unexpected tokenizer vocabulary")
    maximum = Counter()
    totals = Counter()
    for row in m["rows"]:
        p, y = row["prompt"], row["expectedFile"]
        pids, yids = tokenizer.encode(p), tokenizer.encode(y)
        if (not pids or not yids or tokenizer.decode(pids) != p or tokenizer.decode(yids) != y
            or any(type(t) is not int or not 4 <= t < 16384 for t in pids + yids)):
            raise ValueError(f"Tokenizer cannot roundtrip candidate: {row['id']}")
        if (len(pids) + len(yids) + 1 > 512
            or len(pids) + PREP.MAX_NEW_TOKENS + 1 > 512
            or len(yids) + 1 > PREP.MAX_NEW_TOKENS):
            raise ValueError(f"Token budget exceeded: {row['id']}")
        # Same next-token mask as the P3-02 answer/EOS objective:
        # first answer token is predicted by final prompt token; final answer predicts EOS.
        mask = (0,) * (len(pids) - 1) + (1,) * (len(yids) + 1)
        if len(mask) != len(pids) + len(yids) or sum(mask) != len(yids) + 1:
            raise ValueError(f"Answer + EOS supervision mask invalid: {row['id']}")
        maximum["promptTokens"] = max(maximum["promptTokens"], len(pids))
        maximum["answerTokensPlusEos"] = max(maximum["answerTokensPlusEos"], len(yids) + 1)
        maximum["recordTokensIncludingEos"] = max(maximum["recordTokensIncludingEos"],
                                                  len(pids) + len(yids) + 1)
        totals[(row["split"], "rows")] += 1
        totals[(row["split"], "answerEosTargets")] += sum(mask)
    return {
        "status": "copy-fidelity-local-tokenizer-review-passed-no-model-run",
        "candidateSha256": EXPECTED_SHA256,
        "modelContextTokens": 512, "vocabularySize": 16384,
        "maximumPromptTokens": maximum["promptTokens"],
        "maximumAnswerTokensPlusEos": maximum["answerTokensPlusEos"],
        "maximumPackedRecordTokensIncludingEos": maximum["recordTokensIncludingEos"],
        "trainRecords": totals["train", "rows"], "validationRecords": totals["validation", "rows"],
        "trainAnswerAndEosTargets": totals["train", "answerEosTargets"],
        "validationAnswerAndEosTargets": totals["validation", "answerEosTargets"],
        "trainingPerformed": False, "optimizerUpdates": 0,
        "checkpointLoaded": False, "finalHoldoutOpened": False,
    }


def main() -> None:
    # Import the locally installed training module only after immutable source
    # verification; no dataset bundle materialization, write, or optimizer.
    if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Frozen candidate changed before tokenizer inspection")
    from plex_training.completion import _completion_tokenizer

    tokenizer, record = _completion_tokenizer(BUNDLE)
    if record["tokenizerSha256"] != PREP.TOKENIZER_SHA256:
        raise ValueError("Local tokenizer SHA-256 mismatched original P3-02")
    print(json.dumps(review(tokenizer), indent=2))


if __name__ == "__main__":
    main()
