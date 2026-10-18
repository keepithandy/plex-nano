"""Read-only SB01 frozen-tokenizer review. No checkpoint/model/optimizer import.

Only the proposal and existing tokenizer JSON are read. No bundle/stage is built.
The default tokenizer belongs to the completed copy-fidelity training bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sb01_frozen_scoring", HERE / "score_source_binding_microprobe.py")
assert SPEC and SPEC.loader
SCORING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORING)
TOKENIZER = Path.home() / "PlexArtifacts/file-edits/p3-copy-fidelity-v1-bundle/tokenizer.json"


def review(tokenizer: Any) -> dict[str, Any]:
    """Dependency-injected tokenizer checks; the CLI checks byte SHA before load."""
    if type(tokenizer.vocabulary_size) is not int or tokenizer.vocabulary_size != 16384:
        raise ValueError("Expected frozen 16384-token vocabulary")
    maxima = {"promptTokens": 0, "answerTokensPlusEos": 0, "recordTokensIncludingEos": 0}
    targets = 0
    rows = SCORING.frozen_probe()["rows"]
    for row in rows:
        prompt, answer = row["prompt"], row["expectedFile"]
        pids, aids = tokenizer.encode(prompt), tokenizer.encode(answer)
        if (not pids or not aids or tokenizer.decode(pids) != prompt
                or tokenizer.decode(aids) != answer
                or any(type(t) is not int or not 4 <= t < 16384 for t in pids + aids)):
            raise ValueError("Frozen tokenizer roundtrip or ordinary-token check failed")
        if (len(pids) + 192 + 1 > 512 or len(aids) + 1 > 192
                or len(pids) + len(aids) + 1 > 512):
            raise ValueError("Frozen tokenizer exceeds 512/192 token bounds")
        # x = (prompt + answer + EOS)[:-1], y = (...)[1:].
        # Last prompt position predicts first answer, last answer predicts EOS.
        mask = [0] * (len(pids) - 1) + [1] * (len(aids) + 1)
        if len(mask) != len(pids) + len(aids) or sum(mask) != len(aids) + 1:
            raise ValueError("Answer/EOS target-mask alignment failed")
        maxima["promptTokens"] = max(maxima["promptTokens"], len(pids))
        maxima["answerTokensPlusEos"] = max(maxima["answerTokensPlusEos"], len(aids) + 1)
        maxima["recordTokensIncludingEos"] = max(maxima["recordTokensIncludingEos"], len(pids) + len(aids) + 1)
        targets += sum(mask)
    return {
        "status": "sb01-tokenizer-reviewed-no-model-execution",
        "probeSha256": SCORING.PROBE_SHA, "tokenizerSha256": SCORING.TOKENIZER_SHA,
        "records": len(rows), "maximum": maxima, "answerEosTargets": targets,
        "tokenizerByteIdentityVerified": False,
        "modelLoaded": False, "trainingPerformed": False, "optimizerUpdates": 0,
        "evaluationAuthorized": False, "finalHoldoutOpened": False,
    }


def review_file(path: Path = TOKENIZER) -> dict[str, Any]:
    SCORING.frozen_probe()  # fail before tokenizer access if proposal changed
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Frozen tokenizer must be a regular bounded file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCORING.TOKENIZER_SHA:
        raise ValueError("Frozen tokenizer SHA changed")
    from tokenizers import Tokenizer
    from plex_training.tokenizer import PlexTokenizer

    # Load exactly the bytes hashed above, without reopening a mutable path.
    output = review(PlexTokenizer(Tokenizer.from_str(raw.decode("utf-8"))))
    output["tokenizerByteIdentityVerified"] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    args = parser.parse_args()
    try:
        print(json.dumps(review_file(args.tokenizer), sort_keys=True, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"p3-sb01-tokenizer: {exc}\n")


if __name__ == "__main__":
    main()
