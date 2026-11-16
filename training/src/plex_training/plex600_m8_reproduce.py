"""Independent M7 tokenizer reconstruction and M8 output feasibility.

This reconstructs a tokenizer only from the committed, owner-authored M7
preparation-approved dataset, inside a self-cleaning temporary directory.
No model weights, CUDA, optimizer, model training, or final holdout are used.
Measured M8 capacities are reported as M7-comparable only if all original
owner-reported tokenizer and manifest hashes reproduce exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from . import plex600_m7 as m7
from . import plex600_m8_eval as m8
from . import plex600_owner_pilot as owner
from . import plex600_research as research
from . import plex600_setup as setup
from .tokenizer import PlexTokenizer, train_tokenizer

PROTOCOL = "plex600-m8-independent-tokenizer-reconstruction-v1"


def inspect(*, max_cases: int = 36) -> dict[str, Any]:
    m8.plan(max_cases=max_cases, max_new_tokens=64)
    return {
        "protocol": PROTOCOL,
        "status": "plan-only",
        "maxCases": max_cases,
        "dataset": "committed-owner-authored-m7-pilot-only",
        "ownerPinnedTokenizerSha256": m7.TOKENIZER_SHA,
        "ownerPinnedBundleManifestSha256": research.BUNDLE_SHA,
        "modelLoaded": False, "trainingAuthorized": False,
        "optimizerUpdates": 0, "finalHoldoutUsed": False,
        "externalSourcesUsed": False,
    }


def reconstruct(*, max_cases: int = 36) -> dict[str, Any]:
    """Run a bounded CPU-side identity and capacity experiment, not training."""
    baseline = inspect(max_cases=max_cases)
    dataset = owner.DATASET
    receipt = owner.verify(dataset)
    if receipt["status"] != "verified-owner-authored-tokenizer-ready":
        raise ValueError("Committed owner-authored dataset failed source audit")
    if research.digest_file(dataset / "manifest.json", 1024 * 1024) != research.SOURCE_SHA:
        raise ValueError("M7 owner-authored source manifest SHA differs")

    with TemporaryDirectory(prefix="plex600-m8-tokenizer-reconstruction-") as temp:
        bundle = Path(temp) / "new-tokenizer"
        fitted = train_tokenizer(
            dataset, bundle, vocab_size=32_768, min_frequency=2,
            model_family="plex-600m-v1",
            storage_limit_bytes=16 * 1024 * 1024,
        )
        bundle_sha = research.digest_file(bundle / "manifest.json", 1024 * 1024)
        check = setup.check_tokenizer_bundle(bundle)
        if (check["tokenizerSha256"] != fitted["tokenizerSha256"]
                or check["actualVocabularySize"] != fitted["actualVocabularySize"]):
            raise ValueError("Reconstructed bundle fails independent tokenizer verification")

        identity_match = (
            fitted["tokenizerSha256"] == m7.TOKENIZER_SHA
            and bundle_sha == research.BUNDLE_SHA
            and fitted["actualVocabularySize"] == 1191
        )
        result = {
            **baseline,
            "status": (
                "verified-frozen-m7-tokenizer-reconstruction"
                if identity_match else "reconstructed-tokenizer-m7-identity-mismatch"
            ),
            "ownerApprovedSourceVerified": True,
            "reconstructedTokenizerSha256": fitted["tokenizerSha256"],
            "reconstructedBundleManifestSha256": bundle_sha,
            "reconstructedActualVocabularySize": fitted["actualVocabularySize"],
            "frozenM7IdentityMatches": identity_match,
            "comparableM8CapacityMeasured": False,
        }
        if not identity_match:
            # Never pass off an alternative tokenizer's token lengths as
            # those of the owner's frozen M7 checkpoint.
            return result

        bound = setup.check_source_binding(
            bundle, dataset, bundle_manifest_sha256=bundle_sha,
            source_manifest_sha256=research.SOURCE_SHA,
        )
        if bound["sourceBindingVerified"] is not True:
            raise ValueError("Reconstructed M7 tokenizer source binding differs")

        tokenizer = PlexTokenizer.load(bundle)
        cases = m8.choose_cases(max_cases)
        budget32 = m8.token_budget(cases, tokenizer, max_new_tokens=32)
        budget64 = m8.token_budget(cases, tokenizer, max_new_tokens=64)
        result.update({
            "comparableM8CapacityMeasured": True,
            "sourceBindingVerified": True,
            "benchmarkSha256": research.digest_file(research.BENCHMARK),
            "cap32": budget32,
            "cap64": budget64,
        })
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-read-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=36)
    args = parser.parse_args(argv)
    try:
        result = (
            reconstruct(max_cases=args.max_cases)
            if args.reconstruct_read_only else inspect(max_cases=args.max_cases)
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print("Plex M8 independent tokenizer audit refused: " + str(exc),
              file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
