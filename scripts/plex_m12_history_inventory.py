"""Read-only research inventory of the prior prepared Plex 600M owner corpus.

First reuses the original source-bound verifier; then counts previously
prepared language/split data without exporting source text or authorizing
M12 reuse. This does not train, load weights, or access a final holdout.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training/src"))

from plex_training import plex600_owner_pilot as pilot

DATASET = ROOT / "training/pretraining/plex600-owner-authored-pilot-v1"


def inventory(dataset=DATASET):
    verified = pilot.verify(dataset)
    if (verified["trainRecords"] != 120 or verified["validationRecords"] != 24
            or verified["trainingAuthorized"] is not False
            or verified["finalHoldoutOpened"] is not False
            or verified["priorValidationExcluded"] != 72
            or verified["externalHistoricalRecords"] != 0):
        raise ValueError("Prior corpus status or provenance is not the frozen preparation scope")
    groups = {}
    for split in ("train", "validation"):
        items = [json.loads(line) for line in pilot.read(dataset / (split + ".jsonl")).splitlines()]
        counts = Counter(item["language"] for item in items)
        if set(counts) != {"javascript", "html", "css"}:
            raise ValueError("Unexpected language split in historical 600M corpus")
        groups[split] = {
            "records": len(items),
            "byLanguage": {name: counts[name] for name in ("javascript", "html", "css")},
            "groupCount": len({item["splitGroupId"] for item in items}),
            "requestAnswerFormat": all(item["text"].startswith("Request: ")
                                       and "\nAnswer:\n" in item["text"] for item in items),
        }
        if not groups[split]["requestAnswerFormat"]:
            raise ValueError("Historical corpus contract is not request/answer text")
    if groups["train"]["byLanguage"] != {"javascript": 36, "html": 42, "css": 42}:
        raise ValueError("Unexpected historical training language breakdown")
    if groups["validation"]["byLanguage"] != {"javascript": 8, "html": 8, "css": 8}:
        raise ValueError("Unexpected historical internal validation breakdown")
    return {
        "kind": "m12-a-read-only-historical-inventory-v1",
        "historicalVerificationStatus": verified["status"],
        "historicalManifestSha256": verified["manifestSha256"],
        "priorCorpus": groups,
        "previousP2ValidationExcluded": verified["priorValidationExcluded"],
        "javascriptRecordsTotal": 44,
        "priorApprovedScope": "local Plex 600M dataset preparation/tokenizer evaluation only",
        "priorCorpusTask": "request-answer code generation, NOT exact source copying",
        "m12NewOwnerCorpusPrepared": False,
        "m12TrainingAuthorized": False,
        "m12ReuseRightsEstablished": False,
        "independentM12DevelopmentEstablished": False,
        "sealedFinalHoldoutAccessed": False,
        "modelWeightsLoaded": False,
        "trainingPerformed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(inventory(args.dataset_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
