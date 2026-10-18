"""P3-SB05 pure synthetic scorer design. No model loading or execution entrypoint.

Calling this function does not authorize generating model responses. All criteria
come from the immutable SB01 proposal. Lexical namespace counts are diagnostic;
only complete, exact files pass capability gates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

HERE = Path(__file__).resolve().parent
PROBE = HERE / "source-binding-microprobe-v1.json"
PROBE_SHA = "0a30df999f8e1c0f12cf0f6e4a71e0eab814bdd99b3bd83d0e2184637ad31fda"
TOKENIZER_SHA = "2d5102623cf8e8e51925ab5e6ea05716221013538c5b661476aa1ea765af2697"
CHECKPOINT_SHA = "c96800281811d710f92322bac37f31a0386994bc7f64b59fd127b4631e470230"
LANGUAGES = ("html", "css", "javascript")


def frozen_probe() -> dict[str, Any]:
    if PROBE.is_symlink() or not PROBE.is_file() or PROBE.stat().st_size > 1024 * 1024:
        raise ValueError("Microprobe must be the regular frozen proposal")
    raw = PROBE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROBE_SHA:
        raise ValueError("Frozen microprobe SHA changed")
    return json.loads(raw)


def _namespace_lexemes(text: str, namespace: str) -> Counter[str]:
    # Includes full CSS/JS names and HTML attribute values, plus comment words.
    # A multiset match cannot prove syntax or identifier location preservation.
    return Counter(re.findall(
        r"(?<![A-Za-z0-9_-])" + re.escape(namespace) + r"[A-Za-z0-9_-]*", text
    ))


def score_responses(responses: list[dict[str, Any]], *,
                    probe_sha256: str, checkpoint_sha256: str,
                    tokenizer_sha256: str) -> dict[str, Any]:
    """Score supplied responses only; tests use synthetic strings exclusively.

    Identity arguments bind the caller's declared provenance, not independently
    verified model execution. There is deliberately no inference/file-reader CLI.
    """
    if (probe_sha256 != PROBE_SHA or checkpoint_sha256 != CHECKPOINT_SHA
            or tokenizer_sha256 != TOKENIZER_SHA):
        raise ValueError("Response provenance differs from the frozen scoring proposal")
    rows = frozen_probe()["rows"]
    if (not isinstance(responses, list) or len(responses) != 48
            or any(not isinstance(r, dict) for r in responses)
            or [r.get("recordId") for r in responses] != [r["id"] for r in rows]):
        raise ValueError("Expected all 48 frozen response IDs exactly once in order")
    counts: Counter[str] = Counter()
    per_language = {lang: Counter() for lang in LANGUAGES}
    groups: dict[str, list[bool]] = defaultdict(list)
    scored = []
    for row, reply in zip(rows, responses):
        if (type(reply.get("actual")) is not str
                or type(reply.get("generatedTokenCount")) is not int
                or not 0 <= reply["generatedTokenCount"] <= 192
                or any(type(reply.get(key)) is not bool for key in
                       ("stoppedAtEos", "decodeSucceeded", "fullPromptVisible"))
                or (reply["stoppedAtEos"] and reply["generatedTokenCount"] == 192)):
            raise ValueError("Malformed completion flags, text or token budget")
        actual = reply["actual"]
        actual.encode("utf-8")  # reject invalid Unicode, never normalize bytes
        complete = all(reply[key] for key in
                       ("stoppedAtEos", "decodeSucceeded", "fullPromptVisible"))
        copy = row["condition"] == "copy"
        start, end = row["targetStart"], row["targetEnd"]
        before, after = row["inputFile"][:start], row["inputFile"][end:]
        boundaries = (actual.startswith(before) and actual.endswith(after)
                      and len(actual) >= len(before) + len(after))
        extractable = complete and boundaries if not copy else None
        middle_end = len(actual) - len(after) if after else len(actual)
        requested = (actual[len(before):middle_end] == row["requestedLiteral"]
                     if extractable else None)
        exact = complete and actual == row["expectedFile"]
        preserved = complete and (actual == row["inputFile"] if copy else boundaries)
        namespace = row["sourceNamespace"]
        lexical = (_namespace_lexemes(actual, namespace) ==
                   _namespace_lexemes(row["inputFile"], namespace))
        facts = {
            "cases": 1, "copies": int(copy), "edits": int(not copy),
            "exactCopies": int(copy and exact), "exactEdits": int(not copy and exact),
            "exactFiles": int(exact), "unrelatedSourcePreserved": int(preserved),
            "namespaceLexemesMatch": int(lexical),
            "requestedValueExtractable": int(extractable is True),
            "requestedValueCorrect": int(requested is True),
            "requestedValueWrong": int(extractable is True and requested is False),
            "requestedValueUnknown": int(not copy and extractable is not True),
            "eos": int(reply["stoppedAtEos"]),
            "decodeSucceeded": int(reply["decodeSucceeded"]),
            "fullPromptVisible": int(reply["fullPromptVisible"]),
        }
        counts.update(facts)
        per_language[row["language"]].update(facts)
        groups[row["groupId"]].append(exact)
        scored.append({
            **reply, "language": row["language"], "condition": row["condition"],
            "exactCompleteFile": exact, "unrelatedBytesPreserved": preserved,
            "namespaceLexemesMatch": lexical, "requestedValueExtractable": extractable,
            "requestedValueCorrect": requested,
        })
    complete_groups = sum(all(values) for values in groups.values())
    gates = {
        "copies12of12": counts["exactCopies"] == 12,
        "editsAtLeast30of36": counts["exactEdits"] >= 30,
        "perLanguageAtLeast10of12Edits": all(
            per_language[lang]["exactEdits"] >= 10 for lang in LANGUAGES),
        "completeGroupsAtLeast9of12": complete_groups >= 9,
    }
    return {
        "kind": "p3-sb05-synthetic-scorer-design-v1",
        "probeSha256": PROBE_SHA, "declaredCheckpointSha256": checkpoint_sha256,
        "declaredTokenizerSha256": tokenizer_sha256,
        "executionProvenanceIndependentlyVerified": False,
        "totals": dict(counts), "perLanguage": {k: dict(v) for k, v in per_language.items()},
        "fullyCorrectContrastGroups": complete_groups, "contrastGroups": len(groups),
        "gates": gates, "allProposedGatesPassed": all(gates.values()),
        "namespaceLexemeMatchIsNotExactFilePreservation": True,
        "perRecord": scored,
        "modelLoaded": False, "trainingPerformed": False, "optimizerUpdates": 0,
        "liveEvaluationAuthorized": False, "finalHoldoutOpened": False,
    }
