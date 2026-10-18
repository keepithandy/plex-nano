"""SB03 V2: streaming non-holdout ancestry inspection, synthetic-only CI."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "phase3" / "copy-fidelity-v1"
SPEC = importlib.util.spec_from_file_location(
    "p3_sb03_v2_ancestor_audit_tests", HERE / "audit_v2_ancestor_history.py",
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fake_names():
    # Full-size fake list; repeat-free; never touch the owner's private proposal.
    names = [f"x{n:04d}".encode("ascii") for n in range(48)]
    assert len(set(names)) == 48
    return names


def make_inventory(tmp_path: Path, sources: list[tuple[str, bytes]], *, alternate=False):
    files = []
    for name, body in sources:
        f = tmp_path / name
        f.write_bytes(body)
        files.append({
            "path": str(f.resolve()),
            "sha256": hashlib.sha256(body).hexdigest(),
            "role": "ancestor-training-text",
        })
    p = tmp_path / "inventory.json"
    p.write_text(json.dumps({
        "schemaVersion": 1,
        "scope": AUDIT.SCOPE,
        "files": files,
    }), encoding="utf-8")
    return p


def test_sha_and_ascii_matches_across_one_byte_chunks():
    text = b"header X0001 and x0007 at tail "
    digest, total, hits = AUDIT._stream_sha_and_matches(
        io.BytesIO(text), tuple(fake_names()), buffer_size=1
    )
    assert digest == hashlib.sha256(text).hexdigest()
    assert total == len(text)
    assert hits == {"x0001", "x0007"}


def test_streaming_digest_over_binary_chunks_never_fabricates_identifiers():
    text = b"\x00" + b"a" * 5000 + b"\xff"
    d, count, hits = AUDIT._stream_sha_and_matches(
        io.BytesIO(text), tuple(fake_names()), buffer_size=13
    )
    assert d == hashlib.sha256(text).hexdigest()
    assert count == len(text)
    assert hits == set()


def test_valid_inventory_and_sha_fail_closed(tmp_path: Path):
    manifest = make_inventory(tmp_path, [
        ("a.txt", b"safe training ancestor text"),
        ("b.jsonl", b"{\"text\":\"X0001\"}\n"),
    ])
    entries = AUDIT._inventory(manifest)
    assert len(entries) == 2
    assert entries[0]["role"] == "ancestor-training-text"
    # Tamper AFTER capturing expected SHA.
    (tmp_path / "b.jsonl").write_bytes(b"changed")
    monkey_names = tuple(fake_names())
    with (tmp_path / "b.jsonl").open("rb") as handle:
        got = AUDIT._stream_sha_and_matches(handle, monkey_names)[0]
    assert got != entries[1]["sha256"]


def test_audit_matches_only_listed_sources_and_never_certifies_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = make_inventory(tmp_path, [
        ("train.txt", b"uses X0001 in a prior example\n"),
        ("other.jsonl", b"ordinary script without these identifiers\n"),
    ])
    monkeypatch.setattr(AUDIT, "_read_proposal", lambda path: ({}, tuple(fake_names())))
    result = AUDIT.audit(proposal=tmp_path / "ignored.json", inventory=manifest)
    assert result["inventoryFileCount"] == 2
    assert result["verifiedSourceFiles"] == 2
    assert result["filesWithNameCollisions"] == 1
    assert result["collidingNames"] == ["x0001"]
    assert result["caseInsensitive"] is True
    assert result["status"] == "listed-sources-include-training-name-collisions"
    assert result["historicalTrainingLineageComplete"] is False
    assert result["readyForTraining"] is False
    assert result["optimizerUpdates"] == 0
    assert result["finalHoldoutOpened"] is False
    assert result["trainingAuthorized"] is False
    assert result["newInferenceAuthorized"] is False


def test_empty_collision_result_never_authorizes_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = make_inventory(tmp_path, [("plain.md", b"ordinary coding document\n")])
    monkeypatch.setattr(AUDIT, "_read_proposal", lambda path: ({}, tuple(fake_names())))
    r = AUDIT.audit(proposal=tmp_path / "ignored.json", inventory=manifest)
    assert r["status"] == "listed-sources-clear-completeness-unverified"
    assert r["collidingNames"] == []
    assert r["historicalTrainingLineageComplete"] is False
    assert r["trainingAuthorized"] is False
    assert r["readyForTraining"] is False


def test_file_hash_mismatch_refuses_entire_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = make_inventory(tmp_path, [
        ("known.txt", b"expected baseline"),
        ("later.jsonl", b"possible x0001 conflict"),
    ])
    (tmp_path / "later.jsonl").write_bytes(b"modified after manifest")
    monkeypatch.setattr(AUDIT, "_read_proposal", lambda path: ({}, tuple(fake_names())))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        AUDIT.audit(proposal=tmp_path / "ignored.json", inventory=manifest)


def test_duplicate_paths_and_nontext_are_refused(tmp_path: Path):
    manifest = make_inventory(tmp_path, [("x.txt", b"data")])
    j = json.loads(manifest.read_text())
    j["files"].append(j["files"][0])
    manifest.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        AUDIT._inventory(manifest)
    manifest = make_inventory(tmp_path, [("tokens.u16le", b"\x00\x01")])
    with pytest.raises(ValueError, match="plain-text"):
        AUDIT._inventory(manifest)


def test_final_holdout_and_symlinks_are_refused(tmp_path: Path):
    d = tmp_path / "final-holdout"
    d.mkdir()
    source = d / "test.txt"
    source.write_bytes(b"sealed")
    manifest = make_inventory(tmp_path, [("safe.txt", b"ok")])
    j = json.loads(manifest.read_text())
    j["files"][0]["path"] = str(source)
    j["files"][0]["sha256"] = hashlib.sha256(b"sealed").hexdigest()
    manifest.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        AUDIT._inventory(manifest)
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(source)
    except OSError:
        return  # Windows may deny symlink creation without developer mode.
    j["files"][0]["path"] = str(link)
    manifest.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(ValueError, match="regular file"):
        AUDIT._inventory(manifest)


def test_inventory_must_be_explicit_and_sha_pinned(tmp_path: Path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({
        "schemaVersion": 1, "scope": AUDIT.SCOPE, "files": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="1..1000"):
        AUDIT._inventory(empty)
    malformed = tmp_path / "bad.json"
    malformed.write_text(json.dumps({
        "schemaVersion": 1, "scope": AUDIT.SCOPE,
        "files": [{"path": str(empty.resolve()),
                   "sha256": "bad", "role": "ancestor-training-text"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid inventory"):
        AUDIT._inventory(malformed)


def test_actual_v2_proposal_requires_expected_sha(tmp_path: Path):
    candidate = tmp_path / "proposal.json"
    candidate.write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        AUDIT._read_proposal(candidate)


def test_stream_only_no_model_execution_or_file_writes():
    script = (HERE / "audit_v2_ancestor_history.py").read_text(encoding="utf-8")
    for unsafe in (
        "import torch", "optimizer.step(", "model.generate(",
        "read_checkpoint(", "final_holdout", ".write_text(", ".write_bytes(",
        "os.walk(", "glob(", "rglob(", "mkdir(", 'open("xb")',
    ):
        assert unsafe not in script
    assert "trainingAuthorized" in script
    assert "historyCoverageOnlyListedFiles" in script
    assert "stream.read(buffer_size)" in script
