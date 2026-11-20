# Plex Phase 1 · Supplied-file editor completion audit

**Date:** November 20, 2026  
**Scope:** Built-in Python console editor and packaged `plex` dashboard, not model training or a successful Plex 600M source-conditioned edit.  
**Baseline:** [PR #226](https://github.com/keepithandy/plex-nano/pull/226) merged into `master` at `fd302953a33c5b124e34ebc5a86ad7543043aec7`.  
**Audit fixes:** [PR #227](https://github.com/keepithandy/plex-nano/pull/227).

## Exit-gate decision

**PASS for the supported Phase 1 editing subset**, provided PR #227 remains merged and its verified code is present on `master`. The required end-to-end tests demonstrate correct edits, bytes-preserved originals, reviewable diffs and changed spans, undo, create-only saves, and deliberate refusals. This is not a claim that arbitrary JavaScript can be safely renamed or that the M11 model can make an edit.

## What was exercised

The console tests launch `scripts/PlexLocal.py` through standard input in isolated Python with real temporary files. They compare original and saved bytes and check the messages shown to the user.

- File open: UTF-8, UTF-8 BOM, CRLF, invalid UTF-8 refusal, retained trailing newline or missing final newline.
- Editing: supported flat identifier rename (declaration and console.log references), exact copy, set-integer, unique literal replacement, preserved quoted strings and comments, and changed-span ranges.
- Review: unified diff, missing-newline notices, undo, independent fresh paste baseline.
- Saving: source path protected, existing target never overwritten, saved copy byte-exact, saved copy allows a source switch.
- Refusals: nested syntax, duplicate declarations, collisions, reserved identifiers, value mismatch, absent source/identifier, repeated literal occurrence, malformed built-in commands, invalid Unicode surrogate replacement, and unsaved-source switching. The refusal path does not invoke the model.
- `/discard`: explicitly abandons in-memory changes; the original file stays unchanged. Failed open leaves the current session intact.

## CI evidence

[Successful cross-platform Phase 1 audit run](https://github.com/keepithandy/plex-nano/actions/runs/44828473111) on the tested code commit `b2f8a60d55958e2c5ab8517501cbc48f3b09bc56`:

- Ubuntu, Python 3.12: Phase 0 smoke plus **nine** Phase 1 end-to-end tests passed.
- Windows, Python 3.12: Phase 0 smoke plus **nine** Phase 1 end-to-end tests passed.
- Node 24 maintenance: `npm test` passed; the `npm pack` installed entry-point smoke invoked `plex` and passed dashboard, rename, redraw, status, and version checks.

Reproduce the editor checks with:

```bash
python -I -S scripts/plex_phase0_smoke.py
python -I -S scripts/plex_phase1_smoke.py -v
```

The first audit run caught an earlier test that tried to paste over an unsaved edit; the test was updated to save a new copy before switching sources. The subsequent complete run passed on both operating systems. CI runs on GitHub-hosted runners; this audit did not execute the owner's local Windows/WSL/CUDA installation directly.

## Scope and limitations

The editor verifies only its advertised bounded operations. Rename supports flat integer declarations and simple console.log calls, not arbitrary JavaScript, nested or lexical scopes, destructuring, imports, or parser-backed refactors. Explicit malformed commands refuse rather than being silently treated as free-form model requests. Broad natural-language paraphrase handling, independent syntax validation of model suggestions, and model-proposal diff before acceptance remain later-phase work.

Plex 600M M11 step-200 inference remains a separate measurement. The owner's recent free-form console request returned **no code (EOS)**. Earlier measured four-line M11 probes passed only the two familiar training lines exactly and used the wrong identifier on both development lines. Neither built-in editor passes nor the installed-dashboard CI are evidence that the model completed a JavaScript edit.

**Training status:** paused. No new training, checkpoint, tokenizer, or protected evaluation changes were made in this audit.
