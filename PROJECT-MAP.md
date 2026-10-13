# Plex Nano — Start Here in VS Code

**Open the repository root** (the folder containing `README.md`, `training/`, and `package.json`) in VS Code. This is a real, runnable Python model research package with a separate, older TypeScript utility layer. The model is **not yet a reliable coding assistant**: P3-02 trained successfully but failed its 36-task unseen-edit development gate (0/36 exact edits).

## First five places to open

| What you want to find | Open this |
|---|---|
| **Current research status and what comes next** | [Plex-ROADMAP.md](Plex-ROADMAP.md) and [P3-02 experiment report](docs/PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md) |
| **The model (Transformer weights, forward pass)** | [training/src/plex_training/model.py](training/src/plex_training/model.py) and [config.py](training/src/plex_training/config.py) |
| **How training and evaluation are invoked** | [training/src/plex_training/cli.py](training/src/plex_training/cli.py) and [training/README.md](training/README.md) |
| **Current HTML/CSS/JS editing experiment** | [file_edit_replacements.py](training/src/plex_training/file_edit_replacements.py), [file_edit_replacements_run.py](training/src/plex_training/file_edit_replacements_run.py), and [file_edit_replacements_development.py](training/src/plex_training/file_edit_replacements_development.py) |
| **The result and its limitations** | [P3-02 development gate summary](training/pretraining/p3-02-development-gate-result-summary.json) and [frozen training result](training/pretraining/p3-02-first-run-result.json) |

## Explorer map

```text
plex-nano/
├── README.md                       Project overview and current status
├── PROJECT-MAP.md                  YOU ARE HERE — where to find things
├── Plex-ROADMAP.md                 Roadmap and milestone history
├── CHANGELOG.md                    Historical changes
├── AGENTS.md                       Project scope and editing guardrails
├── .vscode/                        VS Code navigation, extensions and safe tasks
├── training/                       PRIMARY ACTIVE MODEL WORKSPACE (Python)
│   ├── README.md                   Python environment and command reference
│   ├── pyproject.toml, uv.lock      Python package and locked dependencies
│   ├── src/plex_training/          Model, tokenizer, CLI, checkpoint, evaluation
│   │   ├── model.py                Transformer model
│   │   ├── config.py               Model configuration
│   │   ├── tokenizer.py            Tokenizer code
│   │   ├── checkpoint.py           Save/load and checkpoint metadata
│   │   ├── cli.py                  Research command dispatcher
│   │   ├── file_edit_replacements*.py  P3-02 task, run and evaluation
│   │   └── ...                     Prior milestone and shared utilities
│   ├── tests/                      Python regression and safety tests
│   ├── pretraining/                Frozen experiment contracts and JSON results
│   ├── phase3/drafts/              Authored P3-01/P3-02 reviewed examples
│   ├── phase2/                     HISTORICAL P2 experiments and data
│   └── artifacts/                  GENERATED LOCAL FILES, ignored by Git
├── docs/                           Detailed experiment reports (see docs/INDEX.md)
├── src/                            LEGACY TypeScript repository utilities
├── tests/                          Tests for that TypeScript layer
├── prompts/, fixtures/             Historical/test prompts and small fixtures
├── scripts/                        Utility scripts
└── package.json                    Legacy TypeScript package/build config
```

**Start in `training/src/plex_training/` for Plex's AI model.** The top-level `src/` folder is an older repository-discovery utility, not the Python model. Keep it working for compatibility, but do not treat it as Plex Nano's present product direction.

## What's real versus a placeholder?

- **Implemented:** the PyTorch Transformer, tokenizer, checkpoint serializer, CPU/CUDA training runner, model CLI, deterministic experiment packs, source safeguards, development scorer, and automated tests.
- **Frozen research records:** `training/pretraining/*.json`, historical `training/phase2/` drafts and data, and milestone documents. Many use the word *draft* because they are reviewed input candidates or historical snapshots; they are **not empty mockups**. Don't move or delete these casually—hashes and references are used to verify experiments.
- **Generated locally, not necessarily visible in GitHub:** `training/artifacts/` and your external `$HOME\PlexArtifacts` checkpoints/evaluation outputs. They are ignored or external by design, **not missing placeholder checkpoints**.
- **Legacy or future product work:** root `src/` and `tests/` are real TypeScript utilities from the earlier direction, but not the active model implementation. A conversational file editor/app and expanded models are not yet built.
- **Model capability:** P3-02 completed 100 training updates, with validation loss 10.5668 → 1.5893; the separately frozen development evaluation failed **0/36 exact edits** and **0/36 preservation**. Don't equate a working training stack with a working code editor.

## Open, run, and test safely

Use VS Code **File → Open Folder** on the repo root. Open its integrated terminal (PowerShell), then:

```powershell
# Inspect the installed local training environment; no optimizer updates
uv run --project training python -m plex_training.cli environment

# See the available Python model/research commands
uv run --project training python -m plex_training.cli --help

# Run Python regression tests (safe; no authorized model training)
uv run --project training pytest training/tests

# Optional: test the legacy TypeScript layer
npm test
```

On Windows the Python interpreter usually lives at `training/.venv/Scripts/python.exe`. Suggested VS Code Python and Pylance extensions appear under **Extensions → Workspace Recommendations**. The checked-in `.vscode/settings.json` hides only disposable local caches/build outputs from the **Explorer** and excludes bulky historical Phase 2 data/drafts from **default text search**. The archive remains present and browsable, and you can override those search exclusions when needed. Safe, explicitly labeled tasks are available under **Terminal → Run Task**; there is **no training task**.

For the exact P3-02 **development-only** local evaluation and its fixed step-100 checkpoint, see [P3-02 experiment notes](docs/PHASE-3-P3-02-EXACT-SMALL-REPLACEMENTS.md). Do **not** rerun it into an existing result directory. More optimizer updates, checkpoint reselection, and final project holdout access remain unauthorized.

## Detailed navigation

- [Documentation index](docs/INDEX.md) — find current milestone reports without scanning the entire docs folder
- [Python training workspace](training/README.md) — setup, commands, research runners
- [Pretraining evidence](training/pretraining/README.md) — historical corpus details plus current milestone pointers
- [Bug Atlas](BUG-ATLAS.md) and [package audit](docs/PACKAGE-BUG-AUDIT.md) — resolved package/reliability findings
- [Project development rules](AGENTS.md), [Python research rules](training/AGENTS.md), and [TypeScript rules](src/AGENTS.md)

## Clean-up policy

The organization pass deliberately **does not rename or relocate** model modules, datasets, P2/P3 manifests, checksum-bound files, checkpoint evidence, or legacy CLI code. Such changes could invalidate reproducibility or tests. New work should prioritize clearer navigation and safe, reviewed changes over cosmetic mass moves.
