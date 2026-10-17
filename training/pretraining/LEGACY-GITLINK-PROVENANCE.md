# Historical raw-source gitlink provenance

The eight paths below were tracked as Git mode `160000` commit references inside `training/pretraining/data/raw/`. The repository has **no `.gitmodules` file or URL mapping** for those entries. They are not ordinary checked-in source files, and standard checkout/submodule initialization cannot retrieve their contents.

On October 17, 2026, these unsupported gitlinks were retired to stop GitHub Actions checkout cleanup from issuing `fatal: No url found for submodule path ... in .gitmodules`. Their exact former Git object references are preserved here.

| Former gitlink under `training/pretraining/data/raw/` | Referenced commit SHA |
|---|---|
| `adamschwartz-magic-of-css` | `066aa596b420029f72b553f027773c5de74c1b21` |
| `andy-set-studio-vanilla-js-state-management` | `818586cf3f010e646ae068e6d09a543f299b5bda` |
| `basic-projects` | `b6305245cea37b7a9cb600b1e3b3aa618dc33c68` |
| `bradtraversy-vanilla-parcel-boilerplate` | `bbb80fc71c9fa1d5ac5a4aa5a143c26b357ef353` |
| `html-css-js-template` | `a25d09d1829b3b38a987dc68a14e8f0c6d2fc6c9` |
| `mailgun-transactional-email-templates` | `e05e59e8c1e0fa7ffbacc7a165a609fd37c590b5` |
| `micku7zu-vanilla-tilt-js` | `48f4ee931d4d91dcdd2f91803db67a9f3a16587e` |
| `sparksuite-simple-html-invoice-template` | `7d74fe34c862b7712f1a23a3f12147a4a8bd53e1` |

This is only repository checkout hygiene and provenance preservation. No raw-source files were read, rewritten, replaced, or moved. The ignored `training/pretraining/data/raw/` location remains available for separately obtained local corpus material, subject to the historical source licensing and contamination rules. Frozen dataset manifests, tokenizer files, model checkpoints, numeric experiment reports, and hash records were not changed. A referenced commit SHA is **not** evidence that its source file bytes were present or usable at checkout.
