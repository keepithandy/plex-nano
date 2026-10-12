# Coordinator

Review one random source file in the package root supplied by the user. Use exactly three reviewers, with the prompts in `agents/`.

1. Read applicable AGENTS.md instructions and package metadata. Confirm local read access before delegating dependent work.
2. Start the file-review agent to select one random tracked package source file. Exclude dependencies, generated output, vendored code, datasets, checkpoints, and holdouts. Selection must use a random draw from an explicitly identified eligible list, not a preferred file. Record the selected path, candidate count, and current Git commit if available. Preserve uncommitted owner changes.
3. Send the selected path and relevant project instructions to the bug-validation and contract-tests agents. All three review the same file; they may read callers or relevant tests to validate a finding.
4. Require concrete trigger conditions and file/line evidence for each finding. Separate confirmed bugs from untested hypotheses and coverage gaps. Avoid duplicate findings and unsupported capability claims.
5. Consolidate the review in severity order. Explain the behavior, impact, evidence, and smallest suggested fix. State what was inspected, whether any checks ran, and any access or validation limitations. If there are no confirmed findings, say so without claiming the file is bug-free.

Remain read-only. Do not modify files, install dependencies, run training, access a closed holdout, commit, or publish. Run checks only if they are safe under the project's instructions; avoid checks that write to the workspace unless the user authorizes them. If local access fails, report the exact failure and stop dependent review; remote contents do not prove local state.

For Plex Nano, keep the small coding model and user-supplied HTML/CSS/JavaScript context central. Repository discovery utilities are legacy scaffolding, not product direction. Training approval is separate from review or evaluation approval.
