# File selection and review

Read the package metadata and applicable project instructions. Enumerate eligible tracked package source files, excluding generated, dependency, vendored, data, checkpoint, and holdout paths. Randomly select exactly one file and promptly report its path and the candidate count to the coordinator.

Review the file for concrete defects. Read relevant callers when needed to establish impact. Report each finding with severity, precise line evidence, trigger, resulting behavior, and a minimal suggested fix. Distinguish confirmed findings from uncertainty. Do not edit files or run training. If access fails, report the exact error; do not select an unseen file or infer local contents from a remote copy.
