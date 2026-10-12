# Three-agent package review swarm

Reusable coordinator instructions and agent prompts for reviewing one randomly selected package source file. These are instructions for an agent host with delegation support, not a standalone executable or a saved live session.

## Use

Give `coordinator.md` to the coordinating agent and supply the package root. The coordinator starts the three roles in `agents/`, passes the selected file to the other reviewers, and consolidates their findings.

All review work is read-only. Do not edit source, run training, access a closed holdout, commit, or send messages externally. Respect applicable project instructions. Report access failures without inventing findings.

## Create a ZIP on Windows

From the parent directory, run:

```powershell
powershell -NoProfile -File .\review-swarm\package.ps1
```

The script creates `review-swarm.zip` alongside this folder and refuses to overwrite an existing archive.
