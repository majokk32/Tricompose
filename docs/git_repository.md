# Git repository usage

The CARC workspace exposes the root `.git` directory as a read-only mount.
This checkout therefore keeps writable Git metadata in the ignored
`.git-tricompose/` directory and provides a repository-local wrapper:

```bash
cd /project2/ruishanl_1185/inference_3mod
./tools/tricompose-git status
./tools/tricompose-git add <path>
./tools/tricompose-git commit -m "Describe the change"
./tools/tricompose-git push
```

The configured remote is:

```text
https://github.com/majokk32/Tricompose.git
```

This workaround applies only to the existing CARC workspace. A normal clone
of the GitHub repository has a standard writable `.git` directory and uses
ordinary `git status`, `git add`, `git commit`, and `git push` commands.

The root `.gitignore` excludes protected/generated artifacts, patient or
medical data formats, checkpoints, weights, environments, caches, logs,
credentials, and downloaded baseline repositories. Never force-add anything
under `artifacts/`, even when Git permits an override.
