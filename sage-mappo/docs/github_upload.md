# Publishing This Folder

Publish only this release directory, not its parent research workspace.
No remote repository has been created and no files have been pushed by the
packaging process.

## Before Upload

1. Replace the pending `LICENSE` notice with the license selected by the
   copyright holder, after checking simulator/source redistribution rights.
2. Review `ANONYMIZATION_REPORT.md` and `docs/protocol_notes.md`.
3. Run the unit tests and smoke commands in the README.
4. Confirm no results, weights, personal documents, credentials or local model
   files are staged. `.gitignore` excludes common artifact types.
5. For anonymous review, use an appropriate account and Git commit identity.
   An anonymous folder name alone cannot anonymize GitHub account/history.

## Git Commands

From this directory, after creating an empty GitHub repository:

```bash
git init
git branch -M main
git add .
git diff --cached --stat
git status --short
git commit -m "Prepare research code release"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

The URL above is a placeholder, not a configured remote. Check repository
visibility before pushing. Authentication should use your normal GitHub login
or credential manager; do not put access tokens in source files or URLs.

## Code Archive

```bash
python scripts/build_code_archive.py
```

This generates `dist/anonymous-hippo-mappo-code.zip` and a per-file SHA-256
manifest. The archive uses an allowlist of source/docs/config directories and
specific input/output README placeholders. Model files, metrics, plots, caches,
private local files and experiment outputs are excluded. The archive does not
include `.git` history or select a license on the owner's behalf.
