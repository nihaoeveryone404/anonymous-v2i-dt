# Anonymization and Release Scope

## Included

- Main research implementation, comparison runners and procedural V2X simulator.
- Optional VMAS integration, with the dependency distributed separately.
- Explicit config presets, smoke tests, raw-metric reporting and reproduction docs.
- Source hashes and semantic fingerprints for the extracted core definitions.

## Excluded

- Workstation-specific absolute environment/model paths.
- Model weights, checkpoints, logs, Excel/CSV results and generated images.
- Manuscript drafts, third-party papers, author kits and personal documents.
- Historical scripts that shift, rank, stretch or synthesize display curves.
- Paper-layout automation, ad hoc merge scripts and unrelated utilities.
- Git history, credentials, account identifiers and editor/agent configuration.

## Checks and Limitations

The selected source was scanned for Windows drive paths, local user-profile
paths and common credential token patterns. Public files contain no configured
GitHub remote, author email or private download location. This is a scoped
release check, not a guarantee that all possible identifying stylistic clues
have been removed.

Comments already present in the simulator were retained. The historical
algorithm/package names were retained for traceability. The source manifest
contains original relative paths and hashes, but not the workstation root.
Generated runtime configs and logs can contain the executing machine's paths;
they are not part of the code archive.

License selection, simulator ownership and any model redistribution rights
remain the copyright holder's responsibility. Git account identity and remote
repository visibility must be checked separately before publication.
