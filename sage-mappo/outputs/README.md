# Generated artifacts

Each run writes into its configured subdirectory. Training produces Excel
workbooks containing raw episode/step metrics and resolved arguments. The safe
baseline writes incremental JSONL logs. The reporting entry point produces CSV
tables and PNG/PDF figures from raw episode metrics.

These files are local outputs, ignored by Git and excluded from the release ZIP.
Long training jobs should also redirect stdout/stderr to a local log. Main and
additional-baseline workbooks are written when the corresponding job completes.
