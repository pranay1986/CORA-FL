# Reproducibility audit

The package retains the pre-result protocol lock, configuration, pinned
dependencies, exact dataset-array fingerprints, partition fingerprints, frozen
traces, a CSV curve and JSON record per run, plot data, tables, figures, and a
SHA-256 manifest.

`python scripts/verify_confirmatory.py` checks:

- the exact 3,780-cell factorial matrix, unique run IDs, and zero failures
- config and implementation hashes against the pre-result lock
- all raw curve and trace checksums, finite metrics, final-value identities, and
  byte-accounting identities
- paired reuse of datasets, partitions, traces, and learning rates
- all 630 validation-only pilot trials, 63 exact argmin selections, no test
  access, and no search-grid boundary selections
- all 105 pair-domain mappings for deterministic topology certificates
- independent recomputation of 1,890 nominal-outage pairs, seed-macro inference,
  all three tables, and the claim-decision JSON
- exactly eight PNG and eight PDF figures, eight plot-data CSVs, and three CSV
  plus three TeX tables

Wall time is hardware- and load-dependent and is reported exactly as measured.
Absolute provenance paths in JSON may point to the originating workspace. The
verifier resolves bundled traces by basename if the package is moved.

Older development raw logs have distinct run prefixes and are retained for
auditability. `results/derived/run_index.csv` names only the active confirmatory
matrix, and every generator filters by the full locked configuration hash.

The publication traffic configuration remains blocked until genuine NPZ files,
source URLs, licenses, checksums, and a missing-value policy are supplied.
`python scripts/check_traffic_data.py` exits with status 2 and never downloads
or generates a substitute. The chronological traffic adapter exists, but no
traffic experiment is represented by the classification results.
