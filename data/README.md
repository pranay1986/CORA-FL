# Data policy

Benchmark data come from the pinned scikit-learn package. Generated manifests
record split sizes and exact-array fingerprints. Real traffic files are intentionally absent.
Supply attributable checksum-locked NPZ tensors and update every `REQUIRED`
field in `configs/publication_traffic.yaml` before running the traffic gate.
Synthetic substitution is forbidden.
