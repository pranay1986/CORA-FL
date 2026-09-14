# Development and rejected-variant log

This file separates inspected development evidence from the untouched
confirmatory experiment. Development results may explain design choices but are
not used as confirmatory observations.

## Initial benchmark

The initial 420-run matrix used three datasets, IID and ordinary Dirichlet
non-IID partitions, two scenarios, seven methods, and seeds 101, 202, 303, 404,
and 505. Under the non-IID one-domain outage, the measured macro final
accuracies were 0.963265 for CORA-FL, 0.962183 for Random-Gossip-GT, 0.961228 for
FedAvg, 0.961131 for Static-RR-GT, 0.959376 for Exponential-GT, 0.955088 for
Ring-GT, and 0.941881 for Similarity-Adaptive-GT. CORA-FL minus
Random-Gossip-GT was +0.001082 with a 95% paired interval of [-0.008245,
+0.010409], so accuracy superiority was not established. Positive degradation
AUC was also not best for CORA-FL: FedAvg measured 0.086014,
Random-Gossip-GT 0.172173, and CORA-FL 0.220614.

## Design changes retained

- MATCHA-GT replaced the less informative static random-overlay control in the
  confirmatory baseline set.
- Domain-correlated non-IID was added as an explicit stress condition. It
  preserves the generated shards and groups similar shards using train-only
  histograms.
- A persistent 60% erasure rate on one surviving domain pair was added to the
  regional-outage scenario, while the base attempted-link erasure remains 5%.
- Accuracy AUC and degradation-from-nominal AUC were separated because they
  answer different questions.
- The confirmatory analysis uses one three-dataset macro-average per seed and
  30 new seeds.

## Rejected CORA-FL refinements

- Budget-preserving vacancy repair in safety phases was evaluated in 480 pilot
  and 480 main development runs. Relative to the original mechanism it changed
  macro final accuracy by -0.001637 and positive degradation AUC by +0.017154,
  where lower degradation AUC is better. It was rejected.
- Updating the reliability estimator during every phase, including fixed safety
  phases, reduced development performance and was rejected.
- A rolling component-connectivity scoring term reduced consensus error but
  reduced predictive utility and was rejected.

The final confirmatory `cora_fl` implementation therefore returns to the
original four-round safety scaffold with coverage-debt, cross-domain, and
observed-reliability scoring in adaptive phases. `cora_fl_v1` remains only as a
development-log alias and is excluded from `configs/confirmatory.yaml`.

No rejected variant is presented as a favorable ablation, and no confirmatory
seed, split, method, metric, or threshold is selected from its outcome.
