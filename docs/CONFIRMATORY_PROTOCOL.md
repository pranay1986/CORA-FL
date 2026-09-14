# CORA-FL confirmatory protocol

Status: frozen before reading any result from the split, pilot seed, or 30 main
seeds in `configs/confirmatory.yaml`. The configuration and implementation
hashes are recorded in `CONFIRMATORY_LOCK.json` before the pilot starts.

## Research question and decision rules

The confirmatory question is whether CORA-FL preserves predictive utility under
a correlated regional outage while adding a deterministic topology guarantee
that the strongest stochastic peer comparator does not provide.

The primary operating regime is domain-correlated Dirichlet non-IID data under
the one-domain outage. Random-Gossip-GT is the primary comparator because it was
the strongest decentralized peer in the inspected development benchmark. For
each of 30 locked seeds, final test accuracy is first averaged equally across
Digits, Breast Cancer, and Wine. If \(d_s\) is CORA-FL minus Random-Gossip-GT
for seed \(s\), the primary non-inferiority hypotheses are

$$
H_0: \mathbb{E}[d_s] \leq -0.005,
\qquad
H_1: \mathbb{E}[d_s] > -0.005.
$$

Non-inferiority passes only when the one-sided 95% Student-t lower confidence
bound exceeds -0.005 absolute accuracy. Strict accuracy superiority is a
separate, more demanding result and is reported only if the two-sided 95%
paired interval lies entirely above zero. It is not inferred from a favorable
point estimate.

The predeclared qualified robustness claim passes only if all three conditions
hold:

1. predictive non-inferiority to Random-Gossip-GT passes
2. exhaustive recomputation verifies CORA-FL's mapping-agnostic four-round
   one-domain connectivity certificate
3. Random-Gossip-GT has no corresponding deterministic schedule certificate.

This decision never licenses a statement of universal accuracy superiority.
The certificate status of every deterministic control, including
Exponential-GT, is reported even if it matches CORA-FL on this property.

## Frozen methods

All peer-to-peer methods use the same stochastic gradient-tracking update,
initialization, float64 softmax-regression model, deterministic minibatches,
partitions, and link traces. Only the peer-selection rule changes.

| Method | Frozen role |
|---|---|
| FedAvg | One-local-step centralized reference over active clients |
| Ring-GT | Alternating matchings whose two-round union is the client-ID ring |
| Random-Gossip-GT | Fresh seeded matching over active clients each round |
| Exponential-GT | Rotating XOR partner-bit matching |
| MATCHA-GT | One independently sampled factor from a complete-graph one-factorization per round |
| Similarity-Adaptive-GT | Greedy matching by train-only label-histogram similarity |
| CORA-FL | Certified safety scaffold interleaved with coverage, domain, and reliability-aware matching |

The MATCHA control uses the symmetric one-matching communication budget. It is
an explicit control derived from the MATCHA mechanism, not a claim of reproducing
every optimization or topology-specific setting from that paper. Split learning
and serpentine/Snake learning change model placement and message semantics, so
they remain qualitative related-work comparators rather than mislabeled topology
controls in this linear-model experiment.

## Frozen data, partitions, and optimization

The three real, bundled scikit-learn classification datasets are Digits, Breast
Cancer Wisconsin Diagnostic, and Wine. A new stratified 70/10/20 split uses seed
4144280299. Standardization is fitted on the training split only. Each exact
array, partition, trace, raw curve, and implementation is SHA-256 fingerprinted.

Eight clients are assigned to four two-client failure domains. The partition
conditions are:

- IID: each split is permuted and divided once without replacement
- Dirichlet non-IID: class allocations use \(\alpha=0.3\), shared across the
  train, validation, and test allocations, with at least five train examples
  per client
- domain-correlated non-IID: the same Dirichlet shards are retained, then greedily
  paired into failure domains by smallest L1 distance between training-only class
  histograms. Validation and test labels do not determine the pairing.

The common batch size is 16 and L2 coefficient is \(10^{-4}\). Learning rates
are selected independently for each dataset, partition, and method using only
final validation loss after 60 nominal rounds. The ten-point grid is
\(\{0.003,0.01,0.03,0.1,0.3,1,3,10,30,100\}\), giving 630 pilot trials and 63
selections. An edge selection at either grid boundary blocks the main run and
requires a documented grid extension and a new lock. Exact ties choose the
smaller learning rate. The test split is not evaluated in the pilot.

## Frozen failure process and run matrix

Nominal traces have no link erasure. In the outage scenario, one two-client
domain is unavailable in zero-indexed update slots 35 through 64. Every link
has 5% erasure probability, while links between one deterministic pair of
surviving domains have 60% erasure probability throughout the scenario. The
failed domain and degraded survivor pair are deterministic functions of the
trace seed. Link events are generated before training and the identical trace
is reused by all methods in a paired dataset-partition-scenario-seed cell.

The main factorial matrix is

$$
3\text{ datasets}\times3\text{ partitions}\times2\text{ scenarios}
\times7\text{ methods}\times30\text{ seeds}=3780\text{ runs}.
$$

Each run lasts 100 communication rounds and is evaluated every five rounds.
Failed or non-finite runs are reported. They are never silently discarded.
Reruns may repair an incidental execution failure only with the identical
config and implementation hashes.

## Frozen estimands and multiplicity

The primary estimand is final outage test accuracy. Secondary estimands are:

- normalized trapezoidal accuracy AUC during the plotted outage interval
- normalized worst-domain-accuracy AUC over the same interval
- unnormalized positive degradation AUC, integrating
  \(\max(a_{\mathrm{nominal}}-a_{\mathrm{outage}},0)\) from outage onset through
  round 100
- peak positive accuracy drop and rounds to recover 95% of paired nominal final
  accuracy
- consensus error, transmitted payload and control bytes, messages, attempted
  and successful edges, and measured wall time.

For inferential comparisons, datasets are macro-averaged within each seed, so
the sample size is 30 rather than 90 pseudo-replicates. Within each secondary
metric, CORA-FL comparisons against the five decentralized peers use Holm
adjustment. Dataset-level and partition-level results remain descriptive and
are reported without converting them into extra independent samples.

Exactly eight figures and three tables are generated from raw logs. Curves are
unsmoothed, every plotted series has a CSV source, and Student-t uncertainty
bands use the appropriate seed-level units.

## Scope and prohibited interpretations

These experiments measure small classification benchmarks, not traffic
forecasting or a deployed smart-city service. They do not demonstrate Byzantine
robustness, differential privacy, secure aggregation, arbitrary node failures,
or asymptotic convergence. Label histograms are metadata and their bytes are
counted, but they are not privacy protected. The finite topology certificate
assumes scheduled safety links succeed. Independent erasures are evaluated
empirically. Any paper claim must preserve these boundaries and report failed
decision rules alongside passed ones.
