# Baseline scope

| Method | Implemented control | Strength | Limitation |
|---|---|---|---|
| FedAvg | One centralized active-client gradient average | Strong connected reference | Coordinator and one logical aggregation point |
| Ring-GT | Alternating natural-ring matchings | Sparse and predictable | Slow and domain-unaware mixing |
| Random-Gossip-GT | Fresh seeded matching over active clients | Failure-aware randomized mixing | No deterministic window certificate |
| Exponential-GT | Rotating XOR partner bit | Short paths and finite-time averaging structure | Power-of-two client count and no data-aware adaptation |
| MATCHA-GT | One sampled factor of a complete-graph one-factorization | Communication-budget-matched stochastic topology | Symmetric special case, not MATCHA's full topology optimization |
| Similarity-Adaptive-GT | Match similar train-only class histograms | Data-aware diagnostic | Can reinforce silos and exposes sketches |
| CORA-FL | Certified scaffold plus adaptive coverage, domain, and reliability phases | Scoped connectivity with data and failure awareness | Metadata overhead and a narrow certificate |

Every decentralized control uses the same gradient-tracking optimizer, model,
batch stream, initialization, partition, trace, and accounting code as CORA-FL.
This isolates topology selection. Static-RR-GT appears only in the inspected
development archive and is excluded from the confirmatory matrix after MATCHA-GT
was added as a closer communication-budget comparator.

These controls are transparent mechanism-level implementations, not claims of
bit-for-bit reproduction of every named paper. In particular, symmetric edge
costs and a complete base graph make the MATCHA activation probabilities equal,
so the control independently samples one matching per round. The full MATCHA
optimization matters for heterogeneous edge costs and base graphs.

Split learning exchanges intermediate activations and gradients across model
partitions. Snake learning sequentially trains assigned layer blocks along a
serpentine route. A direct quantitative comparison requires a partitionable
deep model and a different activation, gradient, latency, and memory ledger.
Ring-GT is therefore not relabeled as either method.

Primary mechanism references include [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[randomized gossip](https://doi.org/10.1109/TIT.2006.874516),
[DIGing](https://arxiv.org/abs/1607.03218),
[D-PSGD](https://arxiv.org/abs/1705.09056),
[MATCHA](https://arxiv.org/abs/1905.09435),
[one-peer exponential graphs](https://arxiv.org/abs/2110.13363),
[SplitNN](https://arxiv.org/abs/1812.00564), and
[Snake Learning](https://arxiv.org/abs/2405.03372).
