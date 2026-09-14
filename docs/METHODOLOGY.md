# Methodology

## Common decentralized update

Each decentralized client stores parameters \(x_i^k\) and a gradient tracker
\(y_i^k\). With the symmetric Metropolis matrix \(W_k\) induced by successful
peer links, the simulator applies

$$
X^{k+1}=W_kX^k-\eta Y^k,
\qquad
Y^{k+1}=W_kY^k+G(X^{k+1},\xi_{k+1})-G(X^k,\xi_k).
$$

Unavailable clients freeze their parameter and tracker state. The evaluation
model is the arithmetic mean over currently active clients. Every method uses
float64 softmax regression, batch size 16, and L2 coefficient \(10^{-4}\).

## CORA-FL peer selection

CORA-FL interleaves two fixed safety matchings and two adaptive matching phases
in a four-round period. The fixed matchings are the two alternating matchings
of a ring whose order groups the two clients in each declared domain. For
normalized local training-label histogram \(h_i\), exponentially averaged
neighbor summary \(m_i^k\), uniform target \(u\), domain \(q_i\), and observed
pair reliability \(r_{ij}^k\), define

$$
d_i^k=[u-m_i^k]_+,
$$

$$
s_{ij}^k=2\left((d_i^k)^\top h_j+(d_j^k)^\top h_i\right)
+0.75\mathbf{1}[q_i\ne q_j]+0.25r_{ij}^k.
$$

Adaptive phases greedily construct a matching in descending score order with
deterministic client-ID tie breaking. Neighbor summaries use EMA rate 0.2 and
pair reliability uses EMA rate 0.1 after attempted adaptive contacts. Safety
phases are not repaired when an endpoint is absent. Development variants that
added repair or extra component scoring are disclosed in `DEVELOPMENT_LOG.md`.

The adaptive control record consists of a float32 class histogram, an int16
domain identifier, and two int64 identifiers per active client. These bytes and
two control messages per active client are charged in every adaptive round.
Each attempted peer exchange charges parameters and trackers in both directions,
including attempts erased by the trace.

## Data and factorial matrix

The confirmatory stratified 70/10/20 split is fixed by seed 4144280299 and
standardization is trained only on the training split. IID partitions divide
each split without replacement. Ordinary non-IID partitions use per-class
Dirichlet allocation with \(\alpha=0.3\) and at least five training examples per
client. Domain-correlated non-IID retains the same shards and pairs similar
training histograms into two-client failure domains.

The main matrix is 3 datasets × 3 partitions × 2 scenarios × 7 methods × 30
seeds = 3,780 runs. In the stress trace, one domain is absent in zero-indexed
update slots 35 through 64. Attempted links have 5% base erasure and links
between one deterministic surviving domain pair have 60% erasure. Every method
uses the identical partition and trace within a paired cell.

Learning rates are selected separately for each dataset, partition, and method
by minimum final validation loss after 60 nominal rounds. Ten candidates from
0.003 through 100 give 630 validation trials. Neither test values nor outage
results enter tuning.

## Metrics and inference

- held-out accuracy and loss every five rounds
- mean squared disagreement among active peer models
- minimum declared-domain test accuracy
- absolute accuracy and worst-domain accuracy AUC during the outage interval
- positive paired degradation AUC from outage onset through round 100
- maximum paired drop and recovery to 95% of paired nominal final accuracy
- payload bytes, control bytes, messages, successful edges, and wall time

Curves are unsmoothed. The primary inferential unit is one equally weighted
three-dataset macro-average per seed. With 30 fixed seeds, intervals use the
Student-t distribution. Secondary peer comparisons use Holm adjustment within
each metric. Full formulas and decision thresholds are frozen in
`CONFIRMATORY_PROTOCOL.md`.

## Certificate and convergence boundary

The union of CORA-FL's two safety matchings is a domain-ordered ring. Removing
any one contiguous two-client domain leaves a connected path, and every cyclic
four-round window contains both fixed matchings. The audit enumerates all 105
ways to partition eight clients into four unlabeled pairs, every failed domain,
and every cyclic window start. The same audit is applied to Ring-GT and
Exponential-GT. Stochastic schedules are labeled as lacking a deterministic
certificate rather than assigned a fictitious deterministic failure count.

This finite structural result is not by itself an optimization-convergence
proof. A paper theorem must separately state smoothness, gradient-noise,
heterogeneity, joint-mixing, and step-size assumptions. A deterministic theorem
under link erasure needs a bounded-erasure assumption, while the independent
erasures used here support empirical or probabilistic conclusions. The
100-round results cannot be described as asymptotic or Byzantine-robust.
