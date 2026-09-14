# CORA-FL reproducibility package

CORA-FL is a serverless learning mechanism for correlated regional outages. It
combines stochastic gradient tracking with a four-round safety scaffold and
adaptive peer selection based on coverage debt, failure-domain diversity, and
observed link reliability.

This repository contains the frozen protocol, code, raw logs, traces, exact
plot data, eight figures, three tables, and an independent verifier for the
confirmatory benchmark. The measurements use three real scikit-learn bundled
classification datasets. They are algorithmic evidence, not smart-city traffic
or field-deployment evidence.

## Confirmatory evaluation

| Item | Frozen value |
|---|---|
| Datasets | Digits, Breast Cancer Wisconsin Diagnostic, Wine |
| Partitions | IID, Dirichlet non-IID, domain-correlated Dirichlet non-IID |
| Clients / two-client failure domains | 8 / 4 |
| Scenarios | nominal, correlated one-domain outage |
| Outage links | 5% base erasure, 60% on one surviving domain pair |
| Methods | FedAvg reference, five decentralized peers, CORA-FL |
| Main repetitions | 30 locked seeds |
| Main runs | 3,780 |
| Pilot | 630 validation-only trials, 63 locked selections |

All decentralized methods share the same gradient-tracking update,
deterministic minibatches, initialization, partitions, and frozen network
traces. Only topology selection changes. FedAvg is a one-local-step centralized
reference. The primary comparator is Random-Gossip-GT in the
domain-correlated non-IID outage condition.

## Measured decision

All 3,780 main runs and 630 pilot trials succeeded. In the primary regime,
CORA-FL achieved 0.971192 mean final accuracy versus 0.970905 for
Random-Gossip-GT. The paired difference was +0.000288 with two-sided 95%
interval [-0.001404, +0.001979]. Strict accuracy superiority was therefore not
established. The one-sided lower bound was -0.001118, which passed the frozen
-0.005 non-inferiority threshold.

CORA-FL passed the 105/105 mapping certificate while Random-Gossip-GT has no
deterministic schedule certificate. The resulting qualified robustness claim
passed. Exponential-GT also passed 105/105 and MATCHA-GT had the highest final
accuracy point estimate at 0.972804. Both facts are retained as claim limits.
See `results/derived/RESULTS.md` and `docs/CLAIM_GUIDANCE.md` for exact wording.

## Run and audit

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_experiments.py --stage pilot
.venv/bin/python scripts/run_experiments.py --stage main --force
.venv/bin/python scripts/make_confirmatory_artifacts.py
.venv/bin/python scripts/verify_confirmatory.py
```

To audit supplied outputs without retraining, run only the last command. The
main runner rejects pilot selections from a different config or implementation.
The pilot blocks the main stage if any selected learning rate lies on either
edge of the search grid.

The eight plots cover accuracy by round and byte, consensus error,
cross-dataset performance, all three partition regimes, absolute and paired
outage utility, communication-accuracy trade-offs, and worst-domain accuracy.
The three tables report predictive performance, system cost plus certificates,
and paired inference. Every plotted series has a companion CSV.

## Claim boundaries

- The primary statistical claim is non-inferiority within 0.005 absolute final
  accuracy against Random-Gossip-GT, using one three-dataset macro-average per
  seed. Strict superiority has a separate confidence-interval rule.
- The structural audit exhausts all 105 assignments of eight clients to four
  unlabeled two-client domains. It reports that Exponential-GT shares the tested
  mapping-agnostic connectivity property rather than hiding that result.
- The certificate covers one declared domain removed over a four-round window
  when scheduled safety links succeed. Independent link erasure is evaluated
  empirically.
- Adaptive label-distribution sketches are counted in communication cost but
  are not differentially private. Byzantine robustness is not implemented.
- Split and Snake/serpentine learning alter model placement and message types.
  They are treated in related work, not renamed as topology controls.
- Real traffic tensors are absent. The traffic stage is blocked and has no
  synthetic fallback.

See `docs/CONFIRMATORY_PROTOCOL.md`, `docs/DEVELOPMENT_LOG.md`,
`docs/METHODOLOGY.md`, `docs/BASELINES.md`, and `docs/REPRODUCIBILITY.md`.
