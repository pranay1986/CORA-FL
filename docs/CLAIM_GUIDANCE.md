# Evidence-based claim guidance

## Claim supported by the locked experiment

The following wording matches the predeclared test and measured values:

> In the domain-correlated non-IID regional-outage benchmark, CORA-FL was
> non-inferior to Random-Gossip-GT in final accuracy within the predeclared
> 0.005 margin. Its mean paired difference was +0.000288, the two-sided 95%
> confidence interval was [-0.001404, +0.001979], and the one-sided lower bound
> was -0.001118. CORA-FL also provided a deterministic four-round one-domain
> connectivity certificate across all 105 two-client domain mappings, which the
> stochastic primary comparator does not provide. This supports a qualified
> robustness advantage without establishing strict accuracy superiority.

The communication comparison can be added descriptively. CORA-FL transmitted
2.797649 MiB on average in the primary regime, 1.42% less than
Random-Gossip-GT's 2.837931 MiB. CORA-FL used 2,192 messages versus 1,480 because
its adaptive phases carry control exchanges. Its final consensus error was
0.243443 versus 0.248406. The byte and consensus differences were not assigned
predeclared significance tests.

## Baseline-specific conclusions

- Ring-GT: CORA-FL had significantly higher outage accuracy AUC by 0.005583
  with 95% interval [0.002253, 0.008913] and Holm-adjusted p = 0.003674. Its
  positive degradation AUC was lower by 0.294197 with interval
  [-0.423683, -0.164711] and adjusted p = 0.000135. CORA-FL certified 105/105
  mappings versus Ring-GT's 2/105. Final accuracy did not differ significantly.
- Similarity-Adaptive-GT: CORA-FL was significantly better on final accuracy,
  outage accuracy AUC, worst-domain AUC, and positive degradation AUC after
  Holm adjustment.
- Random-Gossip-GT: the predeclared non-inferiority and qualified robustness
  claim passed. Strict accuracy superiority and secondary metric superiority
  did not pass.
- Exponential-GT: final accuracy and outage metrics were statistically
  indistinguishable under the locked tests. It shares the 105/105 deterministic
  certificate and has lower communication and consensus-error point estimates.
- MATCHA-GT: MATCHA-GT had the best primary final-accuracy point estimate,
  0.972804 versus CORA-FL's 0.971192. CORA-FL minus MATCHA-GT was -0.001611
  with interval [-0.004029, +0.000806]. Neither accuracy nor robustness
  superiority was established. CORA-FL has the deterministic certificate,
  while this stochastic MATCHA control does not.
- FedAvg: the centralized reference had similar final accuracy and lower
  degradation AUC. It is not a serverless competitor and does not establish a
  decentralized topology guarantee.

## Claims not supported

Do not state that CORA-FL:

- is universally or uniformly superior to all decentralized baselines
- has statistically superior final accuracy to Random-Gossip-GT, Exponential-GT,
  MATCHA-GT, or Ring-GT
- uniquely possesses the tested deterministic certificate
- is state of the art based on these three small classification datasets
- has proven Byzantine robustness, privacy, or convergence under arbitrary link
  erasures
- has been validated on traffic forecasting or a live smart-city deployment

Any manuscript should label the classifier results as controlled algorithmic
evidence and reserve the smart-city traffic setting for a future attributable
dataset and task-appropriate forecasting model.
