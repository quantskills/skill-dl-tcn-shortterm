# TCN grouped batch order stability v31 results

## Decision

`stop_epoch_seeded_no_gain_v31`

The experiment completed on the real five-year Pandadata minute-bar feature cache with three seeds and five ordinary-validation folds. Integrity, mechanism, and speed gates passed. The predictive-effect gate failed. Sealed test data was neither accessed nor authorized.

## What changed

The only behavior change was the date-batch order across epochs:

- control: `fixed_once`, reproducing the v30 behavior where every epoch sees the same deterministic date order;
- candidate: `epoch_seeded`, where every epoch has a different deterministic order and every `(seed, fold, epoch)` is replayable.

The TCN architecture, causal chomp path, WeightNorm, receptive field 511, 480-bar input, grouped SmoothL1, batch cap 128, frozen parent checkpoints, 88 trainable shape parameters, optimizer, learning rate, early stopping, data, and folds were unchanged.

## Integrity and reproduction

- The control reproduced all 15 v30 grouped-control RankIC values exactly; maximum absolute replay error was `0.0`.
- Frozen-parent state drift was `0.0` and parent prediction error was `0.0`.
- Every control unit had one order fingerprint across epochs.
- Every candidate unit had as many unique order fingerprints as completed epochs.
- Candidate/control throughput ratio was `0.9912`, so instrumentation did not materially slow training.

## Predictive results

| seed | control mean RankIC | candidate mean RankIC | paired delta |
|---:|---:|---:|---:|
| 7 | 0.097624 | 0.097511 | -0.000113 |
| 17 | 0.100781 | 0.100733 | -0.000048 |
| 27 | 0.100968 | 0.100960 | -0.000008 |
| all | 0.099791 | 0.099735 | -0.000056 |

Ten of 15 fold/seed units were numerically unchanged. Twelve of 15 candidate units selected epoch 0 or 1, so changing the order from epoch 2 onward could not improve most selected checkpoints.

The paired daily RankIC block bootstrap was also inconclusive:

| scope | paired mean delta | 95% CI |
|---|---:|---:|
| all | -0.000056 | [-0.000247, +0.000135] |
| seed 7 | -0.000113 | [-0.000552, +0.000343] |
| seed 17 | -0.000048 | [-0.000221, +0.000127] |
| seed 27 | -0.000008 | [-0.000315, +0.000308] |

## Gradient and speed diagnosis

- Median epoch gradient-norm CV changed from `0.3687` to `0.3756`; epoch reseeding did not reduce gradient dispersion.
- The observed change in physical-batch CV had only weak association with the RankIC delta (`r=-0.196` on 15 units).
- Model-step TCN/LSTM ratio was `10.040x` and end-to-end ratio was `8.668x` in this instrumented run. Both are above the frozen 3x requirement, though these ratios should be interpreted within the receipt-bound local environment rather than as universal claims.

## Falsified hypothesis

The v30 seed instability was not caused by reusing the same deterministic date order every epoch. Epoch-specific reshuffling worked as designed but did not improve RankIC, did not rescue seed 27, and slightly increased median gradient-norm CV. `epoch_seeded` must therefore not replace `fixed_once` as the project default.

## Next bounded direction

The next unknown is the shape branch's weak or harmful early learning signal, not TCN speed or epoch-order randomness. A subsequent experiment should first explain why 12/15 candidate units select the parent checkpoint at epoch 0 or 1. If a single intervention is tested, the most direct one is an equal-date/equal-horizon reduction of the existing SmoothL1 terms inside each date-grouped batch. That would keep TCN, targets, causal path, optimizer, and SmoothL1 itself fixed while aligning each optimizer step with the date/horizon units used by RankIC. It must be compared against `fixed_once` and must retain the same daily block-bootstrap and 3x speed gates.

## Evidence

- Prompt: `docs/prompts/tcn-grouped-batch-order-stability-v31.md`
- Config: `config/pandadata-tcn-grouped-batch-order-stability-v31.example.json`
- Artifact: `artifacts/tcn-grouped-batch-order-stability-v31`
- Receipt: `419c3caf6654d74662bbc39fc498e82da1e40ebd297c9b6d080c879f9f704a04`
