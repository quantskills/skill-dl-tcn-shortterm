# TCN date/horizon equal SmoothL1 v32 results

## Decision

`stop_date_horizon_equal_no_gain_v32`

The v32 experiment completed on the real five-year Pandadata minute-bar cache with three seeds and five ordinary-validation folds. Integrity and loss-mechanism gates passed. Predictive-effect and candidate/control throughput gates failed. Sealed test data was neither accessed nor authorized.

## Single tested variable

- control: grouped SmoothL1 reduced over every valid stock/horizon label (`label_mean`);
- candidate: SmoothL1 first averaged inside each `(signal_date,horizon)` group and then equally averaged across groups (`date_horizon_mean`).

Both arms used the same frozen TCN parent checkpoint, model seed, fixed date order, causal chomp, WeightNorm, 480-bar input, receptive field 511, 88 trainable shape parameters, optimizer, learning rate, batch cap, early-stopping rule, data and folds.

## Integrity and mechanism evidence

- The v32 control reproduced all 15 v31 fixed-once control RankIC values exactly; maximum absolute replay error was `0.0`.
- Frozen-parent state drift and parent prediction error were both `0.0`.
- The minimal imbalanced fixture changed from label-mean `0.375` to date/horizon-mean `0.75`, proving the reduction contract.
- Real batches had a median of about `23.92` non-empty date/horizon groups and `460.53` valid labels, or `19.22` labels per group.
- Both arms used exactly one fixed date-order fingerprint per unit.

The mechanism was therefore active. However, the PIT top20 cross-sections were already nearly uniform in group size, so label weighting and group weighting were much closer than the hypothesis required.

## Predictive results

| seed | control mean RankIC | candidate mean RankIC | paired delta |
|---:|---:|---:|---:|
| 7 | 0.097624 | 0.097632 | +0.000008 |
| 17 | 0.100781 | 0.100739 | -0.000042 |
| 27 | 0.100968 | 0.101004 | +0.000036 |
| all | 0.099791 | 0.099792 | +0.000001 |

The candidate improved five units, degraded five units, and was unchanged in five units. The paired daily block bootstrap did not resolve an effect:

| scope | paired mean delta | 95% CI |
|---|---:|---:|
| all | +0.000001 | [-0.000139, +0.000148] |
| seed 7 | +0.000008 | [-0.000412, +0.000443] |
| seed 17 | -0.000042 | [-0.000108, +0.000019] |
| seed 27 | +0.000036 | [-0.000037, +0.000107] |

Trained-effect units decreased from 11 to 10. The equal-group loss did not solve the early parent-checkpoint selection pattern.

## Gradient and speed evidence

- Median epoch gradient-norm CV changed from `0.3687` to `0.3658`, a small reduction without a corresponding RankIC gain.
- Candidate/control median throughput was `0.884x`, below the pre-registered `0.90x` gate because the group-wise Python reduction adds work.
- Receipt-bound TCN/LSTM model-step and end-to-end ratios remained `6.530x` and `5.904x`, above the 3x requirement.

Because prediction did not improve, optimizing the failed reduction's implementation would not change the v32 adoption decision.

## Falsified hypothesis

Implicit overweighting of larger dates or horizons is not the main source of the current TCN prediction plateau. The real top20 groups are too uniform for equal-group weighting to materially change the training signal. `date_horizon_mean` must not replace `label_mean` as the default.

## Updated bottleneck

Speed remains solved for the current receipt-bound CPU environment. Date-order randomness and date/horizon weighting have now both been falsified as primary causes. The remaining evidence points to the frozen 88-parameter shape branch itself: it often cannot produce a validation-improving update beyond the parent checkpoint. The next bounded investigation should inspect the shape branch's feature rank, activation scale, per-horizon Jacobian rank and update-to-output transfer before selecting another training intervention. It should not scan another loss or batching parameter without that representation evidence.

## Evidence

- Prompt: `docs/prompts/tcn-date-horizon-equal-smooth-l1-v32.md`
- Config: `config/pandadata-tcn-date-horizon-equal-smooth-l1-v32.example.json`
- Artifact: `artifacts/tcn-date-horizon-equal-smooth-l1-v32`
- Receipt: `3a2d7866b116eb4a6265dc4f447faa83db1c1fd8b35cb4bd6f6ec4bbb831456d`
