from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from skill_dl_tcn_shortterm.baselines import (
    _block_bootstrap_means,
    build_risk_exposure_report,
    run_statistical_baselines,
)


def test_statistical_baselines_share_data_and_report_rankic() -> None:
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    instruments = ["A", "B", "C", "D"]
    index_rows = []
    label_rows = []
    feature_rows = []
    position = 0
    targets = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
    for date in dates:
        for instrument, target in zip(instruments, targets, strict=True):
            sample_id = f"{date}-{instrument}"
            index_rows.append(
                {
                    "sample_position": position,
                    "sample_id": sample_id,
                    "instrument_id": instrument,
                    "signal_date": date,
                }
            )
            feature_rows.append(np.full((2, 3), target, dtype="float32"))
            for horizon in [1, 2, 3, 5]:
                label_rows.append(
                    {
                        "sample_id": sample_id,
                        "instrument_id": instrument,
                        "signal_date": date,
                        "horizon": horizon,
                        "rank_target": target,
                        "valid": True,
                    }
                )
            position += 1
    window_index = pd.DataFrame(index_rows)
    labels = pd.DataFrame(label_rows)
    features = np.stack(feature_rows)
    split_manifest = window_index[
        ["sample_id", "sample_position", "instrument_id", "signal_date"]
    ].copy()
    split_manifest["fold"] = 0
    split_manifest["stage"] = [
        "train" if date <= "2024-01-04" else "validation"
        for date in split_manifest["signal_date"]
    ]
    split_manifest["sealed"] = False

    result = run_statistical_baselines(
        features,
        window_index,
        labels,
        split_manifest,
        seed=7,
    )

    assert set(result.predictions["model"]) == {"constant-zero", "ridge", "lightgbm"}
    assert set(result.predictions["stage"]) == {"validation"}
    assert set(result.predictions["horizon"]) == {1, 2, 3, 5}
    assert result.predictions.groupby(["model", "horizon"]).size().eq(8).all()

    metrics = result.metrics.set_index(["model", "horizon"])
    assert float(cast(Any, metrics.loc[("ridge", 1), "rankic"])) > 0.99
    assert float(cast(Any, metrics.loc[("lightgbm", 1), "rankic"])) > 0.99
    assert np.isnan(float(cast(Any, metrics.loc[("constant-zero", 1), "rankic"])))
    assert (
        metrics["rankic_ci_low"]
        .notna()
        .loc[[index for index in metrics.index if index[0] == "ridge"]]
        .all()
    )
    assert metrics.loc[("ridge", 1), "coverage"] == 1.0
    assert float(cast(Any, metrics.loc[("ridge", 1), "quantile_monotonicity"])) > 0.99
    assert metrics.loc[("ridge", 1), "direction_accuracy"] == 1.0
    assert metrics.loc[("ridge", 1), "direction_accuracy_role"] == "auxiliary"
    assert metrics.loc[("ridge", 1), "comparison_baseline"] in {"ridge", "lightgbm"}
    assert np.isfinite(
        float(cast(Any, metrics.loc[("ridge", 1), "paired_delta_rankic"]))
    )
    assert {"rankic_1d", "rankic_decay_from_1d"}.issubset(result.metrics.columns)


def test_date_block_bootstrap_is_seeded_and_risk_exposures_are_pit_stratified() -> None:
    values = np.asarray([0.1, -0.2, 0.3, 0.4, -0.1], dtype="float64")
    first = _block_bootstrap_means(values, np.random.default_rng(17), draws=16)
    second = _block_bootstrap_means(values, np.random.default_rng(17), draws=16)
    np.testing.assert_array_equal(first, second)

    predictions = pd.DataFrame(
        [
            {
                "model": "ridge",
                "fold": 0,
                "horizon": 1,
                "instrument_id": instrument,
                "signal_date": signal_date,
                "score": score,
                "target": score,
            }
            for signal_date in ["2024-01-02", "2024-01-03"]
            for instrument, score in [("A", -1.0), ("B", 0.0), ("C", 1.0)]
        ]
    )
    universe = pd.DataFrame(
        [
            {
                "instrument_id": instrument,
                "signal_date": signal_date,
                "industry": industry,
                "market_cap": market_cap,
                "adv20": adv20,
                "market_state": "normal",
            }
            for signal_date in ["2024-01-02", "2024-01-03"]
            for instrument, industry, market_cap, adv20 in [
                ("A", "bank", 10.0, 100.0),
                ("B", "bank", 20.0, 200.0),
                ("C", "tech", 30.0, 300.0),
            ]
        ]
    )

    report = build_risk_exposure_report(predictions, universe)

    assert set(report["dimension"]) == {
        "industry",
        "size",
        "liquidity",
        "market_state",
    }
    assert report["scores_unchanged"].eq(True).all()
    assert int(report["sample_count"].sum()) == len(predictions) * 4
