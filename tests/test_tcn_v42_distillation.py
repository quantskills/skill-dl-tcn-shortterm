from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skill_dl_tcn_shortterm import ContractError
from skill_dl_tcn_shortterm.consensus_distillation import (
    blend_training_targets,
    build_fold_consensus_rank_targets,
)


def _teacher_predictions() -> pd.DataFrame:
    rows = []
    for seed, offset in ((7, 0.0), (17, 0.1), (27, -0.1)):
        for position, date in ((0, "2025-01-02"), (1, "2025-01-02"), (2, "2025-01-03"), (3, "2025-01-03")):
            for horizon in (1, 2, 3, 5):
                rows.append(
                    {
                        "seed": seed,
                        "sample_position": position,
                        "signal_date": date,
                        "horizon": horizon,
                        "score": float(position + offset),
                    }
                )
    return pd.DataFrame(rows)


def test_consensus_teacher_is_fold_scoped_ranked_and_complete() -> None:
    targets = build_fold_consensus_rank_targets(
        _teacher_predictions(),
        sample_count=6,
        train_positions=np.asarray([0, 1, 2, 3]),
        expected_seeds=(7, 17, 27),
    )

    assert targets.shape == (6, 4)
    np.testing.assert_allclose(targets[0], -1.0)
    np.testing.assert_allclose(targets[1], 1.0)
    np.testing.assert_allclose(targets[2], -1.0)
    np.testing.assert_allclose(targets[3], 1.0)
    assert np.isnan(targets[4:]).all()

    leaked = pd.concat(
        [
            _teacher_predictions(),
            _teacher_predictions()
            .loc[lambda frame: frame["sample_position"].eq(0)]
            .assign(sample_position=4, signal_date="2025-01-04"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ContractError, match="train-position coverage"):
        build_fold_consensus_rank_targets(
            leaked,
            sample_count=6,
            train_positions=np.asarray([0, 1, 2, 3]),
            expected_seeds=(7, 17, 27),
        )


def test_target_blend_changes_only_valid_train_cells() -> None:
    true_targets = np.arange(24, dtype="float32").reshape(6, 4) / 24
    masks = np.ones((6, 4), dtype="bool")
    masks[1, 2] = False
    teacher = np.full((6, 4), np.nan, dtype="float32")
    teacher[:4] = -0.5

    blended = blend_training_targets(
        true_targets,
        masks,
        teacher,
        train_positions=np.asarray([0, 1, 2, 3]),
        teacher_weight=0.25,
    )

    expected = 0.75 * true_targets[:4] + 0.25 * -0.5
    expected[1, 2] = true_targets[1, 2]
    np.testing.assert_allclose(blended[:4], expected)
    np.testing.assert_array_equal(blended[4:], true_targets[4:])
