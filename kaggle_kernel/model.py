"""LightGBM matcher.

LightGBM is Apache-2.0 licensed and is a few thousand parameters here, far under the
8B-parameter cap. The booster is trained on candidate pairs only; the decision
threshold is chosen later by ``threshold_tuning`` against held-out macro F_0.5.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from features import FEATURE_NAMES

_PAIR_BIG = np.int64(10_000_000_000)


def make_labels(s1_index: np.ndarray, s23_index: np.ndarray, truth: dict[int, np.ndarray]) -> np.ndarray:
    """1 when the candidate pair is a true match, else 0."""
    n = len(s1_index)
    labels = np.zeros(n, np.int8)
    if n == 0:
        return labels
    true_keys: list[np.ndarray] = []
    for row, linked in truth.items():
        if linked is None or len(linked) == 0:
            continue
        true_keys.append(np.int64(row) * _PAIR_BIG + np.asarray(linked, np.int64))
    if not true_keys:
        return labels
    keys = np.concatenate(true_keys)
    keys.sort()
    query = s1_index.astype(np.int64) * _PAIR_BIG + s23_index.astype(np.int64)
    idx = np.searchsorted(keys, query)
    in_range = idx < len(keys)
    labels[in_range] = (keys[idx[in_range]] == query[in_range]).astype(np.int8)
    return labels


def _params(n_rows: int, seed: int) -> dict:
    small = n_rows < 50_000
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.08 if small else 0.05,
        "num_leaves": 31 if small else 63,
        "min_child_samples": 5 if small else 40,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": seed,
        "num_threads": -1,
        "deterministic": True,
        "force_row_wise": True,
    }


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    valid_features: np.ndarray | None = None,
    valid_labels: np.ndarray | None = None,
    seed: int = 0,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    """Train a binary booster. Early stopping is used only when a two-class valid set is given."""
    if len(labels) == 0 or int(labels.sum()) == 0 or int(labels.sum()) == len(labels):
        raise SystemExit(
            "The training pairs contain only one class, so LightGBM has nothing to separate. "
            "On a subset, raise --max-rows so more true matches are included."
        )
    train_set = lgb.Dataset(features, label=labels, feature_name=FEATURE_NAMES)
    rounds = num_boost_round if num_boost_round is not None else (200 if len(labels) < 50_000 else 500)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=50)]
    use_valid = (
        valid_features is not None
        and valid_labels is not None
        and len(valid_labels) > 0
        and len(np.unique(valid_labels)) > 1
        and num_boost_round is None
    )
    if use_valid:
        valid_sets.append(lgb.Dataset(valid_features, label=valid_labels, reference=train_set, feature_name=FEATURE_NAMES))
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(40, verbose=False))
    booster = lgb.train(
        _params(len(labels), seed),
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def predict_proba(booster: lgb.Booster, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, np.float32)
    return booster.predict(features).astype(np.float32, copy=False)


def save_model(booster: lgb.Booster, threshold: float, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / "matcher.txt"))
    (directory / "threshold.txt").write_text(f"{threshold:.6f}\n", encoding="utf-8")
    (directory / "features.txt").write_text("\n".join(FEATURE_NAMES) + "\n", encoding="utf-8")


def load_model(directory: str | Path) -> tuple[lgb.Booster, float]:
    directory = Path(directory)
    booster = lgb.Booster(model_file=str(directory / "matcher.txt"))
    threshold = float((directory / "threshold.txt").read_text(encoding="utf-8").strip())
    return booster, threshold


def print_importance(booster: lgb.Booster, limit: int = 15) -> None:
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:limit]
    print("feature importance (gain):", flush=True)
    for idx in order:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else str(idx)
        print(f"  {name}: {gain[idx]:.1f}", flush=True)
