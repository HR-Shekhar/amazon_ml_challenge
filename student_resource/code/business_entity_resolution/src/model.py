"""LightGBM matcher.

LightGBM is Apache-2.0 licensed and far under the 8B-parameter cap. The booster scores
(Source 2/3 record, Source 1 candidate) pairs; the decision rule and threshold live in
``threshold_tuning``.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from features import FEATURE_NAMES


def _params(n_rows: int, seed: int, objective: str) -> dict:
    small = n_rows < 200_000
    if objective == "lambdarank":
        metric = "ndcg"
        extra = {"ndcg_eval_at": [1, 3], "lambdarank_truncation_level": 20}
    else:
        metric = "binary_logloss"
        extra = {}
    return {
        "objective": objective,
        "metric": metric,
        **extra,
        "learning_rate": 0.1 if small else 0.08,
        "num_leaves": 31 if small else 255,
        "min_child_samples": 5 if small else 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbose": -1,
        "seed": seed,
        "num_threads": -1,
        "force_row_wise": True,
    }


def _group_sizes(q: np.ndarray) -> np.ndarray:
    """Row counts of each contiguous query. ``q`` must be sorted."""
    if len(q) == 0:
        return np.empty(0, np.int32)
    change = np.flatnonzero(np.r_[True, q[1:] != q[:-1], True])
    return np.diff(change).astype(np.int32)


def disagreement_weights(q: np.ndarray, labels: np.ndarray, rank: np.ndarray, heavy: float = 12.0) -> np.ndarray:
    """Upweight queries whose true Source 1 row is in the list but not in first place.

    ``q`` must be sorted. ``rank`` is 0 for the blocker's first candidate. Easy queries
    stay at weight 1 so the booster still sees ordinary pairs.
    """
    weights = np.ones(len(q), np.float32)
    missed = (labels == 1) & (rank > 0)
    if not missed.any():
        return weights
    if len(q) > 1 and np.any(q[1:] < q[:-1]):
        raise RuntimeError("disagreement weights expect queries in sorted order")
    queries = np.unique(q[missed])
    left = np.searchsorted(q, queries, side="left")
    right = np.searchsorted(q, queries, side="right")
    for start, stop in zip(left.tolist(), right.tolist()):
        weights[start:stop] = np.float32(heavy)
    return weights


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    valid_features: np.ndarray | None = None,
    valid_labels: np.ndarray | None = None,
    seed: int = 0,
    max_rounds: int = 1500,
    objective: str = "binary",
    train_group: np.ndarray | None = None,
    valid_group: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    weights: np.ndarray | None = None,
    valid_weights: np.ndarray | None = None,
) -> lgb.Booster:
    """Train a booster, early-stopped on the valid set when it has both classes.

    ``objective='lambdarank'`` trains the true Source 1 record to outrank the other
    candidates of the same Source 2/3 record. ``train_group`` / ``valid_group`` are
    the per-query row counts, and the rows must already be grouped by query.
    """
    if len(labels) == 0 or int(labels.sum()) == 0 or int(labels.sum()) == len(labels):
        raise SystemExit(
            "The training pairs contain only one class, so LightGBM has nothing to separate. "
            "On a subset, raise --max-rows so more true matches are included."
        )
    names = FEATURE_NAMES if feature_names is None else feature_names
    if len(names) != features.shape[1]:
        raise SystemExit(f"feature name count {len(names)} != matrix width {features.shape[1]}")
    train_set = lgb.Dataset(
        features,
        label=labels,
        weight=weights,
        group=None if train_group is None else train_group,
        feature_name=names,
        free_raw_data=True,
    )
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=100)]
    if valid_features is not None and valid_labels is not None and len(np.unique(valid_labels)) > 1:
        valid_sets.append(
            lgb.Dataset(
                valid_features,
                label=valid_labels,
                weight=valid_weights,
                group=None if valid_group is None else valid_group,
                reference=train_set,
                feature_name=names,
            )
        )
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(50, verbose=True))
    return lgb.train(
        _params(len(labels), seed, objective),
        train_set,
        num_boost_round=max_rounds if len(labels) >= 200_000 else 300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def predict_proba(
    boosters: list[lgb.Booster],
    features: np.ndarray,
    chunk: int = 5_000_000,
    columns: np.ndarray | None = None,
) -> np.ndarray:
    """Mean score of the boosters. Binary training returns a probability; ranking returns a score."""
    out = np.zeros(len(features), np.float32)
    if len(features) == 0:
        return out
    for start in range(0, len(features), chunk):
        part = features[start : start + chunk]
        if columns is not None:
            part = np.ascontiguousarray(part[:, columns])
        out[start : start + chunk] = np.mean([b.predict(part) for b in boosters], axis=0)
    return out


def save_models(
    boosters: list[lgb.Booster],
    threshold: float,
    directory: str | Path,
    min_gap: float = 0.0,
    blind_boosters: list[lgb.Booster] | None = None,
    override_gap: float = -1.0,
    blind_margin: float = 0.0,
    blind_threshold: float = 0.5,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("matcher*.txt"):
        old.unlink()
    for i, booster in enumerate(boosters):
        booster.save_model(str(directory / f"matcher_{i}.txt"))
    for old in directory.glob("blind_*.txt"):
        old.unlink()
    for i, booster in enumerate(blind_boosters or []):
        booster.save_model(str(directory / f"blind_{i}.txt"))
    (directory / "threshold.txt").write_text(
        f"{threshold:.6f} {min_gap:.6f} {override_gap:.6f} {blind_margin:.6f} {blind_threshold:.6f}\n",
        encoding="utf-8",
    )
    (directory / "features.txt").write_text("\n".join(FEATURE_NAMES) + "\n", encoding="utf-8")


def load_models(directory: str | Path) -> tuple[list[lgb.Booster], float, float]:
    directory = Path(directory)
    boosters = [lgb.Booster(model_file=str(p)) for p in sorted(directory.glob("matcher_*.txt"))]
    parts = (directory / "threshold.txt").read_text(encoding="utf-8").split()
    threshold = float(parts[0])
    min_gap = float(parts[1]) if len(parts) > 1 else 0.0
    return boosters, threshold, min_gap


def print_importance(booster: lgb.Booster, limit: int = 20) -> None:
    gain = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    order = np.argsort(gain)[::-1][:limit]
    print("feature importance (gain):", flush=True)
    for idx in order:
        name = names[idx] if idx < len(names) else str(idx)
        print(f"  {name}: {gain[idx]:.1f}", flush=True)
