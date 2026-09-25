"""Macro-averaged F_0.5, matching the challenge definition.

Per Source 1 entity:

    F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

The reported score is the unweighted mean over every Source 1 entity. A singleton
(no true matches) scores 1.0 when the prediction is empty and 0.0 otherwise.
"""

from __future__ import annotations

import numpy as np


def f05(precision: float, recall: float) -> float:
    """F_beta with beta = 0.5. Both-zero is defined as 0 (no credit for an empty overlap)."""
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_f05(predicted: set[int], truth: set[int]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    return f05(tp / len(predicted), tp / len(truth))


def macro_f05(predicted: dict[int, set[int]], truth: dict[int, set[int]], entities: np.ndarray) -> float:
    """Mean per-entity F_0.5. Missing keys are empty sets."""
    if len(entities) == 0:
        return 0.0
    total = 0.0
    for entity in entities:
        key = int(entity)
        total += entity_f05(predicted.get(key, set()), set(truth.get(key, ())))
    return total / float(len(entities))


def self_check() -> None:
    """Fail loudly if the scorer drifts from the published example."""
    # Worked example from the problem statement: precision 2/3, recall 1 -> 0.714...
    expect = (1.25 * (2.0 / 3.0) * 1.0) / (0.25 * (2.0 / 3.0) + 1.0)
    got = f05(2.0 / 3.0, 1.0)
    if abs(got - expect) > 1e-12:
        raise AssertionError(f"F0.5 example mismatch: {got} != {expect}")
    example = macro_f05({0: {1, 2, 3}}, {0: {1, 3}}, np.array([0]))
    if abs(example - expect) > 1e-12:
        raise AssertionError(f"macro example mismatch: {example} != {expect}")
    # Correct singleton scores 1, a false match on a singleton scores 0.
    singleton = macro_f05({0: set(), 1: {9}}, {0: set(), 1: set()}, np.array([0, 1]))
    if abs(singleton - 0.5) > 1e-12:
        raise AssertionError(f"singleton scoring mismatch: {singleton}")
