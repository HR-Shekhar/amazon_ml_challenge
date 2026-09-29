"""Pick the decision threshold that maximises held-out macro F_0.5.

0.5 is not a candidate we privilege: every cutoff on the grid is scored the same way,
and ties go to the higher cutoff (fewer false merges).
"""

from __future__ import annotations

import numpy as np

from scoring import entity_f05


def tune_threshold(
    s1_index: np.ndarray,
    s23_index: np.ndarray,
    proba: np.ndarray,
    truth: dict[int, np.ndarray],
    valid_entities: np.ndarray,
    grid: int = 49,
) -> tuple[float, float]:
    """Return ``(threshold, held_out_macro_f05)``."""
    valid_entities = np.asarray(valid_entities, dtype=np.int64)
    truth_sets = {
        int(entity): set(np.asarray(truth.get(int(entity), np.empty(0, np.int64))).tolist())
        for entity in valid_entities
    }
    groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(s1_index):
        hi = int(valid_entities.max()) if len(valid_entities) else 0
        hi = max(hi, int(s1_index.max()))
        valid_flag = np.zeros(hi + 1, dtype=bool)
        valid_flag[valid_entities] = True
        mask = valid_flag[s1_index]
        if mask.any():
            order = np.argsort(s1_index[mask], kind="mergesort")
            grouped_s1 = s1_index[mask][order]
            grouped_s23 = s23_index[mask][order]
            grouped_p = proba[mask][order]
            starts = np.flatnonzero(np.diff(grouped_s1, prepend=grouped_s1[0] - 1))
            bounds = list(starts) + [len(grouped_s1)]
            for a, b in zip(bounds, bounds[1:]):
                groups[int(grouped_s1[a])] = (grouped_s23[a:b], grouped_p[a:b])

    fixed = 0.0
    for entity, truth_set in truth_sets.items():
        if entity not in groups:
            fixed += 1.0 if not truth_set else 0.0

    if not groups:
        score = fixed / float(len(valid_entities)) if len(valid_entities) else 0.0
        print(f"no held-out candidate pairs to score; F_0.5 of empty predictions = {score:.4f}", flush=True)
        return 0.99, score

    thresholds = np.linspace(0.02, 0.98, grid)
    best_t = float(thresholds[0])
    best_f = -1.0
    for threshold in thresholds:
        total = fixed
        cutoff = float(threshold)
        for entity, (linked, scores) in groups.items():
            chosen = set(linked[scores >= cutoff].tolist())
            total += entity_f05(chosen, truth_sets[entity])
        score = total / float(len(valid_entities))
        if score > best_f + 1e-6 or (abs(score - best_f) <= 1e-6 and cutoff > best_t):
            best_f = score
            best_t = cutoff
    print(f"held-out macro F_0.5 = {best_f:.4f} at threshold {best_t:.2f}", flush=True)
    return best_t, best_f
