"""Decision rule and threshold search.

Every Source 2/3 record belongs to at most one Source 1 entity. So each record is linked
to its single highest-probability candidate, and only when that probability clears the
threshold. The threshold is the one that maximises macro F_0.5 over all Source 1
entities (singletons included); ties go to the higher cutoff.
"""

from __future__ import annotations

import numpy as np


def best_with_margin(q: np.ndarray, c: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top candidate per query, its probability, and the gap to the runner-up.

    A query with only one candidate gets gap 1, so a gap rule does not reject it.
    """
    if len(q) == 0:
        empty_i = np.empty(0, np.int64)
        empty_f = np.empty(0, np.float32)
        return empty_i, empty_i, empty_f, empty_f
    order = np.lexsort((-proba, q))
    qs = q[order]
    ps = proba[order]
    first = np.r_[True, qs[1:] != qs[:-1]]
    pick_pos = np.flatnonzero(first)
    second_pos = pick_pos + 1
    has_next = second_pos < len(qs)
    same = np.zeros(len(pick_pos), dtype=bool)
    same[has_next] = qs[second_pos[has_next]] == qs[pick_pos[has_next]]
    second = np.zeros(len(pick_pos), np.float32)
    second[same] = ps[second_pos[same]]
    chosen = order[pick_pos]
    bp = ps[pick_pos].astype(np.float32)
    gap = np.where(same, bp - second, np.float32(1.0)).astype(np.float32)
    return q[chosen].astype(np.int64), c[chosen].astype(np.int64), bp, gap


def best_per_query(q: np.ndarray, c: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query, its top candidate and that candidate's probability."""
    bq, bc, bp, _gap = best_with_margin(q, c, proba)
    return bq, bc, bp


def macro_f05_links(
    link_c: np.ndarray,
    correct: np.ndarray,
    true_count: np.ndarray,
    rows: np.ndarray | None = None,
) -> float:
    """Macro F_0.5 over Source 1 rows, given the chosen links.

    ``link_c`` is the Source 1 row of each link, ``correct`` whether that link is a true
    match, ``true_count[i]`` the number of true matches of Source 1 row ``i``.
    ``rows`` restricts the average to those Source 1 entities.
    """
    n = len(true_count)
    if rows is not None:
        use = np.zeros(n, dtype=bool)
        use[rows] = True
        keep = use[link_c] if len(link_c) else np.empty(0, dtype=bool)
        link_c = link_c[keep]
        correct = correct[keep]
    pred = np.bincount(link_c, minlength=n).astype(np.float64)
    tp = np.bincount(link_c, weights=correct.astype(np.float64), minlength=n)
    truth = true_count.astype(np.float64)
    score = np.zeros(n, np.float64)
    both_empty = (pred == 0) & (truth == 0)
    score[both_empty] = 1.0
    ok = tp > 0
    p = np.divide(tp, pred, out=np.zeros(n), where=ok)
    r = np.divide(tp, truth, out=np.zeros(n), where=ok)
    denom = 0.25 * p + r
    score[ok] = (1.25 * p[ok] * r[ok]) / denom[ok]
    if rows is not None:
        return float(score[rows].mean()) if len(rows) else 0.0
    return float(score.mean()) if n else 0.0


def aligned_picks(
    q: np.ndarray,
    c: np.ndarray,
    proba: np.ndarray,
    blind: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Primary and rank-blind top picks, aligned on the same query order."""
    bq, bc, bp, gap = best_with_margin(q, c, proba)
    lq, lc, lp, lg = best_with_margin(q, c, blind)
    order_p = np.argsort(bq, kind="mergesort")
    order_b = np.argsort(lq, kind="mergesort")
    if len(bq) != len(lq) or not np.array_equal(bq[order_p], lq[order_b]):
        raise RuntimeError("primary and rank-blind picks cover different queries")
    return bq[order_p], bc[order_p], bp[order_p], gap[order_p], lc[order_b], lp[order_b], lg[order_b]


def apply_rule(
    bc: np.ndarray,
    bp: np.ndarray,
    gap: np.ndarray,
    lc: np.ndarray,
    lp: np.ndarray,
    lg: np.ndarray,
    threshold: float,
    min_gap: float,
    override_gap: float,
    blind_margin: float,
    blind_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose a candidate per query. A flip happens only on a close primary call.

    Returns ``(chosen source1, keep mask, flip mask)``.
    """
    flip = (override_gap > 0) & (gap < override_gap) & (lc != bc) & (lg >= blind_margin) & (lp >= blind_threshold)
    chosen = np.where(flip, lc, bc)
    keep = flip | ((~flip) & (bp >= threshold) & (gap >= min_gap))
    return chosen, keep, flip


def tune_threshold(
    q: np.ndarray,
    c: np.ndarray,
    proba: np.ndarray,
    owner: np.ndarray,
    true_count: np.ndarray,
    grid: int = 97,
) -> tuple[float, float, float]:
    """Return ``(threshold, min_gap, macro_f05)`` on out-of-fold probabilities.

    A link is kept when its probability clears the threshold and its gap to the
    runner-up clears ``min_gap``. Ties prefer the higher threshold, then the higher gap.
    """
    bq, bc, bp, gap = best_with_margin(q, c, proba)
    correct = owner[bq] == bc
    best_t, best_g, best_f = 0.5, 0.0, -1.0
    prob_scale = len(bp) > 0 and float(bp.min()) >= 0.0 and float(bp.max()) <= 1.0
    thresholds = np.linspace(0.02, 0.98, grid) if prob_scale else np.quantile(bp, np.linspace(0.02, 0.98, grid))
    observed = gap[(gap > 0) & (gap < 0.999)]
    gap_grid = (0.0, 0.03, 0.06, 0.10, 0.15) if prob_scale or len(observed) < 100 else (0.0,) + tuple(float(x) for x in np.quantile(observed, [0.25, 0.5, 0.75]))
    for g in gap_grid:
        wide = gap >= g
        for t in thresholds:
            sel = wide & (bp >= t)
            f = macro_f05_links(bc[sel], correct[sel], true_count)
            better = f > best_f + 1e-7
            tie = abs(f - best_f) <= 1e-7 and (t > best_t + 1e-9 or (abs(t - best_t) <= 1e-9 and g > best_g))
            if better or tie:
                best_t, best_g, best_f = float(t), float(g), f
    sel = (bp >= best_t) & (gap >= best_g)
    n_true = int((owner >= 0).sum())
    n_links = int(sel.sum())
    n_correct = int(correct[sel].sum())
    print(
        f"held-out macro F_0.5 = {best_f:.5f} at threshold {best_t:.3f} min_gap {best_g:.2f} "
        f"(links {n_links:,}, correct {n_correct:,}, true links {n_true:,}, "
        f"link precision {n_correct / max(1, n_links):.5f}, link recall {n_correct / max(1, n_true):.5f})",
        flush=True,
    )
    if prob_scale:
        for t in (0.85, 0.90, 0.95, 0.98):
            hi = bp >= t
            if not hi.any():
                continue
            print(
                f"  proba>={t:.2f}: links {int(hi.sum()):,} precision {correct[hi].mean():.5f} "
                f"recall {correct[hi].sum() / max(1, n_true):.5f}",
                flush=True,
            )
    return best_t, best_g, best_f


def tune_override(
    q: np.ndarray,
    c: np.ndarray,
    proba: np.ndarray,
    blind: np.ndarray,
    owner: np.ndarray,
    true_count: np.ndarray,
    threshold: float,
    min_gap: float,
    tune_rows: np.ndarray,
    eval_rows: np.ndarray,
) -> tuple[float, float, float, float]:
    """Search a rank-blind flip rule. Return ``(override_gap, blind_margin, blind_threshold, full_f05)``.

    The grid is fit on ``tune_rows`` only. It is kept when it also raises macro F_0.5 on
    ``eval_rows``, which the search never saw. Otherwise the flip is turned off and the
    primary rule stands. ``override_gap <= 0`` means off.
    """
    bq, bc, bp, gap, lc, lp, lg = aligned_picks(q, c, proba, blind)

    def _score(chosen: np.ndarray, keep: np.ndarray, rows: np.ndarray | None) -> float:
        correct = owner[bq] == chosen
        return macro_f05_links(chosen[keep], correct[keep], true_count, rows)

    base_chosen, base_keep, _base_flip = apply_rule(bc, bp, gap, lc, lp, lg, threshold, min_gap, -1.0, 0.0, 0.5)
    primary_tune = _score(base_chosen, base_keep, tune_rows)
    primary_eval = _score(base_chosen, base_keep, eval_rows)
    primary_full = _score(base_chosen, base_keep, None)
    best_f = primary_tune
    best = (-1.0, 0.0, 0.5)
    gap_cuts = (0.03, 0.06, 0.10, 0.15, 0.25, 0.40)
    margins = (0.02, 0.05, 0.10, 0.20, 0.35)
    blind_cuts = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    for override_gap in gap_cuts:
        for margin in margins:
            for blind_t in blind_cuts:
                chosen, keep, _flip = apply_rule(bc, bp, gap, lc, lp, lg, threshold, min_gap, override_gap, margin, blind_t)
                score = _score(chosen, keep, tune_rows)
                if score > best_f + 1e-5:
                    best_f = score
                    best = (float(override_gap), float(margin), float(blind_t))
    chosen, keep, flip = apply_rule(bc, bp, gap, lc, lp, lg, threshold, min_gap, best[0], best[1], best[2])
    eval_f = _score(chosen, keep, eval_rows)
    if best[0] <= 0 or eval_f <= primary_eval + 1e-5:
        print(
            f"rank-blind override off: eval-half primary F_0.5 {primary_eval:.5f}, "
            f"best override on that half {eval_f:.5f} (tune-half {best_f:.5f})",
            flush=True,
        )
        print(
            f"full-data macro F_0.5 = {primary_full:.5f} with the primary rule only",
            flush=True,
        )
        return -1.0, 0.0, 0.5, primary_full
    full_f = _score(chosen, keep, None)
    correct = owner[bq] == chosen
    n_links = int(keep.sum())
    n_correct = int(correct[keep].sum())
    n_true = int((owner >= 0).sum())
    print(
        f"rank-blind override on: eval-half F_0.5 {eval_f:.5f} vs primary {primary_eval:.5f}; "
        f"full-data macro F_0.5 = {full_f:.5f} at override_gap {best[0]:.2f} "
        f"blind_margin {best[1]:.2f} blind_threshold {best[2]:.2f} "
        f"(flips {int(flip.sum()):,}, links {n_links:,}, correct {n_correct:,}, true links {n_true:,}, "
        f"link precision {n_correct / max(1, n_links):.5f}, link recall {n_correct / max(1, n_true):.5f})",
        flush=True,
    )
    return best[0], best[1], best[2], full_f
