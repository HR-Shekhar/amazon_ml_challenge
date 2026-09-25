"""Candidate generation.

Two stages, both run inside each country (the country label is an open string, never a
hard-coded list):

1. Token blocking. Rare name tokens, phonetic-skeleton tokens, a compact-name prefix,
   and street numbers are posted into inverted indexes. Keys that are too common are
   dropped so one popular token cannot pull in thousands of unrelated businesses.
   An exact compact-name hit is weighted high enough that the later cap cannot drop it.
2. TF-IDF nearest neighbours. Hashed word/bigram TF-IDF of the normalized name (plus
   its phonetic skeleton) and of the normalized address; each Source 2/3 record keeps
   its closest Source 1 rows. This is what recovers typos and transliteration.

The union is capped per Source 1 entity. ``recall_check`` reports how many ground-truth
pairs survived — that number is the ceiling for everything downstream.
"""

from __future__ import annotations

import heapq
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from data_loading import name_skeleton

_PAIR_BIG = np.int64(10_000_000_000)


@dataclass
class BlockResult:
    s1_index: np.ndarray
    s23_index: np.ndarray
    votes: np.ndarray
    tfidf: np.ndarray

    def __len__(self) -> int:
        return int(len(self.s1_index))


def _strings(df: pd.DataFrame, column: str) -> np.ndarray:
    if column == "name_skel" and column not in df.columns:
        core = df["name_core"].fillna("").astype(str).tolist()
        return np.array([name_skeleton(x) for x in core], dtype=object)
    return df[column].fillna("").astype(str).to_numpy()


def _row_keys(name_core: str, name_skel: str, name_compact: str, addr_nums: str) -> list[str]:
    keys: list[str] = []
    parts = name_core.split()
    for tok in parts:
        if len(tok) >= 4:
            keys.append("n" + tok)
    for tok in name_skel.split():
        if len(tok) >= 3:
            keys.append("s" + tok)
    if len(name_compact) >= 5:
        keys.append("e" + name_compact)
    if len(name_compact) >= 8:
        keys.append("p" + name_compact[:8])
    first = parts[0] if parts else ""
    for num in addr_nums.split():
        if len(num) >= 3:
            keys.append("d" + num)
            if len(first) >= 3:
                keys.append("x" + first + num)
    return keys


def _df_cap(key: str, n_rows: int) -> int:
    scaled = max(25, min(80, n_rows // 30 or 25))
    if key.startswith("e"):
        return min(12, scaled)
    return scaled


def _token_pairs(s1_local: np.ndarray, s23_local: np.ndarray, cols: dict[str, np.ndarray], max_keep: int) -> tuple[list[int], list[int], list[int]]:
    """Return global-index lists (s1, s23, votes) from the inverted index."""
    fields = ("name_core", "name_skel", "name_compact", "addr_nums")
    s23_keys = [
        _row_keys(*(str(cols[f][j]) for f in fields))
        for j in range(len(s23_local))
    ]
    counts: dict[str, int] = {}
    for keys in s23_keys:
        for key in set(keys):
            counts[key] = counts.get(key, 0) + 1
    postings: dict[str, list[int]] = {}
    for local_j, keys in enumerate(s23_keys):
        for key in set(keys):
            if counts[key] <= _df_cap(key, len(s23_local)):
                postings.setdefault(key, []).append(local_j)

    out_s1: list[int] = []
    out_s23: list[int] = []
    out_votes: list[int] = []
    for local_i, gi in enumerate(s1_local):
        scores: dict[int, int] = {}
        for key in _row_keys(*(str(cols[f + "_s1"][local_i]) for f in fields)):
            bucket = postings.get(key)
            if not bucket:
                continue
            weight = 100 if key.startswith("e") else 1
            for local_j in bucket:
                scores[local_j] = scores.get(local_j, 0) + weight
        if not scores:
            continue
        if len(scores) > max_keep:
            chosen = heapq.nlargest(max_keep, scores.items(), key=lambda kv: kv[1])
        else:
            chosen = scores.items()
        g_s23 = s23_local
        for local_j, vote in chosen:
            out_s1.append(int(gi))
            out_s23.append(int(g_s23[local_j]))
            out_votes.append(int(vote))
    return out_s1, out_s23, out_votes


def _fit_idf(texts: list[str], n_features: int) -> tuple[HashingVectorizer, np.ndarray, np.ndarray]:
    hasher = HashingVectorizer(
        n_features=n_features,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm=None,
        token_pattern=r"\S+",
        dtype=np.float32,
    )
    matrix = hasher.transform(texts)
    df = np.bincount(matrix.indices, minlength=n_features).astype(np.float32)
    idf = (np.log((len(texts) + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    return hasher, idf, df


def _weight(matrix: sp.csr_matrix, idf: np.ndarray, df: np.ndarray, cap: float, apply_cap: bool) -> sp.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    if matrix.nnz == 0:
        return matrix
    scale = idf[matrix.indices]
    if apply_cap:
        scale = scale * (df[matrix.indices] <= cap)
    matrix.data = matrix.data * scale
    matrix.eliminate_zeros()
    return normalize(matrix, norm="l2", copy=False)


def _tfidf_pairs(
    s1_local: np.ndarray,
    s23_local: np.ndarray,
    name_s1: np.ndarray,
    addr_s1: np.ndarray,
    skel_s1: np.ndarray,
    name_s23: np.ndarray,
    addr_s23: np.ndarray,
    skel_s23: np.ndarray,
    topn: int,
    n_features: int,
) -> tuple[list[int], list[int], list[float]]:
    from sparse_dot_topn import sp_matmul_topn

    name1 = [f"{a} {b}".strip() for a, b in zip(name_s1, skel_s1)]
    name2 = [f"{a} {b}".strip() for a, b in zip(name_s23, skel_s23)]
    hasher_n, idf_n, df_n = _fit_idf(name1, n_features)
    hasher_a, idf_a, df_a = _fit_idf(list(addr_s1), n_features)
    cap = float(min(8000, max(300, len(s1_local) // 20 or 300)))
    left = _weight(hasher_n.transform(name1), idf_n, df_n, cap, True)
    right = _weight(hasher_a.transform(list(addr_s1)), idf_a, df_a, cap, True)
    index = sp.hstack([left * np.float32(0.6), right * np.float32(0.8)], format="csr")
    if index.nnz == 0:
        return [], [], []
    index_t = index.T.tocsr()
    threads = max(1, (os.cpu_count() or 2) - 1)
    out_s1: list[int] = []
    out_s23: list[int] = []
    out_score: list[float] = []
    chunk = 20_000
    for start in range(0, len(s23_local), chunk):
        stop = min(start + chunk, len(s23_local))
        q_name = _weight(hasher_n.transform(name2[start:stop]), idf_n, df_n, cap, False)
        q_addr = _weight(hasher_a.transform(list(addr_s23[start:stop])), idf_a, df_a, cap, False)
        query = sp.hstack([q_name * np.float32(0.6), q_addr * np.float32(0.8)], format="csr")
        if query.nnz == 0:
            continue
        hits = sp_matmul_topn(query, index_t, top_n=topn, n_threads=threads, sort=True)
        for row in range(stop - start):
            a, b = hits.indptr[row], hits.indptr[row + 1]
            if a == b:
                continue
            gi23 = int(s23_local[start + row])
            for col, score in zip(hits.indices[a:b], hits.data[a:b]):
                out_s23.append(gi23)
                out_s1.append(int(s1_local[int(col)]))
                out_score.append(float(score))
    return out_s1, out_s23, out_score


def _dedupe_and_cap(
    s1: np.ndarray,
    s23: np.ndarray,
    votes: np.ndarray,
    tfidf: np.ndarray,
    max_candidates: int,
) -> BlockResult:
    if len(s1) == 0:
        empty_i = np.empty(0, np.int32)
        empty_f = np.empty(0, np.float32)
        return BlockResult(empty_i, empty_i, empty_i, empty_f)
    order = np.lexsort((s23, s1))
    s1, s23, votes, tfidf = s1[order], s23[order], votes[order], tfidf[order]
    change = np.empty(len(s1), dtype=bool)
    change[0] = True
    change[1:] = (s1[1:] != s1[:-1]) | (s23[1:] != s23[:-1])
    starts = np.flatnonzero(change)
    votes = np.maximum.reduceat(votes, starts)
    tfidf = np.maximum.reduceat(tfidf, starts)
    s1, s23 = s1[change], s23[change]
    # Exact-name votes are ~100. A TF-IDF hit gets a flat bonus so the cap cannot
    # throw away the nearest neighbours in favour of weak token votes.
    score = votes.astype(np.float32) + np.float32(20.0) * (tfidf > 0) + np.float32(5.0) * tfidf
    order = np.lexsort((-score, s1))
    s1, s23, votes, tfidf = s1[order], s23[order], votes[order], tfidf[order]
    idx = np.arange(len(s1))
    group_mark = np.where(np.diff(s1, prepend=np.int64(s1[0]) - 1) != 0, idx, 0)
    rank = idx - np.maximum.accumulate(group_mark)
    keep = rank < max_candidates
    return BlockResult(
        s1[keep].astype(np.int32, copy=False),
        s23[keep].astype(np.int32, copy=False),
        votes[keep].astype(np.int32, copy=False),
        tfidf[keep].astype(np.float32, copy=False),
    )


def block(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    topn: int = 12,
    max_candidates: int = 40,
    n_features: int = 1 << 20,
) -> BlockResult:
    """Build the candidate pairs the matcher will score. One country at a time."""
    if len(s1) == 0 or len(s23) == 0:
        empty_i = np.empty(0, np.int32)
        return BlockResult(empty_i, empty_i, empty_i, np.empty(0, np.float32))

    c1 = _strings(s1, "country")
    c23 = _strings(s23, "country")
    pieces_s1: list[np.ndarray] = []
    pieces_s23: list[np.ndarray] = []
    pieces_votes: list[np.ndarray] = []
    pieces_tfidf: list[np.ndarray] = []

    for country in sorted(set(c1.tolist()) | set(c23.tolist())):
        s1_local = np.flatnonzero(c1 == country)
        s23_local = np.flatnonzero(c23 == country)
        print(f"blocking {country or '(blank)'}: source1={len(s1_local):,} source2/3={len(s23_local):,}", flush=True)
        if len(s1_local) == 0 or len(s23_local) == 0:
            continue
        cols = {
            "name_core": _strings(s23, "name_core")[s23_local],
            "name_skel": _strings(s23, "name_skel")[s23_local],
            "name_compact": _strings(s23, "name_compact")[s23_local],
            "addr_nums": _strings(s23, "addr_nums")[s23_local],
            "name_core_s1": _strings(s1, "name_core")[s1_local],
            "name_skel_s1": _strings(s1, "name_skel")[s1_local],
            "name_compact_s1": _strings(s1, "name_compact")[s1_local],
            "addr_nums_s1": _strings(s1, "addr_nums")[s1_local],
        }
        t_s1, t_s23, t_votes = _token_pairs(s1_local, s23_local, cols, max_candidates)
        f_s1: list[int] = []
        f_s23: list[int] = []
        f_score: list[float] = []
        if len(s1_local) >= 8 and len(s23_local) >= 8:
            try:
                f_s1, f_s23, f_score = _tfidf_pairs(
                    s1_local,
                    s23_local,
                    cols["name_core_s1"],
                    _strings(s1, "addr_norm")[s1_local],
                    cols["name_skel_s1"],
                    cols["name_core"],
                    _strings(s23, "addr_norm")[s23_local],
                    cols["name_skel"],
                    topn,
                    n_features,
                )
            except Exception as exc:
                print(f"  tfidf blocking skipped for {country}: {exc}", flush=True)
        s1_cat = np.concatenate([np.asarray(t_s1, np.int32), np.asarray(f_s1, np.int32)]) if (t_s1 or f_s1) else np.empty(0, np.int32)
        s23_cat = np.concatenate([np.asarray(t_s23, np.int32), np.asarray(f_s23, np.int32)]) if (t_s1 or f_s1) else np.empty(0, np.int32)
        votes = np.concatenate([np.asarray(t_votes, np.int32), np.zeros(len(f_s1), np.int32)]) if (t_s1 or f_s1) else np.empty(0, np.int32)
        tfidf = np.concatenate([np.zeros(len(t_s1), np.float32), np.asarray(f_score, np.float32)]) if (t_s1 or f_s1) else np.empty(0, np.float32)
        if len(s1_cat) == 0:
            print("  candidates=0", flush=True)
            continue
        capped = _dedupe_and_cap(s1_cat, s23_cat, votes, tfidf, max_candidates)
        print(f"  candidates={len(capped):,}", flush=True)
        pieces_s1.append(capped.s1_index)
        pieces_s23.append(capped.s23_index)
        pieces_votes.append(capped.votes)
        pieces_tfidf.append(capped.tfidf)

    if not pieces_s1:
        empty_i = np.empty(0, np.int32)
        return BlockResult(empty_i, empty_i, empty_i, np.empty(0, np.float32))
    return BlockResult(
        np.concatenate(pieces_s1),
        np.concatenate(pieces_s23),
        np.concatenate(pieces_votes),
        np.concatenate(pieces_tfidf),
    )


def recall_check(candidates: BlockResult, truth: dict[int, np.ndarray], countries: np.ndarray | None = None) -> dict[str, float]:
    """Pair recall of ``candidates`` against ground truth.

    Pair recall is the fraction of true (source1, source2/3) links that appear in the
    candidate set. Entity recall is the fraction of non-singleton Source 1 entities
    whose true links are *all* present. Both are ceilings on the matcher.
    """
    true_s1: list[np.ndarray] = []
    true_s23: list[np.ndarray] = []
    singletons = 0
    for row, linked in truth.items():
        if linked is None or len(linked) == 0:
            singletons += 1
            continue
        true_s1.append(np.full(len(linked), int(row), np.int64))
        true_s23.append(np.asarray(linked, np.int64))
    n_s1 = len(truth)
    n_cand = len(candidates)
    report: dict[str, float] = {
        "source1_entities": float(n_s1),
        "singletons": float(singletons),
        "true_pairs": 0.0,
        "candidate_pairs": float(n_cand),
        "mean_candidates": float(n_cand / n_s1) if n_s1 else 0.0,
        "pair_recall": 1.0 if singletons == n_s1 else 0.0,
        "entity_recall": 1.0 if singletons == n_s1 else 0.0,
    }
    if not true_s1:
        return report
    ts1 = np.concatenate(true_s1)
    ts23 = np.concatenate(true_s23)
    tkeys = ts1 * _PAIR_BIG + ts23
    order_t = np.argsort(tkeys, kind="mergesort")
    tkeys = tkeys[order_t]
    ts1 = ts1[order_t]
    if n_cand:
        ckeys = candidates.s1_index.astype(np.int64) * _PAIR_BIG + candidates.s23_index.astype(np.int64)
        ckeys.sort()
        idx = np.searchsorted(ckeys, tkeys)
        in_range = idx < len(ckeys)
        hit = np.zeros(len(tkeys), dtype=bool)
        hit[in_range] = ckeys[idx[in_range]] == tkeys[in_range]
    else:
        hit = np.zeros(len(tkeys), dtype=bool)
    report["true_pairs"] = float(len(tkeys))
    report["pair_recall"] = float(hit.mean())
    starts = np.flatnonzero(np.diff(ts1, prepend=ts1[0] - 1))
    complete = np.minimum.reduceat(hit.astype(np.int8), starts)
    report["entity_recall"] = float(complete.mean())
    if countries is not None:
        ent_country = countries[ts1[starts]]
        for country in sorted(set(ent_country.tolist())):
            mask = ent_country == country
            report[f"entity_recall[{country}]"] = float(complete[mask].mean()) if mask.any() else 0.0
            # pair recall by the entity's country
            pair_country = countries[ts1]
            pmask = pair_country == country
            report[f"pair_recall[{country}]"] = float(hit[pmask].mean()) if pmask.any() else 0.0
    return report


def format_recall(report: dict[str, float]) -> str:
    lines = ["blocking recall (ceiling for the matcher)"]
    for key in ("source1_entities", "singletons", "true_pairs", "candidate_pairs", "mean_candidates", "pair_recall", "entity_recall"):
        value = report[key]
        if key.endswith("recall") or key == "mean_candidates":
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {int(value)}")
    for key in sorted(k for k in report if k not in lines and "[" in k):
        lines.append(f"  {key}: {report[key]:.4f}")
    return "\n".join(lines)
