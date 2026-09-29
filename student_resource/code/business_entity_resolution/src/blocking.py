"""Candidate generation, query side.

Every Source 2/3 record belongs to at most one Source 1 entity, so candidates are
generated per Source 2/3 record (the "query"): its nearest Source 1 records, inside the
query's country (an open string, never a hard-coded list). Three searches are unioned:

1. Word TF-IDF of name (+ phonetic skeleton) and address, cosine top-n.
2. Character-trigram TF-IDF of the compact name plus address words, cosine top-n. This
   recovers typos, transliteration, and concatenated names.
3. Rare inverted-index keys (name tokens, skeleton tokens, exact compact name, street
   numbers). Keys common in Source 1 are dropped.

Each query keeps its best ``max_candidates`` Source 1 records. ``recall_report`` gives
the fraction of matched queries whose true owner survived (the ceiling downstream).
"""

from __future__ import annotations

import heapq
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from features import _transform

SCORE_NAMES = ("blk_word", "blk_char", "blk_votes")


@dataclass
class Candidates:
    """Pairs sorted by query. ``q`` indexes Source 2/3 rows, ``c`` Source 1 rows."""

    q: np.ndarray
    c: np.ndarray
    scores: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.q))


def _strings(df: pd.DataFrame, column: str) -> np.ndarray:
    return df[column].fillna("").astype(str).to_numpy(dtype=object)


def _row_keys(name_core: str, name_skel: str, name_compact: str, addr_nums: str, _addr_norm: str) -> list[str]:
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


def _n_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------------------
# TF-IDF searches
# ---------------------------------------------------------------------------------------


def _hasher(analyzer: str, ngram: tuple[int, int]) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=1 << 20,
        analyzer=analyzer,
        ngram_range=ngram,
        alternate_sign=False,
        norm=None,
        lowercase=False,
        token_pattern=r"\S+" if analyzer == "word" else None,
        dtype=np.float32,
    )


def _weighted(parts1: list[sp.csr_matrix], parts23: list[sp.csr_matrix], weights: list[float], cap: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """IDF from Source 1, drop index terms with df above ``cap``, L2 per part, then weight."""
    idx_parts, q_parts = [], []
    for m1, m23, w in zip(parts1, parts23, weights):
        df = np.bincount(m1.indices, minlength=m1.shape[1]).astype(np.float32)
        idf = (np.log((m1.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        a = m1.copy()
        a.data *= idf[a.indices] * (df[a.indices] <= cap)
        a.eliminate_zeros()
        b = m23.copy()
        b.data *= idf[b.indices]
        idx_parts.append(normalize(a, copy=False) * np.float32(w))
        q_parts.append(normalize(b, copy=False) * np.float32(w))
    return sp.hstack(idx_parts, format="csr"), sp.hstack(q_parts, format="csr")


def _topn(index: sp.csr_matrix, query: sp.csr_matrix, topn: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sparse_dot_topn import sp_matmul_topn

    index_t = index.T.tocsr()
    qs, cs, ss = [], [], []
    step = 100_000
    for start in range(0, query.shape[0], step):
        part = query[start : start + step]
        if part.nnz == 0:
            continue
        hits = sp_matmul_topn(part, index_t, top_n=topn, n_threads=n_jobs, sort=True)
        rows = np.repeat(np.arange(hits.shape[0], dtype=np.int64), np.diff(hits.indptr)) + start
        qs.append(rows)
        cs.append(hits.indices.astype(np.int64))
        ss.append(hits.data.astype(np.float32))
    if not qs:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(qs), np.concatenate(cs), np.concatenate(ss)


# ---------------------------------------------------------------------------------------
# Token keys
# ---------------------------------------------------------------------------------------

_TOKEN_STATE: dict = {}


def _token_slice(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = bounds
    postings = _TOKEN_STATE["postings"]
    keys23 = _TOKEN_STATE["keys23"]
    keep = _TOKEN_STATE["keep"]
    oq: list[int] = []
    oc: list[int] = []
    ov: list[int] = []
    for j in range(a, b):
        scores: dict[int, int] = {}
        for key in keys23[j]:
            bucket = postings.get(key)
            if bucket is None:
                continue
            w = 20 if key[0] == "e" else 1
            for i in bucket:
                scores[i] = scores.get(i, 0) + w
        if not scores:
            continue
        items = heapq.nlargest(keep, scores.items(), key=itemgetter(1)) if len(scores) > keep else scores.items()
        for i, v in items:
            oq.append(j)
            oc.append(i)
            ov.append(v)
    return np.asarray(oq, np.int64), np.asarray(oc, np.int64), np.asarray(ov, np.float32)


def _token_search(cols1: dict[str, np.ndarray], cols23: dict[str, np.ndarray], keep: int, cap: int, n_jobs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = ("name_core", "name_skel", "name_compact", "addr_nums", "addr_norm")
    keys1 = [set(_row_keys(*vals)) for vals in zip(*(cols1[x] for x in f))]
    counts: dict[str, int] = {}
    for ks in keys1:
        for k in ks:
            counts[k] = counts.get(k, 0) + 1
    postings: dict[str, list[int]] = {}
    for li, ks in enumerate(keys1):
        for k in ks:
            if counts[k] <= cap:
                postings.setdefault(k, []).append(li)
    del keys1, counts
    keys23 = [set(_row_keys(*vals)) for vals in zip(*(cols23[x] for x in f))]
    step = 100_000
    bounds = [(a, min(a + step, len(keys23))) for a in range(0, len(keys23), step)]
    _TOKEN_STATE.update(postings=postings, keys23=keys23, keep=keep)
    try:
        if len(bounds) > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(n_jobs) as pool:
                parts = pool.map(_token_slice, bounds)
        else:
            parts = [_token_slice(b) for b in bounds]
    finally:
        _TOKEN_STATE.clear()
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))


# ---------------------------------------------------------------------------------------
# Union and cap
# ---------------------------------------------------------------------------------------


def _topk_mask(q: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """True at the ``k`` highest ``score`` rows within each query. ``q`` need not be sorted."""
    keep = np.zeros(len(q), dtype=bool)
    if len(q) == 0 or k <= 0:
        return keep
    order = np.lexsort((-score, q))
    qs = q[order]
    start = np.r_[True, qs[1:] != qs[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(q)), 0))
    keep[order[(np.arange(len(q)) - first) < k]] = True
    return keep


def rank_score(scores: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(scores["blk_word"], scores["blk_char"]) + np.float32(0.01) * np.minimum(scores["blk_votes"], 40)


def _union(lists: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], n_c: int, max_candidates: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    q = np.concatenate([x[1] for x in lists])
    c = np.concatenate([x[2] for x in lists])
    uniq, inverse = np.unique(q * np.int64(n_c) + c, return_inverse=True)
    scores = {}
    offset = 0
    for name, lq, _, ls in lists:
        arr = np.zeros(len(uniq), np.float32)
        np.maximum.at(arr, inverse[offset : offset + len(lq)], ls)
        scores[name] = arr
        offset += len(lq)
    uq = uniq // n_c
    uc = uniq % n_c
    # One blended ranking, top ``max_candidates``. This is the candidate list behind
    # the 0.968 submission. Reserved token slots were measured and did not raise recall.
    keep = _topk_mask(uq, rank_score(scores), max_candidates)
    uq, uc = uq[keep], uc[keep]
    scores = {k: v[keep] for k, v in scores.items()}
    order = np.lexsort((-rank_score(scores), uq))
    return uq[order], uc[order], {k: v[order] for k, v in scores.items()}


def block(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    topn: int = 20,
    max_candidates: int = 12,
    token_keep: int = 10,
    token_cap: int = 60,
    n_jobs: int | None = None,
) -> Candidates:
    """Candidate Source 1 rows for every Source 2/3 row, one country at a time."""
    c1 = _strings(s1, "country")
    c23 = _strings(s23, "country")
    out_q, out_c = [], []
    out_s: dict[str, list[np.ndarray]] = {k: [] for k in SCORE_NAMES}
    fields = ("name_core", "name_skel", "name_compact", "addr_norm", "addr_nums")
    all1 = {f: _strings(s1, f) for f in fields}
    all23 = {f: _strings(s23, f) for f in fields}
    for country in sorted(set(c1.tolist()) & set(c23.tolist())):
        t0 = time.time()
        i1 = np.flatnonzero(c1 == country)
        i23 = np.flatnonzero(c23 == country)
        print(f"blocking {country or '(blank)'}: source1={len(i1):,} source2/3={len(i23):,}", flush=True)
        cols1 = {f: all1[f][i1] for f in fields}
        cols23 = {f: all23[f][i23] for f in fields}
        cap = float(min(8000, max(300, len(i1) // 20)))
        jobs = n_jobs or _n_jobs()
        lists = []
        if len(i1) >= 2:
            name1 = np.array([f"{a} {b}".strip() for a, b in zip(cols1["name_core"], cols1["name_skel"])], dtype=object)
            name23 = np.array([f"{a} {b}".strip() for a, b in zip(cols23["name_core"], cols23["name_skel"])], dtype=object)
            word = _hasher("word", (1, 2))
            char = _hasher("char", (3, 3))
            a1 = _transform(word, cols1["addr_norm"], jobs)
            a23 = _transform(word, cols23["addr_norm"], jobs)
            index, query = _weighted([_transform(word, name1, jobs), a1], [_transform(word, name23, jobs), a23], [0.6, 0.8], cap)
            q, c, s = _topn(index, query, topn, jobs)
            lists.append(("blk_word", q, c, s))
            print(f"  word tfidf: {len(q):,} pairs ({time.time() - t0:.0f}s)", flush=True)
            index, query = _weighted([_transform(char, cols1["name_compact"], jobs), a1], [_transform(char, cols23["name_compact"], jobs), a23], [0.7, 0.7], cap)
            del a1, a23
            q, c, s = _topn(index, query, topn, jobs)
            lists.append(("blk_char", q, c, s))
            del index, query
            print(f"  char tfidf: {len(q):,} pairs ({time.time() - t0:.0f}s)", flush=True)
        q, c, s = _token_search(cols1, cols23, token_keep, token_cap, jobs)
        lists.append(("blk_votes", q, c, s))
        print(f"  token keys: {len(q):,} pairs ({time.time() - t0:.0f}s)", flush=True)
        present = {x[0] for x in lists}
        for name in SCORE_NAMES:
            if name not in present:
                lists.append((name, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)))
        uq, uc, scores = _union(lists, len(i1), max_candidates)
        out_q.append(i23[uq].astype(np.int32))
        out_c.append(i1[uc].astype(np.int32))
        for k in SCORE_NAMES:
            out_s[k].append(scores[k])
        print(f"  candidates={len(uq):,} ({time.time() - t0:.0f}s)", flush=True)
    if not out_q:
        return Candidates(np.empty(0, np.int32), np.empty(0, np.int32), {k: np.empty(0, np.float32) for k in SCORE_NAMES})
    q = np.concatenate(out_q)
    c = np.concatenate(out_c)
    order = np.argsort(q, kind="stable")
    return Candidates(q[order], c[order], {k: np.concatenate(v)[order] for k, v in out_s.items()})


def add_source1_scores(cands: Candidates) -> None:
    """How contested each Source 1 candidate is, across all queries.

    ``s1_n_queries``: queries listing this Source 1 row. ``s1_n_top``: queries where it
    is the best-ranked candidate. ``rev_rank`` / ``rev_gap``: this query's rank among all
    queries listing the same Source 1 row, and its gap to the best other one.
    """
    from features import group_rank_gap

    best = rank_score(cands.scores)
    cands.scores["blk_best"] = best
    c = cands.c.astype(np.int64)
    _, inv, counts = np.unique(c, return_inverse=True, return_counts=True)
    cands.scores["s1_n_queries"] = counts[inv].astype(np.float32)
    q_rank, _ = group_rank_gap(cands.q.astype(np.int64), best)
    top = np.bincount(inv, weights=(q_rank == 0).astype(np.float64), minlength=len(counts))
    cands.scores["s1_n_top"] = top[inv].astype(np.float32)
    rev_rank, rev_gap = group_rank_gap(c, best)
    cands.scores["rev_rank"] = rev_rank
    cands.scores["rev_gap"] = rev_gap


def owner_array(n_s23: int, truth: dict[int, np.ndarray]) -> np.ndarray:
    """``owner[j]`` is the Source 1 row that Source 2/3 row ``j`` belongs to, or -1."""
    owner = np.full(n_s23, -1, np.int64)
    for row, linked in truth.items():
        if len(linked):
            owner[np.asarray(linked, np.int64)] = int(row)
    return owner


def recall_report(cands: Candidates, owner: np.ndarray, countries23: np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20)) -> str:
    """Fraction of matched Source 2/3 rows whose owner is in their top-k candidates."""
    order = np.lexsort((-rank_score(cands.scores), cands.q))
    q, c = cands.q[order].astype(np.int64), cands.c[order].astype(np.int64)
    start = np.r_[True, q[1:] != q[:-1]]
    rank = np.arange(len(q)) - np.maximum.accumulate(np.where(start, np.arange(len(q)), 0))
    hit_rank = np.full(len(owner), np.iinfo(np.int64).max, np.int64)
    hit = owner[q] == c
    np.minimum.at(hit_rank, q[hit], rank[hit])
    matched = owner >= 0
    lines = ["blocking recall (Source 2/3 rows whose owner is among their candidates)"]
    lines.append(f"  candidate pairs: {len(q):,}  mean per query: {len(q) / max(1, len(np.unique(q))):.2f}")
    for country in ["(all)"] + sorted(set(countries23[matched].tolist())):
        mask = matched if country == "(all)" else matched & (countries23 == country)
        if not mask.any():
            continue
        parts = [f"r@{k}={np.mean(hit_rank[mask] < k):.4f}" for k in ks]
        lines.append(f"  {country}: n={int(mask.sum()):,} " + " ".join(parts) + f" any={np.mean(hit_rank[mask] < np.iinfo(np.int64).max):.4f}")
    return "\n".join(lines)
