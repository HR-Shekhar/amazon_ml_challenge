"""Pairwise similarity features for candidate pairs.

Every comparison is computed twice where it applies: once on the raw lower-cased text
and once on the normalized / legal-suffix-stripped text. Metrics are Jaccard, Levenshtein,
Jaro-Winkler, token sort/set/partial ratios, and TF-IDF cosine.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz.distance.JaroWinkler import normalized_similarity as jaro_winkler
from rapidfuzz.distance.Levenshtein import normalized_similarity as levenshtein
from rapidfuzz.fuzz import partial_ratio, token_set_ratio, token_sort_ratio
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

FEATURE_NAMES = [
    "jac_name_core",
    "jac_name_norm",
    "jac_name_raw",
    "jac_addr_norm",
    "jac_addr_raw",
    "jac_addr_nums",
    "lev_name_core",
    "lev_name_norm",
    "lev_name_raw",
    "jw_name_core",
    "jw_name_norm",
    "sort_name_core",
    "partial_name_core",
    "lev_addr_norm",
    "lev_addr_raw",
    "set_addr_norm",
    "lev_name_compact",
    "sort_name_skel",
    "cos_name_core",
    "cos_name_norm",
    "cos_name_raw",
    "cos_addr_norm",
    "cos_addr_raw",
    "cos_name_skel",
    "len_ratio_name_core",
    "first_token_eq",
    "compact_contains",
    "addr_both_empty",
    "addr_one_empty",
    "shared_num_frac",
    "block_votes",
    "block_tfidf",
]

_N_FEATURES = 1 << 16
_COSINE_FIELDS = ("name_core", "name_norm", "name_raw", "addr_norm", "addr_raw", "name_skel")


def _jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / float(len(sa) + len(sb) - inter)


def _lev(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(levenshtein(a, b))


def _jw(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(jaro_winkler(a, b))


def _ratio(fn, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(fn(a, b)) / 100.0


def _fit_idf(texts: np.ndarray) -> tuple[HashingVectorizer, np.ndarray]:
    hasher = HashingVectorizer(
        n_features=_N_FEATURES,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm=None,
        token_pattern=r"\S+",
        dtype=np.float32,
    )
    matrix = hasher.transform(texts.tolist())
    df = np.bincount(matrix.indices, minlength=_N_FEATURES).astype(np.float32)
    idf = (np.log((len(texts) + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    return hasher, idf


def _pairwise_cosine(hasher: HashingVectorizer, idf: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0:
        return np.empty(0, np.float32)
    a = hasher.transform(left.tolist()).tocsr(copy=True)
    b = hasher.transform(right.tolist()).tocsr(copy=True)
    if a.nnz:
        a.data *= idf[a.indices]
    if b.nnz:
        b.data *= idf[b.indices]
    a = normalize(a, norm="l2", copy=False)
    b = normalize(b, norm="l2", copy=False)
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32, copy=False)


def _string_block(rows: list[tuple[str, ...]]) -> np.ndarray:
    """String metrics for one chunk. Each tuple is (s1 fields..., s23 fields...)."""
    # Field order on each side: name_core, name_norm, name_raw, addr_norm, addr_raw,
    # addr_nums, name_compact, name_skel. 8 fields, so 16 strings per row.
    out = np.zeros((len(rows), 24), np.float32)
    for i, rec in enumerate(rows):
        nc1, nn1, nr1, an1, ar1, nu1, cc1, sk1, nc2, nn2, nr2, an2, ar2, nu2, cc2, sk2 = rec
        out[i, 0] = _jaccard(nc1, nc2)
        out[i, 1] = _jaccard(nn1, nn2)
        out[i, 2] = _jaccard(nr1, nr2)
        out[i, 3] = _jaccard(an1, an2)
        out[i, 4] = _jaccard(ar1, ar2)
        out[i, 5] = _jaccard(nu1, nu2)
        out[i, 6] = _lev(nc1, nc2)
        out[i, 7] = _lev(nn1, nn2)
        out[i, 8] = _lev(nr1, nr2)
        out[i, 9] = _jw(nc1, nc2)
        out[i, 10] = _jw(nn1, nn2)
        out[i, 11] = _ratio(token_sort_ratio, nc1, nc2)
        out[i, 12] = _ratio(partial_ratio, nc1, nc2)
        out[i, 13] = _lev(an1, an2)
        out[i, 14] = _lev(ar1, ar2)
        out[i, 15] = _ratio(token_set_ratio, an1, an2)
        out[i, 16] = _lev(cc1, cc2)
        out[i, 17] = _ratio(token_sort_ratio, sk1, sk2)
        la, lb = len(nc1), len(nc2)
        out[i, 18] = (min(la, lb) / max(la, lb)) if la and lb else 0.0
        t1 = nc1.split(" ", 1)[0] if nc1 else ""
        t2 = nc2.split(" ", 1)[0] if nc2 else ""
        out[i, 19] = 1.0 if t1 and t1 == t2 else 0.0
        out[i, 20] = 1.0 if len(cc1) >= 5 and len(cc2) >= 5 and (cc1 in cc2 or cc2 in cc1) else 0.0
        empty1 = an1 == ""
        empty2 = an2 == ""
        out[i, 21] = 1.0 if empty1 and empty2 else 0.0
        out[i, 22] = 1.0 if empty1 != empty2 else 0.0
        if nu1 and nu2:
            sa, sb = set(nu1.split()), set(nu2.split())
            out[i, 23] = len(sa & sb) / float(max(len(sa), len(sb)))
    return out


def _chunk_rows(s1: pd.DataFrame, s23: pd.DataFrame, fields: list[str], i1: np.ndarray, i23: np.ndarray) -> list[tuple[str, ...]]:
    """Materialize only this chunk's strings, not the whole source tables."""
    left = [s1[name].iloc[i1].fillna("").astype(str).to_numpy() for name in fields]
    right = [s23[name].iloc[i23].fillna("").astype(str).to_numpy() for name in fields]
    return [tuple(col[r] for col in left) + tuple(col[r] for col in right) for r in range(len(i1))]


def build_feature_matrix(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    s1_index: np.ndarray,
    s23_index: np.ndarray,
    block_votes: np.ndarray | None = None,
    block_tfidf: np.ndarray | None = None,
    chunk_size: int = 4_000,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Return an array of shape ``(n_pairs, len(FEATURE_NAMES))`` aligned with the pairs."""
    n = len(s1_index)
    matrix = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    if n == 0:
        return matrix

    fields = ["name_core", "name_norm", "name_raw", "addr_norm", "addr_raw", "addr_nums", "name_compact", "name_skel"]
    idf = {}
    for name in _COSINE_FIELDS:
        idf[name] = _fit_idf(s1[name].fillna("").astype(str).to_numpy())
    n_jobs = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)
    starts = list(range(0, n, chunk_size))

    def _rows_for(start: int) -> list[tuple[str, ...]]:
        stop = min(start + chunk_size, n)
        return _chunk_rows(s1, s23, fields, s1_index[start:stop], s23_index[start:stop])

    # The string metrics release no GIL, so chunks go to worker processes once there
    # are enough pairs to hide process startup. Cosine stays in this process.
    use_pool = n_jobs > 1 and n >= 8_000
    pool = ProcessPoolExecutor(max_workers=n_jobs) if use_pool else None
    encoded_iter = pool.map(_string_block, map(_rows_for, starts), chunksize=1) if pool else map(_string_block, map(_rows_for, starts))
    try:
        for start, encoded in zip(starts, encoded_iter):
            stop = min(start + chunk_size, n)
            i1 = s1_index[start:stop]
            i23 = s23_index[start:stop]
            matrix[start:stop, 0:18] = encoded[:, 0:18]
            matrix[start:stop, 24:30] = encoded[:, 18:24]
            col = 18
            for name in _COSINE_FIELDS:
                hasher, weights = idf[name]
                left = s1[name].iloc[i1].fillna("").astype(str).to_numpy()
                right = s23[name].iloc[i23].fillna("").astype(str).to_numpy()
                matrix[start:stop, col] = _pairwise_cosine(hasher, weights, left, right)
                col += 1
            if start == 0 or stop == n or (start // chunk_size) % 25 == 0:
                print(f"  features {stop:,}/{n:,}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    if block_votes is not None:
        matrix[:, FEATURE_NAMES.index("block_votes")] = block_votes.astype(np.float32, copy=False)
    if block_tfidf is not None:
        matrix[:, FEATURE_NAMES.index("block_tfidf")] = block_tfidf.astype(np.float32, copy=False)
    return matrix
