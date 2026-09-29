"""Features for (Source 2/3 query, Source 1 candidate) pairs.

Pair features compare the two records: Levenshtein, Jaro-Winkler, token sort/set/partial
ratios, token Jaccard, and TF-IDF cosine, on raw and normalized / legal-suffix-stripped
text. All string metrics run in rapidfuzz's C++ ``cpdist`` over whole chunks.

Query features compare a candidate with the other candidates of the same Source 2/3
record: its rank on a score and its gap to the best other candidate. Every Source 2/3
record belongs to at most one Source 1 entity, so "clearly the best of its candidates"
is the strongest signal. Source 1 features count how contested a Source 1 entity is.

``pairs`` must be sorted by query (``q``).
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

TEXT_FIELDS = ("name_core", "name_norm", "name_raw", "name_compact", "name_skel", "addr_norm", "addr_raw", "addr_nums")

_LEV = Levenshtein.normalized_similarity
_JW = JaroWinkler.normalized_similarity

# (feature, field, scorer, scale)
_STRING_METRICS = [
    ("lev_name_core", "name_core", _LEV, 1.0),
    ("jw_name_core", "name_core", _JW, 1.0),
    ("sort_name_core", "name_core", fuzz.token_sort_ratio, 0.01),
    ("set_name_core", "name_core", fuzz.token_set_ratio, 0.01),
    ("partial_name_core", "name_core", fuzz.partial_ratio, 0.01),
    ("lev_name_norm", "name_norm", _LEV, 1.0),
    ("jw_name_norm", "name_norm", _JW, 1.0),
    ("lev_name_raw", "name_raw", _LEV, 1.0),
    ("lev_name_compact", "name_compact", _LEV, 1.0),
    ("jw_name_compact", "name_compact", _JW, 1.0),
    ("lev_name_skel", "name_skel", _LEV, 1.0),
    ("sort_name_skel", "name_skel", fuzz.token_sort_ratio, 0.01),
    ("lev_addr_norm", "addr_norm", _LEV, 1.0),
    ("sort_addr_norm", "addr_norm", fuzz.token_sort_ratio, 0.01),
    ("set_addr_norm", "addr_norm", fuzz.token_set_ratio, 0.01),
    ("partial_addr_norm", "addr_norm", fuzz.partial_ratio, 0.01),
    ("lev_addr_raw", "addr_raw", _LEV, 1.0),
    ("lev_addr_nums", "addr_nums", _LEV, 1.0),
]
_JACCARD_FIELDS = ("name_core", "name_norm", "addr_norm", "addr_nums")
# (feature, field, analyzer, ngram)
_COSINE = [
    ("cos_name_core", "name_core", "word", (1, 2)),
    ("cos_name_char", "name_compact", "char", (3, 3)),
    ("cos_addr_norm", "addr_norm", "word", (1, 2)),
    ("cos_addr_char", "addr_norm", "char_wb", (3, 3)),
]
_IDF_FIELDS = ("name_core", "addr_norm")
BLOCK_SCORES = ("blk_word", "blk_char", "blk_votes", "blk_best", "s1_n_queries", "s1_n_top", "rev_rank", "rev_gap")
_FLAGS = [
    "addr_both_empty",
    "addr_one_empty",
    "first_token_eq",
    "first_num_eq",
    "compact_contains",
    "len_ratio_name_core",
    "len_ratio_addr",
    "n_name_tokens_q",
    "n_name_tokens_c",
]
# Scores that get a within-query rank and gap-to-best-other feature.
_GROUP_ON = (
    "blk_word",
    "blk_char",
    "blk_votes",
    "blk_best",
    "jw_name_core",
    "lev_name_compact",
    "set_name_core",
    "cos_name_char",
    "set_addr_norm",
    "lev_addr_norm",
    "jac_addr_nums",
    "idf_name_core",
    "quick",
)

PAIR_NAMES = (
    [m[0] for m in _STRING_METRICS]
    + [f"jac_{f}" for f in _JACCARD_FIELDS]
    + ["shared_addr_nums"]
    + [c[0] for c in _COSINE]
    + [f"idf_{f}" for f in _IDF_FIELDS]
    + list(BLOCK_SCORES)
    + _FLAGS
    + ["quick"]
)
GROUP_NAMES = [f"rank_{g}" for g in _GROUP_ON] + [f"gap_{g}" for g in _GROUP_ON] + ["group_size"]
FEATURE_NAMES = PAIR_NAMES + GROUP_NAMES
# The rank-aware model matches the 0.968 feature set: no IDF columns.
PRIMARY_NAMES = [name for name in FEATURE_NAMES if "idf" not in name]
# The tie-break model sees only pair comparisons. Block scores and within-list
# ranks are what make the other model copy the blocker's first place.
BLIND_NAMES = [name for name in PAIR_NAMES if name not in BLOCK_SCORES]


def _column_index(names: list[str]) -> np.ndarray:
    pos = {name: i for i, name in enumerate(FEATURE_NAMES)}
    missing = [name for name in names if name not in pos]
    if missing:
        raise RuntimeError(f"unknown feature columns: {missing}")
    return np.asarray([pos[name] for name in names], dtype=np.intp)


PRIMARY_INDEX = _column_index(PRIMARY_NAMES)
BLIND_INDEX = _column_index(BLIND_NAMES)


def text_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {f: df[f].fillna("").astype(str).to_numpy(dtype=object) for f in TEXT_FIELDS}


_POOL_STATE: dict = {}


def _transform_slice(bounds: tuple[int, int]) -> sp.csr_matrix:
    a, b = bounds
    return _POOL_STATE["hv"].transform(_POOL_STATE["texts"][a:b].tolist()).tocsr()


def _transform(hv: HashingVectorizer, texts: np.ndarray, n_jobs: int) -> sp.csr_matrix:
    """``hv.transform`` over many texts; forked workers when the platform allows it."""
    step = 200_000
    bounds = [(a, min(a + step, len(texts))) for a in range(0, len(texts), step)]
    if n_jobs > 1 and len(bounds) > 1 and "fork" in mp.get_all_start_methods():
        _POOL_STATE.update(hv=hv, texts=texts)
        try:
            with mp.get_context("fork").Pool(n_jobs) as pool:
                parts = pool.map(_transform_slice, bounds)
        finally:
            _POOL_STATE.clear()
    else:
        parts = [hv.transform(texts[a:b].tolist()).tocsr() for a, b in bounds]
    if not parts:
        return sp.csr_matrix((0, hv.n_features), dtype=np.float32)
    return sp.vstack(parts, format="csr")


def _hasher(analyzer: str, ngram: tuple[int, int], binary: bool = False) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=1 << 20,
        analyzer=analyzer,
        ngram_range=ngram,
        alternate_sign=False,
        norm=None,
        binary=binary,
        lowercase=False,
        token_pattern=r"\S+" if analyzer == "word" else None,
        dtype=np.float32,
    )


def vectorize(s1_text: dict[str, np.ndarray], s23_text: dict[str, np.ndarray], n_jobs: int = 1) -> tuple[dict, dict]:
    """Hash every record once. Cosine vectors are IDF-weighted (IDF from Source 1) and
    L2-normalized; Jaccard vectors are binary token sets."""
    v1: dict[str, sp.csr_matrix] = {}
    v23: dict[str, sp.csr_matrix] = {}
    for name, field, analyzer, ngram in _COSINE:
        hv = _hasher(analyzer, ngram)
        m1 = _transform(hv, s1_text[field], n_jobs)
        m23 = _transform(hv, s23_text[field], n_jobs)
        df = np.bincount(m1.indices, minlength=hv.n_features).astype(np.float32)
        idf = (np.log((m1.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        for m in (m1, m23):
            m.data *= idf[m.indices]
        v1[name] = normalize(m1, copy=False)
        v23[name] = normalize(m23, copy=False)
    binary = _hasher("word", (1, 1), binary=True)
    for field in _JACCARD_FIELDS:
        v1[f"jac_{field}"] = _transform(binary, s1_text[field], n_jobs)
        v23[f"jac_{field}"] = _transform(binary, s23_text[field], n_jobs)
    for field in _IDF_FIELDS:
        m1 = v1[f"jac_{field}"]
        df = np.bincount(m1.indices, minlength=m1.shape[1]).astype(np.float32)
        idf = (np.log((m1.shape[0] + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        weighted = m1.copy()
        weighted.data *= idf[weighted.indices]
        v1[f"idf_{field}"] = weighted
        v23[f"idf_{field}"] = v23[f"jac_{field}"]
    return v1, v23


def _rowdot(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def _jaccard(a: sp.csr_matrix, b: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    inter = _rowdot(a, b)
    na = np.diff(a.indptr).astype(np.float32)
    nb = np.diff(b.indptr).astype(np.float32)
    union = na + nb - inter
    jac = np.divide(inter, union, out=np.zeros_like(union), where=union > 0)
    small = np.minimum(na, nb)
    share = np.divide(inter, small, out=np.zeros_like(small), where=small > 0)
    return jac.astype(np.float32), share.astype(np.float32)


def _first(values: np.ndarray) -> np.ndarray:
    return np.array([v.split(" ", 1)[0] if v else "" for v in values], dtype=object)


def _pair_block(q_text: dict[str, np.ndarray], c_text: dict[str, np.ndarray], blocks: dict[str, np.ndarray], q_vec: dict, c_vec: dict) -> np.ndarray:
    n = len(q_text["name_core"])
    out = np.zeros((n, len(PAIR_NAMES)), np.float32)
    col = 0
    empty = {f: (q_text[f] == "") | (c_text[f] == "") for f in TEXT_FIELDS}
    for _, field, scorer, scale in _STRING_METRICS:
        vals = process.cpdist(q_text[field], c_text[field], scorer=scorer, workers=_CPDIST_WORKERS["n"], dtype=np.float32)
        vals = vals * np.float32(scale)
        vals[empty[field]] = 0.0
        out[:, col] = vals
        col += 1
    shared = None
    for field in _JACCARD_FIELDS:
        jac, share = _jaccard(q_vec[f"jac_{field}"], c_vec[f"jac_{field}"])
        out[:, col] = jac
        col += 1
        if field == "addr_nums":
            shared = share
    out[:, col] = shared
    col += 1
    for name, _, _, _ in _COSINE:
        out[:, col] = _rowdot(q_vec[name], c_vec[name])
        col += 1
    for field in _IDF_FIELDS:
        out[:, col] = _rowdot(q_vec[f"idf_{field}"], c_vec[f"idf_{field}"])
        col += 1
    for name in BLOCK_SCORES:
        out[:, col] = blocks[name]
        col += 1
    qa, ca = q_text["addr_norm"], c_text["addr_norm"]
    qe, ce = qa == "", ca == ""
    out[:, col] = qe & ce
    out[:, col + 1] = qe != ce
    fq, fc = _first(q_text["name_core"]), _first(c_text["name_core"])
    out[:, col + 2] = (fq == fc) & (fq != "")
    nq, nc = _first(q_text["addr_nums"]), _first(c_text["addr_nums"])
    out[:, col + 3] = (nq == nc) & (nq != "")
    out[:, col + 4] = np.fromiter(
        (len(a) >= 5 and len(b) >= 5 and (a in b or b in a) for a, b in zip(q_text["name_compact"], c_text["name_compact"])),
        dtype=np.float32,
        count=n,
    )
    lq = np.fromiter((len(x) for x in q_text["name_core"]), np.float32, n)
    lc = np.fromiter((len(x) for x in c_text["name_core"]), np.float32, n)
    out[:, col + 5] = np.divide(np.minimum(lq, lc), np.maximum(lq, lc), out=np.zeros(n, np.float32), where=np.maximum(lq, lc) > 0)
    aq = np.fromiter((len(x) for x in qa), np.float32, n)
    ac = np.fromiter((len(x) for x in ca), np.float32, n)
    out[:, col + 6] = np.divide(np.minimum(aq, ac), np.maximum(aq, ac), out=np.zeros(n, np.float32), where=np.maximum(aq, ac) > 0)
    out[:, col + 7] = np.fromiter((x.count(" ") + 1 if x else 0 for x in q_text["name_core"]), np.float32, n)
    out[:, col + 8] = np.fromiter((x.count(" ") + 1 if x else 0 for x in c_text["name_core"]), np.float32, n)
    col += len(_FLAGS)
    if col != len(PAIR_NAMES) - 1:
        raise RuntimeError(f"pair feature width {col} != {len(PAIR_NAMES) - 1}")
    idx = {name: i for i, name in enumerate(PAIR_NAMES)}
    out[:, col] = (
        out[:, idx["jw_name_core"]]
        + out[:, idx["lev_name_compact"]]
        + out[:, idx["cos_name_char"]]
        + out[:, idx["set_addr_norm"]]
        + out[:, idx["cos_addr_char"]]
        + out[:, idx["jac_addr_nums"]]
    ) / np.float32(6.0)
    return out


def group_rank_gap(key: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank (0 = best) and gap to the best *other* member, within groups of ``key``."""
    n = len(key)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    order = np.lexsort((-value, key))
    k_sorted = key[order]
    v_sorted = value[order]
    start_mask = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    starts = np.flatnonzero(start_mask)
    sizes = np.diff(np.r_[starts, n])
    first = np.repeat(starts, sizes)
    rank_sorted = np.arange(n) - first
    best = v_sorted[first]
    second_idx = np.minimum(first + 1, n - 1)
    has_second = np.repeat(sizes > 1, sizes)
    second = np.where(has_second, v_sorted[second_idx], 0.0)
    gap_sorted = np.where(rank_sorted == 0, v_sorted - second, v_sorted - best)
    rank = np.empty(n, np.float32)
    gap = np.empty(n, np.float32)
    rank[order] = rank_sorted
    gap[order] = gap_sorted
    return rank, gap


def add_group_features(q: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """Within-query comparisons. ``pair`` columns follow PAIR_NAMES."""
    n = len(q)
    out = np.zeros((n, len(GROUP_NAMES)), np.float32)
    idx = {name: i for i, name in enumerate(PAIR_NAMES)}
    g = len(_GROUP_ON)
    for i, name in enumerate(_GROUP_ON):
        rank, gap = group_rank_gap(q, pair[:, idx[name]])
        out[:, i] = rank
        out[:, g + i] = gap
    _, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    out[:, 2 * g] = counts[inverse]
    return out


_CPDIST_WORKERS = {"n": -1}
_FEAT_STATE: dict = {}


def _chunk_features(bounds: tuple[int, int]) -> np.ndarray:
    a, b = bounds
    st = _FEAT_STATE
    qi, ci = st["q"][a:b], st["c"][a:b]
    q_text = {f: st["s23_text"][f][qi] for f in TEXT_FIELDS}
    c_text = {f: st["s1_text"][f][ci] for f in TEXT_FIELDS}
    q_vec = {k: m[qi] for k, m in st["s23_vec"].items()}
    c_vec = {k: m[ci] for k, m in st["s1_vec"].items()}
    part = {k: v[a:b] for k, v in st["blocks"].items()}
    pair = _pair_block(q_text, c_text, part, q_vec, c_vec)
    return np.hstack([pair, add_group_features(qi, pair)])


def query_aligned_bounds(q: np.ndarray, chunk_size: int) -> list[tuple[int, int]]:
    """Slices of roughly ``chunk_size`` pairs that never split one query's candidates."""
    n = len(q)
    bounds = []
    start = 0
    while start < n:
        stop = min(start + chunk_size, n)
        if stop < n:
            stop = int(np.searchsorted(q, q[stop - 1], side="right"))
        bounds.append((start, stop))
        start = stop
    return bounds


def build_features(
    s1_text: dict[str, np.ndarray],
    s23_text: dict[str, np.ndarray],
    s1_vec: dict,
    s23_vec: dict,
    q: np.ndarray,
    c: np.ndarray,
    blocks: dict[str, np.ndarray],
    chunk_size: int = 500_000,
    n_jobs: int = 1,
) -> np.ndarray:
    """Return ``(n_pairs, len(FEATURE_NAMES))`` float32, aligned with ``q`` / ``c``.

    ``q`` must be sorted so each query's candidates are contiguous.
    """
    n = len(q)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    t0 = time.time()
    bounds = query_aligned_bounds(q, chunk_size)
    _FEAT_STATE.update(s1_text=s1_text, s23_text=s23_text, s1_vec=s1_vec, s23_vec=s23_vec, q=q, c=c, blocks=blocks)
    matrix = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    try:
        if n_jobs > 1 and len(bounds) > 1 and "fork" in mp.get_all_start_methods():
            _CPDIST_WORKERS["n"] = 2
            with mp.get_context("fork").Pool(n_jobs) as pool:
                for i, ((a, b), part) in enumerate(zip(bounds, pool.imap(_chunk_features, bounds))):
                    matrix[a:b] = part
                    if i % 20 == 0 or b == n:
                        print(f"  features {b:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
        else:
            for i, (a, b) in enumerate(bounds):
                matrix[a:b] = _chunk_features((a, b))
                if i % 20 == 0 or b == n:
                    print(f"  features {b:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        _FEAT_STATE.clear()
        _CPDIST_WORKERS["n"] = -1
    return matrix
