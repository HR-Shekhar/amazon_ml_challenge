"""Apply the matchers to one split and write the two submission files."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import Candidates, add_source1_scores, block
from features import build_features, text_columns, vectorize
from features import BLIND_INDEX, PRIMARY_INDEX
from model import predict_proba
from threshold_tuning import aligned_picks, apply_rule


def _entity_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame["entity_id"].astype(str).to_numpy()


def _lists_for(n_s1: int, s1_rows: np.ndarray, s23_rows: np.ndarray, s23_ids: np.ndarray) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(n_s1)]
    for row, col in zip(s1_rows.tolist(), s23_rows.tolist()):
        buckets[row].append(s23_ids[col])
    return [sorted(set(bucket)) for bucket in buckets]


def write_id_file(path: Path, id_column: str, source1_ids: np.ndarray, id_lists: list[list[str]]) -> None:
    """Tab-separated, UTF-8, LF line endings, no quoting, empty string when there are no ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"source1_entity_id\t{id_column}\n")
        for entity_id, linked in zip(source1_ids.tolist(), id_lists):
            handle.write(f"{entity_id}\t{','.join(linked)}\n")


def check_submission(matching_path: Path, candidate_path: Path, required_ids: list[str]) -> list[str]:
    """Return a list of format problems. Empty means the files follow the output rules."""

    def _load(path: Path, column: str) -> dict[str, list[str]]:
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            if header != ["source1_entity_id", column]:
                raise ValueError(f"{path.name}: header {header}")
            for line in handle:
                entity, sep, rest = line.rstrip("\n").partition("\t")
                if not sep:
                    raise ValueError(f"{path.name}: row without a tab: {line!r}")
                if entity in rows:
                    raise ValueError(f"{path.name}: duplicate source1 id {entity}")
                rows[entity] = [] if not rest.strip() else rest.split(",")
        return rows

    errors: list[str] = []
    try:
        matched = _load(matching_path, "matched_entity_ids")
        candidates = _load(candidate_path, "candidate_entity_ids")
    except ValueError as exc:
        return [str(exc)]
    if list(matched) != required_ids:
        errors.append("matching_results.tsv rows are not exactly the Source 1 entities, in order, once each.")
    if list(candidates) != required_ids:
        errors.append("candidate_pairs.tsv rows are not exactly the Source 1 entities, in order, once each.")
    seen: set[str] = set()
    for entity in required_ids:
        m_ids = matched.get(entity, [])
        c_ids = candidates.get(entity, [])
        if len(m_ids) != len(set(m_ids)) or len(c_ids) != len(set(c_ids)):
            errors.append(f"{entity}: duplicate id in a list")
        if any(not x.startswith(("S2-", "S3-")) for x in m_ids + c_ids):
            errors.append(f"{entity}: an id is not an S2-/S3- id")
        if not set(m_ids) <= set(c_ids):
            errors.append(f"{entity}: a matched id is missing from candidate_entity_ids")
        if seen.intersection(m_ids):
            errors.append(f"{entity}: a Source 2/3 id is matched to more than one Source 1 entity")
        seen.update(m_ids)
        if len(errors) > 12:
            break
    return errors


def score_split(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    boosters,
    blind_boosters,
    topn: int,
    max_candidates: int,
    n_jobs: int,
) -> tuple[Candidates, np.ndarray, np.ndarray]:
    """Block, featurize, and score every candidate pair with both models."""
    cands = block(s1, s23, topn=topn, max_candidates=max_candidates, n_jobs=n_jobs)
    add_source1_scores(cands)
    t1, t23 = text_columns(s1), text_columns(s23)
    v1, v23 = vectorize(t1, t23, n_jobs)
    features = build_features(t1, t23, v1, v23, cands.q, cands.c, cands.scores, n_jobs=n_jobs)
    del t1, t23, v1, v23
    gc.collect()
    proba = predict_proba(boosters, features, columns=PRIMARY_INDEX)
    blind = predict_proba(blind_boosters, features, columns=BLIND_INDEX) if blind_boosters else np.zeros(len(features), np.float32)
    del features
    gc.collect()
    return cands, proba, blind


def write_submission(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    cands: Candidates,
    proba: np.ndarray,
    threshold: float,
    matching_path: Path,
    candidate_path: Path,
    min_gap: float = 0.0,
    scores_path: Path | None = None,
    blind: np.ndarray | None = None,
    override_gap: float = -1.0,
    blind_margin: float = 0.0,
    blind_threshold: float = 0.5,
) -> None:
    if blind is None:
        blind = proba
    bq, bc, bp, gap, lc, lp, lg = aligned_picks(cands.q, cands.c, proba, blind)
    bc, keep, flip = apply_rule(bc, bp, gap, lc, lp, lg, threshold, min_gap, override_gap, blind_margin, blind_threshold)
    print(f"rank-blind flips kept: {int((keep & flip).sum()):,}", flush=True)
    source1_ids = _entity_ids(s1)
    source23_ids = _entity_ids(s23)
    cand_lists = _lists_for(len(s1), cands.c, cands.q, source23_ids)
    match_lists = _lists_for(len(s1), bc[keep], bq[keep], source23_ids)
    write_id_file(candidate_path, "candidate_entity_ids", source1_ids, cand_lists)
    write_id_file(matching_path, "matched_entity_ids", source1_ids, match_lists)
    if scores_path is not None:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("s23_entity_id\ts1_entity_id\tproba\tgap\n")
            reported_p = np.where(flip, lp, bp)
            reported_g = np.where(flip, lg, gap)
            for qi, ci, p, g in zip(bq.tolist(), bc.tolist(), reported_p.tolist(), reported_g.tolist()):
                handle.write(f"{source23_ids[qi]}\t{source1_ids[ci]}\t{p:.6f}\t{g:.6f}\n")
        print(f"wrote {scores_path}", flush=True)
    problems = check_submission(matching_path, candidate_path, source1_ids.tolist())
    if problems:
        raise SystemExit("output format check failed:\n" + "\n".join(f"  - {p}" for p in problems))
    n_matched = sum(bool(row) for row in match_lists)
    print(f"wrote {matching_path}", flush=True)
    print(f"wrote {candidate_path}", flush=True)
    print(
        f"source1 rows: {len(s1):,}; rows with a match: {n_matched:,}; links: {int(keep.sum()):,}; candidate pairs: {len(cands):,}",
        flush=True,
    )
