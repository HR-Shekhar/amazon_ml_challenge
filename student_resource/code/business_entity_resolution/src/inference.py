"""Apply a trained matcher to one split and write the two submission files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import BlockResult, block
from features import build_feature_matrix
from model import predict_proba


def _entity_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame["entity_id"].astype(str).to_numpy()


def _lists_for(n_s1: int, s1_index: np.ndarray, s23_index: np.ndarray, s23_ids: np.ndarray, mask: np.ndarray) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(n_s1)]
    if mask.any():
        for row, col in zip(s1_index[mask].tolist(), s23_index[mask].tolist()):
            buckets[row].append(s23_ids[col])
    lists: list[list[str]] = []
    for bucket in buckets:
        lists.append(sorted(set(bucket)))
    return lists


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
                ids = [] if not rest.strip() else rest.split(",")
                rows[entity] = ids
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
    for entity in required_ids:
        m_ids = matched.get(entity, [])
        c_ids = candidates.get(entity, [])
        if len(m_ids) != len(set(m_ids)):
            errors.append(f"{entity}: duplicate id in matched_entity_ids")
        if len(c_ids) != len(set(c_ids)):
            errors.append(f"{entity}: duplicate id in candidate_entity_ids")
        for mid in m_ids + c_ids:
            if not mid.startswith(("S2-", "S3-")):
                errors.append(f"{entity}: id {mid} is not an S2-/S3- id")
                break
        if not set(m_ids) <= set(c_ids):
            errors.append(f"{entity}: a matched id is missing from candidate_entity_ids")
        if len(errors) > 12:
            break
    return errors


def run_inference(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    booster,
    threshold: float,
    matching_path: Path,
    candidate_path: Path,
    topn: int,
    max_candidates: int,
) -> BlockResult:
    """Block, score, and write both output files for this split. Returns the candidate pairs."""
    print(f"inference on {len(s1):,} source1 and {len(s23):,} source2/3 records", flush=True)
    candidates = block(s1, s23, topn=topn, max_candidates=max_candidates)
    features = build_feature_matrix(
        s1, s23, candidates.s1_index, candidates.s23_index, candidates.votes, candidates.tfidf
    )
    proba = predict_proba(booster, features)
    keep = proba >= threshold if len(proba) else np.zeros(0, dtype=bool)
    source1_ids = _entity_ids(s1)
    source23_ids = _entity_ids(s23)
    cand_lists = _lists_for(len(s1), candidates.s1_index, candidates.s23_index, source23_ids, np.ones(len(candidates), dtype=bool))
    match_lists = _lists_for(len(s1), candidates.s1_index, candidates.s23_index, source23_ids, keep)
    write_id_file(candidate_path, "candidate_entity_ids", source1_ids, cand_lists)
    write_id_file(matching_path, "matched_entity_ids", source1_ids, match_lists)
    problems = check_submission(matching_path, candidate_path, source1_ids.tolist())
    if problems:
        raise SystemExit("output format check failed:\n" + "\n".join(f"  - {p}" for p in problems))
    n_matched = sum(bool(row) for row in match_lists)
    print(f"wrote {matching_path}", flush=True)
    print(f"wrote {candidate_path}", flush=True)
    print(f"source1 rows: {len(s1):,}; rows with a match: {n_matched:,}; candidate pairs: {len(candidates):,}", flush=True)
    return candidates


def read_header_ok(path: Path) -> bool:
    """Used by tests; csv import keeps the module's dependency list obvious."""
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.readline().rstrip("\n").split("\t")[0] == "source1_entity_id" and csv.excel.delimiter == ","
