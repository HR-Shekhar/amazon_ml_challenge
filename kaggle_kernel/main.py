"""Train, tune, and run the entity-resolution pipeline.

From ``student_resource`` (defaults point at ``dataset/`` and ``output/`` next to this repo):

    python code/business_entity_resolution/src/main.py --max-rows 2000
    python code/business_entity_resolution/src/main.py

``--max-rows`` keeps the run small enough for a first Kaggle check: it still loads the
true matches of the sampled Source 1 entities, so the booster sees both classes.
Drop the flag for the full data. ``--block-only`` stops after printing blocking recall.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

SRC = Path(__file__).resolve().parent
PACKAGE = SRC.parent
STUDENT_RESOURCE = PACKAGE.parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blocking import block, format_recall, recall_check  # noqa: E402
from data_loading import load_split  # noqa: E402
from features import build_feature_matrix  # noqa: E402
from inference import run_inference  # noqa: E402
from model import make_labels, predict_proba, print_importance, save_model, train_model  # noqa: E402
from scoring import macro_f05, self_check  # noqa: E402
from threshold_tuning import tune_threshold  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--data-dir", type=Path, default=STUDENT_RESOURCE / "dataset")
    parser.add_argument("--output-dir", type=Path, default=STUDENT_RESOURCE / "output")
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE / "artifacts" / "cache")
    parser.add_argument("--model-dir", type=Path, default=PACKAGE / "artifacts" / "model")
    parser.add_argument("--max-rows", type=int, default=0, help="Source 1 rows to keep. 0 means the full files.")
    parser.add_argument("--distractor-factor", type=int, default=6, help="Extra Source 2/3 rows per Source 1 row, per file, when --max-rows is set.")
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--topn", type=int, default=12, help="TF-IDF neighbours kept per Source 2/3 record.")
    parser.add_argument("--max-candidates", type=int, default=40, help="Candidate cap per Source 1 entity (the set the model scores).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=4,
        help="Training negatives kept per positive. Held-out entities keep every candidate so F_0.5 stays exact.",
    )
    parser.add_argument("--block-only", action="store_true", help="Print training blocking recall and exit.")
    parser.add_argument("--validate-script", type=Path, default=STUDENT_RESOURCE / "utils" / "validate_submission.py")
    return parser.parse_args()


def _holdout(n_entities: int, valid_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_entities)
    n_valid = int(round(n_entities * valid_frac))
    n_valid = min(max(n_valid, 1), n_entities - 1)
    return perm[n_valid:], perm[:n_valid]


def _predictions(s1_index: np.ndarray, s23_index: np.ndarray, proba: np.ndarray, threshold: float) -> dict[int, set[int]]:
    predicted: dict[int, set[int]] = {}
    if len(proba) == 0:
        return predicted
    keep = proba >= threshold
    for row, col in zip(s1_index[keep].tolist(), s23_index[keep].tolist()):
        predicted.setdefault(int(row), set()).add(int(col))
    return predicted


def _release_ram() -> None:
    """Return freed Arrow and glibc pages so the next large array can allocate."""
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _fit_rows(pair_valid: np.ndarray, labels: np.ndarray, neg_ratio: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices for training (all positives, capped negatives) and the full holdout."""
    valid_idx = np.flatnonzero(pair_valid)
    train_pos = np.flatnonzero(~pair_valid & (labels == 1))
    train_neg = np.flatnonzero(~pair_valid & (labels == 0))
    n_neg = min(len(train_neg), max(neg_ratio, 1) * max(len(train_pos), 1))
    if len(train_neg) > n_neg:
        train_neg = np.random.default_rng(seed).choice(train_neg, size=n_neg, replace=False)
    if len(train_pos) or len(train_neg):
        train_idx = np.concatenate([train_pos, train_neg])
    else:
        train_idx = np.empty(0, np.int64)
    train_idx.sort()
    valid_idx.sort()
    return train_idx, valid_idx


def _validate_official(script: Path, matching: Path, candidate: Path, test_dir: Path) -> None:
    if not script.is_file():
        print(f"validator not found at {script}; skipped", flush=True)
        return
    print(f"running {script.name}", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matching",
            str(matching),
            "--candidate",
            str(candidate),
            "--test-dir",
            str(test_dir),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"validate_submission.py exited {completed.returncode}")


def main() -> None:
    args = _parse()
    self_check()
    max_rows = args.max_rows or None
    t0 = time.time()
    print("loading train", flush=True)
    train = load_split(
        args.data_dir,
        "train",
        cache_dir=args.cache_dir,
        max_rows=max_rows,
        distractor_factor=args.distractor_factor,
    )
    if train.truth is None:
        raise SystemExit(f"no ground truth under {args.data_dir / 'train'}")
    print(f"train source1={len(train.s1):,} source2/3={len(train.s23):,} in {time.time() - t0:.0f}s", flush=True)

    print("generating training candidates", flush=True)
    blocked = block(train.s1, train.s23, topn=args.topn, max_candidates=args.max_candidates)
    countries = train.s1["country"].astype(str).to_numpy()
    report = recall_check(blocked, train.truth, countries)
    print(format_recall(report), flush=True)
    if report["true_pairs"] and report["pair_recall"] < 0.9:
        print("WARNING: blocking pair recall is under 0.90. The matcher cannot recover pairs that blocking dropped.", flush=True)
    if args.block_only:
        return

    print("building training features", flush=True)
    n_entities = len(train.s1)
    features = build_feature_matrix(
        train.s1, train.s23, blocked.s1_index, blocked.s23_index, blocked.votes, blocked.tfidf
    )
    labels = make_labels(blocked.s1_index, blocked.s23_index, train.truth)
    print(f"candidate pairs={len(labels):,} positives={int(labels.sum()):,}", flush=True)
    # The text tables are no longer needed, and keeping them beside a copied feature
    # matrix is what got the full run SIGKILL'd (~76M rows x 32 floats, copied again).
    train.s1 = pd.DataFrame()
    train.s23 = pd.DataFrame()
    _release_ram()

    _, valid_entities = _holdout(n_entities, args.valid_frac, args.seed)
    is_valid = np.zeros(n_entities, dtype=bool)
    is_valid[valid_entities] = True
    pair_valid = is_valid[blocked.s1_index] if len(blocked) else np.zeros(0, dtype=bool)
    train_idx, valid_idx = _fit_rows(pair_valid, labels, args.neg_ratio, args.seed)
    print(
        f"fitting on {len(train_idx):,} pairs ({int((labels[train_idx] == 1).sum()) if len(train_idx) else 0:,} positives, "
        f"negatives capped at {args.neg_ratio} per positive); scoring all {len(valid_idx):,} held-out pairs",
        flush=True,
    )
    train_x = features[train_idx]
    train_y = labels[train_idx]
    valid_x = features[valid_idx]
    valid_y = labels[valid_idx]
    valid_s1 = blocked.s1_index[valid_idx]
    valid_s23 = blocked.s23_index[valid_idx]
    del features, labels, train_idx, valid_idx, pair_valid, is_valid
    _release_ram()
    booster = train_model(train_x, train_y, valid_x, valid_y, seed=args.seed)
    del train_x, train_y
    _release_ram()
    print_importance(booster)

    valid_proba = predict_proba(booster, valid_x)
    del valid_x, valid_y
    _release_ram()
    threshold, heldout = tune_threshold(
        valid_s1,
        valid_s23,
        valid_proba,
        train.truth,
        valid_entities,
    )
    # Score the holdout once more with the public macro function so the log shows the same definition.
    truth_sets = {int(k): set(np.asarray(train.truth.get(int(k), ())).tolist()) for k in valid_entities.tolist()}
    predicted = _predictions(valid_s1, valid_s23, valid_proba, threshold)
    confirm = macro_f05(predicted, truth_sets, valid_entities)
    print(f"held-out macro F_0.5 (confirmed) = {confirm:.4f}  threshold = {threshold:.2f}", flush=True)
    if abs(confirm - heldout) > 1e-6:
        print(f"WARNING: threshold sweep score {heldout:.4f} disagrees with macro_f05 {confirm:.4f}", flush=True)
    save_model(booster, threshold, args.model_dir)
    del blocked, valid_s1, valid_s23, valid_proba, predicted, truth_sets
    _release_ram()

    print("loading test", flush=True)
    test = load_split(
        args.data_dir,
        "test",
        cache_dir=args.cache_dir,
        max_rows=max_rows,
        distractor_factor=args.distractor_factor,
    )
    matching = args.output_dir / "matching_results.tsv"
    candidate = args.output_dir / "candidate_pairs.tsv"
    run_inference(test.s1, test.s23, booster, threshold, matching, candidate, args.topn, args.max_candidates)

    if test.limited:
        print(
            "Official validator was not run: --max-rows does not emit a row for every Source 1 id in the full test file. "
            "The files were checked against the slice. Re-run without --max-rows before submitting.",
            flush=True,
        )
    else:
        _validate_official(args.validate_script, matching, candidate, args.data_dir / "test")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
