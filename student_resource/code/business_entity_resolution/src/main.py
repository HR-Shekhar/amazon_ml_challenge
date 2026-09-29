"""Train, tune, and run the entity-resolution pipeline.

From ``student_resource`` (defaults point at ``dataset/`` and ``output/`` next to this repo):

    python code/business_entity_resolution/src/main.py --max-rows 2000
    python code/business_entity_resolution/src/main.py

Each Source 2/3 record belongs to at most one Source 1 entity, so the pipeline works
per Source 2/3 record: block its nearest Source 1 records, score each pair, and link the
record to its best candidate when the probability clears a threshold tuned for macro
F_0.5. Training uses two folds split by Source 1 entity, so every training pair gets an
out-of-fold probability and the threshold is tuned on the exact challenge metric.

``--max-rows`` keeps a small, truth-preserving slice for a quick check.
``--block-only`` stops after printing blocking recall.
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
import pyarrow as pa

SRC = Path(__file__).resolve().parent
PACKAGE = SRC.parent
# Repo layout is student_resource/code/<package>/src. On Modal the files sit in /root/pipeline.
STUDENT_RESOURCE = PACKAGE.parents[1] if len(PACKAGE.parents) > 1 else PACKAGE
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blocking import add_source1_scores, block, owner_array, recall_report  # noqa: E402
from data_loading import load_split  # noqa: E402
from features import BLIND_INDEX, BLIND_NAMES, FEATURE_NAMES, PRIMARY_INDEX, PRIMARY_NAMES, build_features, text_columns, vectorize  # noqa: E402
from inference import score_split, write_submission  # noqa: E402
from model import disagreement_weights, predict_proba, print_importance, save_models, train_model  # noqa: E402
from scoring import macro_f05, self_check  # noqa: E402
from threshold_tuning import aligned_picks, apply_rule, tune_override, tune_threshold  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--data-dir", type=Path, default=STUDENT_RESOURCE / "dataset")
    parser.add_argument("--output-dir", type=Path, default=STUDENT_RESOURCE / "output")
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE / "artifacts" / "cache")
    parser.add_argument("--model-dir", type=Path, default=PACKAGE / "artifacts" / "model")
    parser.add_argument("--max-rows", type=int, default=0, help="Source 1 rows to keep. 0 means the full files.")
    parser.add_argument("--distractor-factor", type=int, default=6, help="Extra Source 2/3 rows per Source 1 row, per file, when --max-rows is set.")
    parser.add_argument("--topn", type=int, default=20, help="TF-IDF neighbours per Source 2/3 record, per search.")
    parser.add_argument("--max-candidates", type=int, default=12, help="Candidates kept per Source 2/3 record (the set the model scores).")
    parser.add_argument("--fit-frac", type=float, default=0.5, help="Share of each fold's queries used to fit its booster.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=0, help="Worker processes. 0 uses CPU count minus one.")
    parser.add_argument("--block-only", action="store_true", help="Print training blocking recall and exit.")
    parser.add_argument("--validate-script", type=Path, default=STUDENT_RESOURCE / "utils" / "validate_submission.py")
    return parser.parse_args()


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


def _validate_official(script: Path, matching: Path, candidate: Path, test_dir: Path) -> None:
    if not script.is_file():
        print(f"validator not found at {script}; skipped", flush=True)
        return
    print(f"running {script.name}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(script), "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir)],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"validate_submission.py exited {completed.returncode}")


def _cols(features: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros((0, len(cols)), np.float32)
    return np.ascontiguousarray(features[np.ix_(rows, cols)])


def _confirm_with_sets(q, c, proba, blind, truth, n_s1, threshold, min_gap, override_gap, blind_margin, blind_threshold) -> float:
    """Score the chosen links with the set-based scorer, as a check on the fast one."""
    bq, bc, bp, gap, lc, lp, lg = aligned_picks(q, c, proba, blind)
    chosen, keep, _flip = apply_rule(bc, bp, gap, lc, lp, lg, threshold, min_gap, override_gap, blind_margin, blind_threshold)
    predicted: dict[int, set[int]] = {}
    for row, col in zip(chosen[keep].tolist(), bq[keep].tolist()):
        predicted.setdefault(row, set()).add(col)
    truth_sets = {i: set(np.asarray(truth.get(i, ())).tolist()) for i in range(n_s1)}
    return macro_f05(predicted, truth_sets, np.arange(n_s1))


def main() -> None:
    args = _parse()
    self_check()
    max_rows = args.max_rows or None
    n_jobs = args.n_jobs or None
    jobs = n_jobs or 1
    if n_jobs is None:
        import os

        jobs = max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()

    print("loading train", flush=True)
    train = load_split(args.data_dir, "train", cache_dir=args.cache_dir, n_jobs=n_jobs, max_rows=max_rows, distractor_factor=args.distractor_factor)
    if train.truth is None:
        raise SystemExit(f"no ground truth under {args.data_dir / 'train'}")
    print(f"train source1={len(train.s1):,} source2/3={len(train.s23):,} in {time.time() - t0:.0f}s", flush=True)
    n_s1 = len(train.s1)
    owner = owner_array(len(train.s23), train.truth)
    true_count = np.bincount(owner[owner >= 0], minlength=n_s1)

    print("generating training candidates", flush=True)
    cands = block(train.s1, train.s23, topn=args.topn, max_candidates=args.max_candidates, n_jobs=jobs)
    print(recall_report(cands, owner, train.s23["country"].astype(str).to_numpy()), flush=True)
    if args.block_only:
        return
    add_source1_scores(cands)

    print("building training features", flush=True)
    t1, t23 = text_columns(train.s1), text_columns(train.s23)
    v1, v23 = vectorize(t1, t23, jobs)
    features = build_features(t1, t23, v1, v23, cands.q, cands.c, cands.scores, n_jobs=jobs)
    del t1, t23, v1, v23
    _release_ram()
    labels = (owner[cands.q] == cands.c).astype(np.int8)
    print(f"candidate pairs={len(labels):,} positives={int(labels.sum()):,} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    s1_fold = rng.integers(0, 2, n_s1)
    q_fold = np.where(owner >= 0, s1_fold[np.maximum(owner, 0)], rng.integers(0, 2, len(owner)))
    q_fit = rng.random(len(owner)) < args.fit_frac
    q_stop = rng.random(len(owner)) < 0.04
    pair_fold = q_fold[cands.q]
    rank_col = FEATURE_NAMES.index("rank_quick")
    oof = np.zeros(len(labels), np.float32)
    oof_blind = np.zeros(len(labels), np.float32)
    boosters = []
    blinds = []
    for fold in (0, 1):
        fit_idx = np.flatnonzero((pair_fold == fold) & q_fit[cands.q])
        stop_idx = np.flatnonzero((pair_fold != fold) & q_stop[cands.q])
        print(f"fold {fold}: fitting on {len(fit_idx):,} pairs, early stopping on {len(stop_idx):,}", flush=True)
        booster = train_model(
            _cols(features, fit_idx, PRIMARY_INDEX),
            labels[fit_idx],
            _cols(features, stop_idx, PRIMARY_INDEX),
            labels[stop_idx],
            seed=args.seed + fold,
            feature_names=PRIMARY_NAMES,
        )
        _release_ram()
        other = np.flatnonzero(pair_fold != fold)
        oof[other] = predict_proba([booster], features[other], columns=PRIMARY_INDEX)
        train_w = disagreement_weights(cands.q[fit_idx], labels[fit_idx], features[fit_idx, rank_col])
        valid_w = disagreement_weights(cands.q[stop_idx], labels[stop_idx], features[stop_idx, rank_col]) if len(stop_idx) else None
        print(f"fold {fold}: rank-blind heavy rows {int((train_w > 1).sum()):,} of {len(train_w):,}", flush=True)
        blind = train_model(
            _cols(features, fit_idx, BLIND_INDEX),
            labels[fit_idx],
            _cols(features, stop_idx, BLIND_INDEX),
            labels[stop_idx],
            seed=args.seed + fold + 10,
            feature_names=BLIND_NAMES,
            weights=train_w,
            valid_weights=valid_w,
        )
        oof_blind[other] = predict_proba([blind], features[other], columns=BLIND_INDEX)
        del fit_idx, stop_idx, other, train_w, valid_w
        _release_ram()
        boosters.append(booster)
        blinds.append(blind)
        print(f"fold {fold} done ({time.time() - t0:.0f}s)", flush=True)
    print("primary model", flush=True)
    print_importance(boosters[0])
    print("rank-blind model", flush=True)
    print_importance(blinds[0])
    del features
    _release_ram()

    threshold, min_gap, primary_heldout = tune_threshold(cands.q, cands.c, oof, owner, true_count)
    tune_rows = np.flatnonzero(s1_fold == 0)
    eval_rows = np.flatnonzero(s1_fold == 1)
    override_gap, blind_margin, blind_threshold, heldout = tune_override(
        cands.q, cands.c, oof, oof_blind, owner, true_count, threshold, min_gap, tune_rows, eval_rows
    )
    confirm = _confirm_with_sets(
        cands.q, cands.c, oof, oof_blind, train.truth, n_s1, threshold, min_gap, override_gap, blind_margin, blind_threshold
    )
    print(
        f"out-of-fold macro F_0.5 (set-based check) = {confirm:.5f}  threshold = {threshold:.3f} "
        f"min_gap = {min_gap:.2f} override_gap = {override_gap:.2f} (primary-only {primary_heldout:.5f})",
        flush=True,
    )
    if abs(confirm - heldout) > 1e-4:
        print(f"WARNING: fast scorer {heldout:.5f} disagrees with set scorer {confirm:.5f}", flush=True)
    countries = train.s1["country"].astype(str).to_numpy()
    bq, bc, bp, gap, lc, lp, lg = aligned_picks(cands.q, cands.c, oof, oof_blind)
    chosen, keep, _flip = apply_rule(bc, bp, gap, lc, lp, lg, threshold, min_gap, override_gap, blind_margin, blind_threshold)
    bc = chosen
    for country in sorted(set(countries.tolist())):
        rows = np.flatnonzero(countries == country)
        sub = {int(i): set() for i in rows}
        for row, col in zip(bc[keep].tolist(), bq[keep].tolist()):
            if row in sub:
                sub[row].add(col)
        truth_sets = {int(i): set(np.asarray(train.truth.get(int(i), ())).tolist()) for i in rows}
        print(f"  macro F_0.5 [{country}] = {macro_f05(sub, truth_sets, rows):.5f}", flush=True)
    non_ascii = r"[^\x00-\x7F]"
    s1_script = train.s1["name_raw"].str.contains(non_ascii, regex=True, na=False).to_numpy()
    s23_script = train.s23["name_raw"].str.contains(non_ascii, regex=True, na=False).to_numpy()
    for row, linked in train.truth.items():
        if len(linked) and s23_script[np.asarray(linked, np.int64)].any():
            s1_script[int(row)] = True
    script_rows = np.flatnonzero(s1_script)
    if len(script_rows):
        sub = {int(i): set() for i in script_rows}
        wanted = set(script_rows.tolist())
        for row, col in zip(bc[keep].tolist(), bq[keep].tolist()):
            if row in wanted:
                sub[row].add(col)
        truth_sets = {int(i): set(np.asarray(train.truth.get(int(i), ())).tolist()) for i in script_rows}
        print(f"  macro F_0.5 [script-shifted names] = {macro_f05(sub, truth_sets, script_rows):.5f} n={len(script_rows):,}", flush=True)
    save_models(
        boosters,
        threshold,
        args.model_dir,
        min_gap=min_gap,
        blind_boosters=blinds,
        override_gap=override_gap,
        blind_margin=blind_margin,
        blind_threshold=blind_threshold,
    )
    if abs(confirm - heldout) > 1e-3:
        print(f"scorers disagree ({heldout:.5f} vs {confirm:.5f}); test inference skipped", flush=True)
        return
    if heldout < 0.976 and not max_rows:
        print(
            f"held-out macro F_0.5 {heldout:.5f} is below 0.976; test inference skipped. "
            "Do not submit this run.",
            flush=True,
        )
        return
    del cands, oof, oof_blind, owner, train, bq, bc, bp, keep
    _release_ram()

    print("loading test", flush=True)
    test = load_split(args.data_dir, "test", cache_dir=args.cache_dir, n_jobs=n_jobs, max_rows=max_rows, distractor_factor=args.distractor_factor)
    print(f"inference on {len(test.s1):,} source1 and {len(test.s23):,} source2/3 records", flush=True)
    test_cands, proba, blind_proba = score_split(test.s1, test.s23, boosters, blinds, args.topn, args.max_candidates, jobs)
    matching = args.output_dir / "matching_results.tsv"
    candidate = args.output_dir / "candidate_pairs.tsv"
    write_submission(
        test.s1,
        test.s23,
        test_cands,
        proba,
        threshold,
        matching,
        candidate,
        min_gap=min_gap,
        scores_path=args.output_dir / "link_scores.tsv",
        blind=blind_proba,
        override_gap=override_gap,
        blind_margin=blind_margin,
        blind_threshold=blind_threshold,
    )

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
