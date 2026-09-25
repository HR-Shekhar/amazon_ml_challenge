# Business entity resolution

Matches every Source 1 business to the Source 2 and Source 3 records that refer to the same entity. The pipeline only reads the provided TSV files. It does not call external APIs, geocoders, or business registries.

LightGBM (Apache-2.0) is the matcher. A run uses well under a million parameters, inside the 8B-parameter / MIT-or-Apache-2.0 rule.

## Layout

```
src/data_loading.py        load TSV files, normalize names and addresses
src/blocking.py            token blocking + TF-IDF neighbours, plus recall_check
src/features.py            Jaccard, Levenshtein, Jaro-Winkler, TF-IDF cosine
src/model.py               LightGBM train / save / predict
src/scoring.py             macro F_0.5 (the challenge formula)
src/threshold_tuning.py    pick the cutoff on held-out entities
src/inference.py           score the test split and write both TSV files
src/main.py                train -> tune -> infer -> validate
```

Country is treated as an open string. Nothing in the code branches on `{US, India}`.

## Setup

From this directory (`code/business_entity_resolution`):

```bash
pip install -r requirements.txt
```

The data directory must contain `train/` and `test/` with the challenge TSV files. By default that is `student_resource/dataset`, two levels up from this folder.

## Smoke test (do this first)

`--max-rows N` reads the first N Source 1 records and, on the training split, keeps every Source 2/3 record that the ground truth links to them, plus some non-matches. The booster therefore sees both classes, and the run finishes in minutes instead of hours.

```bash
python src/main.py --max-rows 2000
```

On Kaggle, point the flags at the mounted input and the writable working directory:

```bash
python src/main.py \
    --data-dir /kaggle/input/<your-dataset>/dataset \
    --output-dir /kaggle/working/output \
    --cache-dir /kaggle/working/cache \
    --model-dir /kaggle/working/model \
    --max-rows 2000
```

What you should see, in order:

1. A blocking-recall report (`pair_recall` / `entity_recall`) before any model training. That recall is the ceiling for the matcher.
2. A held-out macro F_0.5 and the threshold that produced it (it will not be a hard-coded 0.5).
3. `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

The official validator is skipped on a `--max-rows` run, because those files only cover the slice. An internal check still enforces the format rules against that slice. To stop after the recall report:

```bash
python src/main.py --max-rows 2000 --block-only
```

If the files you uploaded are already a small cut of the data, omit `--max-rows`. The loader detects small files and reads them whole, and the official validator then runs against that test directory.

## Full run

```bash
python src/main.py
```

Same command from `student_resource`:

```bash
python code/business_entity_resolution/src/main.py
```

This writes:

* `student_resource/output/matching_results.tsv`
* `student_resource/output/candidate_pairs.tsv`

and then runs:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Normalized tables are cached under `artifacts/cache/` so a second run skips parsing. Expect on the order of one to three hours on a 12-core machine: the expensive steps are TF-IDF blocking over the full Source 2/3 files and the pairwise string features on the candidate set.

Useful flags: `--topn` (TF-IDF neighbours per Source 2/3 row, default 12), `--max-candidates` (cap per Source 1 entity, default 40), `--valid-frac` (held-out entities used to pick the threshold, default 0.2), `--seed`.

## How a match is decided

Blocking keeps, per country, rare shared name / skeleton / street-number keys and the closest TF-IDF neighbours. The model only scores that set, and `candidate_pairs.tsv` is exactly that set. A pair is written to `matching_results.tsv` when its probability is at least the held-out F_0.5 threshold. Source 1 entities with nothing above the threshold are left blank (the correct output for a singleton).
