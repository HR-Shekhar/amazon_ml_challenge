# Amazon ML Challenge 2026 — Business Entity Resolution

Match each clean Source 1 business to the Source 2 / Source 3 records that refer to the same real-world entity. Source 1 is the deduplicated reference. A Source 1 row may match zero, one, or many messy records. In the labels, every matched Source 2/3 id belongs to exactly one Source 1.

**Public score of the file we keep:** **0.968** macro F₀.₅  
**Submission file:** [`output2-v2/matching_results.tsv`](output2-v2/matching_results.tsv)

The official problem statement is in `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` and `student_resource/README.md`.

## Approach (the 0.968 system)

1. **Normalize text** — transliterate Brahmic scripts, fold accents, strip legal suffixes / honorifics / aliases. Keep `name_core`, a consonant skeleton, a space-free compact name, a normalized address, and address numbers.
2. **Block by country** — search only inside the same open country string. Do not hard-code `{US, India}`; France appears only in test.
3. **Candidate generation** — for every Source 2/3 row, retrieve the top **12** Source 1 rows by unioning:
   - word TF-IDF on name + address
   - character TF-IDF on compact name + address
   - exact keys (name tokens, skeleton tokens, compact name, prefix-8, address numbers, first-token+number; exact compact weight 20)
4. **Score pairs with LightGBM** — binary classifier on string similarities (Levenshtein, Jaro-Winkler, token sort/set/partial), Jaccard, shared address numbers, TF-IDF cosine, blocking scores, and within-list rank / gap features. Two fold models (50/50 by Source 1 entity); test uses the average.
5. **Decide with a tuned threshold** — link a query to its best Source 1 only if probability ≥ **0.690** (chosen on held-out macro F₀.₅, not 0.5). At most one Source 1 per Source 2/3 row.
6. **Emit the contest file** — one row per test Source 1 with the matched S2-/S3- ids.

Held-out macro F₀.₅ for this run was **0.97634** (link precision ≈ 0.994, link recall ≈ 0.946). Blocking recall on all true train links: r@1 ≈ 0.946, r@12 ≈ 0.976.

## Attempt history

| Attempt | Change | Result |
| --- | --- | --- |
| 1 | Source 1 as query; same S2/S3 id allowed on many S1 rows | Public **0.93** |
| 2 | Flip: S2/S3 as query, at most one S1; v2 blocking + LightGBM | Public **0.968** ← kept |
| 3 | Phonetic legal stripping, extra keys, IDF, min-gap | Holdout 0.970 → public **0.95** |
| 4 | Reserved token slots in the candidate list | Any-recall flat (~0.9756); no better submit |
| 5 | Second rank-blind LightGBM for close-call overrides | Override never accepted; holdout 0.97649 |

Later outputs under `output/`, `output3/`, `output5/` are experiments. Do not treat them as the leaderboard file.

## Repo layout

```
student_resource/code/business_entity_resolution/src/   # pipeline (source of truth)
  data_loading.py   load + normalize
  blocking.py       country-partitioned candidate generation
  features.py       pair + within-list features
  model.py          LightGBM train / predict
  scoring.py        macro F_0.5
  threshold_tuning.py
  inference.py      write matching_results.tsv + candidate_pairs.tsv
  main.py           end-to-end entrypoint

modal_train.py      full train on Modal (16 CPU / 128 GiB)
output2-v2/         final public 0.968 submission
student_resource/utils/validate_submission.py
```

`kaggle_code/` and `kaggle_kernel/` are older snapshots and may lag the src tree.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the challenge data at `student_resource/dataset/` with `train/` and `test/` TSV folders (gitignored).

## Run

Smoke test (small slice, minutes):

```bash
python student_resource/code/business_entity_resolution/src/main.py ^
  --max-rows 2000 ^
  --data-dir student_resource/dataset ^
  --output-dir scratch/smoke_out ^
  --model-dir scratch/smoke_model ^
  --cache-dir scratch/smoke_cache
```

Full local train needs a machine with enough RAM for ~124M candidate pairs. On Modal:

```bash
modal run --detach modal_train.py
```

Validate a submission (from `student_resource`):

```bash
python utils/validate_submission.py ^
  --matching path/to/matching_results.tsv ^
  --candidate path/to/candidate_pairs.tsv ^
  --test-dir dataset/test
```

## Constraints we followed

- Metric: macro F₀.₅ (precision weighted 2×)
- No external APIs, geocoders, or registries
- No hard-coded country set
- Matcher: LightGBM (Apache-2.0), far under the 8B-parameter limit
- Threshold tuned on holdout; never defaulted to 0.5

## License note

LightGBM is Apache-2.0. Pipeline code in this repo is for the challenge submission package.
