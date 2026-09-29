"""Load the challenge TSV files and derive normalized text fields.

Every record gets:

* ``name_raw`` / ``addr_raw``   - original text, lower-cased (the "raw" view)
* ``name_norm``                 - transliterated, accent-folded, punctuation-free name
* ``name_core``                 - ``name_norm`` without legal suffixes / honorifics /
                                  alias markers (including transliterations such as
                                  ``praivet`` / ``limatid``), with in-word digit typos repaired
* ``name_compact``              - ``name_core`` with spaces removed (catches
                                  ``bengalurufinvest.com`` vs ``Bengaluru Finvest``)
* ``addr_norm``                 - transliterated, accent-folded address with common
                                  street/unit abbreviations canonicalized
* ``addr_nums``                 - space-joined numeric tokens of the address
* ``name_skel``                 - phonetic consonant skeleton of ``name_core``; maps
                                  transliterations and vowel typos to one key
                                  (``praivet``/``private`` -> ``prvt``)

Nothing here depends on specific country labels: every rule is applied to every record.
"""

from __future__ import annotations

import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
NORMALIZATION_VERSION = "v2"

# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def read_tsv(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Read a challenge TSV as all-string columns with empty strings for missing values.

    Quoting is disabled: the files are plain TSV and names may contain quote characters.
    """
    table = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            include_columns=columns,
            strings_can_be_null=False,
        ),
    )
    df = table.to_pandas()
    return df.fillna("").astype(str)


# --------------------------------------------------------------------------------------
# Transliteration of Brahmic scripts
# --------------------------------------------------------------------------------------
# The nine major Indic scripts occupy parallel 128-codepoint Unicode blocks that share
# (ISCII-derived) layout, so one offset table transliterates all of them. Consonants
# carry an inherent vowel ("a"), removed before a vowel sign / virama and at word end.

_BRAHMIC_BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_INHERENT = "\x01"  # placeholder for the inherent vowel
_KILL = "\x02"  # removes a preceding inherent vowel

_INDEPENDENT_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL_SIGNS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4D: "", 0x55: "", 0x56: "ai", 0x57: "au", 0x62: "li", 0x63: "li",
}
_OTHER_SIGNS = {
    0x01: "n", 0x02: "n", 0x03: "h", 0x3C: "", 0x3D: "", 0x50: "om", 0x64: " ",
    0x65: " ", 0x70: "n", 0x71: "", 0x72: "", 0x73: "",
    0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k",
}


def _build_translit_table() -> dict[int, str]:
    table: dict[int, str] = {}
    for start in _BRAHMIC_BLOCK_STARTS:
        for off, s in _OTHER_SIGNS.items():
            table[start + off] = s
        for off, s in _INDEPENDENT_VOWELS.items():
            table[start + off] = s
        for off, s in _CONSONANTS.items():
            table[start + off] = s + _INHERENT
        for off, s in _VOWEL_SIGNS.items():
            table[start + off] = _KILL + s
        for d in range(10):
            table[start + 0x66 + d] = str(d)
    # Bengali khanda ta; zero-width joiners / non-joiners.
    table[0x09CE] = "t"
    table[0x200C] = ""
    table[0x200D] = ""
    return table


_TRANSLIT_TABLE = _build_translit_table()
_WORD_FINAL_INHERENT = re.compile(_INHERENT + r"(?![a-z])")


def to_ascii(text: str) -> str:
    """Transliterate Brahmic scripts, fold accents, drop anything else non-ASCII."""
    if text.isascii():
        return text.lower()
    s = text.translate(_TRANSLIT_TABLE)
    s = s.replace(_INHERENT + _KILL, "")
    s = _WORD_FINAL_INHERENT.sub("", s)
    s = s.replace(_INHERENT, "a").replace(_KILL, "")
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower()


# --------------------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------------------

LEGAL_TOKENS = frozenset(
    """
    inc incorporated llc llp lp ltd limited pvt private corp corporation co company
    pllc pc plc lc cooperative coop sarl sas sasu sa eurl sci snc ei cie ets gmbh ag bv nv
    """.split()
)
HONORIFIC_TOKENS = frozenset("the dr mr mrs ms smt shri sri shree er prof".split())
ALIAS_TOKENS = frozenset("dba aka fka formerly known doing business as".split())
NAME_STOP_TOKENS = LEGAL_TOKENS | HONORIFIC_TOKENS | ALIAS_TOKENS

_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|in|org|net|co|fr|biz|info|us)(?:\.[a-z]{2})?\b"
)
_ALIAS_SLASH_RE = re.compile(r"\b(?:f/k/a|a/k/a|d/b/a|t/a|m/s)\b")
_APOS_DOT_RE = re.compile(r"['`\u2019.]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})
_LEET_RE = re.compile(r"(?<=[a-z])[013457]+(?=[a-z])")


def _clean_url_segments(s: str) -> str:
    """Drop ``| www.foo.com`` style trailers; turn a bare domain into its label."""
    if "|" in s:
        parts = [p for p in s.split("|") if not _URL_RE.search(p) and "www" not in p]
        s = " ".join(parts) if any(p.strip() for p in parts) else s.replace("|", " ")
    return _URL_RE.sub(r" \1 ", s)


def normalize_name(raw: str) -> tuple[str, str]:
    """Return ``(name_norm, name_core)`` for one business name."""
    s = to_ascii(raw)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _ALIAS_SLASH_RE.sub(" ", s)
    s = _clean_url_segments(s)
    s = _APOS_DOT_RE.sub("", s)
    norm = " ".join(_NON_ALNUM_RE.split(s)).strip()
    fixed = _LEET_RE.sub(lambda m: m.group(0).translate(_LEET), norm)
    tokens = [t for t in fixed.split() if t not in NAME_STOP_TOKENS]
    if not tokens:  # name consisting only of stop tokens: keep it rather than lose it
        tokens = fixed.split()
    return norm, " ".join(tokens)


# Phonetic skeleton: digraphs first, then soft c/g, then merge confusable consonants,
# drop vowels / h, collapse repeats. Legal words written phonetically are dropped.
_SKEL_DIGRAPHS = [("ph", "f"), ("gh", ""), ("ck", "k"), ("sch", "s"), ("sh", "s"), ("ch", "k"),
                  ("th", "t"), ("kh", "k"), ("bh", "b"), ("dh", "d"), ("jh", "j")]
_SKEL_SOFT_C = re.compile(r"c(?=[eiy])")
_SKEL_SOFT_G = re.compile(r"g(?=[ei])")
_SKEL_MAP = str.maketrans({"c": "k", "q": "k", "x": "ks", "z": "s", "j": "s", "w": "v", "h": None})
_SKEL_VOWELS = re.compile(r"[aeiouy]")
_SKEL_REPEAT = re.compile(r"(.)\1+")
SKELETON_STOP = frozenset("prvt pvt lmtd ltd llp llk lls pr l lt lmt nk ink krp krprsn krprtn kmpn".split())
# Transliterations of legal words seen in the training names. A skeleton match alone is not
# enough: smith/summit share a skeleton with "smt", and parvati shares one with "private".
PHONETIC_LEGAL_TOKENS = frozenset(
    """
    praivet piraivet limatid
    bijanes bisines bijines bijhanes bisinas bijanas
    """.split()
)


def skeleton_word(w: str) -> str:
    if w.isdigit():
        return w.lstrip("0") or "0"
    for a, b in _SKEL_DIGRAPHS:
        w = w.replace(a, b)
    w = _SKEL_SOFT_G.sub("j", _SKEL_SOFT_C.sub("s", w)).translate(_SKEL_MAP)
    return _SKEL_REPEAT.sub(r"\1", _SKEL_VOWELS.sub("", w))


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_LEGAL_BY_SKELETON: dict[str, tuple[str, ...]] = {}
for _tok in NAME_STOP_TOKENS:
    if len(_tok) >= 6:
        _LEGAL_BY_SKELETON.setdefault(skeleton_word(_tok), []).append(_tok)
_LEGAL_BY_SKELETON = {k: tuple(v) for k, v in _LEGAL_BY_SKELETON.items()}


def _is_stop_token(token: str) -> bool:
    """Legal, honorific, or alias token, including transliterations and one-edit typos."""
    if token in NAME_STOP_TOKENS or token in PHONETIC_LEGAL_TOKENS:
        return True
    if len(token) < 5:
        return False
    canons = _LEGAL_BY_SKELETON.get(skeleton_word(token))
    if not canons:
        return False
    return any(abs(len(token) - len(canon)) <= 2 and _edit_distance(token, canon) <= 2 for canon in canons)


def name_skeleton(name_core: str) -> str:
    out = [s for s in (skeleton_word(t) for t in name_core.split()) if s and s not in SKELETON_STOP]
    return " ".join(out)


# --------------------------------------------------------------------------------------
# Address normalization
# --------------------------------------------------------------------------------------

ADDRESS_CANONICAL = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd", "lane": "ln",
    "court": "ct", "terrace": "ter", "place": "pl", "trail": "trl", "highway": "hwy",
    "parkway": "pkwy", "circle": "cir", "square": "sq", "suite": "ste", "apartment": "apt",
    "floor": "fl", "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "near": "nr", "opposite": "opp", "sector": "sec", "impasse": "imp", "route": "rte",
    "chemin": "ch", "allee": "all", "mount": "mt", "fort": "ft", "expressway": "expy",
    "freeway": "fwy", "crossing": "xing", "junction": "jn", "colony": "col",
    "apartments": "apt", "apts": "apt", "number": "no",
}
ADDRESS_DROP_TOKENS = frozenset("no city township twp of the po box hno null".split())
_ADDR_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ALNUM_BOUNDARY_RE = re.compile(r"(?<=[0-9])(?=[a-z]{3,})|(?<=[a-z]{3})(?=[0-9])")
_DIGITS_RE = re.compile(r"\d+")


def normalize_address(raw: str) -> tuple[str, str]:
    """Return ``(addr_norm, addr_nums)`` for one address."""
    if not raw:
        return "", ""
    s = to_ascii(raw).replace("&", " and ")
    s = _APOS_DOT_RE.sub("", s)
    s = _ALNUM_BOUNDARY_RE.sub(" ", s)
    out = []
    for t in _ADDR_SPLIT_RE.split(s):
        if not t or t in ADDRESS_DROP_TOKENS:
            continue
        if t[0] == "0" and t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(ADDRESS_CANONICAL.get(t, t))
    nums = _DIGITS_RE.findall(" ".join(out))
    return " ".join(out), " ".join(nums)


# --------------------------------------------------------------------------------------
# Record preparation (parallel) + caching
# --------------------------------------------------------------------------------------


def _normalize_chunk(args: tuple[list[str], list[str]]) -> tuple[list, ...]:
    names, addrs = args
    name_norm, name_core, name_skel, addr_norm, addr_nums, name_raw = [], [], [], [], [], []
    for n in names:
        a, b = normalize_name(n)
        name_norm.append(a)
        name_core.append(b)
        name_skel.append(name_skeleton(b))
        name_raw.append(n.lower())
    for a in addrs:
        x, y = normalize_address(a)
        addr_norm.append(x)
        addr_nums.append(y)
    return name_raw, name_norm, name_core, name_skel, addr_norm, addr_nums


def prepare_records(df: pd.DataFrame, n_jobs: int | None = None, chunk: int = 50_000) -> pd.DataFrame:
    """Add normalized columns to a raw source frame (keeps entity_id / country / raw text)."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    jobs = [(names[i : i + chunk], addrs[i : i + chunk]) for i in range(0, len(df), chunk)]
    cols: list[list[str]] = [[], [], [], [], [], []]
    if n_jobs == 1 or len(jobs) == 1:
        results = map(_normalize_chunk, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        results = pool.map(_normalize_chunk, jobs)
    for res in results:
        for acc, part in zip(cols, res):
            acc.extend(part)
    if n_jobs != 1 and len(jobs) > 1:
        pool.shutdown()
    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"].to_numpy(),
            "country": df["country"].str.strip().str.lower().to_numpy(),
            "name_raw": cols[0],
            "addr_raw": df["business_address"].str.lower().to_numpy(),
            "name_norm": cols[1],
            "name_core": cols[2],
            "name_skel": cols[3],
            "addr_norm": cols[4],
            "addr_nums": cols[5],
        }
    )
    out["name_compact"] = out["name_core"].str.replace(" ", "", regex=False)
    for c in out.columns:
        out[c] = out[c].astype("string[pyarrow]")
    return out


@dataclass
class SplitData:
    """Normalized records of one split.

    ``s1`` holds Source 1; ``s23`` holds Source 2 followed by Source 3 (the match pool).
    Row positions in these frames are the integer ids used throughout the pipeline.
    ``truth`` (train only) maps S1 row -> array of s23 rows it matches.
    """

    name: str
    s1: pd.DataFrame
    s23: pd.DataFrame
    truth: dict[int, np.ndarray] | None = None
    limited: bool = False


def _csv_reader(path: Path, columns: list[str]):
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=8 * 1024 * 1024),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            include_columns=columns,
            strings_can_be_null=False,
        ),
    )


def _table_to_frame(table: pa.Table, columns: list[str]) -> pd.DataFrame:
    if table.num_rows == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    return table.to_pandas().fillna("").astype(str)


def read_tsv_head(path: str | Path, columns: list[str], n: int) -> pd.DataFrame:
    """Read only the first ``n`` data rows of a TSV (the header is not counted)."""
    batches = []
    seen = 0
    for batch in _csv_reader(Path(path), columns):
        batches.append(batch)
        seen += batch.num_rows
        if seen >= n:
            break
    if not batches:
        return _table_to_frame(pa.table({c: pa.array([], type=pa.string()) for c in columns}), columns)
    return _table_to_frame(pa.Table.from_batches(batches).slice(0, n), columns)


def read_tsv_keep_and_extra(path: str | Path, columns: list[str], keep_ids: set[str], extra: int) -> pd.DataFrame:
    """Stream a TSV, keeping every row whose id is in ``keep_ids`` plus ``extra`` other rows.

    Used by ``--max-rows`` so a small training slice still contains the true matches of
    the sampled Source 1 entities (otherwise the slice can have zero positive labels).
    """
    import pyarrow.compute as pc

    if not keep_ids:
        return read_tsv_head(path, columns, extra)
    keep_arr = pa.array(sorted(keep_ids), type=pa.string())
    batches: list[pa.RecordBatch] = []
    n_extra = 0
    n_hits = 0
    scanned = 0
    path = Path(path)
    target_hits = len(keep_ids)
    for batch in _csv_reader(path, columns):
        scanned += batch.num_rows
        in_keep = pc.is_in(batch.column(0), value_set=keep_arr)
        hit = batch.filter(in_keep)
        if hit.num_rows:
            batches.append(hit)
            n_hits += hit.num_rows
        if n_extra < extra:
            rest = batch.filter(pc.invert(in_keep)).slice(0, extra - n_extra)
            if rest.num_rows:
                batches.append(rest)
                n_extra += rest.num_rows
        if scanned // 2_000_000 != (scanned - batch.num_rows) // 2_000_000:
            print(f"  scanned {scanned:,} rows of {path.name} (matches kept {n_hits:,})", flush=True)
        if n_hits >= target_hits and n_extra >= extra:
            break
    if not batches:
        return _table_to_frame(pa.table({c: pa.array([], type=pa.string()) for c in columns}), columns)
    print(f"  kept {pa.Table.from_batches(batches).num_rows:,} rows from {path.name} (scanned {scanned:,})", flush=True)
    return _table_to_frame(pa.Table.from_batches(batches), columns)


def _cache_path(cache_dir: Path | None, split: str, k: int, tag: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{split}_source{k}_{NORMALIZATION_VERSION}_{tag}.parquet"


def _load_cached_or_prepare(raw: pd.DataFrame, cache: Path | None, n_jobs: int | None) -> pd.DataFrame:
    if cache is not None and cache.exists():
        return pd.read_parquet(cache, dtype_backend="pyarrow").astype("string[pyarrow]")
    df = prepare_records(raw, n_jobs=n_jobs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def _sources_are_small(data_dir: Path, split: str, max_rows: int) -> bool:
    """True when the on-disk files are already about ``max_rows`` (or smaller)."""
    limit = max(1_000_000, max_rows * 1200)
    for k in (1, 2, 3):
        path = data_dir / split / f"{split}_source{k}.tsv"
        if path.stat().st_size > limit:
            return False
    return True


def _load_source(data_dir: Path, split: str, k: int, cache_dir: Path | None, n_jobs: int | None) -> pd.DataFrame:
    cache = _cache_path(cache_dir, split, k, "full")
    # Accept the cache written before the "_full" suffix existed.
    legacy = None if cache_dir is None else cache_dir / f"{split}_source{k}_{NORMALIZATION_VERSION}.parquet"
    if cache is not None and not cache.exists() and legacy is not None and legacy.exists():
        cache = legacy
    if cache is not None and cache.exists():
        return pd.read_parquet(cache, dtype_backend="pyarrow").astype("string[pyarrow]")
    raw = read_tsv(data_dir / split / f"{split}_source{k}.tsv", SOURCE_COLUMNS)
    return _load_cached_or_prepare(raw, cache, n_jobs)


def load_truth(path: str | Path, s1: pd.DataFrame, s23: pd.DataFrame) -> dict[int, np.ndarray]:
    """Parse the ground-truth file into ``{s1_row: np.array(s23_rows)}``."""
    gt = read_tsv(path, GT_COLUMNS)
    s1_ids = s1["entity_id"].astype(str).to_numpy()
    s23_ids = s23["entity_id"].astype(str).to_numpy()
    gt = gt[gt["source1_entity_id"].isin(set(s1_ids))].copy()
    s1_pos = pd.Series(np.arange(len(s1)), index=s1_ids)
    s23_pos = pd.Series(np.arange(len(s23)), index=s23_ids)
    exploded = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    exploded["m"] = exploded["m"].fillna("").str.strip()
    exploded = exploded[exploded["m"] != ""]
    rows1 = s1_pos.reindex(exploded["source1_entity_id"].to_numpy()).to_numpy()
    rows23 = s23_pos.reindex(exploded["m"].to_numpy()).to_numpy()
    ok = ~(np.isnan(rows1) | np.isnan(rows23))
    rows1, rows23 = rows1[ok].astype(np.int64), rows23[ok].astype(np.int64)
    truth: dict[int, np.ndarray] = {
        int(i): np.empty(0, np.int64) for i in s1_pos.reindex(gt["source1_entity_id"]).dropna().astype(int)
    }
    if len(rows1) == 0:
        return truth
    order = np.argsort(rows1, kind="stable")
    rows1, rows23 = rows1[order], rows23[order]
    splits = np.flatnonzero(np.diff(rows1)) + 1
    for grp1, grp23 in zip(np.split(rows1, splits), np.split(rows23, splits)):
        if len(grp1):
            truth[int(grp1[0])] = grp23
    return truth


def _matched_ids_for(gt_path: Path, s1_ids: set[str]) -> set[str]:
    gt = read_tsv(gt_path, GT_COLUMNS)
    keep = gt["source1_entity_id"].isin(s1_ids)
    matched: set[str] = set()
    for cell in gt.loc[keep, "matched_entity_ids"]:
        if cell:
            matched.update(part.strip() for part in cell.split(",") if part.strip())
    return matched


def _load_limited(data_dir: Path, split: str, cache_dir: Path | None, n_jobs: int | None, max_rows: int, distractor_factor: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a truth-preserving slice.

    Source 1 is the first ``max_rows`` records. On the training split, every Source 2/3
    record that the ground truth links to those Source 1 entities is kept, plus
    ``max_rows * distractor_factor`` extra rows from each file so the matcher sees
    non-matches. The test split has no labels, so each test file is just the first
    ``max_rows`` rows.
    """
    tag = f"rows{max_rows}"
    caches = [_cache_path(cache_dir, split, k, tag) for k in (1, 2, 3)]
    if all(c is not None and c.exists() for c in caches):
        frames = [pd.read_parquet(c, dtype_backend="pyarrow").astype("string[pyarrow]") for c in caches]
        s23 = pd.concat(frames[1:], ignore_index=True)
        return frames[0], s23

    print(f"loading a {max_rows}-row slice of {split} (streaming source files)", flush=True)
    s1_raw = read_tsv_head(data_dir / split / f"{split}_source{1}.tsv", SOURCE_COLUMNS, max_rows)
    gt_path = data_dir / split / f"{split}_ground_truth.tsv"
    extra = max_rows * distractor_factor
    if gt_path.exists():
        keep_ids = _matched_ids_for(gt_path, set(s1_raw["entity_id"]))
        print(f"  {len(keep_ids):,} true match ids to retain from source 2/3", flush=True)
        s2_raw = read_tsv_keep_and_extra(data_dir / split / f"{split}_source2.tsv", SOURCE_COLUMNS, keep_ids, extra)
        s3_raw = read_tsv_keep_and_extra(data_dir / split / f"{split}_source3.tsv", SOURCE_COLUMNS, keep_ids, extra)
    else:
        s2_raw = read_tsv_head(data_dir / split / f"{split}_source2.tsv", SOURCE_COLUMNS, max_rows)
        s3_raw = read_tsv_head(data_dir / split / f"{split}_source3.tsv", SOURCE_COLUMNS, max_rows)
    frames = []
    for raw, cache in ((s1_raw, caches[0]), (s2_raw, caches[1]), (s3_raw, caches[2])):
        frames.append(_load_cached_or_prepare(raw, cache, n_jobs=1 if len(raw) < 80_000 else n_jobs))
    s23 = pd.concat(frames[1:], ignore_index=True)
    return frames[0], s23


def load_split(
    data_dir: str | Path,
    split: str,
    cache_dir: str | Path | None = None,
    n_jobs: int | None = None,
    max_rows: int | None = None,
    distractor_factor: int = 6,
) -> SplitData:
    """Load and normalize all sources of ``split`` ("train" or "test").

    ``max_rows`` caps Source 1 (and, on a labeled split, keeps the true matches of
    those entities). Leave it unset to load everything. Files that are already
    smaller than ``max_rows`` are loaded in full, so a hand-cut Kaggle subset works
    with or without the flag.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    limited = bool(max_rows) and not _sources_are_small(data_dir, split, int(max_rows))
    if limited:
        s1, s23 = _load_limited(data_dir, split, cache_dir, n_jobs, int(max_rows), distractor_factor)
    else:
        s1 = _load_source(data_dir, split, 1, cache_dir, n_jobs)
        s2 = _load_source(data_dir, split, 2, cache_dir, n_jobs)
        s3 = _load_source(data_dir, split, 3, cache_dir, n_jobs)
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s2, s3
    truth = None
    gt_path = data_dir / split / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        truth = load_truth(gt_path, s1, s23)
    return SplitData(name=split, s1=s1, s23=s23, truth=truth, limited=limited)
