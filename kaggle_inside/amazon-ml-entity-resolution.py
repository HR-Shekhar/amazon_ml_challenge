import subprocess
import sys
from pathlib import Path

SEARCH = [
    Path("/kaggle/input/amazon-ml-entity-code"),
    Path("/kaggle/input/datasets/h1manshushekhar/amazon-ml-entity-code"),
    Path("/kaggle/working"),
    Path("/kaggle/src"),
    Path(__file__).resolve().parent,
    Path.cwd(),
]


def find_root() -> Path:
    for folder in SEARCH:
        if (folder / "requirements.txt").is_file() and (folder / "main.py").is_file():
            return folder
    found = []
    kaggle = Path("/kaggle")
    if kaggle.is_dir():
        found = [str(p) for p in kaggle.rglob("main.py")]
    raise SystemExit(
        "Could not find requirements.txt and main.py. "
        f"cwd={Path.cwd()} searched={SEARCH} main.py hits={found}"
    )


def find_data() -> Path:
    options = [
        Path("/kaggle/input/amazon-ml-entity-data"),
        Path("/kaggle/input/datasets/h1manshushekhar/amazon-ml-entity-data"),
    ]
    for folder in options:
        if (folder / "train" / "train_source1.tsv").is_file():
            return folder
    listing = []
    inp = Path("/kaggle/input")
    if inp.is_dir():
        listing = [str(p) for p in inp.rglob("train_source1.tsv")]
    raise SystemExit(f"Could not find the dataset. hits={listing}")


root = find_root()
data = find_data()
print(f"pipeline root: {root}", flush=True)
print(f"data dir: {data}", flush=True)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")]
)
subprocess.check_call(
    [
        sys.executable,
        str(root / "main.py"),
        "--data-dir",
        str(data),
        "--output-dir",
        "/kaggle/working/output",
        "--cache-dir",
        "/kaggle/working/cache",
        "--model-dir",
        "/kaggle/working/model",
    ],
    cwd=str(root),
)
