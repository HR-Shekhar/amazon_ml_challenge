"""Run the entity-resolution pipeline on a Modal CPU.

    modal run --detach modal_train.py
    modal run --detach modal_train.py --args "--block-only"

The dataset must already be on the ``amazon-ml-entity`` volume:

    modal volume put amazon-ml-entity student_resource/dataset/train dataset/train
    modal volume put amazon-ml-entity student_resource/dataset/test dataset/test

Outputs land on that same volume, under ``output5/``, ``model5/``, and ``cache/``:

    modal volume get amazon-ml-entity output5 output5
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "student_resource" / "code" / "business_entity_resolution" / "src"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.4.6",
        "pandas==3.0.6",
        "pyarrow==25.0.1",
        "scipy==1.17.1",
        "scikit-learn==1.9.1",
        "sparse-dot-topn==1.2.0",
        "rapidfuzz==3.14.6",
        "lightgbm==4.7.0",
        "joblib==1.6.0",
    )
    .add_local_dir(str(SRC), remote_path="/root/pipeline", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file(str(VALIDATOR), remote_path="/root/utils/validate_submission.py")
)

volume = modal.Volume.from_name("amazon-ml-entity", create_if_missing=True)
app = modal.App("amazon-ml-entity-resolution")


def _commit_periodically() -> None:
    """Persist the normalized cache if the run is interrupted after loading."""
    time.sleep(20 * 60)
    while True:
        try:
            volume.commit()
            print("committed volume", flush=True)
        except Exception as exc:
            print(f"volume commit failed: {exc}", flush=True)
        time.sleep(15 * 60)


@app.function(
    image=image,
    cpu=16,
    memory=131072,
    timeout=23 * 60 * 60,
    volumes={"/vol": volume},
)
def train(extra: str = "") -> None:
    data = Path("/vol/dataset")
    needed = data / "train" / "train_source1.tsv"
    if not needed.is_file():
        found = [str(path) for path in data.rglob("*")][:40] if data.exists() else []
        raise SystemExit(f"dataset not found at {needed}. hits={found}")

    print(f"starting pipeline on Modal CPU {extra}", flush=True)
    threading.Thread(target=_commit_periodically, daemon=True).start()
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "16"
    code = subprocess.call(
        [
            sys.executable,
            "/root/pipeline/main.py",
            "--data-dir",
            str(data),
            "--output-dir",
            "/vol/output5",
            "--cache-dir",
            "/vol/cache",
            "--model-dir",
            "/vol/model5",
            "--validate-script",
            "/root/utils/validate_submission.py",
            "--n-jobs",
            "16",
            *shlex.split(extra),
        ],
        env=env,
    )
    volume.commit()
    print(f"volume committed; pipeline exit {code}", flush=True)
    if code != 0:
        raise SystemExit(code)


@app.local_entrypoint()
def main(args: str = "") -> None:
    # spawn returns as soon as the cloud job is scheduled, so a dropped local
    # connection cannot cancel the run.
    call = train.spawn(args)
    print(f"spawned {call.object_id}", flush=True)
