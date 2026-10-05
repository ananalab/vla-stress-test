"""Loading the per-episode CSVs written by vla_stress.evaluate."""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def load(name: str, root: Path = RESULTS) -> pd.DataFrame | None:
    """All rows of results/<name>.csv (and its shards), or None if the run does not exist yet."""
    files = sorted(glob.glob(str(root / f"{name}.csv")) + glob.glob(str(root / f"{name}.shard*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    for c in ["success", "task_id", "episode", "steps"]:
        df[c] = df[c].astype(int)
    df["magnitude"] = df["magnitude"].astype(float)
    df["intensity"] = df["intensity"].astype(float)
    df["duration_s"] = df["duration_s"].astype(float)
    return df
