from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET_ID = "FutureMa/EvasionBench"
DEFAULT_SPLIT = "train"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT = Path("data/raw/evasionbench.parquet")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file in a streaming-safe way."""
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_evasionbench(
    output_path: str | Path = DEFAULT_OUTPUT,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_SPLIT,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Download EvasionBench split deterministically and persist parquet output."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        dataset_id,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
    )
    dataframe = dataset.to_pandas()
    dataframe.to_parquet(output, index=False)

    return {
        "dataset_id": dataset_id,
        "split": split,
        "revision": revision,
        "cache_dir": cache_dir or "",
        "output_path": str(output),
        "row_count": int(len(dataframe)),
        "schema": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "checksum_sha256": sha256_file(output),
    }
