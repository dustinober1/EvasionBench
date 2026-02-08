from datasets import load_dataset
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_evasionbench(save_path: str = None):
    """Download EvasionBench dataset from HuggingFace and save to parquet."""
    if save_path is None:
        save_path = os.path.join(DATA_PATH, "evasionbench.parquet")
    ds = load_dataset("FutureMa/EvasionBench")
    # Save train split or full dataset as parquet
    if "train" in ds:
        df = ds["train"].to_pandas()
    else:
        df = ds.to_pandas()
    df.to_parquet(save_path)
    return save_path
