"""Download EvasionBench dataset and save to data/raw/"""
from src.data import download_evasionbench
import os


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    out = download_evasionbench(save_path="data/raw/evasionbench.parquet")
    print(f"Saved dataset to {out}")
