"""Lexical and n-gram analyses for phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.analysis.artifacts import ensure_phase3_layout, write_artifact_index

REQUIRED_COLUMNS = ("answer", "label")


def _validate(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in str(text).lower().split() if tok]


def _lexical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for label, group in frame.assign(label=frame["label"].astype(str)).groupby("label", dropna=False):
        tokens = []
        lengths = []
        for text in group["answer"].astype(str):
            tks = _tokenize(text)
            tokens.extend(tks)
            lengths.extend([len(t) for t in tks])
        token_count = len(tokens)
        unique_count = len(set(tokens))
        records.append(
            {
                "label": label,
                "num_texts": int(group.shape[0]),
                "token_count": token_count,
                "unique_token_count": unique_count,
                "type_token_ratio": float(unique_count / token_count) if token_count else 0.0,
                "avg_word_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values("label").reset_index(drop=True)


def _top_ngrams_for_label(texts: list[str], ngram_range: tuple[int, int], top_k: int) -> list[dict]:
    if not texts:
        return []
    vectorizer = CountVectorizer(lowercase=True, ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(texts)
    sums = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    table = pd.DataFrame({"ngram": terms, "count": sums})
    table = table.sort_values(["count", "ngram"], ascending=[False, True]).head(top_k).reset_index(drop=True)
    return table.to_dict(orient="records")


def _plot_top_ngrams(top_rows: list[dict], path: Path, title: str) -> None:
    if not top_rows:
        return
    frame = pd.DataFrame(top_rows).sort_values(["count", "ngram"], ascending=[True, True])
    plt.figure(figsize=(8, 4))
    plt.barh(frame["ngram"], frame["count"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_lexical(
    frame: pd.DataFrame,
    output_root: str | Path,
    *,
    source_data: str | Path,
    sections: Iterable[str] | None = None,
    top_k: int = 15,
) -> list[Path]:
    _validate(frame)
    sections_set = set(sections or ["lexical", "ngrams"])

    layout = ensure_phase3_layout(output_root)
    out_dir = layout["lexical"]
    generated: list[Path] = []

    if "lexical" in sections_set:
        summary = _lexical_summary(frame)
        csv_path = out_dir / "lexical_summary.csv"
        json_path = out_dir / "lexical_summary.json"
        summary.to_csv(csv_path, index=False)
        summary.to_json(json_path, orient="records", indent=2)
        generated.extend([csv_path, json_path])

    if "ngrams" in sections_set:
        payload: dict[str, dict] = {}
        for label, group in frame.assign(label=frame["label"].astype(str)).groupby("label", dropna=False):
            texts = group["answer"].astype(str).tolist()
            uni = _top_ngrams_for_label(texts, (1, 1), top_k)
            bi = _top_ngrams_for_label(texts, (2, 2), top_k)
            payload[str(label)] = {"unigrams": uni, "bigrams": bi}

            uni_csv = out_dir / f"top_unigrams_{label}.csv"
            pd.DataFrame(uni).to_csv(uni_csv, index=False)
            bi_csv = out_dir / f"top_bigrams_{label}.csv"
            pd.DataFrame(bi).to_csv(bi_csv, index=False)
            generated.extend([uni_csv, bi_csv])

            uni_plot = out_dir / f"top_unigrams_{label}.png"
            _plot_top_ngrams(uni, uni_plot, f"Top Unigrams ({label})")
            bi_plot = out_dir / f"top_bigrams_{label}.png"
            _plot_top_ngrams(bi, bi_plot, f"Top Bigrams ({label})")
            for plot in (uni_plot, bi_plot):
                if plot.exists():
                    generated.append(plot)

        ngram_json = out_dir / "top_ngrams.json"
        ngram_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated.append(ngram_json)

    write_artifact_index(
        output_root,
        stage="lexical",
        generated_files=generated,
        source_data=source_data,
        metadata={"sections": sorted(sections_set), "top_k": top_k},
    )
    generated.append(Path(output_root) / "artifact_index.json")
    return generated
