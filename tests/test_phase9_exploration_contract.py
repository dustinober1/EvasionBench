from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _exploration_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    labels = ["direct", "intermediate", "fully_evasive"]
    for idx in range(120):
        label = labels[idx % 3]
        if label == "direct":
            question = f"What is gross margin for segment {idx % 5}?"
            answer = "Gross margin improved and the drivers are transparent."
        elif label == "intermediate":
            question = f"How should we compare seasonality assumptions {idx % 4}?"
            answer = (
                "Seasonality varies by quarter and we are still balancing priorities."
            )
        else:
            question = f"Will you disclose exact churn details for cohort {idx % 6}?"
            answer = "We cannot disclose that detail at this time."
        rows.append({"question": question, "answer": answer, "label": label})

    frame = pd.DataFrame(rows)
    data_path = tmp_path / "data/processed/evasionbench_prepared.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_path, index=False)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2500)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=0.8,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                ),
            ),
        ]
    )
    features = frame["question"] + " [SEP] " + frame["answer"]
    model.fit(features, frame["label"])

    model_root = tmp_path / "artifacts/models/phase8/logreg"
    model_root.mkdir(parents=True, exist_ok=True)
    with (model_root / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream)

    selected_summary_path = tmp_path / "artifacts/models/phase8/selected_model.json"
    _write_json(
        selected_summary_path,
        {
            "best_model_family": "logreg",
            "artifact_root": "artifacts/models/phase8/logreg",
            "metrics": {
                "accuracy": 0.65,
                "f1_macro": 0.55,
                "precision_macro": 0.56,
                "recall_macro": 0.54,
            },
        },
    )

    output_root = tmp_path / "artifacts/analysis/phase9/exploration"
    return data_path, selected_summary_path, output_root


def test_phase9_exploration_contract(tmp_path: Path) -> None:
    data_path, selected_summary_path, output_root = _exploration_fixture(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_phase9_exploration.py",
            "--input",
            str(data_path),
            "--selected-model-summary",
            str(selected_summary_path),
            "--output-root",
            str(output_root),
            "--project-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    expected = [
        output_root / "temporal_summary.json",
        output_root / "segment_summary.json",
        output_root / "question_intent_error_map.csv",
        output_root / "artifact_index.json",
    ]
    for path in expected:
        assert path.exists(), str(path)

    temporal = json.loads(
        (output_root / "temporal_summary.json").read_text(encoding="utf-8")
    )
    segment = json.loads(
        (output_root / "segment_summary.json").read_text(encoding="utf-8")
    )
    assert temporal["status"] == "skipped"
    assert segment["status"] == "skipped"

    question_map = pd.read_csv(output_root / "question_intent_error_map.csv")
    assert set(question_map.columns) == {
        "question_type",
        "n_samples",
        "n_errors",
        "error_rate",
        "intermediate_to_fully_evasive",
        "fully_evasive_to_intermediate",
    }
    assert not question_map.empty

    index = json.loads(
        (output_root / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert index["phase"] == "phase9"
    assert index["stage"] == "exploration"
