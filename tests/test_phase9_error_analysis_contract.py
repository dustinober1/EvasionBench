from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _phase9_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    labels = ["direct", "intermediate", "fully_evasive"]
    rows = []
    for idx in range(180):
        label = labels[idx % 3]
        if label == "direct":
            question = f"What was revenue in quarter {idx % 4}?"
            answer = f"Revenue was {100 + (idx % 11)} million with guidance clarity."
        elif label == "intermediate":
            question = f"Can you compare margin trends in quarter {idx % 4}?"
            answer = (
                "Margin trends are mixed and depend on seasonality; "
                f"we are balancing investment and efficiency {idx % 7}."
            )
        else:
            question = f"Will you disclose customer churn in quarter {idx % 4}?"
            answer = (
                "We cannot provide that detail at this time and prefer not to comment "
                f"until filings are complete {idx % 5}."
            )
        rows.append({"question": question, "answer": answer, "label": label})

    frame = pd.DataFrame(rows)
    data_path = tmp_path / "data/processed/evasionbench_prepared.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_path, index=False)

    text = frame["question"] + " [SEP] " + frame["answer"]
    X_train, X_test, y_train, y_test = train_test_split(
        text,
        frame["label"],
        test_size=0.25,
        random_state=42,
        stratify=frame["label"],
    )

    baseline_model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=0.25,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                ),
            ),
        ]
    )
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)

    selected_model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=4000)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    C=1.2,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                ),
            ),
        ]
    )
    selected_model.fit(X_train, y_train)
    selected_pred = selected_model.predict(X_test)

    baseline_root = tmp_path / "artifacts/models/phase5/logreg"
    selected_root = tmp_path / "artifacts/models/phase8/logreg"
    baseline_root.mkdir(parents=True, exist_ok=True)
    selected_root.mkdir(parents=True, exist_ok=True)

    with (selected_root / "model.pkl").open("wb") as stream:
        pickle.dump(selected_model, stream)

    baseline_report = classification_report(
        y_test,
        baseline_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    baseline_confusion = confusion_matrix(y_test, baseline_pred, labels=labels)
    _write_json(baseline_root / "classification_report.json", baseline_report)
    _write_json(
        baseline_root / "confusion_matrix.json",
        {"labels": labels, "confusion_matrix": baseline_confusion.tolist()},
    )

    selected_report = classification_report(
        y_test,
        selected_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    selected_confusion = confusion_matrix(y_test, selected_pred, labels=labels)
    _write_json(selected_root / "classification_report.json", selected_report)
    _write_json(
        selected_root / "confusion_matrix.json",
        {"labels": labels, "confusion_matrix": selected_confusion.tolist()},
    )

    selected_summary_path = tmp_path / "artifacts/models/phase8/selected_model.json"
    _write_json(
        selected_summary_path,
        {
            "best_model_family": "logreg",
            "artifact_root": "artifacts/models/phase8/logreg",
            "metrics": {
                "accuracy": float((selected_pred == y_test).mean()),
                "f1_macro": float(selected_report["macro avg"]["f1-score"]),
                "precision_macro": float(selected_report["macro avg"]["precision"]),
                "recall_macro": float(selected_report["macro avg"]["recall"]),
            },
        },
    )

    phase5_summary_path = (
        tmp_path / "artifacts/models/phase5/model_comparison/summary.json"
    )
    _write_json(
        phase5_summary_path,
        {
            "best_model_family": "logreg",
            "models": {
                "logreg": {
                    "accuracy": float((baseline_pred == y_test).mean()),
                    "f1_macro": float(baseline_report["macro avg"]["f1-score"]),
                    "precision_macro": float(baseline_report["macro avg"]["precision"]),
                    "recall_macro": float(baseline_report["macro avg"]["recall"]),
                    "artifact_root": str(baseline_root),
                }
            },
        },
    )

    output_root = tmp_path / "artifacts/analysis/phase9/error_analysis"
    return data_path, selected_summary_path, phase5_summary_path, output_root


def test_phase9_error_analysis_contract(tmp_path: Path) -> None:
    (
        data_path,
        selected_summary_path,
        phase5_summary_path,
        output_root,
    ) = _phase9_fixture(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_error_profiles.py",
            "--input",
            str(data_path),
            "--selected-model-summary",
            str(selected_summary_path),
            "--phase5-summary",
            str(phase5_summary_path),
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

    expected_files = [
        output_root / "error_summary.json",
        output_root / "misclassification_routes.csv",
        output_root / "hard_cases.md",
        output_root / "class_failure_heatmap.png",
        output_root / "artifact_index.json",
    ]
    for path in expected_files:
        assert path.exists(), str(path)

    summary = json.loads(
        (output_root / "error_summary.json").read_text(encoding="utf-8")
    )
    assert summary["analysis_version"] == "phase9-error-analysis-v1"
    assert "hardest_classes" in summary
    assert "per_class_deltas" in summary

    index = json.loads(
        (output_root / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert index["phase"] == "phase9"
    assert index["stage"] == "error_analysis"
    assert "misclassification_routes.csv" in index["generated_files"]
