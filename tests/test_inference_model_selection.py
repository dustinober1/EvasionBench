from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.inference import EvasionPredictor, HeuristicPredictor, load_model


def _train_text_pipeline() -> Pipeline:
    frame = pd.DataFrame(
        [
            {
                "question": "How is revenue",
                "answer": "Revenue grew strongly",
                "label": "direct",
            },
            {
                "question": "Can you share guidance",
                "answer": "We cannot discuss guidance",
                "label": "fully_evasive",
            },
            {
                "question": "What about margin",
                "answer": "There are moving parts, some pressure",
                "label": "intermediate",
            },
            {
                "question": "How is demand",
                "answer": "Demand remains resilient",
                "label": "direct",
            },
            {
                "question": "Will you quantify",
                "answer": "It depends and we will update later",
                "label": "intermediate",
            },
            {
                "question": "Can you answer directly",
                "answer": "No comment at this stage",
                "label": "fully_evasive",
            },
        ]
    )
    features = frame["question"] + " [SEP] " + frame["answer"]
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(features, frame["label"])
    return model


def _write_pipeline_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_train_text_pipeline(), f)


def test_load_model_prefers_phase8_selected_model(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_pipeline_model(artifacts_root / "models" / "phase8" / "logreg" / "model.pkl")

    selected = {
        "best_model_family": "logreg",
        "metrics": {"accuracy": 0.7, "f1_macro": 0.6},
    }
    selected_path = artifacts_root / "models" / "phase8" / "selected_model.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    predictor = load_model(artifacts_root=artifacts_root)
    assert isinstance(predictor, EvasionPredictor)
    assert predictor.model_name == "logreg"

    result = predictor.predict_single("How is revenue", "Revenue grew strongly")
    assert result["prediction"] in {"direct", "intermediate", "fully_evasive"}
    assert result["model_name"] == "logreg"


def test_load_model_env_override(monkeypatch, tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_pipeline_model(artifacts_root / "models" / "phase5" / "tree" / "model.pkl")
    monkeypatch.setenv("EVASION_MODEL_NAME", "tree")

    predictor = load_model(artifacts_root=artifacts_root)
    assert isinstance(predictor, EvasionPredictor)
    assert predictor.model_name == "tree"


def test_load_model_returns_heuristic_when_default_missing(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    predictor = load_model(artifacts_root=artifacts_root)
    assert isinstance(predictor, HeuristicPredictor)
