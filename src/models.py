from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ModelWrapper:
    """Minimal model wrapper interface."""

    def __init__(self, model: Any):
        self.model = model

    def predict(self, questions, answers):
        raise NotImplementedError


def _build_features(frame: pd.DataFrame) -> pd.Series:
    question = frame.get("question", "").fillna("").astype(str)
    answer = frame.get("answer", "").fillna("").astype(str)
    return (question + " [SEP] " + answer).astype(str)


def train_tfidf_logreg(
    frame: pd.DataFrame,
    target_col: str = "label",
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    if target_col not in frame.columns:
        raise ValueError(f"Missing target column: {target_col}")

    labels = frame[target_col].fillna("unknown").astype(str)
    features = _build_features(frame)

    stratify = labels
    class_counts = labels.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
