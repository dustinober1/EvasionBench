from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
    missing = [col for col in ("question", "answer") if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")

    question = frame["question"].fillna("").astype(str)
    answer = frame["answer"].fillna("").astype(str)
    return (question + " [SEP] " + answer).astype(str)


def _split_data(
    frame: pd.DataFrame,
    *,
    target_col: str,
    random_state: int,
    test_size: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    if target_col not in frame.columns:
        raise ValueError(f"Missing target column: {target_col}")

    labels = frame[target_col].fillna("unknown").astype(str)
    features = _build_features(frame)

    stratify = labels
    class_counts = labels.value_counts()
    stratify_used = True
    if len(class_counts) < 2 or class_counts.min() < 2:
        stratify = None
        stratify_used = False

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test, {
        "method": "train_test_split",
        "stratify": stratify_used,
        "random_state": random_state,
        "test_size": test_size,
    }


def train_tfidf_logreg(
    frame: pd.DataFrame,
    *,
    target_col: str = "label",
    random_state: int = 42,
    test_size: float = 0.2,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 1,
    max_features: int | None = 5000,
    c: float = 1.0,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, split_metadata = _split_data(
        frame,
        target_col=target_col,
        random_state=random_state,
        test_size=test_size,
    )

    vectorizer_params = {
        "ngram_range": (ngram_min, ngram_max),
        "min_df": min_df,
        "max_features": max_features,
    }
    classifier_params = {
        "C": c,
        "class_weight": class_weight,
        "solver": solver,
        "max_iter": 1000,
        "random_state": random_state,
    }

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", LogisticRegression(**classifier_params)),
        ]
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer_params": vectorizer_params,
        "classifier_params": classifier_params,
        "split_metadata": split_metadata,
    }


def train_tree_or_boosting(
    frame: pd.DataFrame,
    *,
    model_family: str,
    target_col: str = "label",
    random_state: int = 42,
    test_size: float = 0.2,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 1,
    max_features: int | None = 2000,
    n_estimators: int = 80,
    max_depth: int | None = None,
    learning_rate: float = 0.1,
    boosting_max_iter: int = 50,
) -> dict[str, Any]:
    X_train_raw, X_test_raw, y_train, y_test, split_metadata = _split_data(
        frame,
        target_col=target_col,
        random_state=random_state,
        test_size=test_size,
    )

    vectorizer_params = {
        "ngram_range": (ngram_min, ngram_max),
        "min_df": min_df,
        "max_features": max_features,
    }
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    if model_family == "tree":
        estimator_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        model = RandomForestClassifier(**estimator_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_bundle: dict[str, Any] = {
            "model": model,
            "predictions": predictions,
        }
    elif model_family == "boosting":
        n_components = min(256, max(2, X_train.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_train_reduced = svd.fit_transform(X_train)
        X_test_reduced = svd.transform(X_test)
        model_params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
            "max_iter": boosting_max_iter,
        }
        model = HistGradientBoostingClassifier(**model_params)
        model.fit(X_train_reduced, y_train)
        predictions = model.predict(X_test_reduced)
        estimator_params = {**model_params, "svd_components": n_components}
        model_bundle = {
            "model": model,
            "predictions": predictions,
            "svd": svd,
        }
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    return {
        **model_bundle,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer": vectorizer,
        "vectorizer_params": vectorizer_params,
        "estimator_params": estimator_params,
        "split_metadata": split_metadata,
    }
