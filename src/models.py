from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import transformers
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from transformers.trainer_utils import set_seed


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


def train_transformer(
    frame: pd.DataFrame,
    *,
    target_col: str = "label",
    random_state: int = 42,
    test_size: float = 0.2,
    model_name: str = "distilbert-base-uncased",
    max_epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = 512,
) -> dict[str, Any]:
    """Train a transformer model (DistilBERT) for binary classification.

    Args:
        frame: Input DataFrame with question, answer, and target columns
        target_col: Name of the target column (default: "label")
        random_state: Random seed for reproducibility
        test_size: Fraction of data for testing
        model_name: Hugging Face model name (default: distilbert-base-uncased)
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for optimizer
        max_length: Maximum sequence length for tokenization

    Returns:
        Dictionary containing model, tokenizer, trainer, predictions, and metadata
    """
    # Set random seeds for reproducibility
    set_seed(random_state)
    torch.manual_seed(random_state)

    # Prepare features (Q + A text)
    if "question" not in frame.columns or "answer" not in frame.columns:
        raise ValueError("DataFrame must contain 'question' and 'answer' columns")

    if target_col not in frame.columns:
        raise ValueError(f"DataFrame must contain target column: {target_col}")

    # Combine question and answer
    frame["text"] = (
        frame["question"].fillna("").astype(str) + " [SEP] " + frame["answer"].fillna("").astype(str)
    )

    # Encode labels to integers
    unique_labels = sorted(frame[target_col].dropna().unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError(f"Expected exactly 2 unique labels, found {len(unique_labels)}")

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    frame["label_id"] = frame[target_col].map(label2id)

    # Split data
    stratify = frame["label_id"] if min(frame["label_id"].value_counts()) >= 2 else None
    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Remove columns we don't need
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask", "label_id"]]
    )
    test_dataset = test_dataset.remove_columns(
        [col for col in test_dataset.column_names if col not in ["input_ids", "attention_mask", "label_id"]]
    )

    # Rename label_id to labels
    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    # Set format for PyTorch
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Detect hardware and adjust batch size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        per_device_batch_size = 16
        fp16 = True
        gradient_checkpointing = False
    else:
        per_device_batch_size = 4
        fp16 = False
        gradient_checkpointing = True

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        precision = precision_score(labels, predictions, average="macro", zero_division=0)
        recall = recall_score(labels, predictions, average="macro", zero_division=0)

        return {
            "accuracy": accuracy,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./tmp_transformer_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size * 2,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir="./tmp_transformer_logs",
        logging_strategy="epoch",
        save_total_limit=1,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",  # Disable default logging
        seed=random_state,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()

    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = predictions.label_ids

    # Convert predictions back to string labels
    y_pred_labels = [id2label[int(idx)] for idx in y_pred]
    y_true_labels = [id2label[int(idx)] for idx in y_true]

    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "y_pred": y_pred_labels,
        "y_true": y_true_labels,
        "eval_results": eval_results,
        "split_metadata": {
            "method": "train_test_split",
            "stratify": stratify is not None,
            "random_state": random_state,
            "test_size": test_size,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "model_config": {
            "model_name": model_name,
            "num_labels": 2,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "num_epochs": max_epochs,
            "per_device_batch_size": per_device_batch_size,
            "fp16": fp16,
            "gradient_checkpointing": gradient_checkpointing,
            "device": device,
        },
        "label2id": label2id,
        "id2label": id2label,
    }
