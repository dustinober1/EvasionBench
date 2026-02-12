"""Model inference module for EvasionBench."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL_FALLBACK = "boosting"


def _patch_numpy_pickle_compat() -> None:
    """Handle cross-version NumPy bit generator pickle compatibility."""
    try:
        from numpy.random import _pickle as np_pickle

        np_pickle.BitGenerators.setdefault(np.random.PCG64, np.random.PCG64)
        np_pickle.BitGenerators.setdefault(np.random.PCG64DXSM, np.random.PCG64DXSM)
    except Exception:
        return


class EvasionPredictor:
    """Load and run inference with trained EvasionBench models."""

    def __init__(
        self,
        model_path: str | Path,
        model_type: str = "boosting",
        model_name: str | None = None,
    ):
        """
        Initialize predictor with a trained model.

        Args:
            model_path: Path to model pickle file
            model_type: "pipeline" for direct text models, otherwise bundle-based
            model_name: Logical model family name
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model_name = model_name or model_type

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        _patch_numpy_pickle_compat()
        with open(self.model_path, "rb") as f:
            if model_type == "pipeline":
                self.model = pickle.load(f)
                self.vectorizer = None
                self.svd = None
            else:
                bundle = pickle.load(f)
                self.model = bundle["model"]
                self.vectorizer = bundle["vectorizer"]
                self.svd = bundle.get("svd")

        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        elif hasattr(self.model, "steps"):
            self.classes_ = self.model.steps[-1][1].classes_
        else:
            raise ValueError("Loaded model does not expose class labels")

    def _build_features(self, questions: list[str], answers: list[str]) -> pd.Series:
        """Build features from question-answer pairs using [SEP] token."""
        questions_clean = [str(q) if q else "" for q in questions]
        answers_clean = [str(a) if a else "" for a in answers]
        return pd.Series(
            [f"{q} [SEP] {a}" for q, a in zip(questions_clean, answers_clean)]
        )

    def _predict_with_probabilities(
        self, features: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.vectorizer is None:
            predictions = self.model.predict(features)
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(features)
            else:
                probas = self._fallback_proba(predictions)
            return predictions, probas

        X = self.vectorizer.transform(features)
        if self.svd is not None:
            X = self.svd.transform(X)

        predictions = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
        else:
            probas = self._fallback_proba(predictions)

        return predictions, probas

    def _fallback_proba(self, predictions: np.ndarray) -> np.ndarray:
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        probs = np.zeros((len(predictions), len(self.classes_)), dtype=float)
        for row_idx, pred in enumerate(predictions):
            probs[row_idx, class_to_idx[pred]] = 1.0
        return probs

    def predict(
        self,
        questions: str | list[str],
        answers: str | list[str],
        return_proba: bool = True,
    ) -> dict[str, Any]:
        """
        Predict evasion labels for question-answer pairs.

        Args:
            questions: Single question or list of questions
            answers: Single answer or list of answers
            return_proba: If True, return probability scores

        Returns:
            Dictionary with predictions and optional confidence details.
        """
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]

        if len(questions) != len(answers):
            raise ValueError("Number of questions and answers must match")

        features = self._build_features(questions, answers)

        if return_proba:
            predictions, probas = self._predict_with_probabilities(features)
        else:
            if self.vectorizer is None:
                predictions = self.model.predict(features)
            else:
                X = self.vectorizer.transform(features)
                if self.svd is not None:
                    X = self.svd.transform(X)
                predictions = self.model.predict(X)

        result = {
            "predictions": predictions.tolist(),
        }

        if return_proba:
            proba_dicts = []
            confidences = []
            for proba_row in probas:
                proba_dict = {
                    label: float(prob) for label, prob in zip(self.classes_, proba_row)
                }
                proba_dicts.append(proba_dict)
                confidences.append(float(np.max(proba_row)))

            result["probabilities"] = proba_dicts
            result["confidence"] = confidences

        return result

    def predict_single(
        self,
        question: str,
        answer: str,
        return_proba: bool = True,
    ) -> dict[str, Any]:
        """
        Predict evasion label for a single question-answer pair.

        Args:
            question: Question text
            answer: Answer text
            return_proba: If True, return probability scores

        Returns:
            Prediction dictionary for a single sample.
        """
        result = self.predict([question], [answer], return_proba=return_proba)

        output = {
            "prediction": result["predictions"][0],
            "model_name": self.model_name,
        }
        if return_proba:
            output["probabilities"] = result["probabilities"][0]
            output["confidence"] = result["confidence"][0]

        return output


class HeuristicPredictor:
    """Fallback predictor used when serialized artifacts are not loadable."""

    def _score(self, question: str, answer: str) -> dict[str, float]:
        text = f"{question} {answer}".lower()
        evasive_markers = [
            "cannot",
            "can't",
            "unable",
            "not discuss",
            "no comment",
            "decline",
        ]
        hedging_markers = ["maybe", "likely", "approximately", "depends", "around"]

        evasive_hits = sum(marker in text for marker in evasive_markers)
        hedge_hits = sum(marker in text for marker in hedging_markers)

        if evasive_hits > 0:
            return {"direct": 0.1, "intermediate": 0.2, "fully_evasive": 0.7}
        if hedge_hits > 0:
            return {"direct": 0.2, "intermediate": 0.6, "fully_evasive": 0.2}
        return {"direct": 0.7, "intermediate": 0.2, "fully_evasive": 0.1}

    def predict_single(self, question: str, answer: str) -> dict[str, Any]:
        probabilities = self._score(question, answer)
        prediction = max(probabilities, key=probabilities.get)
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": float(probabilities[prediction]),
            "model_name": "heuristic_fallback",
        }


def _selected_model_path(artifacts_root: Path) -> Path:
    return artifacts_root / "models" / "phase8" / "selected_model.json"


def load_selected_model_summary(
    artifacts_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return selected model metadata if phase-8 artifacts exist."""
    if artifacts_root is None:
        root = Path(__file__).resolve().parents[1] / "artifacts"
    else:
        root = Path(artifacts_root)

    selected_path = _selected_model_path(root)
    if not selected_path.exists():
        return None

    try:
        return json.loads(selected_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_default_model_name(artifacts_root: Path) -> str:
    env_name = os.getenv("EVASION_MODEL_NAME")
    if env_name:
        return env_name

    selected = load_selected_model_summary(artifacts_root)
    if selected:
        for key in ("best_model_family", "model_family", "winner_family"):
            value = selected.get(key)
            if isinstance(value, str) and value:
                return value

    return DEFAULT_MODEL_FALLBACK


def _resolve_model_dir(artifacts_root: Path, model_name: str) -> Path | None:
    candidates = [
        artifacts_root / "models" / "phase8" / model_name,
        artifacts_root / "models" / "phase5" / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_model(
    model_name: str | None = None,
    artifacts_root: str | Path | None = None,
) -> EvasionPredictor | HeuristicPredictor:
    """
    Load a trained EvasionBench model by name.

    Resolution order:
    1. Explicit `model_name` argument.
    2. `EVASION_MODEL_NAME` environment variable.
    3. `artifacts/models/phase8/selected_model.json` winner.
    4. Fallback to boosting.
    """
    if artifacts_root is None:
        root = Path(__file__).resolve().parents[1] / "artifacts"
    else:
        root = Path(artifacts_root)

    requested_name = model_name or _resolve_default_model_name(root)
    explicit_name = model_name is not None

    model_dir = _resolve_model_dir(root, requested_name)

    if (
        model_dir is None
        and not explicit_name
        and requested_name != DEFAULT_MODEL_FALLBACK
    ):
        requested_name = DEFAULT_MODEL_FALLBACK
        model_dir = _resolve_model_dir(root, requested_name)

    if model_dir is None:
        if explicit_name:
            raise ValueError(
                f"Model directory not found for '{requested_name}' under {root / 'models'}"
            )
        return HeuristicPredictor()

    model_path = model_dir / "model.pkl"
    model_type = "pipeline"
    if not model_path.exists():
        model_path = model_dir / "model_bundle.pkl"
        model_type = requested_name

    if not model_path.exists():
        if explicit_name:
            raise ValueError(f"Model file not found in {model_dir}")
        return HeuristicPredictor()

    try:
        return EvasionPredictor(
            model_path=model_path,
            model_type=model_type,
            model_name=requested_name,
        )
    except Exception:
        return HeuristicPredictor()
