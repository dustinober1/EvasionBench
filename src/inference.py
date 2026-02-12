"""Model inference module for EvasionBench."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

    def __init__(self, model_path: str | Path, model_type: str = "boosting"):
        """
        Initialize predictor with a trained model.

        Args:
            model_path: Path to model pickle file
            model_type: Model type - "logreg" or "boosting"/"tree" (has bundle)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        _patch_numpy_pickle_compat()
        with open(self.model_path, "rb") as f:
            if model_type == "logreg":
                # Logreg is a sklearn Pipeline (TfidfVectorizer + LogisticRegression)
                self.model = pickle.load(f)
                self.vectorizer = None
                self.svd = None
            else:
                # Tree/boosting models have a bundle
                bundle = pickle.load(f)
                self.model = bundle["model"]
                self.vectorizer = bundle["vectorizer"]
                self.svd = bundle.get("svd")  # Only for boosting

        # Get class labels from model
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        else:
            # For pipeline models
            self.classes_ = self.model.steps[-1][1].classes_

    def _build_features(self, questions: list[str], answers: list[str]) -> pd.Series:
        """Build features from question-answer pairs using [SEP] token."""
        questions_clean = [str(q) if q else "" for q in questions]
        answers_clean = [str(a) if a else "" for a in answers]
        return pd.Series(
            [f"{q} [SEP] {a}" for q, a in zip(questions_clean, answers_clean)]
        )

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
            Dictionary with:
            - predictions: List of predicted labels
            - probabilities: List of probability dicts (if return_proba=True)
            - confidence: List of confidence scores (max probability)
        """
        # Normalize inputs to lists
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]

        if len(questions) != len(answers):
            raise ValueError("Number of questions and answers must match")

        # Build features
        features = self._build_features(questions, answers)

        # Make predictions based on model type
        if self.model_type == "logreg":
            # Pipeline handles everything
            predictions = self.model.predict(features)
            if return_proba:
                probas = self.model.predict_proba(features)
        else:
            # Tree/boosting: need to vectorize manually
            X = self.vectorizer.transform(features)
            if self.svd is not None:
                X = self.svd.transform(X.toarray())
            predictions = self.model.predict(X)
            if return_proba:
                probas = self.model.predict_proba(X)

        # Format results
        result = {
            "predictions": predictions.tolist(),
        }

        if return_proba:
            # Convert probabilities to list of dicts
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
            Dictionary with:
            - prediction: Predicted label
            - probabilities: Probability dict (if return_proba=True)
            - confidence: Confidence score (if return_proba=True)
        """
        result = self.predict([question], [answer], return_proba=return_proba)

        # Unpack single result
        output = {"prediction": result["predictions"][0]}
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


def load_model(
    model_name: str = "boosting", artifacts_root: str | Path | None = None
) -> EvasionPredictor | HeuristicPredictor:
    """
    Load a trained EvasionBench model by name.

    Args:
        model_name: Model name - "boosting", "tree", or "logreg"
        artifacts_root: Root directory for artifacts (defaults to project root / artifacts)

    Returns:
        EvasionPredictor instance
    """
    if artifacts_root is None:
        # Default to project root / artifacts
        artifacts_root = Path(__file__).resolve().parents[1] / "artifacts"
    else:
        artifacts_root = Path(artifacts_root)

    model_dir = artifacts_root / "models" / "phase5" / model_name

    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    # Determine model file and type
    if model_name == "logreg":
        model_path = model_dir / "model.pkl"
        model_type = "logreg"
    else:
        model_path = model_dir / "model_bundle.pkl"
        model_type = model_name

    try:
        return EvasionPredictor(model_path, model_type=model_type)
    except Exception:
        return HeuristicPredictor()
