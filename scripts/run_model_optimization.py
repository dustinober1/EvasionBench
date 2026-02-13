"""Run phase-8 model optimization with CV + holdout selection."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import compute_classification_metrics, write_evaluation_artifacts

DEFAULT_ACCURACY_FLOOR = 0.6431560071727436
SUPPORTED_FAMILIES = ("logreg", "tree", "boosting")


class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    """SVD wrapper that safely adapts component count to sparse feature size."""

    def __init__(self, n_components: int = 256, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._svd: TruncatedSVD | None = None
        self.actual_components_: int | None = None

    def fit(self, X, y=None):
        n_features = int(X.shape[1])
        components = min(self.n_components, max(1, n_features - 1))
        self.actual_components_ = components
        self._svd = TruncatedSVD(
            n_components=components,
            random_state=self.random_state,
        )
        self._svd.fit(X)
        return self

    def transform(self, X):
        if self._svd is None:
            raise RuntimeError("SafeTruncatedSVD must be fit before transform")
        return self._svd.transform(X)


@dataclass(frozen=True)
class Candidate:
    trial_id: str
    family: str
    estimator: Any
    params: dict[str, Any]


@dataclass(frozen=True)
class CandidateResult:
    trial_id: str
    family: str
    params: dict[str, Any]
    cv_metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    holdout_fully_evasive_f1: float


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared parquet dataset")
    parser.add_argument(
        "--output-root",
        default="artifacts/models/phase8",
        help="Output directory for phase-8 optimization artifacts",
    )
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--selection-metric",
        choices=["f1_macro", "accuracy", "precision_macro", "recall_macro"],
        default="f1_macro",
    )
    parser.add_argument("--accuracy-floor", type=float, default=DEFAULT_ACCURACY_FLOOR)
    parser.add_argument(
        "--families",
        choices=["all", *SUPPORTED_FAMILIES],
        default="all",
        help="Model families to optimize",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional cap on number of candidates (0 means no cap)",
    )
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment-name", default="evasionbench-phase8-optimization")
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid"],
        default="none",
        help="Enable calibration candidates for logistic regression",
    )
    return parser.parse_args()


def _selected_families(raw: str) -> list[str]:
    if raw == "all":
        return list(SUPPORTED_FAMILIES)
    return [raw]


def _build_features(frame: pd.DataFrame) -> pd.Series:
    question = frame["question"].fillna("").astype(str)
    answer = frame["answer"].fillna("").astype(str)
    return (question + " [SEP] " + answer).astype(str)


def _validate_input(frame: pd.DataFrame, target_col: str) -> None:
    missing = [
        col for col in ("question", "answer", target_col) if col not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _split_holdout(
    frame: pd.DataFrame,
    *,
    target_col: str,
    random_state: int,
    holdout_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels = frame[target_col].fillna("unknown").astype(str)
    stratify = labels if labels.value_counts().min() >= 2 else None

    train_df, holdout_df = train_test_split(
        frame,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify,
    )

    split_info = {
        "method": "stratified_train_holdout",
        "random_state": random_state,
        "holdout_size": holdout_size,
        "stratify": stratify is not None,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
    }
    return train_df, holdout_df, split_info


def _logreg_candidates(random_state: int, calibration_method: str) -> list[Candidate]:
    class_weight_options: list[Any] = [
        "balanced",
        {"direct": 1.0, "intermediate": 1.2, "fully_evasive": 2.5},
    ]
    c_options = [0.3, 1.0]
    ngram_options = [(1, 2)]
    min_df_options = [2]
    max_features_options = [10000]
    solver_options = ["liblinear"]

    candidates: list[Candidate] = []
    trial_idx = 1

    for c_value in c_options:
        for ngram_range in ngram_options:
            for min_df in min_df_options:
                for max_features in max_features_options:
                    for solver in solver_options:
                        for class_weight in class_weight_options:
                            params = {
                                "C": c_value,
                                "ngram_range": ngram_range,
                                "min_df": min_df,
                                "max_features": max_features,
                                "solver": solver,
                                "class_weight": class_weight,
                            }
                            pipeline = Pipeline(
                                steps=[
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            ngram_range=ngram_range,
                                            min_df=min_df,
                                            max_features=max_features,
                                        ),
                                    ),
                                    (
                                        "clf",
                                        LogisticRegression(
                                            C=c_value,
                                            solver=solver,
                                            class_weight=class_weight,
                                            max_iter=1000,
                                            random_state=random_state,
                                        ),
                                    ),
                                ]
                            )
                            trial_id = f"logreg_{trial_idx:04d}"
                            candidates.append(
                                Candidate(
                                    trial_id=trial_id,
                                    family="logreg",
                                    estimator=pipeline,
                                    params={**params, "calibration": "none"},
                                )
                            )

                            if calibration_method != "none":
                                calibrated = CalibratedClassifierCV(
                                    estimator=pipeline,
                                    method=calibration_method,
                                    cv=3,
                                )
                                candidates.append(
                                    Candidate(
                                        trial_id=f"{trial_id}_cal",
                                        family="logreg",
                                        estimator=calibrated,
                                        params={
                                            **params,
                                            "calibration": calibration_method,
                                            "calibration_cv": 3,
                                        },
                                    )
                                )

                            trial_idx += 1

    return candidates


def _tree_candidates(random_state: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    trial_idx = 1

    for max_features in (5000,):
        for max_depth in (None, 10):
            for n_estimators in (200,):
                pipeline = Pipeline(
                    steps=[
                        (
                            "tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=1,
                                max_features=max_features,
                            ),
                        ),
                        (
                            "clf",
                            RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                class_weight="balanced",
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                candidates.append(
                    Candidate(
                        trial_id=f"tree_{trial_idx:03d}",
                        family="tree",
                        estimator=pipeline,
                        params={
                            "max_features": max_features,
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                            "ngram_range": (1, 2),
                            "min_df": 1,
                        },
                    )
                )
                trial_idx += 1

    return candidates


def _boosting_candidates(random_state: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    trial_idx = 1

    for max_features in (5000,):
        for max_depth in (None, 10):
            for learning_rate in (0.1,):
                for max_iter in (100,):
                    pipeline = Pipeline(
                        steps=[
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    ngram_range=(1, 2),
                                    min_df=1,
                                    max_features=max_features,
                                ),
                            ),
                            (
                                "svd",
                                SafeTruncatedSVD(
                                    n_components=256,
                                    random_state=random_state,
                                ),
                            ),
                            (
                                "clf",
                                HistGradientBoostingClassifier(
                                    learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    max_iter=max_iter,
                                    random_state=random_state,
                                ),
                            ),
                        ]
                    )
                    candidates.append(
                        Candidate(
                            trial_id=f"boosting_{trial_idx:03d}",
                            family="boosting",
                            estimator=pipeline,
                            params={
                                "max_features": max_features,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "max_iter": max_iter,
                                "n_components": 256,
                                "ngram_range": (1, 2),
                                "min_df": 1,
                            },
                        ),
                    )
                    trial_idx += 1

    return candidates


def _candidate_space(
    families: list[str], random_state: int, calibration_method: str
) -> list[Candidate]:
    candidates: list[Candidate] = []
    if "logreg" in families:
        candidates.extend(_logreg_candidates(random_state, calibration_method))
    if "tree" in families:
        candidates.extend(_tree_candidates(random_state))
    if "boosting" in families:
        candidates.extend(_boosting_candidates(random_state))
    return candidates


def _evaluate_candidate(
    candidate: Candidate,
    train_text: pd.Series,
    train_labels: pd.Series,
    holdout_text: pd.Series,
    holdout_labels: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
) -> CandidateResult:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, float]] = []
    for train_idx, valid_idx in cv.split(train_text, train_labels):
        X_train_fold = train_text.iloc[train_idx]
        X_valid_fold = train_text.iloc[valid_idx]
        y_train_fold = train_labels.iloc[train_idx]
        y_valid_fold = train_labels.iloc[valid_idx]

        candidate.estimator.fit(X_train_fold, y_train_fold)
        pred_valid = candidate.estimator.predict(X_valid_fold)
        fold_metrics.append(
            compute_classification_metrics(y_true=y_valid_fold, y_pred=pred_valid)
        )

    cv_metrics = {
        "accuracy": float(sum(m["accuracy"] for m in fold_metrics) / len(fold_metrics)),
        "f1_macro": float(sum(m["f1_macro"] for m in fold_metrics) / len(fold_metrics)),
        "precision_macro": float(
            sum(m["precision_macro"] for m in fold_metrics) / len(fold_metrics)
        ),
        "recall_macro": float(
            sum(m["recall_macro"] for m in fold_metrics) / len(fold_metrics)
        ),
    }

    candidate.estimator.fit(train_text, train_labels)
    holdout_pred = candidate.estimator.predict(holdout_text)
    holdout_metrics = compute_classification_metrics(
        y_true=holdout_labels,
        y_pred=holdout_pred,
    )
    report = classification_report(
        holdout_labels,
        holdout_pred,
        output_dict=True,
        zero_division=0,
    )
    fully_evasive_f1 = float(report.get("fully_evasive", {}).get("f1-score", 0.0))

    return CandidateResult(
        trial_id=candidate.trial_id,
        family=candidate.family,
        params=candidate.params,
        cv_metrics=cv_metrics,
        holdout_metrics=holdout_metrics,
        holdout_fully_evasive_f1=fully_evasive_f1,
    )


def _to_trial_rows(results: list[CandidateResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "trial_id": result.trial_id,
                "family": result.family,
                "params": json.dumps(result.params, sort_keys=True),
                "cv_accuracy": result.cv_metrics["accuracy"],
                "cv_f1_macro": result.cv_metrics["f1_macro"],
                "cv_precision_macro": result.cv_metrics["precision_macro"],
                "cv_recall_macro": result.cv_metrics["recall_macro"],
                "holdout_accuracy": result.holdout_metrics["accuracy"],
                "holdout_f1_macro": result.holdout_metrics["f1_macro"],
                "holdout_precision_macro": result.holdout_metrics["precision_macro"],
                "holdout_recall_macro": result.holdout_metrics["recall_macro"],
                "holdout_fully_evasive_f1": result.holdout_fully_evasive_f1,
            }
        )
    return rows


def _select_winner(
    results: list[CandidateResult],
    *,
    selection_metric: str,
    accuracy_floor: float,
) -> tuple[CandidateResult, str]:
    eligible = [r for r in results if r.holdout_metrics["accuracy"] >= accuracy_floor]
    rule = "accuracy_floor_enforced"

    pool = eligible
    if not pool:
        pool = results
        rule = "accuracy_floor_relaxed"

    ordered = sorted(
        pool,
        key=lambda r: (
            float(r.holdout_metrics[selection_metric]),
            float(r.holdout_fully_evasive_f1),
        ),
        reverse=True,
    )
    return ordered[0], rule


def _persist_winner_model(
    candidate: Candidate,
    train_text: pd.Series,
    train_labels: pd.Series,
    winner_dir: Path,
) -> Path:
    winner_dir.mkdir(parents=True, exist_ok=True)
    candidate.estimator.fit(train_text, train_labels)
    model_path = winner_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(candidate.estimator, f)
    return model_path


def _metadata(
    *,
    candidate: Candidate,
    split_info: dict[str, Any],
    args: argparse.Namespace,
    train_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    return {
        "model_family": candidate.family,
        "split_seed": args.random_state,
        "test_size": args.holdout_size,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_config": {
            "combined_feature": "question [SEP] answer",
        },
        "model_config": candidate.params,
        "split_metadata": split_info,
        "evaluation_protocol": "stratified_5fold_cv_plus_holdout",
        "cv_folds": args.cv_folds,
        "selection_metric": args.selection_metric,
        "accuracy_floor": args.accuracy_floor,
        "git_sha": _git_sha(),
    }


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.input)
    _validate_input(frame, args.target_col)
    frame = frame.copy()
    frame[args.target_col] = frame[args.target_col].fillna("unknown").astype(str)

    train_df, holdout_df, split_info = _split_holdout(
        frame,
        target_col=args.target_col,
        random_state=args.random_state,
        holdout_size=args.holdout_size,
    )

    train_text = _build_features(train_df)
    train_labels = train_df[args.target_col].astype(str)
    holdout_text = _build_features(holdout_df)
    holdout_labels = holdout_df[args.target_col].astype(str)

    families = _selected_families(args.families)
    candidates = _candidate_space(families, args.random_state, args.calibration_method)
    if args.max_trials > 0:
        candidates = candidates[: args.max_trials]

    if not candidates:
        raise ValueError("No optimization candidates available")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    results: list[CandidateResult] = []
    candidate_index = {candidate.trial_id: candidate for candidate in candidates}

    with mlflow.start_run(run_name="phase8-optimization"):
        for candidate in candidates:
            result = _evaluate_candidate(
                candidate,
                train_text,
                train_labels,
                holdout_text,
                holdout_labels,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
            )
            results.append(result)

        winner_result, winner_rule = _select_winner(
            results,
            selection_metric=args.selection_metric,
            accuracy_floor=args.accuracy_floor,
        )
        winner_candidate = candidate_index[winner_result.trial_id]

        trials_df = pd.DataFrame(_to_trial_rows(results))
        trials_df = trials_df.sort_values(
            by=["holdout_f1_macro", "holdout_accuracy", "trial_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        trials_path = output_root / "optimization_trials.csv"
        trials_df.to_csv(trials_path, index=False)

        cv_summary = {
            "evaluation_protocol": "stratified_5fold_cv_plus_holdout",
            "selection_metric": args.selection_metric,
            "accuracy_floor": args.accuracy_floor,
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
            "split": split_info,
            "families": families,
            "trial_count": len(results),
            "winner_rule": winner_rule,
            "winner_trial_id": winner_result.trial_id,
            "git_sha": _git_sha(),
        }
        cv_summary_path = output_root / "cv_summary.json"
        cv_summary_path.write_text(
            json.dumps(cv_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        winner_dir = output_root / winner_result.family
        model_path = _persist_winner_model(
            winner_candidate,
            train_text,
            train_labels,
            winner_dir,
        )

        y_pred = winner_candidate.estimator.predict(holdout_text)
        holdout_metrics = compute_classification_metrics(holdout_labels, y_pred)
        metadata = _metadata(
            candidate=winner_candidate,
            split_info=split_info,
            args=args,
            train_rows=len(train_df),
            test_rows=len(holdout_df),
        )
        eval_paths = write_evaluation_artifacts(
            winner_dir,
            holdout_labels,
            y_pred,
            holdout_metrics,
            metadata,
        )

        holdout_metrics_path = output_root / "holdout_metrics.json"
        holdout_metrics_path.write_text(
            json.dumps(holdout_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        selected_payload = {
            "best_model_family": winner_result.family,
            "winner_trial_id": winner_result.trial_id,
            "selection_metric": args.selection_metric,
            "winner_rule": winner_rule,
            "accuracy_floor": args.accuracy_floor,
            "metrics": holdout_metrics,
            "deltas_vs_accuracy_floor": {
                "accuracy_delta": holdout_metrics["accuracy"] - args.accuracy_floor,
            },
            "holdout_fully_evasive_f1": winner_result.holdout_fully_evasive_f1,
            "model_path": str(model_path),
            "artifact_root": str(winner_dir),
            "evaluation_protocol": "stratified_5fold_cv_plus_holdout",
            "cv_folds": args.cv_folds,
            "random_state": args.random_state,
            "git_sha": _git_sha(),
        }
        selected_path = output_root / "selected_model.json"
        selected_path.write_text(
            json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        mlflow.log_params(
            {
                "pipeline_stage": "phase-08-model-optimization",
                "selection_metric": args.selection_metric,
                "accuracy_floor": args.accuracy_floor,
                "cv_folds": args.cv_folds,
                "holdout_size": args.holdout_size,
                "families": json.dumps(families),
                "trial_count": len(results),
                "winner_trial_id": winner_result.trial_id,
                "winner_family": winner_result.family,
            }
        )
        mlflow.log_metrics(
            {
                "winner_accuracy": holdout_metrics["accuracy"],
                "winner_f1_macro": holdout_metrics["f1_macro"],
                "winner_precision_macro": holdout_metrics["precision_macro"],
                "winner_recall_macro": holdout_metrics["recall_macro"],
                "winner_fully_evasive_f1": winner_result.holdout_fully_evasive_f1,
            }
        )

        for path in [
            trials_path,
            cv_summary_path,
            holdout_metrics_path,
            selected_path,
            model_path,
            *eval_paths,
        ]:
            mlflow.log_artifact(str(path), artifact_path="phase8")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "winner_family": winner_result.family,
                "winner_trial_id": winner_result.trial_id,
                "metrics": holdout_metrics,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
