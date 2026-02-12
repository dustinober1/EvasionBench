import sys
from pathlib import Path
from typing import Optional  # noqa: UP035

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model

app = FastAPI(title="EvasionBench Detector API")

# Lazy model loading
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = load_model()
    return _predictor


class QAPair(BaseModel):
    question: str
    answer: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model_name: Optional[str] = None  # noqa: UP045
    explanation: Optional[list] = None  # noqa: UP045


@app.get("/health")
async def health():
    predictor = get_predictor()
    model_name = getattr(
        predictor,
        "model_name",
        getattr(predictor, "model_type", type(predictor).__name__),
    )
    return {"status": "ok", "model": model_name}


@app.post("/predict", response_model=PredictionResponse)
async def predict(pair: QAPair):
    # Run real inference
    predictor = get_predictor()
    result = predictor.predict_single(pair.question, pair.answer)
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_name=result.get("model_name"),
    )
