from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from pathlib import Path
import sys

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
        _predictor = load_model("boosting")
    return _predictor


class QAPair(BaseModel):
    question: str
    answer: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: Optional[list] = None


@app.get("/health")
async def health():
    return {"status": "ok", "model": "boosting"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(pair: QAPair):
    # Run real inference
    predictor = get_predictor()
    result = predictor.predict_single(pair.question, pair.answer)
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )
