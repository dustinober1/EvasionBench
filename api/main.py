from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="EvasionBench Detector API")


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
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(pair: QAPair):
    # Placeholder: load model and run inference
    return PredictionResponse(
        prediction="direct",
        confidence=0.9,
        probabilities={"direct": 0.9, "intermediate": 0.09, "fully_evasive": 0.01},
    )
