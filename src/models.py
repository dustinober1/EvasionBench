from typing import Any


class ModelWrapper:
    """Minimal model wrapper interface."""

    def __init__(self, model: Any):
        self.model = model

    def predict(self, questions, answers):
        raise NotImplementedError
