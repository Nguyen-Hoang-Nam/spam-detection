from pydantic import BaseModel
from typing import Any


class SpamPrediction(BaseModel):
    prediction: int
    probability: Any
