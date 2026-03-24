from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_forward_return: float
    state: int
    timestamp: str