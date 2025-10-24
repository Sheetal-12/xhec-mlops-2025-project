from pydantic import BaseModel
from typing import List


class BatchPredictionOutput(BaseModel):
    predictions: List
