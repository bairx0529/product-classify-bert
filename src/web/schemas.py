from pydantic import BaseModel


class PredictRequest(BaseModel):
    title: str


class PredictResponse(BaseModel):
    title: str
    label: str
