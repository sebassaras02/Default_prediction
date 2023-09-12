from fastapi import FastAPI
import uvicorn
from .test import pipeline_testing
from .models import PredictionRequest, PredictionResponse

app = FastAPI(docs_url="/")

@app.post('/predict')
def make_model_prediction(request : PredictionRequest) -> int:
    prediction = pipeline_testing(dict(request))
    return prediction


