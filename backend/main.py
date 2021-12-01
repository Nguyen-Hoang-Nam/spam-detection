from models.spam import SpamPrediction

from os.path import dirname, join, realpath

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import uvicorn

app = FastAPI(title="Spam detection", description="Check spam message", version="1.0")


origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open(join(dirname(realpath(__file__)), "models/spam.pkl"), "rb") as f:
    model = load(f)


@app.post("/predict", response_model=SpamPrediction)
async def predict(data: str):
    prediction = model.predict([data])
    probability = model.predict_proba([data]).tolist()

    return {
        "prediction": prediction,
        "probability": probability,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")
