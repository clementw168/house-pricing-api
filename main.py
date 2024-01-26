from fastapi import FastAPI

from src.models import TrainTest
from src.services import dao, inference_service

app = FastAPI()


@app.get("/")
async def root() -> dict:
    return {"message": "Inference API is running"}


@app.get("/predict/{train_or_test}/{row_id}")
async def predict(train_or_test: TrainTest, row_id: int) -> dict:
    is_train = train_or_test == TrainTest.TRAIN

    features, price = dao.get_row(row_id, is_train)
    predicted_price = inference_service.predict(features)

    return {"predicted_price": float(predicted_price), "actual_price": float(price)}


@app.get("/row/{train_or_test}/{row_id}")
async def row(train_or_test: TrainTest, row_id: int) -> dict:
    is_train = train_or_test == TrainTest.TRAIN

    features, price = dao.get_row(row_id, is_train)

    return {"features": features.to_dict(), "price": float(price)}
