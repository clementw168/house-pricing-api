from fastapi import FastAPI

from src.models import TrainTest
from src.services.dao import DAO
from src.services.inference import InferenceService


class AppCreator(object):
    def __init__(self, dao: DAO, inference_service: InferenceService):
        self.fastapi_app = FastAPI()
        self.dao = dao
        self.inference_service = inference_service

        @self.fastapi_app.get("/")
        async def root() -> dict:
            return {"message": "Inference API is running"}

        @self.fastapi_app.get("/predict/{train_or_test}/{row_id}")
        async def predict(train_or_test: TrainTest, row_id: int) -> dict:
            is_train = train_or_test == TrainTest.TRAIN

            features, price = dao.get_row(row_id, is_train)
            predicted_price = inference_service.predict(features)

            return {
                "predicted_price": float(predicted_price),
                "actual_price": float(price),
            }

        @self.fastapi_app.get("/row/{train_or_test}/{row_id}")
        async def row(train_or_test: TrainTest, row_id: int) -> dict:
            is_train = train_or_test == TrainTest.TRAIN

            features, price = dao.get_row(row_id, is_train)

            return {"features": features.to_dict(), "price": float(price)}

    def get_app(self) -> FastAPI:
        return self.fastapi_app
