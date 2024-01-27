import pytest
from fastapi.testclient import TestClient

from src.app import AppCreator
from src.services.dao import DAO
from src.services.inference import InferenceService


@pytest.fixture
def client() -> TestClient:
    app = AppCreator(DAO(), InferenceService()).get_app()

    return TestClient(app)


def test_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Inference API is running"}


def test_predict(client: TestClient) -> None:
    response = client.get("/predict/test/0")

    assert response.status_code == 200
    assert response.json()["actual_price"] == 154500


def test_row(client: TestClient) -> None:
    response = client.get("/row/test/0")

    assert response.status_code == 200
    assert response.json()["price"] == 154500
    assert response.json()["features"] == {
        "OverallQual": 6,
        "YearBuilt": 1963,
        "TotalBsmtSF": 1059,
        "GrLivArea": 1068,
    }
