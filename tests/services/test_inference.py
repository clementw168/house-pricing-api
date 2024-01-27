import numpy as np
import pandas as pd
import pytest

from src.services.inference import InferenceService


@pytest.fixture
def inference_service() -> InferenceService:
    return InferenceService()


def test_model_loading(inference_service: InferenceService) -> None:
    assert inference_service.model is not None


def test_predict(inference_service: InferenceService) -> None:
    features = pd.Series([0] * 4)
    prediction = inference_service.predict(features)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
