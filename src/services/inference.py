import os

import joblib
import numpy as np
import pandas as pd

from src.constants import MODEL_FILE


class InferenceService(object):
    def __init__(self):
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model file {MODEL_FILE} not found. Train the model first."
            )

        self.model = joblib.load(MODEL_FILE)

    def predict(self, features: pd.Series) -> np.ndarray:
        return self.model.predict(features.to_numpy().reshape(1, -1))
