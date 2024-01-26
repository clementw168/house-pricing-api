import os.path as osp

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src.constants import DATA_FILE, DATA_LINK, RANDOM_STATE, SPLIT_RATIO, TARGET


class DAO(object):
    def __init__(self):
        if not osp.exists(DATA_FILE):
            self.download_data()

        (
            self.train_features,
            self.test_features,
            self.train_label,
            self.test_label,
        ) = self.load_data()

    def download_data(self) -> None:
        r = requests.get(DATA_LINK)
        with open(DATA_FILE, "wb") as f:
            f.write(r.content)

    def load_data(self) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        dataframe = pd.read_csv(DATA_FILE)
        price = dataframe.pop(TARGET)

        train_features, test_features, train_label, test_label = train_test_split(
            dataframe, price, test_size=SPLIT_RATIO, random_state=RANDOM_STATE
        )

        print(
            f"Training features shape: {train_features.shape}\n"
            f"Training label shape: {train_label.shape}\n"
            f"Testing features shape: {test_features.shape}\n"
            f"Testing label shape: {test_label.shape}\n"
        )

        return train_features, test_features, train_label, test_label

    def get_row(
        self, row_id: int, is_train: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if is_train:
            return self.train_features.iloc[row_id], self.train_label.iloc[row_id]

        return self.test_features.iloc[row_id], self.test_label.iloc[row_id]
