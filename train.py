import os

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.constants import MODEL_FILE
from src.services import dao

if __name__ == "__main__":
    if os.path.exists(MODEL_FILE):
        print("Model already trained. Skipping training.")
        exit(0)

    print("---- Training Gradient boosting regression model ----")
    print("train features shape:", dao.train_features.shape)
    print("train label shape:", dao.train_label.shape)
    print("test features shape:", dao.test_features.shape)
    print("test label shape:", dao.test_label.shape)

    model = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
    model.fit(dao.train_features, dao.train_label)

    print(
        "Training set: mse = {}, r2 = {}".format(
            mean_squared_error(dao.train_label, model.predict(dao.train_features)),
            r2_score(dao.train_label, model.predict(dao.train_features)),
        )
    )

    print(
        "Test set: mse = {}, r2 = {}".format(
            mean_squared_error(dao.test_label, model.predict(dao.test_features)),
            r2_score(dao.test_label, model.predict(dao.test_features)),
        )
    )

    print("Saving model to {}".format(MODEL_FILE))

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
