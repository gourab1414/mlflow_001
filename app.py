import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import  urlparse
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import  logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metric(actual, predicted):
    mse = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, r2
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    train, test = train_test_split(data)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    with mlflow.start_run():
        lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
        lr.fit(train_x, train_y)
        predicted = lr.predict(test_x)
        mse, r2 = eval_metric(test_y, predicted)

        print("mse", mse)
        print("r2", r2)


        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        predictions_train = lr.predict(train_x)
        signature = infer_signature(train_x, predictions_train)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)

        mlflow.sklearn.log_model(lr, "wine_model")


