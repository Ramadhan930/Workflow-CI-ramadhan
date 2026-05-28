import mlflow
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="financial_data_preprocessed_train.csv")
parser.add_argument("--test", default="financial_data_preprocessed_test.csv")
args = parser.parse_args()

train = pd.read_csv(args.train)
test = pd.read_csv(args.test)
X_train = train.drop('expenses', axis=1)
y_train = train['expenses']
X_test = test.drop('expenses', axis=1)
y_test = test['expenses']

mlflow.autolog()
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)