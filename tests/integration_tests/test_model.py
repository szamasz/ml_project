import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

df_1 = pd.read_csv("tests/integration_tests/data/apartments_1.csv")
#df_1 = pd.read_csv('data/apartments_1.csv')

model_name = "apartments_selected_columns_linear"

model=mlflow.pyfunc.load_model(f"models:/{model_name}/1")

y_pred = model.predict(df_1)
y_true = df_1["price"]

mape = mean_absolute_percentage_error(y_true,y_pred)
print(f"Price predicted with mape: {mape}")

print("Integration test sucessful")
