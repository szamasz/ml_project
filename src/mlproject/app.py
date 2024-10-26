from datetime import datetime

import click
import mlflow
import pandas as pd
import streamlit as st
from pydantic import BaseModel, ValidationError, confloat

current_year = datetime.now().year


def get_model(name, alias="best"):
    return mlflow.pyfunc.load_model(f"models:/{name}@{alias}")


def predict_from_data(name, input_df, alias="best"):
    try:
        best_model = get_model(name, alias)
    except Exception:
        print(f"Model {name} with alias {alias} not found")
        e = Exception(f"Model {name} with alias {alias} not found")
        return e, None
    run_id = best_model.metadata.run_id
    mape = float("{:.3f}".format(dict(dict(mlflow.get_run(run_id))["data"])["metrics"]["validation_mape"]))
    result = best_model.predict(input_df)[0]
    error = result * mape
    return result, error


class SelectedColumnsModel(BaseModel):
    city: str
    squareMeters: confloat(ge=0)
    centreDistance: confloat(ge=0)
    buildYear: confloat(ge=1000, le=current_year)
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)


@click.command()
@click.option("--run_name", type=click.STRING, required=True)
@click.option("--alias", type=click.STRING, required=False)
def app(run_name, alias):
    st.title("Apartment's price predictor")
    cols = ["city", "squareMeters", "centreDistance", "buildYear", "latitude", "longitude"]
    city = st.selectbox(
        "Select city:",
        [
            "szczecin",
            "gdynia",
            "krakow",
            "poznan",
            "bialystok",
            "gdansk",
            "wroclaw",
            "radom",
            "rzeszow",
            "lodz",
            "katowice",
            "lublin",
            "czestochowa",
            "warszawa",
            "bydgoszcz",
        ],
    )

    squareMeters = st.text_input("Enter square meteres:", value=50)
    centreDistance = st.text_input("Enter distance to the city centre (km):", value=5)
    buildYear = st.text_input("Enter build year", value=1990)
    latitude = st.text_input("Enter latitude:", value=52.112795)
    longitude = st.text_input("Enter longitude", value=19.211946)
    popup_container = st.empty()

    if not alias:
        alias = "best"

    if st.button("Execute Code"):
        data = [city, squareMeters, centreDistance, buildYear, latitude, longitude]
        data_input = {k: v for k, v in zip(cols, data)}

        try:
            SelectedColumnsModel(**data_input)
            input_df = pd.DataFrame(data_input, index=[0])
            prediction, error = predict_from_data(run_name, input_df, alias=alias)
            if isinstance(prediction, Exception):
                popup_container.markdown(
                    "<div style='background-color: red; padding: 10px; border-radius: 5px;'>"
                    f"<p style='color: white;'>{prediction}</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p style='font-size:20px;'>Expected price for this apartment is: {prediction:20,.2f} +/- {error:20,.2f}zł</p>",
                    unsafe_allow_html=True,
                )
        except ValidationError as e:
            popup_container.markdown(
                "<div style='background-color: red; padding: 10px; border-radius: 5px;'>"
                f"<p style='color: white;'>Validation error: {e}</p>"
                "</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    app()
