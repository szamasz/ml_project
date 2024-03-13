import streamlit as st
from lib.utils import predict_model
import pandas as pd
from pydantic import BaseModel, confloat, ValidationError
from datetime import datetime
import mlflow

current_year = datetime.now().year

def predict_from_data(name, input_df, alias = "best"):
    best_model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
    result = best_model.predict(input_df)[0]
    return result

class SelectedColumnsModel(BaseModel):
    city: str
    squareMeters: confloat(ge=0)
    centreDistance: confloat(ge=0)
    buildYear: confloat(ge=1000,le=current_year)
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)

def app():
    st.title("Apartment's price predictor")
    cols = ['city', 'squareMeters','centreDistance','buildYear','latitude','longitude']
    city = st.selectbox("Select city:", ['szczecin', 'gdynia', 'krakow', 'poznan', 'bialystok', 'gdansk', 'wroclaw', 'radom', 'rzeszow', 'lodz', 'katowice', 'lublin','czestochowa', 'warszawa', 'bydgoszcz'])

    squareMeters = st.text_input("Enter square meteres:", value=50)
    centreDistance = st.text_input("Enter distance to the city centre (km):", value=5)
    buildYear = st.text_input("Enter build year", value=1990)
    latitude = st.text_input("Enter latitude:", value=52.112795   )
    longitude = st.text_input("Enter longitude", value=19.211946)
    popup_container = st.empty()


    if st.button("Execute Code"):
        #output_result = predict_model('apartments_selected_columns')

        data=[city, squareMeters, centreDistance, buildYear,latitude, longitude]
        data_input = {k:v for k,v in zip(cols,data)}

        try:
            SelectedColumnsModel(**data_input)
        except ValidationError as e:
            popup_container.markdown("<div style='background-color: red; padding: 10px; border-radius: 5px;'>"
                                 f"<p style='color: white;'>Validation error: {e}</p>"
                                 "</div>", unsafe_allow_html=True)

        input_df = pd.DataFrame(data_input, index=[0])
        prediction = predict_from_data('apartments_selected_columns', input_df, alias = "best")

        st.markdown("<p style='font-size:20px;'>Expected price for this apartment is: {:20,.2f}zł</p>".format(prediction), unsafe_allow_html=True)

if __name__ == "__main__":
    app()