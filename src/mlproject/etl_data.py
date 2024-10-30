from datetime import datetime

import pandera as pa
from pandera.typing import Series
from pandera.typing.common import Category, Float64, Int16, Int64

from mlproject.optunasetup.lib.utils import load_raw_data

current_year = datetime.now().year


class ApartmentsSchema(pa.DataFrameModel):
    city: Series[Category] = pa.Field(
        dtype_kwargs={
            "categories": [
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
            "ordered": False,
        },
    )
    type: Series[Category] = pa.Field(
        dtype_kwargs={"categories": ["apartmentBuilding", "blockOfFlats", "tenement", "Other"], "ordered": False},
    )
    squareMeters: Series[Float64] = pa.Field(ge=0, nullable=True)
    floor: Series[Int16] = pa.Field(ge=0, nullable=True)
    floorCount: Series[Int16] = pa.Field(ge=0, nullable=True)
    buildYear: Series[Int16] = pa.Field(ge=1000, le=current_year, nullable=True)
    latitude: Series[Float64] = pa.Field(ge=-90, le=90)
    longitude: Series[Float64] = pa.Field(ge=-90, le=90)
    centreDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    poiCount: Series[Int16] = pa.Field(ge=0, nullable=True)
    schoolDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    clinicDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    postOfficeDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    kindergartenDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    restaurantDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    collegeDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    pharmacyDistance: Series[Float64] = pa.Field(ge=0, nullable=True)
    ownership: Series[Category] = pa.Field(
        dtype_kwargs={"categories": ["condominium", "cooperative"], "ordered": False},
    )
    hasParkingSpace: Series[Category] = pa.Field(dtype_kwargs={"categories": ["no", "yes"], "ordered": False})
    hasBalcony: Series[Category] = pa.Field(dtype_kwargs={"categories": ["no", "yes"], "ordered": False})
    hasElevator: Series[Category] = pa.Field(dtype_kwargs={"categories": ["no", "yes", "Other"], "ordered": False})
    hasSecurity: Series[Category] = pa.Field(dtype_kwargs={"categories": ["no", "yes"], "ordered": False})
    hasStorageRoom: Series[Category] = pa.Field(dtype_kwargs={"categories": ["no", "yes"], "ordered": False})
    price: Series[Int64] = pa.Field(ge=0, nullable=True)


apartments_dtype = {
    "city": "string",
    "type": "string",
    "squareMeters": "Float64",
    "floor": "Int16",
    "floorCount": "Int16",
    "buildYear": "Int16",
    "latitude": "Float64",
    "longitude": "Float64",
    "centreDistance": "Float64",
    "poiCount": "Int16",
    "schoolDistance": "Float64",
    "clinicDistance": "Float64",
    "postOfficeDistance": "Float64",
    "kindergartenDistance": "Float64",
    "restaurantDistance": "Float64",
    "collegeDistance": "Float64",
    "pharmacyDistance": "Float64",
    "ownership": "string",
    "hasParkingSpace": "string",
    "hasBalcony": "string",
    "hasElevator": "string",
    "hasSecurity": "string",
    "hasStorageRoom": "string",
    "price": "Int64",
}


def process_data():
    df_apartments = load_raw_data("apartments")

    df_apartments_1 = df_apartments.drop(["id", "condition", "buildingMaterial", "rooms"], axis=1)

    df_apartments_1 = df_apartments_1.drop_duplicates()

    df_apartments_1.loc[df_apartments_1["type"].isna(), ["type"]] = "Other"
    df_apartments_1.loc[df_apartments_1["hasElevator"].isna(), ["hasElevator"]] = "Other"

    df_apartments_1[df_apartments_1.select_dtypes("object").columns.to_list()] = df_apartments_1[
        df_apartments_1.select_dtypes("object").columns.to_list()
    ].astype("category")

    df_apartments_1[["floor", "floorCount", "buildYear", "poiCount"]] = df_apartments_1[
        ["floor", "floorCount", "buildYear", "poiCount"]
    ].astype("Int16")

    ApartmentsSchema.validate(df_apartments_1)

    df_apartments_1.to_csv("data/apartments.csv", index=False)
