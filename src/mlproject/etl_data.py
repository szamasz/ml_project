import os
from datetime import datetime
from pathlib import Path

import pandera as pa
from pandera.typing import Series
from pandera.typing.common import Category, Float64, Int16, Int64

from mlproject.optunasetup.lib.utils import load_raw_data

current_year = datetime.now().year

model_directory = "data/05_model_input/"


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


def handle_reference(dataset, df):
    ref_name = f"ref_{dataset}.csv"
    cur_dir = os.path.abspath(os.curdir)
    directory = Path(cur_dir) / Path(model_directory)
    files = [file.name for file in directory.iterdir() if file.is_file() and file.name == ref_name]
    if len(files) == 0:
        df.to_csv(f"{model_directory}/{ref_name}", index=False)


def process_data(dataset, detect_drift):
    df_apartments = load_raw_data(dataset)

    df_apartments_1 = df_apartments.drop(["id", "condition", "buildingMaterial", "rooms"], axis=1)

    df_apartments_1 = df_apartments_1.drop_duplicates()

    df_apartments_1 = df_apartments_1[
        df_apartments_1["ownership"] != "udział"
    ]  # dropping 1 rows that have unexpected value in month 10 CSV and 2 rows in month 11 CSV

    df_apartments_1.loc[df_apartments_1["type"].isna(), ["type"]] = "Other"
    df_apartments_1.loc[df_apartments_1["hasElevator"].isna(), ["hasElevator"]] = "Other"

    df_apartments_1[df_apartments_1.select_dtypes("object").columns.to_list()] = df_apartments_1[
        df_apartments_1.select_dtypes("object").columns.to_list()
    ].astype("category")

    df_apartments_1[["floor", "floorCount", "buildYear", "poiCount"]] = df_apartments_1[
        ["floor", "floorCount", "buildYear", "poiCount"]
    ].astype("Int16")

    ApartmentsSchema.validate(df_apartments_1)

    df_apartments_1.to_csv(f"{model_directory}/{dataset}.csv", index=False)

    if detect_drift:
        handle_reference(dataset, df_apartments_1)
