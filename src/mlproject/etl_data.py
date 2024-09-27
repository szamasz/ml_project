from mlproject.optunasetup.lib.utils import load_raw_data

def process_data():
    df_apartments = load_raw_data('apartments')

    df_apartments_1 = df_apartments.drop(["id","condition","buildingMaterial","rooms"], axis=1)

    df_apartments_1 = df_apartments_1.drop_duplicates()

    df_apartments_1.loc[df_apartments_1['type'].isna(),['type']]='Other'
    df_apartments_1.loc[df_apartments_1['hasElevator'].isna(),['hasElevator']]='Other'

    df_apartments_1[df_apartments_1.select_dtypes("object").columns.to_list()]  = df_apartments_1[df_apartments_1.select_dtypes("object").columns.to_list()].astype('category')

    df_apartments_1.to_csv("data/05_model_input/apartments.csv", index=False)