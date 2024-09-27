
from mlproject.optunasetup.lib.algos import init_learner
from mlproject.optunasetup.lib.columns import choose_columns
import pytest
from optuna.trial import create_trial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import mlproject.optunasetup.lib.exceptions as exceptions
from unittest.mock import create_autospec
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from mlproject.optunasetup.lib.encoders import init_encoder
from mlproject.optunasetup.lib.imputers import init_imputers
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator

parameters = [
    'linear',LinearRegression
]

@pytest.fixture
def get_trial():
    trial = create_trial(value = 1.0)
    mck_trial = create_autospec(trial, spec_set=True)
    yield mck_trial
    del trial, mck_trial

@pytest.mark.parametrize(
        "algo_name,algo_class",
        [
            ("linear",LinearRegression),
            ("forest",RandomForestRegressor),
            ("extra_forest",ExtraTreesRegressor),
            pytest.param('other_algo',LinearRegression,marks=pytest.mark.xfail(raises=exceptions.UnsupportedAlgorithm))
        ]        
) 
def test_alg_selection(get_trial,algo_name,algo_class):
    get_trial.suggest_categorical = lambda x,y: algo_name
    assert isinstance(init_learner(get_trial),algo_class)

config_test = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': [], 'exclude_columns': [], 'include_only_columns': ['city', 'squareMeters', 'centreDistance', 'buildYear', 'latitude', 'longitude']}}

config_inc_exc = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': [], 'exclude_columns': ['squareMeters'], 'include_only_columns': ['city']}}
config_par_exc = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': ['city'], 'exclude_columns': ['squareMeters','city'], 'include_only_columns': []}}
config_par_inc = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': ['city'], 'exclude_columns': [], 'include_only_columns': ['squareMeters','city']}}

all_columns = ['city', 'type', 'squareMeters', 'rooms', 'floor', 'floorCount', 'buildYear', 'latitude', 'longitude', 'centreDistance', 'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
       'pharmacyDistance', 'ownership', 'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
@pytest.mark.parametrize(
        "config, all_columns,error_msg",
        [
            pytest.param(config_inc_exc,all_columns,"You can't use exclude and include options"),
            pytest.param(config_par_exc,all_columns,"Column name cant be in parametrized and excluded at the same time"),
            pytest.param(config_par_inc,all_columns,"Column name cant be in parametrized and included at the same time")
        ]        
) 
def test_column_selection_exceptions(config, all_columns,error_msg):
    with pytest.raises(expected_exception=Exception,match=error_msg):
        choose_columns(config, all_columns)
# take all columns
config_1 = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': [], 'exclude_columns': [], 'include_only_columns': []}}
remain_1 = all_columns
parametrized_1 = []
excluded_1 = []

#include columsm without parametrized
config_2 = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': [], 'exclude_columns': [], 'include_only_columns': ['city', 'squareMeters', 'centreDistance']}}
remain_2 = ['city', 'squareMeters', 'centreDistance']
parametrized_2 = []
excluded_2 = ['type', 'rooms', 'floor', 'floorCount', 'buildYear', 'latitude', 'longitude', 'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
       'pharmacyDistance', 'ownership', 'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']

#include columns with parametirzed
config_3 = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': ['hasStorageRoom','buildYear', 'latitude', 'longitude'], 'exclude_columns': [], 'include_only_columns': ['city', 'squareMeters', 'centreDistance']}}
remain_3 = ['city', 'squareMeters', 'centreDistance']
parametrized_3 = ['hasStorageRoom','buildYear', 'latitude', 'longitude']
excluded_3 = ['type', 'rooms', 'floor', 'floorCount', 'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance', 'pharmacyDistance', 'ownership', 'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity']

#exclude columns without parametrized
config_4 = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': [], 'exclude_columns': ['city', 'squareMeters', 'centreDistance'], 'include_only_columns': []}}
excluded_4 = ['city', 'squareMeters', 'centreDistance']
parametrized_4 = []
remain_4 = ['type', 'rooms', 'floor', 'floorCount', 'buildYear', 'latitude', 'longitude', 'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
       'pharmacyDistance', 'ownership', 'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']

#exclude columns with parametrized
config_5 = {'apartments': {'source': 'test', 'target': 'price', 'parametrized_columns': ['buildYear', 'latitude', 'longitude'], 'exclude_columns': ['city', 'squareMeters', 'centreDistance'], 'include_only_columns': []}}
excluded_5 = ['city', 'squareMeters', 'centreDistance']
parametrized_5 = ['buildYear', 'latitude', 'longitude']
remain_5 = ['type', 'rooms', 'floor', 'floorCount', 'poiCount', 'schoolDistance', 'clinicDistance', 'postOfficeDistance', 'kindergartenDistance', 'restaurantDistance', 'collegeDistance',
       'pharmacyDistance', 'ownership', 'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']


@pytest.mark.parametrize(
        "config, all_columns,result_columns",
        [
            (config_1,all_columns,{'remain_columns': remain_1,'parametrized_columns':parametrized_1, 'excluded_columns':excluded_1}),
            (config_2,all_columns,{'remain_columns': remain_2,'parametrized_columns':parametrized_2, 'excluded_columns':excluded_2}),
            (config_3,all_columns,{'remain_columns': remain_3,'parametrized_columns':parametrized_3, 'excluded_columns':excluded_3}),
            (config_4,all_columns,{'remain_columns': remain_4,'parametrized_columns':parametrized_4, 'excluded_columns':excluded_4}),
            (config_5,all_columns,{'remain_columns': remain_5,'parametrized_columns':parametrized_5, 'excluded_columns':excluded_5})
        ]        
) 
def test_column_selection(config, all_columns,result_columns):
        remain_columns, parametrized_columns, exclude_columns = choose_columns(config, all_columns)
        assert set(result_columns['remain_columns']) == set(remain_columns)
        assert set(result_columns['parametrized_columns']) == set(parametrized_columns)
        assert set(result_columns['excluded_columns']) == set(exclude_columns)

@pytest.mark.parametrize(
        "enc_name,encoder_class",
        [
            ("ordinal",OrdinalEncoder),
            ("onehot",OneHotEncoder)
        ]        
)
def test_encoders(get_trial,enc_name,encoder_class):
    get_trial.suggest_categorical = lambda x,y: enc_name
    assert isinstance(init_encoder(get_trial),encoder_class)

@pytest.mark.parametrize(
        "imputer_combination,expected_imputers",
        [
            (({'imputing_method':'knn','indicate_missing':'no'}),[KNNImputer]),
            (({'imputing_method':'simple','indicate_missing':'no'}),[SimpleImputer]),
            (({'imputing_method':'knn','indicate_missing':'yes'}),[KNNImputer,MissingIndicator]),
            (({'imputing_method':'simple','indicate_missing':'yes'}),[SimpleImputer,MissingIndicator]),
        ]        
)
def test_imputers(get_trial,imputer_combination,expected_imputers):
    get_trial.suggest_categorical = lambda x,y: imputer_combination[x]
    imputer_transformer = init_imputers(get_trial)
    transformations = imputer_transformer[1].transformer_list
    assert len(transformations) == len(expected_imputers)
    for tr, tr_exp in zip(transformations,expected_imputers):
        assert isinstance(tr[1],tr_exp)