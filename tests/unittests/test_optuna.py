
from mlproject.optunasetup.lib.algos import init_learner
from mlproject.optunasetup.lib.utils import get_reduced_features, evaluate_model, save_best_study
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
from optuna import create_study
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal
from unittest.mock import Mock, MagicMock
import pickle
import base64
from unittest.mock import call
from hashlib import sha256

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

@pytest.fixture(scope="session")
def get_dataframes():
    X_train = DataFrame({
        'city': ['Warszawa', 'Gdansk', 'Wroclaw'],
        'floor': [1, 5, 6],
        'squareMeters': [21.2, 85.0, 99.3],
        'hasElevator': [False, True, False],
        'clinicDistance': [21.2, 85.0, 99.3],
    })
    
    X_val = DataFrame({
        'city': ['Krakow', 'Gdansk', 'Bydgoszcz'],
        'floor': [12, 2, 6],
        'squareMeters': [31.2, 35.0, 39.3],
        'hasElevator': [True, False, True],
        'clinicDistance': [31.2, 35.0, 39.3],
    })
    expected_X_train = DataFrame({
        'city': ['Warszawa', 'Gdansk', 'Wroclaw'],
        'squareMeters': [21.2, 85.0, 99.3],
        'hasElevator': [False, True, False]
    })
    
    expected_X_val = DataFrame({
        'city': ['Krakow', 'Gdansk', 'Bydgoszcz'],
        'squareMeters': [31.2, 35.0, 39.3],
        'hasElevator': [True, False, True]
    })
    y_train = Series([100, 200, 300])
    y_val = Series([110, 210, 310])
    return X_train, X_val, expected_X_train, expected_X_val, y_train, y_val

def test_get_reduced_features(get_dataframes):
    X_train, X_val, expected_X_train, expected_X_val, _, _ = get_dataframes
    params = {'city':True, 'squareMeters':True, 'floor': False, 'floorCount': False, 'poiCount': False, 'schoolDistance': False, 'clinicDistance': False, 'postOfficeDistance': False, 'kindergartenDistance': False, 'restaurantDistance': False, 'collegeDistance': False, 'pharmacyDistance': False, 'type': False, 'ownership': False, 'hasParkingSpace': False, 'hasBalcony': False, 'hasElevator': False, 'hasSecurity': False, 'hasStorageRoom': False, 'imputing_method': 'knn', 'indicate_missing': 'yes', 'with_centering': True, 'with_scaling': True, 'encoding_method': 'ordinal', 'algorithm': 'forest', 'n_estimators': 999, 'max_depth': 13, 'max_features': 0.5909393194013565, 'bootstrap': False}
    columns = ['city', 'floor', 'squareMeters', 'clinicDistance']

    X_train_selected, X_val_selected = get_reduced_features(X_train, X_val, params, columns)
    
    assert_frame_equal(X_train_selected, expected_X_train)
    assert_frame_equal(X_val_selected, expected_X_val)


def test_evaluate_model(get_dataframes, mocker):
    
    X_train_selected, X_val_selected, _, _, y_train, y_val = get_dataframes
    
    mock_best_model = Mock()
    mock_pipeline = Mock()
    mock_pipeline.predict.return_value = Series([110, 210, 310])  
    mock_pipeline.fit.return_value = None  

    mocker.patch('base64.b64decode', return_value=None)
    mocker.patch('pickle.loads', return_value=mock_pipeline)
    mocker.patch('mlproject.optunasetup.lib.utils.infer_signature', return_value='mock_signature')
    mocker.patch('mlproject.optunasetup.lib.utils.mean_absolute_percentage_error', return_value=0.01)

    # Call the function being tested
    pipeline, signature, validation_mape = evaluate_model(mock_best_model, X_train_selected, X_val_selected, y_train, y_val)

    # Assert that the pipeline's methods are called correctly
    mock_pipeline.fit.assert_called_once_with(X_train_selected, y_train)
    mock_pipeline.predict.assert_called_once_with(X_val_selected)

    # Verify the output values
    assert pipeline == mock_pipeline
    assert signature == 'mock_signature'
    assert validation_mape == 0.01

def test_save_best_study(get_dataframes, mocker):
    
    X_train, X_val, _, _, y_train, y_val = get_dataframes

    experiment_name = "mock_experiment_name"
    columns = ['col1']
    target="mock_target"
    study = MagicMock()
    study.best_trial.params = {'param1':'value'}
    study.best_trial.value = -0.5
    study.user_attrs['best_model'] = 52
    mlflow = Mock()
    validation_mape = 0.07

    mocker.patch('mlproject.optunasetup.lib.utils.plot_param_importances', return_value=1)
    mocker.patch('mlproject.optunasetup.lib.utils.plot_optimization_history', return_value=2)

    mocker.patch('mlproject.optunasetup.lib.utils.sha256', return_value=sha256(b"abcd"))
    mocker.patch('mlproject.optunasetup.lib.utils.evaluate_model', return_value=('mock_pipeline_object', 'mock_signature_object', validation_mape))
    save_best_study(study, experiment_name, X_train, y_train, X_val, y_val, columns, target, mlflow)
    assert mlflow.log_params.call_count == 3
    mlflow.log_params.assert_has_calls([call({'param1':'value'}),call({'target': target}),call({'hash': sha256(b"abcd").hexdigest()[:1024]})])

    assert mlflow.log_metrics.call_count == 2
    mlflow.log_metrics.assert_has_calls([call({"train_mape": study.best_trial.value*(-1)}),call({"validation_mape": validation_mape})])

    mlflow.sklearn.log_model.assert_called_once_with('mock_pipeline_object',artifact_path="ml_project", signature='mock_signature_object', registered_model_name=experiment_name)

    assert mlflow.log_figure.call_count == 2
    mlflow.log_figure.assert_has_calls([call(1,'param_importances.html'),call(2,'optimization_history.html')])