import pytest
from ..utils import load_yaml

def test_load_yaml(parameters_dict):
    assert isinstance(parameters_dict, dict)
    assert 'io_parameters' in parameters_dict
    assert 'model_parameters' in parameters_dict

def test_io_parameter_validation(parameters):
    print(f'parameters: {parameters}')
    io_parameters, network, model_parameters = parameters
    assert io_parameters.data_tiled_uri == "http://data/tiled/uri"
    assert io_parameters.data_tiled_api_key == "a1b2c3"
    assert io_parameters.mask_tiled_uri == "http://mask/tiled/uri"
    assert io_parameters.mask_tiled_api_key == "d4e5f6"
    assert io_parameters.seg_tiled_uri == "http://seg/tiled/uri"
    assert io_parameters.uid_save == "pytest1"
    assert io_parameters.uid_retrieve == "pytest1"
    assert io_parameters.models_dir == "."