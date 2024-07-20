import numpy as np
import torch
import os

def test_load_yaml(parameters_dict):
    assert isinstance(parameters_dict, dict)
    assert 'io_parameters' in parameters_dict
    assert 'model_parameters' in parameters_dict

def test_io_parameter_validation(io_parameters):
    assert io_parameters.data_tiled_uri == "http://data/tiled/uri"
    assert io_parameters.data_tiled_api_key == "a1b2c3"
    assert io_parameters.mask_tiled_uri == "http://mask/tiled/uri"
    assert io_parameters.mask_tiled_api_key == "d4e5f6"
    assert io_parameters.seg_tiled_uri == "http://seg/tiled/uri"
    assert io_parameters.uid_save == "pytest1"
    assert io_parameters.uid_retrieve == "pytest1"
    assert io_parameters.models_dir == "."

def test_network_name(network_name):
    assert network_name == 'DLSIA TUNet'

def test_model_parameter_validation(model_parameters):
    assert model_parameters.network == 'DLSIA TUNet'
    assert model_parameters.num_classes == 1 
    assert model_parameters.num_epochs == 2 
    assert model_parameters.optimizer =='Adam'
    assert model_parameters.criterion == 'CrossEntropyLoss'
    assert model_parameters.weights == '[1.0]'
    assert model_parameters.learning_rate == 0.1
    assert model_parameters.activation == 'ReLU'
    assert model_parameters.normalization == 'BatchNorm2d'
    assert model_parameters.convolution == 'Conv2d'
    assert model_parameters.qlty_window == 3
    assert model_parameters.qlty_step == 1
    assert model_parameters.qlty_border == 0
    assert model_parameters.shuffle_train == True
    assert model_parameters.batch_size_train == 1 
    assert model_parameters.batch_size_val == 1 
    assert model_parameters.batch_size_inference == 2 
    assert model_parameters.val_pct == 0.2
    assert model_parameters.depth == 2
    assert model_parameters.base_channels == 4
    assert model_parameters.growth_rate == 1
    assert model_parameters.hidden_rate == 1

# TODO: Test examples for other model classes

def test_normalization(normed_data):
    assert type(normed_data) == np.ndarray
    assert np.min(normed_data) == 0
    assert np.max(normed_data) == 1

def test_tensor_conversion(data_tensor, mask_tensor):
    assert type(data_tensor) == torch.Tensor
    assert data_tensor.shape == (2, 4, 4)
    assert type(mask_tensor) == torch.Tensor
    assert mask_tensor.shape == (2, 4, 4)

def test_qlty_object(qlty_object):
    # Detailed qlty tests should be part of the qlty package, thus not in the scope of this test
    assert qlty_object
    assert qlty_object.Y == 4
    assert qlty_object.X == 4
    assert qlty_object.window == (3, 3)
    assert qlty_object.step == (1, 1)
    assert qlty_object.border == None
    assert qlty_object.border_weight == 0.2

def test_cropped_pairs(patched_data_mask_pair):
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]
    print(patched_data.shape)
    print(patched_data)
    print(patched_mask.shape)
    print(patched_mask)
    assert patched_data.shape == (8, 1, 3, 3)
    assert patched_mask.shape == (8, 3, 3)

def test_device(device):
    assert device
    assert device.type == 'cpu'

def test_dir_creation(model_directory):
    assert os.path.exists(model_directory)
    assert os.path.isdir(model_directory)


