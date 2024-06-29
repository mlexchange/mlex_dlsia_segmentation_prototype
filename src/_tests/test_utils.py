def test_load_yaml(parameters_dict):
    assert isinstance(parameters_dict, dict)
    assert 'io_parameters' in parameters_dict
    assert 'model_parameters' in parameters_dict

def test_io_parameter_validation(parameters):
    io_parameters = parameters[0]
    assert io_parameters.data_tiled_uri == "http://data/tiled/uri"
    assert io_parameters.data_tiled_api_key == "a1b2c3"
    assert io_parameters.mask_tiled_uri == "http://mask/tiled/uri"
    assert io_parameters.mask_tiled_api_key == "d4e5f6"
    assert io_parameters.seg_tiled_uri == "http://seg/tiled/uri"
    assert io_parameters.uid_save == "pytest1"
    assert io_parameters.uid_retrieve == "pytest1"
    assert io_parameters.models_dir == "."

def test_network_name(parameters):
    network = parameters[1]
    assert network == 'TUNet'

def test_model_parameter_validation(parameters):
    model_parameters = parameters[2]
    assert model_parameters.network == 'TUNet'
    assert model_parameters.num_classes == 3 
    assert model_parameters.num_epochs == 3 
    assert model_parameters.optimizer =='Adam'
    assert model_parameters.criterion == 'CrossEntropyLoss'
    assert model_parameters.weights == '[1.0, 2.0, 0.5]'
    assert model_parameters.learning_rate == 0.1
    assert model_parameters.activation == 'ReLU'
    assert model_parameters.normalization == 'BatchNorm2d'
    assert model_parameters.convolution == 'Conv2d'
    assert model_parameters.qlty_window == 64
    assert model_parameters.qlty_step == 32
    assert model_parameters.qlty_border == 8
    assert model_parameters.shuffle_train == True
    assert model_parameters.batch_size_train == 1 
    assert model_parameters.batch_size_val == 1 
    assert model_parameters.batch_size_inference == 2 
    assert model_parameters.val_pct == 0.2
    assert model_parameters.depth == 4
    assert model_parameters.base_channels == 8
    assert model_parameters.growth_rate == 2
    assert model_parameters.hidden_rate == 1

# TODO: Test examples for other model classes


