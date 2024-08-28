import os
import time

import numpy as np
import pytest
import torch

from network import build_network

from ..train import build_criterion, prepare_data_and_mask, train_network
from ..utils import construct_dataloaders, normalization, qlty_crop


# Test to ensure that the loaded YAML is a dictionary and contains necessary keys
def test_load_yaml(parameters_dict):
    assert isinstance(parameters_dict, dict)
    assert "io_parameters" in parameters_dict
    assert "model_parameters" in parameters_dict


EXPECTED_UIDS = {
    "src/_tests/example_msdnet.yaml": {
        "uid_save": "pytest1",
        "uid_retrieve": "pytest1",
    },
    "src/_tests/example_tunet.yaml": {"uid_save": "pytest2", "uid_retrieve": "pytest2"},
    "src/_tests/example_tunet3plus.yaml": {
        "uid_save": "pytest3",
        "uid_retrieve": "pytest3",
    },
    "src/_tests/example_smsnet_ensemble.yaml": {
        "uid_save": "pytest4",
        "uid_retrieve": "pytest4",
    },
    "src/_tests/example_bad_params.yaml": {
        "uid_save": "bad_save",
        "uid_retrieve": "bad_retrieve",
    },
}


def test_io_parameter_validation(io_parameters, parameters_dict):
    yaml_file = parameters_dict["config_file_name"]

    if "example_bad_params.yaml" in yaml_file:
        pytest.skip("Skipping test for bad parameters")
    else:
        # Fetch the expected values for uid_save and uid_retrieve
        expected_uids = EXPECTED_UIDS.get(yaml_file, {})
        expected_uid_save = expected_uids.get("uid_save", "default_save")
        expected_uid_retrieve = expected_uids.get("uid_retrieve", "default_retrieve")

        assert io_parameters.data_tiled_uri == "http://data/tiled/uri"
        assert io_parameters.data_tiled_api_key == "a1b2c3"
        assert io_parameters.mask_tiled_uri == "http://mask/tiled/uri"
        assert io_parameters.mask_tiled_api_key == "d4e5f6"
        assert io_parameters.seg_tiled_uri == "http://seg/tiled/uri"
        assert io_parameters.seg_tiled_api_key == "g7h8i9"
        assert io_parameters.uid_save == expected_uid_save
        assert io_parameters.uid_retrieve == expected_uid_retrieve
        assert io_parameters.models_dir == "."


# Common parameters that all models share
COMMON_PARAMS = {
    "num_classes": 2,
    "num_epochs": 2,
    "optimizer": "Adam",
    "criterion": "CrossEntropyLoss",
    "learning_rate": 0.1,
    "activation": "ReLU",
    "normalization": "BatchNorm2d",
    "convolution": "Conv2d",
    "qlty_window": 4,
    "qlty_step": 2,
    "qlty_border": 1,
    "shuffle_train": True,
    "batch_size_train": 1,
    "batch_size_val": 1,
    "batch_size_inference": 2,
    "val_pct": 0.2,
}

# Specific parameters for each model
SPECIFIC_PARAMS = {
    "src/_tests/example_msdnet.yaml": {
        "network": "DLSIA MSDNet",
        "weights": "[1.0, 0.5]",
        "layer_width": 1,
        "num_layers": 3,
        "custom_dilation": True,
        "dilation_array": "[1, 2, 4]",
    },
    "src/_tests/example_tunet.yaml": {
        "network": "DLSIA TUNet",
        "weights": "[1.0, 0.5]",
        "depth": 2,
        "base_channels": 4,
        "growth_rate": 2,
        "hidden_rate": 1,
    },
    "src/_tests/example_tunet3plus.yaml": {
        "network": "DLSIA TUNet3+",
        "weights": "[1.0, 2.0]",
        "depth": 2,
        "base_channels": 4,
        "growth_rate": 2,
        "hidden_rate": 1,
        "carryover_channels": 4,
    },
    "src/_tests/example_smsnet_ensemble.yaml": {
        "network": "DLSIA SMSNetEnsemble",
        "weights": "[2.0, 0.1]",
        "num_networks": 2,
        "layers": 3,
        "alpha": 0.0,
        "gamma": 0.0,
        "hidden_channels": None,
        "dilation_choices": "[1, 2, 3]",
        "max_trial": 2,
    },
    "src/_tests/example_bad_params.yaml": {
        # Placeholder for bad params case
    },
}


def test_network_name(network_name, parameters_dict):
    yaml_file = parameters_dict["config_file_name"]
    if "example_bad_params.yaml" in yaml_file:
        # Check if the fixture returned an exception instead of a valid network name
        assert isinstance(network_name, AssertionError), "Expected an assertion error"
        assert "Unsupported Network: Void Model" in str(network_name)
    else:
        # For valid cases, proceed with normal assertions
        expected_network = SPECIFIC_PARAMS[yaml_file]["network"]
        assert network_name == expected_network


def test_model_parameter_validation(model_parameters, parameters_dict):
    yaml_file = parameters_dict["config_file_name"]
    if "example_bad_params.yaml" in yaml_file:
        pytest.skip("Skipping test for bad parameters")
    else:
        # Apply common assertions
        for key, expected_value in COMMON_PARAMS.items():
            assert getattr(model_parameters, key) == expected_value

        # Apply model-specific assertions
        if yaml_file in SPECIFIC_PARAMS:
            for key, expected_value in SPECIFIC_PARAMS[yaml_file].items():
                assert getattr(model_parameters, key) == expected_value


@pytest.fixture
def raw_data(tiled_dataset):
    data, _ = prepare_data_and_mask(tiled_dataset)
    yield data


@pytest.fixture
def mask_array(tiled_dataset):
    _, mask = prepare_data_and_mask(tiled_dataset)
    yield mask


def test_data_and_mask(raw_data, mask_array):
    assert raw_data.shape == (2, 6, 6)
    assert mask_array.shape == (2, 6, 6)


@pytest.fixture
def normed_data(raw_data):
    normed_data = normalization(raw_data)
    yield normed_data


def test_normalization(normed_data):
    assert type(normed_data) is np.ndarray
    assert np.min(normed_data) == 0
    assert np.max(normed_data) == 1


@pytest.fixture
def patched_data_mask_pair(qlty_object, normed_data, mask_array):
    data_tensor = torch.from_numpy(normed_data)
    mask_tensor = torch.from_numpy(mask_array)
    patched_data, patched_mask = qlty_crop(
        qlty_object, data_tensor, is_training=True, masks=mask_tensor
    )
    yield patched_data, patched_mask


def test_cropped_pairs(patched_data_mask_pair):
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]
    assert patched_data.shape == (8, 1, 4, 4)
    assert patched_mask.shape == (8, 4, 4)


@pytest.fixture
def training_dataloaders(patched_data_mask_pair, model_parameters):
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]
    train_loader, val_loader = construct_dataloaders(
        patched_data, model_parameters, is_training=True, masks=patched_mask
    )
    yield train_loader, val_loader


def test_train_and_val_loader(
    training_dataloaders, model_parameters, patched_data_mask_pair
):
    train_loader = training_dataloaders[0]
    val_loader = training_dataloaders[1]
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]

    assert train_loader
    assert len(train_loader) == 7
    batch_size_train = model_parameters.batch_size_train
    # Checking each batch
    for image_batch, mask_batch in train_loader:
        assert (
            len(image_batch) == batch_size_train
            or len(image_batch) == len(patched_data) % batch_size_train
        )
        assert (
            len(mask_batch) == batch_size_train
            or len(mask_batch) == len(patched_mask) % batch_size_train
        )
        assert image_batch.shape[1:] == patched_data.shape[1:]
        assert mask_batch.shape[1:] == patched_mask.shape[1:]
        assert image_batch.dtype == torch.float64
        assert mask_batch.dtype == torch.int8

    assert val_loader
    assert len(val_loader) == 1
    batch_size_val = model_parameters.batch_size_val
    # Checking each batch
    for image_batch, mask_batch in val_loader:
        assert (
            len(image_batch) == batch_size_val
            or len(image_batch) == len(patched_data) % batch_size_val
        )
        assert (
            len(mask_batch) == batch_size_val
            or len(mask_batch) == len(patched_mask) % batch_size_val
        )
        assert image_batch.shape[1:] == patched_data.shape[1:]
        assert mask_batch.shape[1:] == patched_mask.shape[1:]
        assert image_batch.dtype == torch.float64
        assert mask_batch.dtype == torch.int8
    # TODO: Add cases when val_loader is None due to low val_pct


@pytest.fixture
def networks(network_name, patched_data_mask_pair, model_parameters):
    patched_data = patched_data_mask_pair[0]
    networks = build_network(
        network_name=network_name,
        data_shape=patched_data.shape,
        num_classes=model_parameters.num_classes,
        parameters=model_parameters,
    )
    yield networks


def test_build_networks(
    network_name,
    networks,
):
    assert networks
    assert type(networks) is list
    if network_name == "DLSIA SMSNetEnsemble":
        assert len(networks) == 2
    else:
        assert len(networks) == 1
    assert networks[0]


@pytest.fixture
def criterion(model_parameters, device):
    if isinstance(model_parameters, AssertionError):
        pytest.skip("Skipping test due to unsupported network in parameters")
    criterion = build_criterion(model_parameters, device)
    yield criterion


def test_criterion(criterion):
    assert criterion
    assert type(criterion) is torch.nn.modules.loss.CrossEntropyLoss


@pytest.fixture
def trained_network(
    network_name,
    networks,
    io_parameters,
    model_parameters,
    device,
    model_directory,
    training_dataloaders,
    criterion,
):
    # Record the training start time
    start_time = time.time()
    net = train_network(
        network_name=network_name,
        networks=networks,
        io_parameters=io_parameters,
        model_parameters=model_parameters,
        device=device,
        model_dir=model_directory,
        train_loader=training_dataloaders[0],
        criterion=criterion,
        val_loader=training_dataloaders[1],
        use_dvclive=False,
        use_savedvcexp=False,
    )
    yield net, start_time


def test_model_training(trained_network, model_directory, io_parameters, network_name):
    trained_model = trained_network[0]
    start_time = trained_network[1]
    check_point = os.path.join(model_directory, "net_checkpoint")
    assert os.path.exists(check_point)
    assert trained_model
    model_path = os.path.join(
        model_directory, f"{io_parameters.uid_save}_{network_name}1.pt"
    )
    assert os.path.exists(model_path)
    # Get the file modification time
    file_mod_time = os.path.getmtime(model_path)
    # Check if the file was modified after training start time
    assert (
        file_mod_time > start_time
    ), "The model .pt file is not the new one just saved."
    # TODO: Positive test cases using dvclive, figure out the way to run dvc init --no-scm with pytest
    # dvc_path = os.path.join(model_directory, "dvc_metrics")
    # assert os.path.exists(dvc_path)
    # assert os.path.isdir(dvc_path)
