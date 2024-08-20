import os
import time

import numpy as np
import pytest
import torch

from network import build_network

from ..train import build_criterion, prepare_data_and_mask, train_network
from ..utils import construct_dataloaders, normalization, qlty_crop


def test_load_yaml(parameters_dict):
    assert isinstance(parameters_dict, dict)
    assert "io_parameters" in parameters_dict
    assert "model_parameters" in parameters_dict


def test_io_parameter_validation(io_parameters):
    assert io_parameters.data_tiled_uri == "http://data/tiled/uri"
    assert io_parameters.data_tiled_api_key == "a1b2c3"
    assert io_parameters.mask_tiled_uri == "http://mask/tiled/uri"
    assert io_parameters.mask_tiled_api_key == "d4e5f6"
    assert io_parameters.seg_tiled_uri == "http://seg/tiled/uri"
    assert io_parameters.seg_tiled_api_key == "g7h8i9"
    assert io_parameters.uid_save == "pytest1"
    assert io_parameters.uid_retrieve == "pytest1"
    assert io_parameters.models_dir == "."


def test_network_name(network_name):
    assert network_name == "DLSIA TUNet"


def test_model_parameter_validation(model_parameters):
    assert model_parameters.network == "DLSIA TUNet"
    assert model_parameters.num_classes == 2
    assert model_parameters.num_epochs == 2
    assert model_parameters.optimizer == "Adam"
    assert model_parameters.criterion == "CrossEntropyLoss"
    assert model_parameters.weights == "[1.0, 0.5]"
    assert model_parameters.learning_rate == 0.1
    assert model_parameters.activation == "ReLU"
    assert model_parameters.normalization == "BatchNorm2d"
    assert model_parameters.convolution == "Conv2d"
    assert model_parameters.qlty_window == 4
    assert model_parameters.qlty_step == 2
    assert model_parameters.qlty_border == 1
    assert model_parameters.shuffle_train is True
    assert model_parameters.batch_size_train == 1
    assert model_parameters.batch_size_val == 1
    assert model_parameters.batch_size_inference == 2
    assert model_parameters.val_pct == 0.2
    assert model_parameters.depth == 2
    assert model_parameters.base_channels == 4
    assert model_parameters.growth_rate == 2
    assert model_parameters.hidden_rate == 1


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
def data_tensor(normed_data):
    data_tensor = torch.from_numpy(normed_data)
    yield data_tensor


@pytest.fixture
def mask_tensor(mask_array):
    mask_tensor = torch.from_numpy(mask_array)
    yield mask_tensor


@pytest.fixture
def patched_data_mask_pair(qlty_object, data_tensor, mask_tensor):
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
def networks(network_name, tiled_dataset, model_parameters):
    networks = build_network(
        network_name=network_name,
        data_shape=tiled_dataset.data_client.shape,  # TODO: Double check if this needs to be switched to the patch dim
        num_classes=model_parameters.num_classes,
        parameters=model_parameters,
    )
    yield networks


def test_build_networks(networks):
    assert networks
    assert type(networks) is list
    assert len(networks) == 1
    assert networks[0]
    # TODO: Test more aspects of the built network


@pytest.fixture
def criterion(model_parameters, device):
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
