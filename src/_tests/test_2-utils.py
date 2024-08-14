import os
import numpy as np
import tiled
import torch

from utils import find_device

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


# TODO: Test examples for other model classes


def test_normalization(normed_data):
    assert type(normed_data) is np.ndarray
    assert np.min(normed_data) == 0
    assert np.max(normed_data) == 1


def test_cropped_pairs(patched_data_mask_pair):
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]
    assert patched_data.shape == (8, 1, 4, 4)
    assert patched_mask.shape == (8, 4, 4)


def test_find_device_cuda_available(monkeypatch):
    # Monkey patch torch.cuda.is_available to return True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = find_device()
    assert device.type == "cuda", "Device should be cuda when CUDA is available"


def test_find_device_cuda_not_available(monkeypatch):
    # Monkey patch torch.cuda.is_available to return False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = find_device()
    assert device.type == "cpu", "Device should be cpu when CUDA is not available"


def test_dir_creation(model_directory):
    assert os.path.exists(model_directory)
    assert os.path.isdir(model_directory)


# TODO: Discuss error and fix this pytest
# def test_seg_parent_container(client):
#     print(client.uri)
#     print(client)
#     last_container = ensure_parent_containers(client.uri)
#     assert last_container


def test_seg_client(seg_client):
    assert seg_client
    assert type(seg_client) is tiled.client.array.ArrayClient
    assert (
        seg_client.uri
        == "http://local-tiled-app/api/v1/metadata/results/pytest1/seg_result"
    )
    assert seg_client.shape == (2, 6, 6)
    metadata = {
        "data_uri": "http://local-tiled-app/api/v1/metadata/reconstructions/recon1",
        "mask_uri": "http://local-tiled-app/api/v1/metadata/uid0001/mask",
        "mask_idx": [1, 3],
        "uid": "pytest1",
        "model": "DLSIA TUNet",
    }
    assert seg_client.metadata == metadata
