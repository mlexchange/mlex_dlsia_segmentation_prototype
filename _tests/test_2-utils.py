import numpy as np
import pytest
import torch

from mlex_dlsia.dataset import TiledDataset
from mlex_dlsia.utils.dataloaders import (
    construct_inference_dataloaders,
    construct_train_dataloaders,
)
from mlex_dlsia.utils.params_validation import validate_parameters
from mlex_dlsia.utils.tiled import allocate_array_space


def test_construct_train_dataloaders(tiled_masked_dataset, validated_params):
    io_parameters, model_parameters = validated_params
    train_loader, val_loader = construct_train_dataloaders(
        tiled_masked_dataset, model_parameters
    )
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    data, mask = next(iter(train_loader))
    assert data.shape[0] == mask.shape[0] == model_parameters.batch_size_train


def test_construct_inference_dataloaders(tiled_dataset, validated_params):
    io_parameters, model_parameters = validated_params
    for idx in range(len(tiled_dataset)):
        inference_loader = construct_inference_dataloaders(
            tiled_dataset[idx], model_parameters
        )
        assert isinstance(inference_loader, torch.utils.data.DataLoader)
        assert inference_loader.batch_size == model_parameters.batch_size_inference


def test_validate_parameters(parameters_dict):
    # Check if this is the bad_params case
    if "bad_params" in parameters_dict.get("config_file_name", ""):
        # For bad params, we EXPECT an exception
        with pytest.raises(Exception) as exc_info:
            io_params, model_params = validate_parameters(parameters_dict)
        assert exc_info.value is not None
    else:
        # For valid params, it should work without exception
        try:
            io_params, model_params = validate_parameters(parameters_dict)
        except Exception as e:
            pytest.fail(f"validate_parameters raised an exception: {e}")

        assert io_params is not None
        assert model_params.network in [
            "DLSIA MSDNet",
            "DLSIA TUNet",
            "DLSIA TUNet3+",
            "DLSIA SMSNetEnsemble",
        ]
        assert model_params is not None


def test_normalization():
    # Create a sample image with known percentiles
    image = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
    image = torch.from_numpy(image)
    normalized_image = TiledDataset._normalize(image, low_pct=0, high_pct=100)

    # Check that the min and max values are approximately 0 and 1
    assert torch.isclose(normalized_image.min(), torch.tensor(0.0))
    assert torch.isclose(normalized_image.max(), torch.tensor(1.0))

    # Check that specific values are normalized correctly
    expected_normalized = (image - 0) / (255 - 0)
    assert torch.allclose(normalized_image, expected_normalized)


def test_allocate_array_space(tiled_dataset, client):
    last_container = client.create_container("test_container")
    uid = "test_uid"
    model_name = "DLSIA MSDNet"
    array_name = "segmentation_result"

    array_client = allocate_array_space(
        tiled_dataset,
        last_container,
        uid,
        model_name,
        array_name,
    )

    assert array_client.uri.split("/")[-1] == array_name

    # Check that the metadata is set correctly
    assert array_client.metadata["uid"] == uid
    assert array_client.metadata["model"] == model_name
    assert array_client.metadata["data_uri"] == tiled_dataset.data_client.uri
