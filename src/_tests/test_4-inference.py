import numpy as np
import pytest
import tiled
import torch

from ..tiled_dataset import TiledDataset
from ..utils import (
    allocate_array_space,
    construct_dataloaders,
    load_dlsia_network,
    normalization,
    qlty_crop,
    segment_single_frame,
)


@pytest.fixture
def loaded_network(network_name, model_directory):
    net = load_dlsia_network(network_name=network_name, model_dir=model_directory)
    yield net


@pytest.fixture
def seg_tiled_dataset(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"],
        is_training=False,
    )
    yield tiled_dataset


@pytest.fixture
def seg_client(client, seg_tiled_dataset, io_parameters, network_name):

    result_container = client.create_container("results")
    array_client = allocate_array_space(
        tiled_dataset=seg_tiled_dataset,
        last_container=result_container,
        uid=io_parameters.uid_save,
        model_name=network_name,
        array_name="seg_result",
    )
    yield array_client


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


@pytest.fixture
def inference_patches(seg_tiled_dataset, qlty_object):
    image = seg_tiled_dataset[0]
    image = normalization(image)
    image = torch.from_numpy(image)
    patches = qlty_crop(qlty_object=qlty_object, images=image, is_training=False)
    yield patches


def test_inference_cropping(inference_patches):
    assert inference_patches.shape == (4, 1, 4, 4)


@pytest.fixture
def inference_dataloader(inference_patches, model_parameters):
    inference_loader = construct_dataloaders(
        inference_patches, model_parameters, is_training=False
    )
    yield inference_loader


def test_inference_loader(inference_dataloader, model_parameters, inference_patches):
    assert inference_dataloader
    assert len(inference_dataloader) == 2
    batch_size_inference = model_parameters.batch_size_train
    # Checking each batch
    for image_batch in inference_dataloader:
        assert (
            len(image_batch) == batch_size_inference
            or len(image_batch) == len(inference_patches) % batch_size_inference
        )
        assert image_batch[0].shape[1:] == inference_patches.shape[1:]
        assert image_batch[0].dtype == torch.float64


@pytest.fixture
def prediction(loaded_network, inference_dataloader, device):
    softmax = torch.nn.Softmax(dim=1)
    prediction = segment_single_frame(
        network=loaded_network,
        dataloader=inference_dataloader,
        final_layer=softmax,
        device=device,
    )
    yield prediction


def test_prediction(prediction):
    assert type(prediction) is torch.Tensor
    assert len(prediction) == 4


@pytest.fixture
def result(prediction, qlty_object, seg_client):
    stitched_prediction, _ = qlty_object.stitch(prediction)
    result = torch.argmax(stitched_prediction, dim=1).numpy().astype(np.int8)
    seg_client.write_block(result, block=(0, 0, 0))
    yield result


def test_result(result):
    assert result.shape == (1, 6, 6)


# TODO: Discuss error and fix this pytest
# def test_seg_parent_container(client):
#     print(client.uri)
#     print(client)
#     last_container = ensure_parent_containers(client.uri)
#     assert last_container
