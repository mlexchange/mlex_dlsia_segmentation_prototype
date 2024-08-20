import numpy as np

from ..tiled_dataset import TiledDataset


def test_with_mask_training(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"],
        is_training=True,
    )
    assert tiled_dataset
    assert tiled_dataset.mask_idx == [1, 3]
    assert len(tiled_dataset) == 2
    assert len(tiled_dataset[0]) == 2
    # Check data
    assert tiled_dataset[0][0].shape == (6, 6)
    assert tiled_dataset[0][0].dtype == np.uint8
    # Check mask
    assert tiled_dataset[0][1].shape == (6, 6)
    assert tiled_dataset[0][1].dtype == np.int8
    assert np.all(tiled_dataset[0][1])  # should be all 1s


def test_with_mask_inference(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"],
        is_training=False,
    )
    assert tiled_dataset
    assert tiled_dataset.mask_idx == [1, 3]
    assert len(tiled_dataset) == 2
    # Check data
    assert tiled_dataset[0].shape == (6, 6)
    assert tiled_dataset[0].dtype == np.uint8


def test_no_mask_inference(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        is_training=False,
    )
    assert tiled_dataset
    assert len(tiled_dataset) == 5
    # Check data
    assert tiled_dataset[0].shape == (6, 6)
    assert tiled_dataset[0].dtype == np.uint8


# TODO: Test qlty cropping within tiled_dataset.
# Since this part has been moved to the training script and performed outside,
# this is not on higher priority.
