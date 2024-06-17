import numpy as np

from ..tiled_dataset import TiledDataset


def test_with_mask_training(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"],
        is_training=True,
    )
    assert tiled_dataset
    assert tiled_dataset.mask_idx == [1]
    assert len(tiled_dataset) == 1
    assert len(tiled_dataset[0]) == 2
    # Check data
    assert tiled_dataset[0][0].shape == (3, 3)
    assert not np.all(tiled_dataset[0][0])  # should be all 0s
    # Check mask
    assert tiled_dataset[0][1].shape == (3, 3)
    assert np.all(tiled_dataset[0][1])  # should be all 1s


def test_with_mask_inference(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"],
        is_training=False,
    )
    assert tiled_dataset
    assert tiled_dataset.mask_idx == [1]
    assert len(tiled_dataset) == 1
    # Check data
    assert tiled_dataset[0].shape == (3, 3)
    assert not np.all(tiled_dataset[0])  # should be all 0s


def test_no_mask_inference(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        is_training=False,
    )
    assert tiled_dataset
    assert len(tiled_dataset) == 2
    # Check data
    assert tiled_dataset[0].shape == (3, 3)
    assert not np.all(tiled_dataset[0])  # should be all 0s


# TODO: Test qlty cropping within tiled_dataset.
# Since this part has been moved to the training script and performed outside,
# this is not on higher priority.
