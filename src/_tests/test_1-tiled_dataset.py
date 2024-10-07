import numpy as np
import pytest

from ..tiled_dataset import TiledDataset, TiledMaskedDataset


def test_tiled_dataset(client):
    tiled_dataset = TiledDataset(data_tiled_client=client["reconstructions"]["recon1"])
    assert tiled_dataset
    assert len(tiled_dataset) == 5
    assert tiled_dataset.shape == (5, 6, 6)
    # Check data
    for idx in range(5):
        assert tiled_dataset[idx].shape == (6, 6)
        assert tiled_dataset[idx].dtype == np.uint8
        assert np.array_equal(
            tiled_dataset[idx], client["reconstructions"]["recon1"][idx,]
        )


@pytest.mark.parametrize(
    "selected_indices, expected_len", [(None, 5), ([0, 2, 4], 3), ([0, 1, 2, 3, 4], 5)]
)
def test_tiled_dataset_selected_indices(client, selected_indices, expected_len):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        selected_indices=selected_indices,
    )
    assert tiled_dataset
    assert len(tiled_dataset) == expected_len
    assert tiled_dataset.shape == (expected_len, 6, 6)
    # Check data for each index
    for idx in range(expected_len):
        assert tiled_dataset[idx].shape == (6, 6)
        assert tiled_dataset[idx].dtype == np.uint8
        mapped_idx = selected_indices[idx] if selected_indices else idx
        assert np.array_equal(
            tiled_dataset[idx], client["reconstructions"]["recon1"][mapped_idx,]
        )


def test_tiled_masked_dataset(client):
    mask_indices = client["uid0001"].metadata["mask_idx"]
    tiled_masked_dataset = TiledMaskedDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"]["mask"],
        selected_indices=mask_indices,
    )

    assert tiled_masked_dataset
    assert len(tiled_masked_dataset) == 2
    assert tiled_masked_dataset.shape == (2, 6, 6)
    # Check data and mask
    for idx in range(2):
        mapped_idx = mask_indices[idx]
        assert tiled_masked_dataset[idx][0].shape == (6, 6)
        assert tiled_masked_dataset[idx][0].dtype == np.uint8
        assert np.array_equal(
            tiled_masked_dataset[idx][0],
            client["reconstructions"]["recon1"][mapped_idx,],
        )
        assert tiled_masked_dataset[idx][1].shape == (6, 6)
        assert tiled_masked_dataset[idx][1].dtype == np.int8
        assert np.array_equal(
            tiled_masked_dataset[idx][1], client["uid0001"]["mask"][idx,]
        )
