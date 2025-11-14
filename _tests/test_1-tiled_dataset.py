import pytest
import torch

from mlex_dlsia.dataset import TiledDataset, TiledMaskedDataset


def test_tiled_dataset(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        qlty_window=6,
        qlty_step=2,
        qlty_border=1,
    )
    assert tiled_dataset
    assert len(tiled_dataset) == 5
    assert tiled_dataset.shape == (5, 12, 12)
    # Check data
    for idx in range(5):
        assert tiled_dataset[idx].shape == (16, 1, 6, 6)
        assert tiled_dataset[idx].dtype == torch.float32


@pytest.mark.parametrize(
    "selected_indices, expected_len", [(None, 5), ([0, 2, 4], 3), ([0, 1, 2, 3, 4], 5)]
)
def test_tiled_dataset_selected_indices(client, selected_indices, expected_len):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        selected_indices=selected_indices,
        qlty_window=6,
        qlty_step=2,
        qlty_border=1,
    )
    assert tiled_dataset
    assert len(tiled_dataset) == expected_len
    assert tiled_dataset.shape == (expected_len, 12, 12)
    # Check data for each index
    for idx in range(expected_len):
        assert tiled_dataset[idx].shape == (16, 1, 6, 6)
        assert tiled_dataset[idx].dtype == torch.float32


def test_tiled_masked_dataset(client):
    mask_indices = client["uid0001"].metadata["mask_idx"]
    tiled_masked_dataset = TiledMaskedDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"]["mask"],
        selected_indices=mask_indices,
        qlty_window=6,
        qlty_step=2,
        qlty_border=1,
    )

    assert tiled_masked_dataset
    assert len(tiled_masked_dataset) == 32
    assert tiled_masked_dataset.shape == (2, 12, 12)
    # Check data and mask
    for idx in range(2):
        assert tiled_masked_dataset[idx][0].shape == (1, 6, 6)
        assert tiled_masked_dataset[idx][0].dtype == torch.float32

        assert tiled_masked_dataset[idx][1].shape == (6, 6)
        assert tiled_masked_dataset[idx][1].dtype == torch.int8
