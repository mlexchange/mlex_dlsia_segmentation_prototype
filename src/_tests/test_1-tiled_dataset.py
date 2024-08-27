import numpy as np
import pytest

from ..tiled_dataset import TiledDataset


@pytest.mark.parametrize(
    "mask_client, is_training, is_full_inference, expected_len, expected_shape, expected_dtype, check_mask",
    [
        # Test with mask, during training
        (
            {"mask_client": "uid0001", "mask_idx": [1, 3]},
            True,
            False,
            2,
            (6, 6),
            np.uint8,
            True,
        ),
        # Test with mask, quick inference
        (
            {"mask_client": "uid0001", "mask_idx": [1, 3]},
            False,
            False,
            2,
            (6, 6),
            np.uint8,
            False,
        ),
        # Test with mask, full inference
        (
            {"mask_client": "uid0001", "mask_idx": [1, 3]},
            False,
            True,
            5,
            (6, 6),
            np.uint8,
            False,
        ),
        # Test without mask, full inference
        (None, False, True, 5, (6, 6), np.uint8, False),
    ],
)
def test_tiled_dataset(
    client,
    mask_client,
    is_training,
    is_full_inference,
    expected_len,
    expected_shape,
    expected_dtype,
    check_mask,
):
    mask_tiled_client = client[mask_client["mask_client"]] if mask_client else None

    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=mask_tiled_client,
        is_training=is_training,
        is_full_inference=is_full_inference,
    )

    assert tiled_dataset
    assert len(tiled_dataset) == expected_len

    # Check data
    if is_training:
        assert len(tiled_dataset[0]) == 2  # data and mask
        assert tiled_dataset[0][0].shape == expected_shape
        assert tiled_dataset[0][0].dtype == expected_dtype
        # Check mask
        assert tiled_dataset[0][1].shape == expected_shape
        assert tiled_dataset[0][1].dtype == np.int8
        if check_mask:
            assert np.all(tiled_dataset[0][1])  # should be all 1s
    else:
        assert tiled_dataset[0].shape == expected_shape
        assert tiled_dataset[0].dtype == expected_dtype
