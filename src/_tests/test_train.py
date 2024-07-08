def test_data_and_mask(raw_data, mask_array):
    assert raw_data.shape == (2, 3, 3)
    assert mask_array.shape == (2, 3, 3)


# TODO: check dir creation? How to handle file system change during pytest?

# TODO: load TiledDataset from fixture client, test already done.

# TODO: test data and mask array dim and shape

# TODO: test train_loader and val_loader from crop_split_load func, check length

# TODO: test build_network. How to deal with lengthy func? test all network options?

# TODO: test weights and criterion?

# TODO: test dvc?

# TODO: test trainer building

# TODO: test 1 epoch, check param saving
