def test_data_and_mask(data_and_mask):
    print(f'data_and_mask: {data_and_mask}, {data_and_mask[0]}, {data_and_mask[1]}')
    data, mask = data_and_mask[0], data_and_mask[1]
    assert len(data) == 2
    assert len(mask) == 2


# TODO: check dir creation? How to handle file system change during pytest?

# TODO: load TiledDataset from fixture client, test already done.

# TODO: test data and mask array dim and shape

# TODO: test train_loader and val_loader from crop_split_load func, check length

# TODO: test build_network. How to deal with lengthy func? test all network options?

# TODO: test weights and criterion?

# TODO: test dvc?

# TODO: test trainer building

# TODO: test 1 epoch, check param saving
