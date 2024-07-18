import torch

def test_data_and_mask(raw_data, mask_array):
    assert raw_data.shape == (2, 3, 3)
    assert mask_array.shape == (2, 3, 3)

def test_train_and_val_loader(training_dataloaders, model_parameters, patched_data_mask_pair):
    train_loader = training_dataloaders[0]
    val_loader = training_dataloaders[1]
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]
    
    assert train_loader
    assert len(train_loader) == 7 
    batch_size_train = model_parameters.batch_size_train
    # Checking each batch
    for image_batch, mask_batch in train_loader:
        assert len(image_batch) == batch_size_train or len(image_batch) == len(patched_data) % batch_size_train
        assert len(mask_batch) == batch_size_train or len(mask_batch) == len(patched_mask) % batch_size_train
        assert image_batch.shape[1:] == patched_data.shape[1:]
        assert mask_batch.shape[1:] == patched_mask.shape[1:]
        assert image_batch.dtype == torch.float64
        assert mask_batch.dtype == torch.int8
    
    assert val_loader
    assert len(val_loader) == 1
    batch_size_val = model_parameters.batch_size_val
    # Checking each batch
    for image_batch, mask_batch in val_loader:
        assert len(image_batch) == batch_size_val or len(image_batch) == len(patched_data) % batch_size_val
        assert len(mask_batch) == batch_size_val or len(mask_batch) == len(patched_mask) % batch_size_val
        assert image_batch.shape[1:] == patched_data.shape[1:]
        assert mask_batch.shape[1:] == patched_mask.shape[1:]
        assert image_batch.dtype == torch.float64
        assert mask_batch.dtype == torch.int8
    # TODO: Add cases when val_loader is None due to low val_pct

# TODO: check dir creation? How to handle file system change during pytest?

# TODO: load TiledDataset from fixture client, test already done.

# TODO: test data and mask array dim and shape

# TODO: test train_loader and val_loader from crop_split_load func, check length

# TODO: test build_network. How to deal with lengthy func? test all network options?

# TODO: test weights and criterion?

# TODO: test dvc?

# TODO: test trainer building

# TODO: test 1 epoch, check param saving
