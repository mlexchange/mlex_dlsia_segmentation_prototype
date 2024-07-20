import torch

def test_data_and_mask(raw_data, mask_array):
    assert raw_data.shape == (2, 4, 4)
    assert mask_array.shape == (2, 4, 4)

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

def test_build_networks(networks):
    print(networks[0])
    assert networks
    assert type(networks) == list
    assert len(networks) == 1
    assert networks[0]
    # TODO: Test more aspects of the built network

def test_criterion(criterion):
    assert criterion
    assert type(criterion) is torch.nn.modules.loss.CrossEntropyLoss

def test_model_training(trained_network):
    assert trained_network
