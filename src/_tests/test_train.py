import torch
import os

def test_data_and_mask(raw_data, mask_array):
    assert raw_data.shape == (2, 6, 6)
    assert mask_array.shape == (2, 6, 6)

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
    assert networks
    assert type(networks) == list
    assert len(networks) == 1
    assert networks[0]
    # TODO: Test more aspects of the built network

def test_criterion(criterion):
    assert criterion
    assert type(criterion) is torch.nn.modules.loss.CrossEntropyLoss

def test_model_training(trained_network, model_directory, io_parameters, network_name):
    trained_model = trained_network[0]
    start_time = trained_network[1]
    check_point = os.path.join(
            model_directory, "net_checkpoint"
        )
    assert os.path.exists(check_point)
    assert trained_model
    model_path = os.path.join(
            model_directory, f"{io_parameters.uid_save}_{network_name}1.pt"
        )
    assert os.path.exists(model_path)
    # Get the file modification time
    file_mod_time = os.path.getmtime(model_path)
    # Check if the file was modified after training start time
    assert file_mod_time > start_time, "The model .pt file is not the new one just saved."
    dvc_path = os.path.join(
            model_directory, 'dvc_metrics'
        )
    assert os.path.exists(dvc_path)
    assert os.path.isdir(dvc_path)
    # TODO: Negative test cases when not use dvclive, keep in mind of pre-existing directories
    

