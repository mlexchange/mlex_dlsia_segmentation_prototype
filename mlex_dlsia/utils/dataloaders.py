from torch.utils.data import DataLoader, TensorDataset, random_split


def construct_train_dataloaders(dataset, parameters):
    """
    This function constructs dataloaders for training, validation, or inference.
    Input:
        dataset: TiledDataset or TensorDataset, the dataset to create dataloaders from
        parameters: class, pydantic validated model parameters
    Output:
        train_loader, val_loader or inference_loader: DataLoader objects
    """
    train_loader_params = {
        "batch_size": parameters.batch_size_train,
        "shuffle": parameters.shuffle_train,
    }

    val_loader_params = {
        "batch_size": parameters.batch_size_val,
        "shuffle": False,
    }

    val_pct = parameters.val_pct
    val_size = max(int(val_pct * len(dataset)), 1) if len(dataset) > 1 else 0

    if val_size == 0:
        train_loader = DataLoader(dataset, **train_loader_params)
        val_loader = None
    else:
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_data, **train_loader_params)
        val_loader = DataLoader(val_data, **val_loader_params)

    return train_loader, val_loader


def construct_inference_dataloaders(patched_data, parameters):
    """
    This function constructs dataloaders for inference.
    Input:
        patched_data: Tensor, the patched data to create dataloaders from
        parameters: class, pydantic validated model parameters
    Output:
        inference_loader: DataLoader object
    """
    dataset = TensorDataset(patched_data)
    inference_loader_params = {
        "batch_size": parameters.batch_size_inference,
        "shuffle": False,
    }

    inference_loader = DataLoader(dataset, **inference_loader_params)

    return inference_loader
