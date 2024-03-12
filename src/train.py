import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision import transforms

from network import build_network
from parameters import (
    IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)
from seg_utils import crop_split_load, train_segmentation, train_val_split
from tiled_dataset import TiledDataset
from utils import create_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Open the YAML file for all parameters
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)

    # Validate and load I/O related parameters
    io_parameters = parameters["io_parameters"]
    io_parameters = IOParameters(**io_parameters)
    # Check whether mask_uri has been provided as this is a requirement for training.
    assert io_parameters.mask_tiled_uri, "Mask URI not provided for training."

    # Detect which model we have, then load corresponding parameters
    raw_parameters = parameters["model_parameters"]
    network = raw_parameters["network"]

    model_parameters = None
    if network == "MSDNet":
        model_parameters = MSDNetParameters(**raw_parameters)
    elif network == "TUNet":
        model_parameters = TUNetParameters(**raw_parameters)
    elif network == "TUNet3+":
        model_parameters = TUNet3PlusParameters(**raw_parameters)
    elif network == "SMSNetEnsemble":
        model_parameters = SMSNetEnsembleParameters(**raw_parameters)

    assert model_parameters, f"Received Unsupported Network: {network}"

    print("Parameters loaded successfully.")

    # Create Result Directory if not existed
    create_directory(io_parameters.uid_save)

    dataset = TiledDataset(
        data_tiled_uri=io_parameters.data_tiled_uri,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        mask_tiled_uri=io_parameters.mask_tiled_uri,
        mask_tiled_api_key=io_parameters.mask_tiled_api_key,
        is_training=True,
        using_qlty=False,
        qlty_window=model_parameters.qlty_window,
        qlty_step=model_parameters.qlty_step,
        qlty_border=model_parameters.qlty_border,
        transform=transforms.ToTensor()
        )

    # Load all labeled data and masks into memory
    data = dataset.data_client[dataset.mask_idx]
    mask = np.array(dataset.mask_client)
    
    # train_loader, val_loader = train_val_split(dataset, model_parameters)
    train_loader, val_loader = crop_split_load(data, mask, model_parameters)

    # Build network
    networks = build_network(
        network=network,
        data_shape=dataset.data_client.shape,
        num_classes=model_parameters.num_classes,
        parameters=model_parameters,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"      MEMORY ALLOCATED   {torch.cuda.memory_allocated(0)}")
    torch.cuda.empty_cache()
    print(f'Training will be processed on: {device}')


    # Define criterion and optimizer
    criterion = getattr(nn, model_parameters.criterion)
    # Convert the string to a list of floats

    weights = [float(x) for x in model_parameters.weights.strip('[]').split(',')]
    weights = torch.tensor(weights,dtype=torch.float).to(device)
    criterion = criterion(weight=weights,
                          ignore_index=-1, 
                          size_average=None
                          )    

    for idx, net in enumerate(networks):
        print(f"{network}: {idx+1}/{len(networks)}")
        optimizer = getattr(optim, model_parameters.optimizer)
        optimizer = optimizer(net.parameters(), lr=model_parameters.learning_rate)
        net = net.to(device)
        net, results = train_segmentation(
            net,
            train_loader,
            val_loader,
            model_parameters.num_epochs,
            criterion,
            optimizer,
            device,
            savepath=io_parameters.uid_save,
            saveevery=None,
            scheduler=None,
            show=0,
            use_amp=False,
            clip_value=None,
        )
        # Save network parameters

        model_params_path = f"{io_parameters.uid_save}/{io_parameters.uid_save}_{network}{idx+1}.pt"
        print(f"!!!!!!!   model_params_path {model_params_path}")

        net.save_network_parameters(model_params_path)
        # Clear out unnecessary variables from device memory
        torch.cuda.empty_cache()

    print(f"{network} trained successfully.")
