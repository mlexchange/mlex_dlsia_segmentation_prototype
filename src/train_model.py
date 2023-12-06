import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from parameters import TrainingParameters
from tiled_dataset import TiledDataset
from encryption import decrypt
from network import build_network
from train_segmentation import train_segmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_uri', help='tiled uri to training data')
    parser.add_argument('mask_uri', help='tiled uri to mask')
    parser.add_argument('data_api_key', help='tiled api key for training data')
    parser.add_argument('mask_api_key', help='tiled api key for mask')
    parser.add_argument('parameters', help='training parameters')
    args = parser.parse_args()

    # Load parameters
    parameters = TrainingParameters(**json.loads(args.parameters))

    # Prepare dataloders
    DECRYPTION_KEY = os.getenv('DECRYPTION_KEY')
    data_api_key = decrypt(args.data_api_key, DECRYPTION_KEY)
    mask_api_key = decrypt(args.mask_api_key, DECRYPTION_KEY)

    dataset = TiledDataset(
        data_uri=args.data_uri,
        mask_uri=args.mask_uri,
        data_api_key=data_api_key,
        mask_api_key=mask_api_key,
        transform=transforms.ToTensor()
        )
    trainloader = DataLoader(dataset, **parameters.dataloaders)
    validationloader = None

    # Build network
    net = build_network(
        network=parameters.network,
        num_classes=dataset.mask_client.max()+1,
        img_size=dataset.data_client.shape[-2:],
        num_layers=parameters.num_layers,
        activation=parameters.activation,
        normalization=parameters.normalization,
        final_layer=parameters.final_layer,
        custom_dilation=parameters.custom_dilation,
        max_dilation=parameters.max_dilation,
        dilation_array=parameters.dilation_array,
        depth=parameters.depth,
        base_channels=parameters.base_channels,
        growth_rate=parameters.growth_rate,
        hidden_rate=parameters.hidden_rate,
        carryover_channels=parameters.carryover_channels,
        )

    # Define criterion and optimizer
    criterion = getattr(nn, parameters.criterion.value)
    criterion = criterion(ignore_index=-1, size_average=None)    
    optimizer = getattr(optim, parameters.optimizer.value)
    optimizer = optimizer(net.parameters(), lr = parameters.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, results = train_segmentation(
        net,
        trainloader,
        validationloader,
        parameters.num_epochs,
        criterion,
        optimizer,
        device,
        savepath=None,
        saveevery=None,
        scheduler=None,
        show=0,
        use_amp=False,
        clip_value=None
        )

    # Save network parameters
    net.save_network_parameters(save_tunet_path)

    # Clear out unnecessary variables from device memory
    torch.cuda.empty_cache()

    # Load annotated data for segmentation using trained model
    test_loader_params = {
        'batch_size': parameters.batch_size,
        'shuffle': False
        }
    test_loader = DataLoader(dataset, **test_loader_params)

    # Start segmentation
    net.to(device)   # send network to GPU
    for batch in test_loader:
        with torch.no_grad():
            data, _ = batch
            # Necessary data recasting
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            # Input passed through networks here
            output = net(data)
            # Individual output passed through argmax to get class prediction
            prediction = torch.argmax(output.cpu().data, dim=1)
            torch.save(prediction, save_tunet_path + f'preds_tunet-batch{n}.pt')