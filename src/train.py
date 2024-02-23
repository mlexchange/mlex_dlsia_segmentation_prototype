import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from parameters import TrainingParameters
from tiled_dataset import TrainingDataset
from utils import create_directory
from network import build_network
from seg_utils import train_val_split, train_segmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_tiled_uri', help='tiled uri to training data')
    parser.add_argument('mask_tiled_uri', help='tiled uri to mask')
    parser.add_argument('data_tiled_api_key', help='tiled api key for training data')
    parser.add_argument('mask_tiled_api_key', help='tiled api key for mask')
    parser.add_argument('mask_idx', help='mask index from data')
    parser.add_argument('shift', help='pixel shifts for mask')
    parser.add_argument('save_path', help = 'save path for models and outputs')
    parser.add_argument('uid', help='uid for segmentation instance')
    parser.add_argument('parameters', help='training parameters')
    
    args = parser.parse_args()

    # Create Result Directory if not existed
    create_directory(args.save_path)

    # Load parameters
    parameters = TrainingParameters(**json.loads(args.parameters))
    
    dataset = TrainingDataset(
        data_tiled_uri=args.data_tiled_uri,
        mask_tiled_uri=args.mask_tiled_uri,
        mask_idx=json.loads(args.mask_idx), # Convert str to list
        data_tiled_api_key=args.data_tiled_api_key,
        mask_tiled_api_key=args.mask_tiled_api_key,
        shift=args.shift,
        transform=transforms.ToTensor()
        )

    train_loader, val_loader = train_val_split(dataset, parameters.dataloaders)
    
    # Build network
    net = build_network(
        network=parameters.network,
        data_shape=dataset.data_client.shape,
        num_classes=parameters.num_classes,
        parameters=parameters,
        )

    # Define criterion and optimizer
    criterion = getattr(nn, parameters.criterion.value)
    criterion = criterion(ignore_index=-1, size_average=None)    
    optimizer = getattr(optim, parameters.optimizer.value)
    optimizer = optimizer(net.parameters(), lr = parameters.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, results = train_segmentation(
        net,
        train_loader,
        val_loader,
        parameters.num_epochs,
        criterion,
        optimizer,
        device,
        savepath=args.save_path,
        saveevery=None,
        scheduler=None,
        show=0,
        use_amp=False,
        clip_value=None
        )

    # Save network parameters
    model_params_path = f'{args.save_path}/{args.uid}_{parameters.network}.pt'
    net.save_network_parameters(model_params_path)

    print('Network Training Successful.')
    # Clear out unnecessary variables from device memory
    torch.cuda.empty_cache()

    
