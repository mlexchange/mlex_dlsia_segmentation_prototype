import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from parameters import TrainingParameters
from tiled_dataset import TiledDataset
from utils import decrypt, save_seg_to_tiled
from network import build_network
from seg_utils import train_val_split, train_segmentation, segment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recon_uri', help='tiled uri to training data')
    parser.add_argument('mask_uri', help='tiled uri to mask')
    parser.add_argument('seg_uri', help='tiled uri to segmentation result')
    parser.add_argument('recon_api_key', help='tiled api key for training data')
    parser.add_argument('mask_api_key', help='tiled api key for mask')
    parser.add_argument('seg_api_key', help='tiled api key for segmentation result')
    parser.add_argument('mask_idx', help='mask index from recon data')
    parser.add_argument('shift', help='pixel shifts for mask')
    parser.add_argument('save_path', help = 'save path for outputs')
    parser.add_argument('uid', help='uid for segmentation instance')
    parser.add_argument('parameters', help='training parameters')
    args = parser.parse_args()

    # Load parameters
    parameters = TrainingParameters(**json.loads(args.parameters))

    # # TODO: Decryption
    # DECRYPTION_KEY = os.getenv('DECRYPTION_KEY')
    # data_api_key = decrypt(args.data_api_key, DECRYPTION_KEY)
    # mask_api_key = decrypt(args.mask_api_key, DECRYPTION_KEY)

    dataset = TiledDataset(
        recon_uri=args.recon_uri,
        mask_uri=args.mask_uri,
        seg_uri=args.seg_uri,
        mask_idx=args.mask_idx,
        recon_api_key=args.recon_api_key,
        mask_api_key=args.mask_api_key,
        seg_api_key=args.seg_api_key,
        shift=args.shift,
        transform=transforms.ToTensor()
        )
    train_loader, val_loader, test_loader = train_val_split(dataset, **parameters.dataloaders)

    # Build network
    net = build_network(
        network=parameters.network,
        num_classes=parameters.num_classes,
        img_size=dataset.recon_client.shape[-2:],
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
    net.save_network_parameters(args.save_path)

    # Clear out unnecessary variables from device memory
    torch.cuda.empty_cache()

    # Start segmentation
    seg = segment(net, device, test_loader)
    
    # Save results back to Tiled
    # TODO: Change the hard-coding of container keys
    container_keys = ["mlex_store", 'rec20190524_085542_clay_testZMQ_8bit', 'results']
    container_keys.append(args.uid)
    
    seg_result_uri, seg_result_metadata = save_seg_to_tiled(seg, dataset, container_keys, parameters.network)

