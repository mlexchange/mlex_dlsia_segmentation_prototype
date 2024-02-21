import argparse
import json

from parameters import TrainingParameters
from tiled_dataset import InferenceDataset

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from network import load_network
from seg_utils import segment
from utils import save_seg_to_tiled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_tiled_uri', help='tiled uri to training data')
    parser.add_argument('mask_tiled_uri', help='tiled uri to mask')
    parser.add_argument('seg_tiled_uri', help='tiled uri to segmentation result')
    parser.add_argument('data_tiled_api_key', help='tiled api key for training data')
    parser.add_argument('mask_tiled_api_key', help='tiled api key for mask')
    parser.add_argument('seg_tiled_api_key', help='tiled api key for segmentation result')   
    parser.add_argument('mask_idx', help='mask index from data')
    parser.add_argument('save_path', help = 'save path for models and outputs')
    parser.add_argument('uid', help='uid for segmentation instance')
    parser.add_argument('parameters', help='training parameters')
    
    args = parser.parse_args()

    # Load parameters
    parameters = TrainingParameters(**json.loads(args.parameters))
    
    dataset = InferenceDataset(
        data_tiled_uri=args.data_tiled_uri,
        mask_tiled_uri=args.mask_tiled_uri,
        seg_tiled_uri=args.seg_tiled_uri,
        mask_idx=json.loads(args.mask_idx), # Convert str to list
        data_tiled_api_key=args.data_tiled_api_key,
        mask_tiled_api_key=args.mask_tiled_api_key,
        seg_tiled_api_key=args.seg_tiled_api_key,
        transform=transforms.ToTensor()
        )
    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    inference_loader_params = {'batch_size': parameters.dataloaders.batch_size_inference,
                               'shuffle': parameters.dataloaders.shuffle_inference}
    # Build Dataloaders
    inference_loader = DataLoader(dataset, **inference_loader_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Network
    model_params_path = f'{args.save_path}/{args.uid}_{parameters.network}.pt'
    net = load_network(parameters.network, model_params_path)

    # Start segmentation
    seg_result = segment(net, device, inference_loader)
    
    # Save results back to Tiled
    # TODO: Change the hard-coding of container keys
    container_keys = ["mlex_store", 'rec20190524_085542_clay_testZMQ_8bit', 'results']
    container_keys.append(args.uid)
    
    seg_result_uri, seg_result_metadata = save_seg_to_tiled(seg_result, dataset, container_keys, args.uid, parameters.network)
