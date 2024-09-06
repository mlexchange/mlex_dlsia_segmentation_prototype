import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dlsia.core.train_scripts import Trainer
from qlty.qlty2D import NCYXQuilt

from network import build_network
from tiled_dataset import initialize_tiled_datasets
from utils import (
    allocate_array_space,
    construct_dataloaders,
    create_directory,
    ensure_parent_containers,
    find_device,
    normalization,
    qlty_crop,
    segment_single_frame,
    validate_parameters,
)


def prepare_data_and_mask(tiled_masked_dataset):
    """
    This function extracts the data and mask array stack from the tiled dataset class
    Input:
        tiled_masked_dataset: class
    Output:
        data: np.ndarray, all labeled raw image stack
        mask: np.ndarray, all masks stack
    """
    # Load all labeled data and masks into memory
    # TODO: Check if this has limitations in regards to number of possible slices
    # Directly accessing client data is not the intended use of the dataset class
    data = tiled_masked_dataset.data_client[tiled_masked_dataset.selected_indices, :]
    mask = tiled_masked_dataset.mask_client[:]
    return data, mask


def build_criterion(model_parameters, device, ignore_index=-1, size_average=None):
    """
    This function builds the criterion used for model training based on weights provided from the parameters,
    and pass to the device.
    Input:
        model_parameters: class, pydantic validated model parameters
        device: torch.device object, cpu or gpu
    Output:
        criterion:
    """
    # Define criterion and optimizer
    criterion = getattr(nn, model_parameters.criterion)
    # Convert the string to a list of floats
    weights = [float(x) for x in model_parameters.weights.strip("[]").split(",")]
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = criterion(
        weight=weights, ignore_index=ignore_index, size_average=size_average
    )
    return criterion


def train_network(
    network_name,
    networks,
    io_parameters,
    model_parameters,
    device,
    model_dir,
    train_loader,
    criterion,
    val_loader=None,
    use_dvclive=False,
    use_savedvcexp=False,
):
    """
    This function builds the dlsia trainer based on the network (or ensemble) provided and performs training
    Input:
        network_name: str, model name to be trained
        networks: list, constructed dlsia models
        io_parameters: class, pydantic validated io parameters from yaml file, in order to use uid in saved model name
        model_parameters: class, pydantic validated model parameters from yaml file
        device: torch.device, either cpu or gpu
        model_dir: str, path to save the model
        train_loader: torch dataloader for training set
        criterion: pre-built criterion for training
        val_loader: torch dataloaderfor validation
        use_dvclive: bool, whether to use dvclive to store model performance
        use_savedvcexp: bool, whether to use dvcexp
    Output:
        net: trained model
    """
    for idx, net in enumerate(networks):
        print(f"{network_name}: {idx+1}/{len(networks)}")
        optimizer = getattr(optim, model_parameters.optimizer)
        optimizer = optimizer(net.parameters(), lr=model_parameters.learning_rate)
        net = net.to(device)

        if use_dvclive:
            from dvclive import Live

            dvclive_savepath = f"{model_dir}/dvc_metrics"
            dvclive = Live(dvclive_savepath, report="html", save_dvc_exp=use_savedvcexp)
        else:
            dvclive = None

        trainer = Trainer(
            net,
            train_loader,
            val_loader,
            model_parameters.num_epochs,
            criterion,
            optimizer,
            device,
            dvclive=dvclive,
            savepath=model_dir,
            saveevery=None,
            scheduler=None,
            show=0,
            use_amp=False,
            clip_value=None,
        )
        net, results = trainer.train_segmentation()  # training happens here
        # Save network parameters
        model_params_path = os.path.join(
            model_dir, f"{io_parameters.uid_save}_{network_name}{idx+1}.pt"
        )
        net.save_network_parameters(model_params_path)
        # Clear out unnecessary variables from device memory
        torch.cuda.empty_cache()
    print(f"{network_name} trained successfully.")
    return net


def train(args):
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)
    io_parameters, network_name, model_parameters = validate_parameters(parameters)
    dataset = initialize_tiled_datasets(io_parameters, is_training=True)
    data, mask = prepare_data_and_mask(dataset)
    data = normalization(data)
    data = torch.from_numpy(data)
    mask = torch.from_numpy(mask)
    qlty_object = NCYXQuilt(
        X=dataset.shape[-1],
        Y=dataset.shape[-2],
        window=(model_parameters.qlty_window, model_parameters.qlty_window),
        step=(model_parameters.qlty_step, model_parameters.qlty_step),
        border=(model_parameters.qlty_border, model_parameters.qlty_border),
        border_weight=0.2,
    )
    patched_data, patched_mask = qlty_crop(
        qlty_object, data, is_training=True, masks=mask
    )
    train_loader, val_loader = construct_dataloaders(
        patched_data, model_parameters, is_training=True, masks=patched_mask
    )
    # Build network
    networks = build_network(
        network_name=network_name,
        data_shape=patched_data.shape,  # TODO: Double check if this needs to be switched to the patch dim
        # data_shape=dataset.data_client.shape,
        num_classes=model_parameters.num_classes,
        parameters=model_parameters,
    )
    device = find_device()
    torch.cuda.empty_cache()
    print(f"Training will be processed on: {device}")
    criterion = build_criterion(model_parameters, device)
    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_save)
    # Create Result Directory if not existed
    create_directory(model_dir)
    net = train_network(
        network_name=network_name,
        networks=networks,
        io_parameters=io_parameters,
        model_parameters=model_parameters,
        device=device,
        model_dir=model_dir,
        train_loader=train_loader,
        criterion=criterion,
        val_loader=val_loader,
        use_dvclive=True,
        use_savedvcexp=False,
    )
    return io_parameters, network_name, model_parameters, qlty_object, device, net


def partial_inference(
    io_parameters, network_name, model_parameters, qlty_object, device, net
):
    dataset = initialize_tiled_datasets(io_parameters, is_training=True)
    torch.cuda.empty_cache()
    print(f"Partial Inference will be processed on: {device}")
    # Allocate Result space in Tiled
    last_container = ensure_parent_containers(
        io_parameters.seg_tiled_uri, io_parameters.seg_tiled_api_key
    )
    seg_client = allocate_array_space(
        tiled_dataset=dataset,
        last_container=last_container,
        uid=io_parameters.uid_save,
        model_name=network_name,
        array_name="seg_result",
    )
    softmax = torch.nn.Softmax(dim=1)
    for idx in range(len(dataset)):
        image = dataset[idx]
        image = normalization(image)
        image = torch.from_numpy(image)
        patches = qlty_crop(qlty_object=qlty_object, images=image, is_training=False)
        inference_loader = construct_dataloaders(
            patches, model_parameters, is_training=False
        )
        prediction = segment_single_frame(
            network=net, dataloader=inference_loader, final_layer=softmax, device=device
        )
        stitched_prediction, _ = qlty_object.stitch(prediction)
        result = torch.argmax(stitched_prediction, dim=1).numpy().astype(np.int8)
        seg_client.write_block(result, block=(idx, 0, 0))
        print(f"Frame {idx+1} result saved to Tiled")
    print("Segmentation preview completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()
    io_parameters, network_name, model_parameters, qlty_object, device, net = train(
        args
    )
    partial_inference(
        io_parameters, network_name, model_parameters, qlty_object, device, net
    )
