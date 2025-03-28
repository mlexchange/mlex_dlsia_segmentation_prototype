import argparse
import glob
import os

import torch
import yaml
from qlty.qlty2D import NCYXQuilt
from tiled.client import from_uri
from torchvision import transforms

from network import baggin_smsnet_ensemble, load_network
from parameters import (
    IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)
from seg_utils import crop_seg_save
from tiled_dataset import TiledDataset
from utils import allocate_array_space

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

    # Detect which model we have, then load corresponding parameters
    raw_parameters = parameters["model_parameters"]
    network = raw_parameters["network"]

    model_parameters = None
    if network == "DLSIA MSDNet":
        model_parameters = MSDNetParameters(**raw_parameters)
    elif network == "DLSIA TUNet":
        model_parameters = TUNetParameters(**raw_parameters)
    elif network == "DLSIA TUNet3+":
        model_parameters = TUNet3PlusParameters(**raw_parameters)
    elif network == "DLSIA SMSNetEnsemble":
        model_parameters = SMSNetEnsembleParameters(**raw_parameters)

    assert model_parameters, f"Received Unsupported Network: {network}"

    print("Parameters loaded successfully.")

    data_tiled_client = from_uri(
        io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
    )
    mask_tiled_client = None
    if io_parameters.mask_tiled_uri:
        mask_tiled_client = from_uri(
            io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
        )
    dataset = TiledDataset(
        data_tiled_client,
        mask_tiled_client=mask_tiled_client,
        is_training=False,
        using_qlty=False,
        qlty_window=model_parameters.qlty_window,
        qlty_step=model_parameters.qlty_step,
        qlty_border=model_parameters.qlty_border,
        transform=transforms.ToTensor(),
    )

    qlty_inference = NCYXQuilt(
        X=dataset.data_client.shape[-1],
        Y=dataset.data_client.shape[-2],
        window=(model_parameters.qlty_window, model_parameters.qlty_window),
        step=(model_parameters.qlty_step, model_parameters.qlty_step),
        border=(model_parameters.qlty_border, model_parameters.qlty_border),
        border_weight=0.2,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    print(f"Inference will be processed on: {device}")

    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_retrieve)

    # Load Network
    if network == "SMSNetEnsemble":
        net = baggin_smsnet_ensemble(model_dir)
    else:
        net_files = glob.glob(os.path.join(model_dir, "*.pt"))
        net = load_network(network, net_files[0])

    # Allocate Result space in Tiled
    seg_client = allocate_array_space(
        tiled_dataset=dataset,
        seg_tiled_uri=io_parameters.seg_tiled_uri,
        seg_tiled_api_key=io_parameters.seg_tiled_api_key,
        uid=io_parameters.uid_save,
        job_name=io_parameters.job_name,
        model=network,
        array_name="seg_result",
    )
    print(
        f"Result space allocated in Tiled and segmentation will be saved in {seg_client.uri}."
    )

    for idx in range(len(dataset)):
        seg_result = crop_seg_save(
            net=net,
            device=device,
            image=dataset[idx],
            qlty_object=qlty_inference,
            parameters=model_parameters,
            tiled_client=seg_client,
            frame_idx=idx,
        )
    print("Segmentation completed.")
    # # # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    # # inference_loader_params = {'batch_size': model_parameters.batch_size_inference,
    # #                            'shuffle': False}
    # # # Build Dataloaders
    # # inference_loader = DataLoader(dataset, **inference_loader_params, collate_fn=custom_collate)

    # # # Start segmentation and save frame by frame
    # # frame_count = segment(net, device, inference_loader, dataset.qlty_object, seg_client)
