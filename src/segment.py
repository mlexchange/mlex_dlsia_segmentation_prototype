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
from utils import (
    load_yaml,
    validate_parameters,
    initialize_tiled_datasets,
    build_qlty_object,
    find_device,
    load_dlsia_network,
    allocate_array_space,
)

def full_inference(args):
    parameters = load_yaml(args.yaml_path)
    io_parameters, network_name, model_parameters = validate_parameters(parameters)
    dataset = initialize_tiled_datasets(io_parameters, is_training=False)
    qlty_object = build_qlty_object(
        width = dataset.data_client.shape[-1],
        height = dataset.data_client.shape[-2],
        window = model_parameters.qlty_window,
        step = model_parameters.qlty_step,
        border = model_parameters.qlty_border,
        border_weight = 0.2,
    )
    device = find_device()
    torch.cuda.empty_cache()
    print(f"Full Inference will be processed on: {device}")
    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_retrieve)
    net = load_dlsia_network(network_name=network_name, model_dir=model_dir)
    # Allocate Result space in Tiled
    seg_client = allocate_array_space(
        tiled_dataset=dataset,
        seg_tiled_uri=io_parameters.seg_tiled_uri,
        seg_tiled_api_key=io_parameters.seg_tiled_api_key,
        uid=io_parameters.uid_save,
        model=network_name,
        array_name="seg_result",
    )
    for idx in range(len(dataset)):
        seg_result = crop_seg_save(
            net=net,
            device=device,
            image=dataset[idx],
            qlty_object=qlty_object,
            parameters=model_parameters,
            tiled_client=seg_client,
            frame_idx=idx,
        )
    print("Segmentation completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()
    full_inference(args)
    
    # data_tiled_client = from_uri(
    #     io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
    # )
    # mask_tiled_client = None
    # if io_parameters.mask_tiled_uri:
    #     mask_tiled_client = from_uri(
    #         io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
    #     )
    # dataset = TiledDataset(
    #     data_tiled_client,
    #     mask_tiled_client=mask_tiled_client,
    #     is_training=False,
    #     using_qlty=False,
    #     qlty_window=model_parameters.qlty_window,
    #     qlty_step=model_parameters.qlty_step,
    #     qlty_border=model_parameters.qlty_border,
    #     transform=transforms.ToTensor(),
    # )

    # qlty_inference = NCYXQuilt(
    #     X=dataset.data_client.shape[-1],
    #     Y=dataset.data_client.shape[-2],
    #     window=(model_parameters.qlty_window, model_parameters.qlty_window),
    #     step=(model_parameters.qlty_step, model_parameters.qlty_step),
    #     border=(model_parameters.qlty_border, model_parameters.qlty_border),
    #     border_weight=0.2,
    # )

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # torch.cuda.empty_cache()
    # print(f"Inference will be processed on: {device}")

    # model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_retrieve)

    # # Load Network
    # if network == "SMSNetEnsemble":
    #     net = baggin_smsnet_ensemble(model_dir)
    # else:
    #     net_files = glob.glob(os.path.join(model_dir, "*.pt"))
    #     net = load_network(network, net_files[0])

    # # Allocate Result space in Tiled
    # seg_client = allocate_array_space(
    #     tiled_dataset=dataset,
    #     seg_tiled_uri=io_parameters.seg_tiled_uri,
    #     seg_tiled_api_key=io_parameters.seg_tiled_api_key,
    #     uid=io_parameters.uid_save,
    #     model=network,
    #     array_name="seg_result",
    # )
    

    # for idx in range(len(dataset)):
    #     seg_result = crop_seg_save(
    #         net=net,
    #         device=device,
    #         image=dataset[idx],
    #         qlty_object=qlty_inference,
    #         parameters=model_parameters,
    #         tiled_client=seg_client,
    #         frame_idx=idx,
    #     )
    # print("Segmentation completed.")