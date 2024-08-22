import argparse
import os

import numpy as np
import torch
import yaml
from qlty.qlty2D import NCYXQuilt

from utils import (
    allocate_array_space,
    construct_dataloaders,
    ensure_parent_containers,
    find_device,
    initialize_tiled_datasets,
    load_dlsia_network,
    normalization,
    qlty_crop,
    segment_single_frame,
    validate_parameters,
)


def full_inference(args):
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)
    io_parameters, network_name, model_parameters = validate_parameters(parameters)
    dataset = initialize_tiled_datasets(io_parameters, is_training=False)
    qlty_object = NCYXQuilt(
        X=dataset.data_client.shape[-1],
        Y=dataset.data_client.shape[-2],
        window=(model_parameters.qlty_window, model_parameters.qlty_window),
        step=(model_parameters.qlty_step, model_parameters.qlty_step),
        border=(model_parameters.qlty_border, model_parameters.qlty_border),
        border_weight=0.2,
    )
    device = find_device()
    torch.cuda.empty_cache()
    print(f"Full Inference will be processed on: {device}")
    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_retrieve)
    net = load_dlsia_network(network_name=network_name, model_dir=model_dir)
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
