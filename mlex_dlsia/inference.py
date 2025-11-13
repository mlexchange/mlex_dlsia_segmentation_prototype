import logging

import numpy as np
import torch

from mlex_dlsia.utils.dataloaders import construct_inference_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference(dataset, net, seg_client, model_parameters, device):
    torch.cuda.empty_cache()
    logging.info(f"Full Inference will be processed on: {device}")

    # Inference
    softmax = torch.nn.Softmax(dim=1)
    net.eval().to(device)
    for idx in range(len(dataset)):
        inference_loader = construct_inference_dataloaders(
            dataset[idx], model_parameters
        )
        prediction = _segment_single_frame(
            network=net, dataloader=inference_loader, final_layer=softmax, device=device
        )
        stitched_prediction, _ = dataset.qlty_object.stitch(prediction)
        result = torch.argmax(stitched_prediction, dim=1).numpy().astype(np.int8)
        seg_client.write_block(result, block=(idx, 0, 0))
        logging.info(f"Frame {idx+1} result saved to Tiled")


def _segment_single_frame(network, dataloader, final_layer, device):
    """
    This function segments a single frame using the given neural network and returns the segmentation results.

    Input:
        network: torch.nn.Module, The neural network model used for segmentation.
        dataloader: torch.utils.data.DataLoader, DataLoader providing the data batches to be segmented.
        final_layer: callable, A function or layer that processes the network output to obtain the final segmentation.
        device: torch.device, The device (CPU or GPU) on which to perform the computations.

    Output:
        results: torch.Tensor, A tensor containing the concatenated segmentation results for all batches.
    """
    results = []
    for batch in dataloader:
        with torch.no_grad():
            torch.cuda.empty_cache()
            patches = batch[0].type(torch.FloatTensor)
            tmp = final_layer(network(patches.to(device))).cpu()
            results.append(tmp)
    results = torch.cat(results)
    return results
