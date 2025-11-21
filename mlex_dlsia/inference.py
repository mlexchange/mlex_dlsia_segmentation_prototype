import logging

import numpy as np
import torch

from mlex_dlsia.utils.dataloaders import construct_inference_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_inference(dataset, net, seg_client, model_parameters, device):
    torch.cuda.empty_cache()
    logging.info(f"Starting inference on {len(dataset)} frames using device: {device}")

    # Determine if ensemble and whether to apply softmax
    is_ensemble = model_parameters.network == "DLSIA SMSNetEnsemble"
    final_layer = None if is_ensemble else torch.nn.Softmax(dim=1)
    segment_fn = (
        _segment_single_frame_ensemble if is_ensemble else _segment_single_frame
    )

    net.eval().to(device)
    for idx in range(len(dataset)):
        inference_loader = construct_inference_dataloaders(
            dataset[idx], model_parameters
        )
        prediction = segment_fn(
            network=net,
            dataloader=inference_loader,
            final_layer=final_layer,
            device=device,
        )

        stitched_prediction, _ = dataset.qlty_object.stitch(prediction)
        result = torch.argmax(stitched_prediction, dim=1).numpy().astype(np.int8)
        seg_client.write_block(result, block=(idx, 0, 0))
        logging.info(f"Frame {idx+1} result saved to Tiled")

        if device != "cpu" and (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    pass


def _segment_single_frame_ensemble(network, dataloader, device, final_layer=None):
    """
    Segment a single frame using an ensemble network.

    The ensemble already returns softmax probabilities, so no additional
    activation is needed.

    Args:
        network: Ensemble network model
        dataloader: DataLoader providing patches
        final_layer: Not used (kept for API consistency)
        device: Computation device

    Returns:
        Concatenated predictions for all patches
    """
    results = []
    for batch in dataloader:
        with torch.no_grad():
            torch.cuda.empty_cache()
            patches = batch[0].float().to(device)
            mean, _ = network(patches, device=device, return_std=True)
            results.append(mean.cpu())
    results = torch.cat(results)
    return results


def _segment_single_frame(network, dataloader, final_layer, device):
    """
    Segment a single frame using a single network.

    Args:
        network: Single network model
        dataloader: DataLoader providing patches
        final_layer: Activation layer (e.g., Softmax)
        device: Computation device

    Returns:
        Concatenated predictions for all patches
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
