import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from dlsia.core.train_scripts import Trainer

from mlex_dlsia.network import baggin_smsnet_ensemble

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _build_criterion(model_parameters, device, ignore_index=-1, size_average=None):
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


def run_train(
    train_loader,
    val_loader,
    io_parameters,
    networks,
    model_parameters,
    device,
    use_dvclive=True,
    use_savedvcexp=False,
):
    """
    Run the training process for the given networks.
    Input:
        train_loader: DataLoader object for training data
        val_loader: DataLoader object for validation data
        io_parameters: class, pydantic validated I/O parameters
        networks: list of nn.Module objects to be trained
        model_parameters: class, pydantic validated model parameters
        device: torch.device object, cpu or gpu
        use_dvclive: bool, whether to use dvclive for logging
        use_savedvcexp: bool, whether to save dvclive experiments
    Output:
        net: nn.Module object, trained network
    """
    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_save)
    os.makedirs(model_dir, exist_ok=True)

    torch.cuda.empty_cache()
    criterion = _build_criterion(model_parameters, device)

    network_name = model_parameters.network
    trained_nets = []
    for idx, net in enumerate(networks):
        logger.info(f"{network_name}: {idx+1}/{len(networks)}")
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
        net, _ = trainer.train_segmentation()  # training happens here
        # Save network parameters
        model_params_path = os.path.join(
            model_dir, f"{io_parameters.uid_save}_{network_name}{idx+1}.pt"
        )
        net.save_network_parameters(model_params_path)
        trained_nets.append(net)
        # Clear out unnecessary variables from device memory
        torch.cuda.empty_cache()
    logger.info(f"{network_name} trained successfully.")

    if model_parameters.network == "DLSIA SMSNetEnsemble":
        net = baggin_smsnet_ensemble(networks=trained_nets)
    else:
        net = trained_nets[0]
    return net
