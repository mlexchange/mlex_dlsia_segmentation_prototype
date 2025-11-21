import logging
import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from dlsia.core.train_scripts import Trainer

from mlex_dlsia.network import baggin_smsnet_ensemble

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _build_criterion(model_parameters, device, ignore_index=-1):
    """
    This function builds the criterion used for model training based on weights provided from the parameters,
    and pass to the device.
    Input:
        model_parameters: class, pydantic validated model parameters
        device: torch.device object, cpu or gpu
        ignore_index: int, index to ignore in the loss calculation
    Output:
        criterion:
    """
    # Define criterion and optimizer
    criterion = getattr(nn, model_parameters.criterion)
    # Convert the string to a list of floats
    weights = [float(x) for x in model_parameters.weights.strip("[]").split(",")]
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = criterion(weight=weights, ignore_index=ignore_index)
    return criterion


def run_train(
    train_loader,
    val_loader,
    io_parameters,
    networks,
    model_parameters,
    device,
    model_dir,
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
        model_dir: str, directory to save model parameters and metrics
        use_dvclive: bool, whether to use dvclive for logging
        use_savedvcexp: bool, whether to save dvclive experiments
    Output:
        net: nn.Module object, trained network
    """
    mlflow.set_experiment(io_parameters.uid_save)
    print(f"Setting MLflow experiment name: {io_parameters.uid_save}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")

        # NEW: Log hyperparameters
        mlflow.log_params(
            {
                "network": model_parameters.network,
                "num_classes": model_parameters.num_classes,
                "num_epochs": model_parameters.num_epochs,
                "optimizer": model_parameters.optimizer,
                "criterion": model_parameters.criterion,
                "learning_rate": model_parameters.learning_rate,
                "batch_size_train": model_parameters.batch_size_train,
                "batch_size_val": model_parameters.batch_size_val,
                "val_pct": model_parameters.val_pct,
            }
        )

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
                dvclive = Live(
                    dvclive_savepath, report="html", save_dvc_exp=use_savedvcexp
                )
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

            trained_nets.append(net)
            # Clear out unnecessary variables from device memory
            torch.cuda.empty_cache()
        logger.info(f"{network_name} trained successfully.")

        if model_parameters.network == "DLSIA SMSNetEnsemble":
            net = baggin_smsnet_ensemble(networks=trained_nets)
        else:
            net = trained_nets[0]

        # Log model to MLflow
        mlflow.pytorch.log_model(
            net, f"model_{idx+1}", registered_model_name=io_parameters.uid_save
        )
        print(f"Model logged to MLflow with name: {io_parameters.uid_save}")

    # Log DVC metrics to MLflow
    if use_dvclive and os.path.exists(dvclive_savepath):
        mlflow.log_artifacts(dvclive_savepath, artifact_path="dvc_metrics")
        print(f"DVC metrics logged to MLflow from {dvclive_savepath}")
    return net
