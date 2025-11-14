import argparse
import logging

import yaml
from dlsia.core.helpers import get_device

from mlex_dlsia.dataset import initialize_tiled_datasets
from mlex_dlsia.inference import run_inference
from mlex_dlsia.network import build_network, load_network
from mlex_dlsia.train import run_train
from mlex_dlsia.utils.dataloaders import construct_train_dataloaders
from mlex_dlsia.utils.params_validation import validate_parameters
from mlex_dlsia.utils.tiled import prepare_tiled_containers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    parser.add_argument(
        "--train", type=bool, default=True, help="whether to train the model"
    )

    # Load parameters
    args = parser.parse_args()
    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)
    logging.info("Parameters loaded from yaml file.")

    io_parameters, model_parameters = validate_parameters(parameters)
    logging.info("Parameters validated successfully.")

    device = get_device()
    logger.info(f"Process will be processed on: {device}")

    if args.train:
        dataset = initialize_tiled_datasets(
            io_parameters, model_parameters, is_training=args.train
        )
        logging.info("Tiled datasets initialized successfully.")

        # Build network
        # TODO: Assumes that the last channel is the number of channels in the image with a max of 4 channels
        qlty_window = model_parameters.qlty_window
        last_channel = dataset.data_client.shape[-1]
        networks = build_network(
            network_name=model_parameters.network,
            in_channels=last_channel if last_channel <= 4 else 1,
            image_shape=(qlty_window, qlty_window),
            num_classes=model_parameters.num_classes,
            parameters=parameters,  # Pass the raw parameters dictionary for network construction
        )

        train_loader, val_loader = construct_train_dataloaders(
            dataset, model_parameters, is_training=True
        )

        net = run_train(
            train_loader, val_loader, io_parameters, networks, model_parameters, device
        )
        logging.info("Training completed successfully.")
    else:
        net = load_network(model_parameters.network, io_parameters.models_dir)
        logging.info("Model loaded successfully for inference.")

    # Prepare dataset for inference
    dataset = initialize_tiled_datasets(
        io_parameters, model_parameters, is_training=False
    )

    seg_client = prepare_tiled_containers(
        io_parameters, dataset, model_parameters.network
    )

    run_inference(
        dataset,
        net,
        seg_client,
        model_parameters,
        device,
    )
    logging.info("Inference completed successfully.")
