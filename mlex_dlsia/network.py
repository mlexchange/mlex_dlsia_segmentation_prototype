import logging
import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from dlsia.core import helpers
from dlsia.core.networks import msdnet, smsnet, tunet, tunet3plus
from dlsia.core.networks.baggins import model_baggin

from mlex_dlsia.parameters import (
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_network(
    network_name,
    in_channels,
    image_shape,
    num_classes,
    parameters,
):
    if network_name == "DLSIA MSDNet":
        parameters = MSDNetParameters(**parameters)
    elif network_name == "DLSIA TUNet":
        parameters = TUNetParameters(**parameters)
    elif network_name == "DLSIA TUNet3+":
        parameters = TUNet3PlusParameters(**parameters)
    elif network_name == "DLSIA SMSNetEnsemble":
        parameters = SMSNetEnsembleParameters(**parameters)
    else:
        raise ValueError(f"Unsupported network: {network_name}")

    out_channels = num_classes

    if parameters.activation is not None:
        activation = getattr(nn, parameters.activation)
        activation = activation()

    if parameters.normalization is not None:
        normalization = getattr(nn, parameters.normalization)

    if parameters.convolution is not None:
        convolution = getattr(nn, parameters.convolution)

    if network_name == "DLSIA MSDNet":
        network = build_msdnet(
            in_channels,
            out_channels,
            parameters,
            activation,
            normalization,
            convolution,
        )
    elif network_name == "DLSIA TUNet":
        network = build_tunet(
            in_channels,
            out_channels,
            image_shape,
            parameters,
            activation,
            normalization,
        )
    elif network_name == "DLSIA TUNet3+":
        network = build_tunet3plus(
            in_channels,
            out_channels,
            image_shape,
            parameters,
            activation,
            normalization,
        )
    elif network_name == "DLSIA SMSNetEnsemble":
        network = build_smsnet_ensemble(
            in_channels,
            out_channels,
            parameters,
        )
    return network


def load_network(model_name):
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(model_name)[0]
    run_id = model_version.run_id

    cfg = model_version.tags
    is_ensemble = cfg.get("network") == "DLSIA SMSNetEnsemble"

    # Download artifacts directly from MLflow under the pyfunc model artifacts dir.
    # Newer runs store nets as model/artifacts/net_*.pt.
    artifact_infos = client.list_artifacts(run_id, path="model/artifacts")
    net_artifact_paths = sorted(
        [
            info.path
            for info in artifact_infos
            if not info.is_dir and os.path.basename(info.path).startswith("net_")
        ]
    )

    if not net_artifact_paths:
        # Backward-compatible fallback for older layout: model/net_*.pt
        artifact_infos = client.list_artifacts(run_id, path="model")
        net_artifact_paths = sorted(
            [
                info.path
                for info in artifact_infos
                if not info.is_dir and os.path.basename(info.path).startswith("net_")
            ]
        )

    n_nets = int(cfg.get("n_nets", len(net_artifact_paths) or 1))
    if len(net_artifact_paths) < n_nets:
        raise FileNotFoundError(
            f"Expected {n_nets} model artifacts but found {len(net_artifact_paths)} for run {run_id}."
        )

    local_net_paths = [
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
        for path in net_artifact_paths[:n_nets]
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets = [torch.load(path, map_location=device) for path in local_net_paths]

    if is_ensemble:
        return model_baggin(models=nets, model_type="classification")
    else:
        return nets[0]


# ============================MSDNet==================================#
def build_msdnet(
    in_channels,
    out_channels,
    msdnet_parameters,
    activation,
    normalization,
    convolution,
):

    if not msdnet_parameters.custom_dilation:
        logging.info(f"Using maximum dilation: {msdnet_parameters.max_dilation}")
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=msdnet_parameters.num_layers,
            layer_width=msdnet_parameters.layer_width,
            max_dilation=msdnet_parameters.max_dilation,
            activation=activation,
            normalization=normalization,
            convolution=convolution,
        )
    else:
        dilation_array = [
            int(x) for x in msdnet_parameters.dilation_array.strip("[]").split(",")
        ]
        dilation_array = np.array(dilation_array)
        logging.info(f"Using custom dilation: {dilation_array}")
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=msdnet_parameters.num_layers,
            layer_width=msdnet_parameters.layer_width,
            custom_msdnet=dilation_array,
            activation=activation,
            normalization=normalization,
            convolution=convolution,
        )
    return [network]


# ============================TUNet==================================#
def build_tunet(
    in_channels,
    out_channels,
    image_shape,
    tunet_parameters,
    activation,
    normalization,
):
    network = tunet.TUNet(
        image_shape=image_shape,
        in_channels=in_channels,
        out_channels=out_channels,
        depth=tunet_parameters.depth,
        base_channels=tunet_parameters.base_channels,
        growth_rate=tunet_parameters.growth_rate,
        hidden_rate=tunet_parameters.hidden_rate,
        activation=activation,
        normalization=normalization,
    )
    return [network]


# ============================TUNet3+==================================#
def build_tunet3plus(
    in_channels,
    out_channels,
    image_shape,
    tunet3plus_parameters,
    activation,
    normalization,
):

    network = tunet3plus.TUNet3Plus(
        image_shape=image_shape,
        in_channels=in_channels,
        out_channels=out_channels,
        depth=tunet3plus_parameters.depth,
        base_channels=tunet3plus_parameters.base_channels,
        growth_rate=tunet3plus_parameters.growth_rate,
        hidden_rate=tunet3plus_parameters.hidden_rate,
        carryover_channels=tunet3plus_parameters.carryover_channels,
        activation=activation,
        normalization=normalization,
    )
    return [network]


# ============================SMSNet Ensemble==================================#
def construct_2dsms_ensembler(
    n_networks,
    in_channels,
    out_channels,
    layers,
    alpha=0.0,
    gamma=0.0,
    hidden_channels=None,
    dilation_choices=[1, 2, 3, 4],
    P_IL=0.995,
    P_LO=0.995,
    P_IO=True,
    parameter_bounds=None,
    max_trial=100,
    network_type="Regression",
    parameter_counts_only=False,
):
    networks = []

    layer_probabilities = {
        "LL_alpha": alpha,
        "LL_gamma": gamma,
        "LL_max_degree": layers,
        "LL_min_degree": 1,
        "IL": P_IL,
        "LO": P_LO,
        "IO": P_IO,
    }

    if parameter_counts_only:
        assert parameter_bounds is None

    if hidden_channels is None:
        hidden_channels = [3 * out_channels]

    for _ in range(n_networks):
        ok = False
        count = 0
        while not ok:
            count += 1
            this_net = smsnet.random_SMS_network(
                in_channels=in_channels,
                out_channels=out_channels,
                layers=layers,
                dilation_choices=dilation_choices,
                hidden_out_channels=hidden_channels,
                layer_probabilities=layer_probabilities,
                sizing_settings=None,
                dilation_mode="Edges",
                network_type=network_type,
            )
            pcount = helpers.count_parameters(this_net)
            if parameter_bounds is not None:
                if pcount > min(parameter_bounds):
                    if pcount < max(parameter_bounds):
                        ok = True
                        networks.append(this_net)
                if count > max_trial:
                    logging.error("Could not generate network, check bounds")
            else:
                ok = True
                if parameter_counts_only:
                    networks.append(pcount)
                else:
                    networks.append(this_net)
    return networks


def build_smsnet_ensemble(
    in_channels,
    out_channels,
    ensemble_parameters,
):
    dilation_choices = [
        int(x) for x in ensemble_parameters.dilation_choices.strip("[]").split(",")
    ]
    list_of_networks = construct_2dsms_ensembler(
        n_networks=ensemble_parameters.num_networks,
        in_channels=in_channels,
        out_channels=out_channels,
        layers=ensemble_parameters.layers,
        alpha=ensemble_parameters.alpha,
        gamma=ensemble_parameters.gamma,
        hidden_channels=ensemble_parameters.hidden_channels,
        dilation_choices=dilation_choices,
        parameter_bounds=None,
        max_trial=ensemble_parameters.max_trial,
        network_type="Classification",
        parameter_counts_only=False,
    )
    logging.info(f"Number of SMSNet constructed: {len(list_of_networks)}")
    return list_of_networks


def baggin_smsnet_ensemble(mlflow_model_name):
    """
    Create an ensemble model from SMSNet networks loaded from MLflow registry.

    Input:
        mlflow_model_name: str, name of the model in MLflow registry
    Output:
        ensemble: bagged ensemble model
    """
    # Load all versions from MLflow registry
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{mlflow_model_name}'")

    # Load all models
    list_of_smsnet = []
    for mv in sorted(model_versions, key=lambda x: int(x.version)):
        model_uri = f"models:/{mlflow_model_name}/{mv.version}"
        logging.info(f"Loading model version {mv.version} from {model_uri}")
        model = mlflow.pytorch.load_model(model_uri)
        list_of_smsnet.append(model)
    logging.info(f"Loaded {len(list_of_smsnet)} models from MLflow registry")

    ensemble = model_baggin(models=list_of_smsnet, model_type="classification")
    logging.info(f"Ensemble created with {len(list_of_smsnet)} models")
    return ensemble
