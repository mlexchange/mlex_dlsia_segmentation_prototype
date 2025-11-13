import pytest
import torch
from dlsia.core.helpers import get_device

from mlex_dlsia.inference import run_inference
from mlex_dlsia.network import baggin_smsnet_ensemble, build_network


def test_run_inference(
    tiled_dataset, validated_params_with_raw, prepare_tiled_containers
):
    io_parameters, model_parameters, raw_parameters = validated_params_with_raw
    try:
        net = build_network(
            model_parameters.network,
            in_channels=1,
            image_shape=(model_parameters.qlty_window, model_parameters.qlty_window),
            num_classes=model_parameters.num_classes,
            parameters=raw_parameters,
        )
        assert all(isinstance(n, torch.nn.Module) for n in net)

        if model_parameters.network == "DLSIA SMSNetEnsemble":
            net = baggin_smsnet_ensemble(networks=net)
        else:
            net = net[0]

        device = get_device()
        seg_client = prepare_tiled_containers

        run_inference(
            tiled_dataset,
            net,
            seg_client,
            model_parameters,
            device,
        )
    except Exception as e:
        pytest.fail(f"run_inference raised an exception: {e}")
