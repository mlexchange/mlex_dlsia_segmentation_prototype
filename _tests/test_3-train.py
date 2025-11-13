import pytest
import torch
from dlsia.core.helpers import get_device

from mlex_dlsia.network import build_network
from mlex_dlsia.train import run_train


def _get_mock_data(parameters):
    num_samples = 20
    channels = 1
    height = parameters.qlty_window
    width = parameters.qlty_window

    data = torch.randn(num_samples, channels, height, width)
    targets = torch.randint(
        0, parameters.num_classes, (num_samples, channels, height, width)
    )
    dataset = torch.utils.data.TensorDataset(data, targets)
    train_data, val_data = torch.utils.data.random_split(dataset, [16, 4])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False)
    return train_loader, val_loader


def test_train_network(validated_params_with_raw):
    io_parameters, model_parameters, raw_parameters = validated_params_with_raw
    print(f"validated model parameters: {model_parameters}")
    train_loader, val_loader = _get_mock_data(model_parameters)
    item, mask = next(iter(train_loader))
    try:
        net = build_network(
            model_parameters.network,
            in_channels=1,
            image_shape=(model_parameters.qlty_window, model_parameters.qlty_window),
            num_classes=model_parameters.num_classes,
            parameters=raw_parameters,
        )
        assert all(isinstance(n, torch.nn.Module) for n in net)

        device = get_device()
        run_train(
            train_loader,
            val_loader,
            io_parameters,
            net,
            model_parameters,
            device,
            use_dvclive=False,
        )
    except Exception as e:
        pytest.fail(f"run_train raised an exception: {e}")
