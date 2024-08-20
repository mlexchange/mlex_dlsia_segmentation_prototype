import os

import numpy as np
import pytest
import yaml
from qlty.qlty2D import NCYXQuilt
from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.server.app import build_app

from ..tiled_dataset import TiledDataset
from ..utils import create_directory, find_device, validate_parameters


@pytest.fixture
def catalog(tmpdir):
    adapter = from_uri(
        f"sqlite+aiosqlite:///{tmpdir}/catalog.db",
        writable_storage=str(tmpdir),
        init_if_not_exists=True,
    )
    yield adapter


@pytest.fixture
def app(catalog):
    app = build_app(catalog)
    yield app


@pytest.fixture
def context(app):
    with Context.from_app(app) as context:
        yield context


@pytest.fixture
def client(context):
    "Fixture for tests which only read data"
    client = from_context(context)
    recons_container = client.create_container("reconstructions")
    recons_container.write_array(
        np.random.randint(0, 256, size=(5, 6, 6), dtype=np.uint8), key="recon1"
    )
    masks_container = client.create_container(
        "uid0001", metadata={"mask_idx": ["1", "3"]}
    )  # 2 slices
    masks_container.write_array(np.ones((2, 6, 6), dtype=np.int8), key="mask")
    yield client


@pytest.fixture
def tiled_dataset(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"],
        is_training=True,
    )
    yield tiled_dataset


@pytest.fixture
def parameters_dict():
    yaml_path = "src/_tests/example_tunet.yaml"
    with open(yaml_path, "r") as file:
        # Load parameters
        parameters_dict = yaml.safe_load(file)
    yield parameters_dict


@pytest.fixture
def io_parameters(parameters_dict):
    io_parameters, _, _ = validate_parameters(parameters_dict)
    yield io_parameters


@pytest.fixture
def network_name(parameters_dict):
    _, network_name, _ = validate_parameters(parameters_dict)
    yield network_name


@pytest.fixture
def model_parameters(parameters_dict):
    _, _, model_parameters = validate_parameters(parameters_dict)
    yield model_parameters


@pytest.fixture
def model_directory(io_parameters):
    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_save)
    # Create Result Directory if not existed
    create_directory(model_dir)
    return model_dir


@pytest.fixture
def qlty_object(tiled_dataset, model_parameters):
    qlty_object = NCYXQuilt(
        X=tiled_dataset.data_client.shape[-1],
        Y=tiled_dataset.data_client.shape[-2],
        window=(model_parameters.qlty_window, model_parameters.qlty_window),
        step=(model_parameters.qlty_step, model_parameters.qlty_step),
        border=(model_parameters.qlty_border, model_parameters.qlty_border),
        border_weight=0.2,
    )
    yield qlty_object


@pytest.fixture
def device():
    device = find_device()
    yield device
