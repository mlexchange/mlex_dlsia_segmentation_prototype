import numpy as np
import pytest
from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.server.app import build_app
import time

from ..utils import (
    load_yaml, 
    validate_parameters, 
    normalization, 
    array_to_tensor,
    build_qlty_object,
    crop_data_mask_pair,
    construct_dataloaders,
    find_device,
    create_directory
)
from ..train import (
    prepare_data_and_mask,
    build_criterion,
    train_network
)
from ..tiled_dataset import TiledDataset
from network import build_network
import os

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
    recons_container.write_array(np.random.randint(0, 256, size=(5, 6, 6), dtype=np.uint8), key="recon1")
    masks_container = client.create_container("uid0001", metadata={"mask_idx": ["1", "3"]}) # 2 slices
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
def yaml_path():
    yaml_path = f'src/_tests/example_tunet.yaml'
    yield yaml_path


@pytest.fixture
def parameters_dict(yaml_path):
    parameters_dict = load_yaml(yaml_path)
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
def raw_data(tiled_dataset):
    data, _ = prepare_data_and_mask(tiled_dataset)
    yield data


@pytest.fixture
def mask_array(tiled_dataset):
    _, mask = prepare_data_and_mask(tiled_dataset)
    yield mask


@pytest.fixture
def normed_data(raw_data):
   normed_data = normalization(raw_data)
   yield normed_data


@pytest.fixture
def data_tensor(normed_data):
    data_tensor = array_to_tensor(normed_data)
    yield data_tensor


@pytest.fixture
def mask_tensor(mask_array):
    mask_tensor = array_to_tensor(mask_array)
    yield mask_tensor


@pytest.fixture
def qlty_object(tiled_dataset, model_parameters):
    qlty_object = build_qlty_object(
        width = tiled_dataset.data_client.shape[-1],
        height = tiled_dataset.data_client.shape[-2],
        window = model_parameters.qlty_window,
        step = model_parameters.qlty_step,
        border = model_parameters.qlty_border,
        border_weight = 0.2
    )
    yield qlty_object

@pytest.fixture
def patched_data_mask_pair(qlty_object, data_tensor, mask_tensor):
    patched_data, patched_mask = crop_data_mask_pair(qlty_object, data_tensor, mask_tensor)
    yield patched_data, patched_mask  

@pytest.fixture
def training_dataloaders(patched_data_mask_pair, model_parameters):
    patched_data = patched_data_mask_pair[0]
    patched_mask = patched_data_mask_pair[1]
    train_loader, val_loader = construct_dataloaders(patched_data, model_parameters, training = True, masks = patched_mask)
    yield train_loader, val_loader

@pytest.fixture
def networks(network_name, tiled_dataset, model_parameters):
    networks = build_network(
        network=network_name,
        data_shape=tiled_dataset.data_client.shape, # TODO: Double check if this needs to be switched to the patch dim
        num_classes=model_parameters.num_classes,
        parameters=model_parameters,
    )
    yield networks

@pytest.fixture
def device():
    device = find_device()
    yield device

@pytest.fixture
def criterion(model_parameters, device):
    criterion = build_criterion(model_parameters, device)
    yield criterion

@pytest.fixture
def model_directory(io_parameters):
    model_dir = os.path.join(io_parameters.models_dir, io_parameters.uid_save)
    # Create Result Directory if not existed
    create_directory(model_dir)
    return model_dir

@pytest.fixture
def trained_network(
    network_name,
    networks,
    io_parameters,
    model_parameters,
    device,
    model_directory,
    training_dataloaders,
    criterion,
    ):
    # Record the training start time
    start_time = time.time()
    net = train_network(
        network_name = network_name, 
        networks = networks,
        io_parameters = io_parameters, 
        model_parameters = model_parameters, 
        device = device, 
        model_dir = model_directory,
        train_loader = training_dataloaders[0],
        criterion = criterion,
        val_loader = training_dataloaders[1],
        use_dvclive=True,
        use_savedvcexp=False
    )
    yield net, start_time




