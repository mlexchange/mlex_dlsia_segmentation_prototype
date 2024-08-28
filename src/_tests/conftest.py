import shutil
import tempfile
from pathlib import Path

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
        is_full_inference=False,
    )
    yield tiled_dataset


@pytest.fixture(
    params=[
        "src/_tests/example_msdnet.yaml",
        "src/_tests/example_tunet.yaml",
        "src/_tests/example_tunet3plus.yaml",
        "src/_tests/example_smsnet_ensemble.yaml",
        pytest.param(
            "src/_tests/example_bad_params.yaml", marks=pytest.mark.bad_params
        ),
    ]
)
def parameters_dict(request):
    yaml_path = request.param
    with open(yaml_path, "r") as file:
        parameters_dict = yaml.safe_load(file)
    parameters_dict["config_file_name"] = (
        yaml_path  # Adding the file name to the dictionary for testing purpose
    )
    yield parameters_dict


@pytest.fixture
def io_parameters(parameters_dict):
    try:
        # Validate parameters during fixture setup
        io_parameters, _, _ = validate_parameters(parameters_dict)
        # Pass the valid parameters to the tests
        yield io_parameters
    except AssertionError as e:
        # Handle the exception and set a flag or do something else
        yield e


@pytest.fixture
def network_name(parameters_dict):
    try:
        # Validate parameters during fixture setup
        _, network_name, _ = validate_parameters(parameters_dict)
        # Pass the valid parameters to the tests
        yield network_name
    except AssertionError as e:
        # Handle the exception and set a flag or do something else
        yield e


@pytest.fixture
def model_parameters(parameters_dict):
    try:
        # Validate parameters during fixture setup
        _, _, model_parameters = validate_parameters(parameters_dict)
        # Pass the valid parameters to the tests
        yield model_parameters
    except AssertionError as e:
        # Handle the exception and set a flag or do something else
        yield e


@pytest.fixture(scope="session")
def session_temp_directory():
    # Create a temporary directory for the session
    temp_dir = tempfile.mkdtemp()

    # Yield the directory path to the tests
    yield temp_dir

    # Cleanup: Remove the directory after all tests are done
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_directory(io_parameters, session_temp_directory):
    if isinstance(io_parameters, AssertionError):
        pytest.skip("Skipping test due to unsupported network in parameters")

    # Convert session_temp_directory to Path object
    temp_path = Path(session_temp_directory)

    # Construct the model directory path
    model_dir = temp_path / io_parameters.models_dir / io_parameters.uid_save

    # Create Result Directory if not existed
    create_directory(model_dir)

    yield str(model_dir)  # Convert to string for compatibility with external code


@pytest.fixture
def qlty_object(tiled_dataset, model_parameters):
    if isinstance(model_parameters, AssertionError):
        pytest.skip("Skipping test due to unsupported network in parameters")

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
