import numpy as np
import pytest
import yaml
from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.server.app import build_app
from tiled.structures.array import ArrayStructure
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import DataSource

from mlex_dlsia.dataset import TiledDataset, TiledMaskedDataset
from mlex_dlsia.utils.params_validation import validate_parameters


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
        np.random.randint(0, 256, size=(5, 12, 12), dtype=np.uint8), key="recon1"
    )
    masks_container = client.create_container("uid0001", metadata={"mask_idx": [1, 3]})
    # 2 slices
    masks_container.write_array(np.ones((2, 12, 12), dtype=np.int8), key="mask")
    yield client


@pytest.fixture
def tiled_dataset(client):
    tiled_dataset = TiledDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        qlty_window=8,
        qlty_step=2,
        qlty_border=1,
    )
    yield tiled_dataset


@pytest.fixture
def tiled_masked_dataset(client):
    mask_indices = client["uid0001"].metadata["mask_idx"]
    tiled_masked_dataset = TiledMaskedDataset(
        data_tiled_client=client["reconstructions"]["recon1"],
        mask_tiled_client=client["uid0001"]["mask"],
        selected_indices=mask_indices,
        qlty_window=8,
        qlty_step=2,
        qlty_border=1,
    )
    yield tiled_masked_dataset


@pytest.fixture(
    params=[
        "_tests/example_msdnet.yaml",
        "_tests/example_tunet.yaml",
        "_tests/example_tunet3plus.yaml",
        "_tests/example_smsnet_ensemble.yaml",
        pytest.param(
            "_tests/example_bad_params.yaml",
            marks=pytest.mark.bad_params,
            id="bad_params",
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


@pytest.fixture()
def validated_params(parameters_dict, include_raw=False):
    if "bad_params" in parameters_dict.get("config_file_name", ""):
        pytest.skip("Skipping bad_params configuration for this test")
    else:
        io_params, model_params = validate_parameters(parameters_dict)
        yield io_params, model_params


@pytest.fixture()
def validated_params_with_raw(parameters_dict):
    if "bad_params" in parameters_dict.get("config_file_name", ""):
        pytest.skip("Skipping bad_params configuration for this test")
    else:
        io_params, model_params = validate_parameters(parameters_dict)
        yield io_params, model_params, parameters_dict.get("model_parameters")


@pytest.fixture()
def prepare_tiled_containers(client):
    num_frames = 5
    height = 12
    width = 12

    array_shape = (num_frames, height, width)
    structure = ArrayStructure.from_array(np.zeros(array_shape, dtype=np.int8))
    structure.chunks = ((1,) * array_shape[0], (array_shape[1],), (array_shape[2],))

    # Create the array in the container
    array_client = client.new(
        key="segmentation_results",  # Or whatever key name you want
        structure_family=StructureFamily.array,
        data_sources=[
            DataSource(
                structure=structure,
                structure_family=StructureFamily.array,
            )
        ],
        metadata={"test": "inference"},
    )

    return array_client  # Return the array client, not the container
