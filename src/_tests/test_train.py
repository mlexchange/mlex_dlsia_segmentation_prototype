import numpy as np
import pytest
from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.server.app import build_app

from ..tiled_dataset import TiledDataset


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
    recons_container.write_array(np.zeros((2, 3, 3), dtype=np.int8), key="recon1")
    masks_container = client.create_container("masks", metadata={"mask_idx": ["0"]})
    masks_container.write_array(np.zeros((1, 3, 3), dtype=np.int8), key="mask1")
    yield client


@pytest.mark.asyncio
async def test_tiled_dataset(client):
    tiled_dataset = TiledDataset(
        client["reconstructions"]["recon1"],
    )
    assert tiled_dataset
    assert tiled_dataset[0].shape == (3, 3)


@pytest.mark.asyncio
async def test_tiled_dataset_with_masks(client):
    tiled_dataset = TiledDataset(
        client["reconstructions"]["recon1"], mask_tiled_client=client["masks"]
    )
    assert tiled_dataset[0].shape == (3, 3)
