from ..tiled_dataset import TiledDataset


def test_tiled_dataset(client):
    tiled_dataset = TiledDataset(
        client["reconstructions"]["recon1"],
    )
    assert tiled_dataset
    assert tiled_dataset[0].shape == (3, 3)


def test_tiled_dataset_with_masks(client):
    tiled_dataset = TiledDataset(
        client["reconstructions"]["recon1"], mask_tiled_client=client["mask"]
    )
    assert tiled_dataset[0].shape == (3, 3)
