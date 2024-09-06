import torch
from tiled.client import from_uri


class TiledDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset class for data (or index-based subset of data) retrieved through Tiled.

    Parameters:
        data_tiled_client (tiled.client): The client object used to access the data through Tiled.
        selected_indices (List[int]): List of indices to iterate over a subset of the data.
    Attributes:
        shape (Tuple[int]): The shape of the data set.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data at the given index as a numpy array.
    """

    def __init__(self, data_tiled_client, selected_indices=None):
        self.data_client = data_tiled_client
        self.selected_indices = selected_indices

    def __len__(self):
        if not (self.selected_indices is None):
            return len(self.selected_indices)
        return len(self.data_client)

    def __getitem__(self, idx):
        if not (self.selected_indices is None):
            idx = self.selected_indices[idx]
        data = self.data_client[idx,]
        return data

    @property
    def shape(self):
        if not (self.selected_indices is None):
            # Update shape of the data set based on the selected indices
            # List / tuple conversion is needed for mutability
            data_client_shape = list(self.data_client.shape)
            data_client_shape[0] = len(self.selected_indices)
            return tuple(data_client_shape)
        return self.data_client.shape


class TiledMaskedDataset(TiledDataset):
    """
        PyTorch dataset class for data and and subset of data retrieved through Tiled.
    Args:
        data_tiled_client (tiled.client): The tiled client for accessing the data.
        mask_tiled_client (tiled.client): The tiled client for accessing segmentation masks.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data and mask at the given index.
    """

    def __init__(self, data_tiled_client, mask_tiled_client):
        if "mask_idx" not in mask_tiled_client.metadata:
            raise KeyError(
                "The mask client does not have the required 'mask_idx' metadata."
            )

        selected_indices = mask_tiled_client.metadata["mask_idx"]
        super().__init__(data_tiled_client, selected_indices)

        if "mask" not in mask_tiled_client.keys():
            raise KeyError("The mask client does not have the required 'mask' key.")
        self.mask_client = mask_tiled_client["mask"]

    def __len__(self):
        return len(self.mask_client)

    def __getitem__(self, idx):
        data = self.data_client[self.selected_indices[idx],]
        mask = self.mask_client[idx,]
        return data, mask


def initialize_tiled_datasets(io_parameters, is_training=False):
    """
    This function takes Tiled configurations from the io_parameter class, builds the client and constructs TiledDataset.
    Input:
        io_parameters: class, all io parameters in Pydantic class
        is_training: bool, whether the dataset is used for training or inference
    Output:
        dataset: TiledDataset or TiledMaskedDataset

    """
    try:
        data_tiled_client = from_uri(
            io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
        )
    except Exception as e:
        raise ValueError(f"Error initializing data tiled client: {e}")

    if is_training and io_parameters.mask_tiled_uri:
        try:
            mask_tiled_client = from_uri(
                io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
            )
            dataset = TiledMaskedDataset(data_tiled_client, mask_tiled_client)
        except Exception as e:
            raise ValueError(f"Error initializing mask tiled client: {e}")
    else:
        dataset = TiledDataset(data_tiled_client=data_tiled_client)
    return dataset
