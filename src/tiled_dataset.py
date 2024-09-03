import torch
from tiled.client import from_uri


class TiledDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_tiled_client,
        mask_tiled_client=None,
        is_training=False,
        is_full_inference=False,
    ):
        """
        Args:
            data_tiled_uri:      str,    Tiled URI of the input data
            data_tiled_api_key:  str,    Tiled API key for input data access
            mask_tiled_uri:      str,    Tiled URI of mask
            mask_tiled_api_key:  str,    Tiled API key for mask access
            is_training:         bool,   Whether this is a training instance
            is_full_inference:   bool,   Whether to perform full inference

        Return:
            ml_data:        tuple, (data_tensor, mask_tensor)
        """

        self.data_client = data_tiled_client
        self.mask_client = None
        if mask_tiled_client:
            self.mask_client = mask_tiled_client["mask"]
            self.mask_idx = [int(idx) for idx in mask_tiled_client.metadata["mask_idx"]]
        else:
            self.mask_client = None
            self.mask_idx = None

        self.is_training = is_training
        self.is_full_inference = is_full_inference

    def __len__(self):
        if self.is_full_inference:
            return len(self.data_client)
        else:
            return len(self.mask_client)

    def __getitem__(self, idx):
        if self.is_training:
            data = self.data_client[self.mask_idx[idx],]
            mask = self.mask_client[idx,]
            return data, mask
        else:
            if not self.is_full_inference:
                data = self.data_client[self.mask_idx[idx],]
                return data

            else:
                data = self.data_client[idx,]
                return data


def initialize_tiled_datasets(
    io_parameters, is_training=False, is_full_inference=False
):
    """
    This function takes tiled uris from the io_parameter class, build the client and construct TiledDataset.
    Input:
        io_parameters: class, all io parameters in pydantic class
        is_training: bool, whether the dataset is used for training or inference
        is_full_inference: bool, whether the process is used for full inference
    Output:
        dataset: class, TiledDataset

    """
    data_tiled_client = from_uri(
        io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
    )
    mask_tiled_client = None
    if io_parameters.mask_tiled_uri:
        mask_tiled_client = from_uri(
            io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
        )
    dataset = TiledDataset(
        data_tiled_client=data_tiled_client,
        mask_tiled_client=mask_tiled_client,
        is_training=is_training,
        is_full_inference=is_full_inference,
    )
    return dataset
