import torch
from tiled.client import from_uri


class TiledDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_uri,
            mask_uri,
            data_api_key=None,
            mask_api_key=None,
            transform=None):
        '''
        Args:
            data_uri:       str, Tiled URI of data
            mask_uri:       str, Tiled URI of mask
            data_api_key:   str, Tiled API key for data access
            mask_api_key:   str, Tiled API key for mask access
            transform:      callable, if not given return PIL image

        Return:
            ml_data:        tuple, (data_tensor, mask_tensor)
        '''
        self.data_client = from_uri(data_uri, api_key=data_api_key)
        self.mask_client = from_uri(mask_uri, api_key=mask_api_key)
        self.list_indx = [10, 201, 222, 493]
        self.transform = transform

    def __len__(self):
        return len(self.mask_client)

    def __getitem__(self, idx):
        data = self.data_client[self.list_indx[idx],]
        mask = self.mask_client[idx,]-2
        if self.transform:
            return self.transform(data), mask
        else:
            return data, mask