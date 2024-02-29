import torch
from tiled.client import from_uri

class TiledDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_tiled_uri,
            mask_tiled_uri=None,
            mask_idx=None,
            data_tiled_api_key=None,
            mask_tiled_api_key=None,
            shift=0,
            transform=None):
        '''
        Args:
            data_tiled_uri:      str,    Tiled URI of the input data
            mask_tiled_uri:      str,    Tiled URI of mask
            mask_idx:            list,   List for slice indexs that maps the mask to corresponding input images  
            data_tiled_api_key:  str,    Tiled API key for input data access
            mask_tiled_api_key:  str,    Tiled API key for mask access
            shift:               int,    Optional shift for pixel values in masks to bring down the class id to start from 0, (and -1 for unlabeled pixles)
            transform:           callable, if not given return PIL image

        Return:
            ml_data:        tuple, (data_tensor, mask_tensor)
        '''
        self.data_client = from_uri(data_tiled_uri, api_key=data_tiled_api_key)
        if mask_tiled_uri:
            self.mask_client = from_uri(mask_tiled_uri, api_key=mask_tiled_api_key)
        else:
            self.mask_client = None
        self.mask_idx = list(mask_idx) if mask_idx else []
        self.shift = int(shift)
        self.transform = transform

    def __len__(self):
        return len(self.mask_idx)

    def __getitem__(self, idx):
        data = self.data_client[self.mask_idx[idx],]
        if self.mask_client:
            mask = self.mask_client[idx,].astype('int') - self.shift # Conversion to int is needed as switching unlabeled pixels to -1 would cause trouble in uint8 format
            if self.transform:
                return self.transform(data), mask
            else:
                return data, mask
        else:
            if self.transform:
                return self.transform(data)
            else:
                return data