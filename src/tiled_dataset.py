from    qlty            import  cleanup
from    qlty.qlty2D     import  NCYXQuilt
from    tiled.client    import  from_uri
import  torch

class TiledDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_tiled_uri,
            mask_tiled_uri=None,
            mask_idx=None,
            data_tiled_api_key=None,
            mask_tiled_api_key=None,
            shift=0,
            qlty_window=50,
            qlty_step=30,
            qlty_border=3,
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
        # this object handles unstitching and stitching
        self.qlty_object = NCYXQuilt(X=self.data_client.shape[-1], 
                                     Y=self.data_client.shape[-2],
                                     window = (qlty_window, qlty_window),
                                     step = (qlty_step, qlty_step),
                                     border = (qlty_border, qlty_border)
                                     )

    def __len__(self):
        return len(self.mask_idx)

    def __getitem__(self, idx):
        data = self.data_client[self.mask_idx[idx],]
        print('+++++++++++++++++++++')
        print(f'slice shape: {data.shape}')
        # Change to 4d array for qlty requirement
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)

        if self.mask_client:
            mask = self.mask_client[idx,].astype('int') - self.shift # Conversion to int is needed as switching unlabeled pixels to -1 would cause trouble in uint8 format
            # Change to 3d array for qlty requirement of labels 
            mask = torch.from_numpy(mask).unsqueeze(0)
            data_patches, mask_patches = self.qlty_object.unstitch_data_pair(data, mask)
            border_tensor = self.qlty_object.border_tensor()
            clean_data_patches, clean_mask_patches, _ = cleanup.weed_sparse_classification_training_pairs_2D(
                data_patches, 
                mask_patches, 
                missing_label=-1, 
                border_tensor=border_tensor
                )
            print('=======================')
            print(f'clean_data_patches shape: {clean_data_patches.shape}')
            print(f'clean_mask_patches shape: {clean_mask_patches.shape}')

            return clean_data_patches, clean_mask_patches

        else:
            data_patches = self.qlty_object.unstitch(data)
            print('=======================')
            print(f'data_patches shape: {data_patches.shape}')
            return data_patches