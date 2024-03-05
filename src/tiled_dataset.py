from    qlty            import  cleanup
from    qlty.qlty2D     import  NCYXQuilt
from    tiled.client    import  from_uri
import  torch

class TiledDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_tiled_uri,
            data_tiled_api_key=None,
            mask_tiled_uri=None,
            mask_tiled_api_key=None,
            workflow_type=None,
            qlty_window=50,
            qlty_step=30,
            qlty_border=3,
            transform=None):
        '''
        Args:
            data_tiled_uri:      str,    Tiled URI of the input data
            data_tiled_api_key:  str,    Tiled API key for input data access
            mask_tiled_uri:      str,    Tiled URI of mask
            mask_tiled_api_key:  str,    Tiled API key for mask access
            workflow_type:       str,    Training, Inference-Preview or Inference-Full
            qlty_window:         int,    patch size for qlty cropping
            qlty_step:           int,    shifting window for qlty
            qlty_border:         int,    border size for qlty
            transform:           callable, if not given return PIL image

        Return:
            ml_data:        tuple, (data_tensor, mask_tensor)
        '''
        self.data_client = from_uri(data_tiled_uri, api_key=data_tiled_api_key)
        if mask_tiled_uri:
            self.mask_client_one_up = from_uri(mask_tiled_uri, api_key=mask_tiled_api_key)
            self.mask_client = self.mask_client_one_up['mask']
            self.mask_idx = [int(idx) for idx in self.mask_client_one_up.metadata['mask_idx']]
        else:
            self.mask_client = None
            self.mask_idx = None
        self.transform = transform
        # this object handles unstitching and stitching
        self.qlty_object = NCYXQuilt(X=self.data_client.shape[-1], 
                                     Y=self.data_client.shape[-2],
                                     window = (qlty_window, qlty_window),
                                     step = (qlty_step, qlty_step),
                                     border = (qlty_border, qlty_border)
                                     )
        self.workflow_type = workflow_type

    def __len__(self):
        if self.workflow_type == 'Inference-Full':
            return len(self.data_client)
        else:
            return len(self.mask_client)

    def __getitem__(self, idx):
        if self.workflow_type == 'Training':    
            data = self.data_client[self.mask_idx[idx],]
            # Change to 4d array for qlty requirement
            data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            mask = self.mask_client[idx,]
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
            return clean_data_patches, clean_mask_patches
        
        elif self.workflow_type == 'Inference-Preview':
            data = self.data_client[self.mask_idx[idx],]
            # Change to 4d array for qlty requirement
            data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            data_patches = self.qlty_object.unstitch(data)
            return data_patches

        else:
            data = self.data_client[idx,]
            # Change to 4d array for qlty requirement
            data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            data_patches = self.qlty_object.unstitch(data)
            return data_patches