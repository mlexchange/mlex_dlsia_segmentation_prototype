import torch
from qlty import cleanup
from qlty.qlty2D import NCYXQuilt


class TiledDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_tiled_client,
        mask_tiled_client=None,
        is_training=None,
        using_qlty=False,
        qlty_window=50,
        qlty_step=30,
        qlty_border=3,
        transform=None,
    ):
        """
        Args:
            data_tiled_uri:      str,    Tiled URI of the input data
            data_tiled_api_key:  str,    Tiled API key for input data access
            mask_tiled_uri:      str,    Tiled URI of mask
            mask_tiled_api_key:  str,    Tiled API key for mask access
            is_training:         bool,   Whether this is a training instance
            qlty_window:         int,    patch size for qlty cropping
            qlty_step:           int,    shifting window for qlty
            qlty_border:         int,    border size for qlty
            transform:           callable, if not given return PIL image

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

        self.transform = transform
        if using_qlty:
            # this object handles unstitching and stitching
            self.qlty_object = NCYXQuilt(
                X=self.data_client.shape[-1],
                Y=self.data_client.shape[-2],
                window=(qlty_window, qlty_window),
                step=(qlty_step, qlty_step),
                border=(qlty_border, qlty_border),
            )
        self.is_training = is_training
        self.using_qlty = using_qlty

    def __len__(self):
        if self.mask_client:
            return len(self.mask_client)
        else:
            return len(self.data_client)

    def __getitem__(self, idx):
        if self.is_training:
            data = self.data_client[self.mask_idx[idx],]
            mask = self.mask_client[idx,]

            if self.using_qlty:
                # Change to 4d array for qlty requirement
                data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                # Change to 3d array for qlty requirement of labels
                mask = torch.from_numpy(mask).unsqueeze(0)
                data_patches, mask_patches = self.qlty_object.unstitch_data_pair(
                    data, mask
                )
                border_tensor = self.qlty_object.border_tensor()
                clean_data_patches, clean_mask_patches, _ = (
                    cleanup.weed_sparse_classification_training_pairs_2D(
                        data_patches,
                        mask_patches,
                        missing_label=-1,
                        border_tensor=border_tensor,
                    )
                )
                return clean_data_patches, clean_mask_patches
            else:
                return data, mask

        else:
            if self.mask_client:
                data = self.data_client[self.mask_idx[idx],]
                if self.using_qlty:
                    # Change to 4d array for qlty requirement
                    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                    data_patches = self.qlty_object.unstitch(data)
                    return data_patches
                else:
                    return data

            else:
                data = self.data_client[idx,]
                if self.using_qlty:
                    # Change to 4d array for qlty requirement
                    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                    data_patches = self.qlty_object.unstitch(data)
                    return data_patches
                else:
                    return data
