import einops
import torch
from qlty import cleanup
from qlty.qlty2D import NCYXQuilt
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

    def __init__(
        self,
        data_tiled_client,
        selected_indices=None,
        qlty_window=64,
        qlty_step=32,
        qlty_border=16,
    ):
        self.data_client = data_tiled_client
        self.selected_indices = selected_indices
        self.qlty_window = qlty_window
        self.qlty_step = qlty_step
        self.qlty_border = qlty_border
        self._set_qlty_object()

    def __len__(self):
        if not (self.selected_indices is None):
            return len(self.selected_indices)
        return len(self.data_client)

    def __getitem__(self, idx):
        if not (self.selected_indices is None):
            # Index mapping to get the data at the selected index
            idx = self.selected_indices[idx]
        data = self.data_client[idx]
        data = self._normalize(torch.from_numpy(data))
        if len(data.shape) == 2:  # Assumes single image without channel dimension
            data = data.unsqueeze(0).unsqueeze(-1)  # Add batch and channel dimensions
        elif len(data.shape) == 3:  # Add channel dimension if missing
            data = data.unsqueeze(0)
        data = einops.rearrange(data, "N Y X C -> N C Y X")
        patched_data = self._crop_images_to_patches(data)
        return patched_data

    @property
    def shape(self):
        if not (self.selected_indices is None):
            # Update shape of the data set based on the selected indices
            # List / tuple conversion is needed for mutability
            data_client_shape = list(self.data_client.shape)
            data_client_shape[0] = len(self.selected_indices)
            return tuple(data_client_shape)
        return self.data_client.shape

    @staticmethod
    def _normalize(image, low_pct=1, high_pct=99):
        """
        This method normalizes the given image (stack) by clipping to the given percentiles
        Input:
            image: torch.Tensor, single image or the image stack array
            low_pct: float, low percentile for clipping
            high_pct: float, high percentile for clipping
        Output:
            normalized_image: torch.Tensor, normalized array
        """
        # Convert to float if not already
        image_float = image.float() if not image.is_floating_point() else image
        low = torch.quantile(image_float.ravel(), low_pct / 100)
        high = torch.quantile(image_float.ravel(), high_pct / 100)
        return torch.clamp((image_float - low) / (high - low), 0, 1)

    def _set_qlty_object(self, qlty_border_weight=0.1):
        """Create and return a QLTY object for given image shape and parameters."""
        self.qlty_object = NCYXQuilt(
            X=self.shape[-1],
            Y=self.shape[-2],
            window=(self.qlty_window, self.qlty_window),
            step=(self.qlty_step, self.qlty_step),
            border=(self.qlty_border, self.qlty_border),
            border_weight=qlty_border_weight,
        )

    def _crop_images_to_patches(
        self,
        data,
        masks=None,
        cleanup_sparse=False,
    ):
        """Crop data and masks into patches using QLTY."""
        if masks is not None:
            patched_images, patched_masks = self.qlty_object.unstitch_data_pair(
                data, masks
            )
        else:
            patched_images = self.qlty_object.unstitch(data)
            return patched_images

        if cleanup_sparse:
            if masks is None:
                raise ValueError(
                    "Masks must be provided for cleaning up sparse classification training pairs."
                )
            # Clean up unlabeled patches
            patched_images, patched_masks, _ = (
                cleanup.weed_sparse_classification_training_pairs_2D(
                    patched_images,
                    patched_masks,
                    missing_label=-1,
                    border_tensor=self.qlty_object.border_tensor(),
                )
            )
        return patched_images, patched_masks


class TiledMaskedDataset(TiledDataset):
    """
        PyTorch dataset class for data and and subset of data retrieved through Tiled.
    Args:
        data_tiled_client (tiled.client): The tiled client for accessing the data.
        mask_tiled_client (tiled.client): The tiled client for accessing segmentation masks.
        selected_indices (List[int]): List of indices that map consecutive mask indices to data indices.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data and mask at the given index.
    """

    def __init__(
        self,
        data_tiled_client,
        mask_tiled_client,
        selected_indices,
        qlty_window=64,
        qlty_step=32,
        qlty_border=16,
    ):
        super().__init__(
            data_tiled_client, selected_indices, qlty_window, qlty_step, qlty_border
        )
        self.mask_client = mask_tiled_client
        self.patched_data, self.patched_mask = self._load_all_data()

    def __len__(self):
        return len(self.mask_client)

    def __getitem__(self, idx):
        data = self.patched_data[idx]
        mask = self.patched_mask[idx]
        return data, mask

    def _load_all_data(self):
        # Load all labeled data and masks into memory
        # TODO: Check if this has limitations in regards to number of possible slices
        # Directly accessing client data is not the intended use of the dataset class
        data = self.data_client[self.selected_indices, :]
        mask = self.mask_client[:]
        data = self._normalize(torch.from_numpy(data))
        if len(data.shape) == 4:  # Assumes last dimension is channels
            data = einops.rearrange(data, "N Y X C -> N C Y X")
        else:
            data = data.unsqueeze(1)  # Add channel dimension
        mask = torch.from_numpy(mask)
        patched_data, patched_mask = self._crop_images_to_patches(
            data,
            masks=mask,
            cleanup_sparse=True,
        )
        return patched_data, patched_mask


def initialize_tiled_datasets(io_parameters, is_training=False):
    """
    This function takes Tiled configurations from the io_parameter class, builds the client and constructs TiledDataset.
    Input:
        io_parameters: IOParameters, all io parameters
        is_training: bool, whether the dataset is used for training or inference
    Output:
        dataset: TiledDataset or TiledMaskedDataset

    """
    try:
        data_tiled_client = from_uri(
            io_parameters.data_tiled_uri, api_key=io_parameters.data_tiled_api_key
        )
    except Exception as e:
        raise Exception(f"Error initializing data tiled client: {e}")

    if io_parameters.mask_tiled_uri:
        try:
            mask_tiled_client = from_uri(
                io_parameters.mask_tiled_uri, api_key=io_parameters.mask_tiled_api_key
            )
            if "mask_idx" not in mask_tiled_client.metadata:
                raise KeyError(
                    "The mask client does not have the required 'mask_idx' metadata."
                )
            selected_indices = mask_tiled_client.metadata["mask_idx"]

            if "mask" not in mask_tiled_client.keys():
                raise KeyError("The mask client does not have the required 'mask' key.")
            mask_tiled_client = mask_tiled_client["mask"]
            if is_training:
                dataset = TiledMaskedDataset(
                    data_tiled_client, mask_tiled_client, selected_indices
                )
            else:
                dataset = TiledDataset(data_tiled_client, selected_indices)
        except KeyError as e:
            raise KeyError(f"Missing information in mask tiled client: {e}")
        except Exception as e:
            raise Exception(f"Error initializing mask tiled client: {e}")

    else:
        dataset = TiledDataset(data_tiled_client=data_tiled_client)
    return dataset
