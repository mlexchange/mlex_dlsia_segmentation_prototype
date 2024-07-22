import os
from urllib.parse import urlparse, urlunparse
from qlty.qlty2D import NCYXQuilt
from qlty import cleanup
import numpy as np
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure
from network import baggin_smsnet_ensemble, load_network
import yaml
import glob
from parameters import (
    IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)
from tiled_dataset import TiledDataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_yaml(yaml_path):
    '''
    This function loads parameters from the yaml file.
    Input:
        yaml_path: str, path of the yaml file.
    Output:
        parameters: dict, dictionary of all parameters.
    '''
    # Open the YAML file for all parameters
    with open(yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)
        return parameters

def validate_parameters(parameters):
    '''
    This function extracts parameters from the whole parameter dict 
    and performs pydantic validation for both io-related and model-related parameters. 
    Input:
        parameters: dict, parameters from the yaml file as a whole.
    Output:
        io_parameters: class, all io parameters in pydantic class
        network: str, name of the selected algorithm
        model_parameters: class, all model specific parameters in pydantic class
    '''
    # Validate and load I/O related parameters
    io_parameters = parameters["io_parameters"]
    io_parameters = IOParameters(**io_parameters)
    # Check whether mask_uri has been provided as this is a requirement for training.
    assert io_parameters.mask_tiled_uri, "Mask URI not provided for training."

    # Detect which model we have, then load corresponding parameters
    raw_model_parameters = parameters["model_parameters"]
    network = raw_model_parameters["network"]

    model_parameters = None
    if network == "DLSIA MSDNet":
        model_parameters = MSDNetParameters(**raw_model_parameters)
    elif network == "DLSIA TUNet":
        model_parameters = TUNetParameters(**raw_model_parameters)
    elif network == "DLSIA TUNet3+":
        model_parameters = TUNet3PlusParameters(**raw_model_parameters)
    elif network == "DLSIA SMSNetEnsemble":
        model_parameters = SMSNetEnsembleParameters(**raw_model_parameters)

    assert model_parameters, f"Received Unsupported Network: {network}"

    print("Parameters loaded successfully.")
    return io_parameters, network, model_parameters

def initialize_tiled_datasets(io_parameters, is_training=True):
    '''
    This function takes tiled uris from the io_parameter class, build the client and construct TiledDataset.
    Input:
        io_parameters: class, all io parameters in pydantic class
        is_training: bool, whether the dataset is used for training or inference
    Output:
        dataset: class, TiledDataset

    '''
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
        using_qlty=False,
        transform=transforms.ToTensor(),
    )
    return dataset
 
def normalization(image):
    '''
    This function normalizes the given image (stack) by clipping to 1% and 99% percentiles
    Input:
        image: np.ndarray, single image or the image stack array
    Output:
        normed_image: np.ndarray, normalized array
    
    '''
    # Normalize by clipping to 1% and 99% percentiles
    low = np.percentile(image.ravel(), 1)
    high = np.percentile(image.ravel(), 99)
    normed_image = np.clip((image - low) / (high - low), 0, 1)
    return normed_image

def array_to_tensor(array):
    '''
    This function converts numpy array to tensor
    Input:
        array: np.ndarray
    Output:
        tensor: pytorch tensor
    '''
    tensor = torch.from_numpy(array)
    return tensor

def build_qlty_object(width, height, window, step, border, border_weight=0.2):
    '''
    This function builds qlty object based on provided parameters.
    Input:
        width: int, width of the image to be cropped (2d_array.shape[-1])
        height: int, height of the image to be cropped (2d_array.shape[-2])
        window: int, cropping window size
        step: int, moving step between tiles, affecting overlap
        border: int, 
        border_weight: float, default set to 0.2    
    Output:
        qlty_object: class, qlty object needed for cropping and stitching
    '''
    qlty_object = NCYXQuilt(
        X = width,
        Y = height,
        window=(window, window),
        step=(step, step),
        border=(border, border),
        border_weight=border_weight,
    )
    return qlty_object

def crop_data_mask_pair(qlty_object, images, masks):
    '''
    This function crops the image and data pair into small tiles defined by the qlty_object, followed by cleaning of unlabeled patches
    Input:
        qlty_object: class, pre-built qlty object 
        images: torch.Tensor, normalized image stack in tensor format
        masks: torch.Tensor, masks in tensor
    Output:
        patched_images: patch stack of cropped image tiles in tensor form
        patched_masks: corresponding stack of cropped mask tiles in tensor form
    '''
    if images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim == 2:
        images = images.unsqueeze(0).unsqueeze(0)

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    
    # Crop
    patched_images, patched_masks = qlty_object.unstitch_data_pair(images, masks)
    # Clean up unlabeled patches
    patched_images, patched_masks, _ = (
        cleanup.weed_sparse_classification_training_pairs_2D(
            patched_images,
            patched_masks,
            missing_label=-1,
            border_tensor=qlty_object.border_tensor(),
        )
    )
    return patched_images, patched_masks

def construct_dataloaders(images, parameters, training = False, masks = None):
    '''
    This function takes the given image stack and construct them into a pytorch dataloader. 
    Handling both training scenario (where masks are provided as ground truth) 
    and inference (where only images are needed).
    When setting training = True, this function will also random split the dataset into training and validation
    set based on the validation percentage given in the parameters.
    Input:
        images: pytorch tensor, processed image stack in tensor form
        parameters: class, pydantic validated parameters 
        training: bool, default set to False for inference, when set to True this is referred as training case
        masks: pytorch tensor, corresponding mask stack in tensor form for ground truth
    Output:
        train_loader: pytorch dataloader for model training
        val_loader: pytorch dataloader for model validation
        inference_loader: pytorch dataloader for inference
    '''
    if training:
        assert masks is not None, "Error: missing mask information when constructing training dataloaders."
        dataset = TensorDataset(images, masks)
        # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
        train_loader_params = {
            "batch_size": parameters.batch_size_train,
            "shuffle": parameters.shuffle_train,
            }
        val_loader_params = {
            "batch_size": parameters.batch_size_val, 
            "shuffle": False,
            }
        val_pct = parameters.val_pct
        val_size = max(int(val_pct * len(dataset)), 1) if len(dataset) > 1 else 0
        if val_size == 0:
            train_loader = DataLoader(dataset, **train_loader_params)
            val_loader = None
        else:
            train_size = len(dataset) - val_size    
            train_data, val_data = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_data, **train_loader_params)
            val_loader = DataLoader(val_data, **val_loader_params)
        return train_loader, val_loader
    else:
        dataset = TensorDataset(images)
        inference_loader_params = {
            "batch_size": parameters.batch_size_inference,
            "shuffle": False,
            }
        inference_loader = DataLoader(dataset, **inference_loader_params)
        return inference_loader

def find_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# Create directory
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Local directory '{path}' created.")
    else:
        print(f"Local directory '{path}' already exsists.")

def load_dlsia_network(network_name, model_dir):
    '''
    This function loads pre-trained DLSIA network. Support both single network and ensembles.
    Input:
        network: str, name of the DLSIA network to be loaded.
        model_dir: str, path of the saved network.
    Output:
        net: loaded pre-trained network 
    '''
    if network_name == "DLSIA SMSNetEnsemble":
        net = baggin_smsnet_ensemble(model_dir)
    else:
        net_files = glob.glob(os.path.join(model_dir, "*.pt"))
        net = load_network(network_name, net_files[0])
    return net

def ensure_parent_containers(tiled_uri, tiled_api_key):
    parsed_url = urlparse(tiled_uri)
    path = parsed_url.path
    # Splitting path into parts
    path_parts = path.split("/")[1:]  # Split and remove the first empty element
    root_path = "/".join(path_parts[:3])
    tiled_root = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            root_path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment,
        )
    )

    last_container = from_uri(tiled_root, api_key=tiled_api_key)

    container_parts = path_parts[
        3:
    ]  # Ignoring the api/v1/metadata as being covered above during authentication
    for part in container_parts:
        if part in last_container.keys():
            last_container = last_container[part]
        else:
            last_container = last_container.create_container(key=part)
    return last_container


# Tiled Saving
def allocate_array_space(
    tiled_dataset,
    seg_tiled_uri,
    seg_tiled_api_key,
    uid,
    model,
    array_name,
):

    last_container = ensure_parent_containers(seg_tiled_uri, seg_tiled_api_key)

    print(f"@@@@@@@@@@    last_container   {last_container.uri}")
    assert (
        uid not in last_container.keys()
    ), f"uid_save: {uid} already existed in Tiled Server"

    last_container = last_container.create_container(key=uid)

    array_shape = (
        tiled_dataset.mask_client.shape
        if tiled_dataset.mask_client
        else tiled_dataset.data_client.shape
    )
    # In case we have 2d array for the single mask case, where the ArrayClient will return a 2d array.
    if len(array_shape) == 2:
        array_shape = (1,) + array_shape
    structure = ArrayStructure.from_array(np.zeros(array_shape, dtype=np.int8))

    # For now, only save image 1 by 1 regardless of the batch_size_inference.
    structure.chunks = ((1,) * array_shape[0], (array_shape[1],), (array_shape[2],))

    mask_uri = None
    if tiled_dataset.mask_client is not None:
        mask_uri = tiled_dataset.mask_client.uri

    metadata = {
        "data_uri": tiled_dataset.data_client.uri,
        "mask_uri": mask_uri,
        "mask_idx": tiled_dataset.mask_idx,
        "uid": uid,
        "model": model,
    }

    array_client = last_container.new(
        structure_family="array",
        structure=structure,
        key=array_name,
        metadata=metadata,
    )

    print(
        f"Result space allocated in Tiled and segmentation will be saved in {array_client.uri}."
    )

    return array_client
