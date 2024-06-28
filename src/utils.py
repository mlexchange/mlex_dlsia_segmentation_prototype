import os
from urllib.parse import urlparse, urlunparse

import numpy as np
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure

import yaml
from parameters import (
    IOParameters,
    MSDNetParameters,
    SMSNetEnsembleParameters,
    TUNet3PlusParameters,
    TUNetParameters,
)

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
    if network == "MSDNet":
        model_parameters = MSDNetParameters(**raw_model_parameters)
    elif network == "TUNet":
        model_parameters = TUNetParameters(**raw_model_parameters)
    elif network == "TUNet3+":
        model_parameters = TUNet3PlusParameters(**raw_model_parameters)
    elif network == "SMSNetEnsemble":
        model_parameters = SMSNetEnsembleParameters(**raw_model_parameters)

    assert model_parameters, f"Received Unsupported Network: {network}"

    print("Parameters loaded successfully.")
    return io_parameters, network, model_parameters



# Create directory
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Local directory '{path}' created.")
    else:
        print(f"Local directory '{path}' already exsists.")


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
    return array_client
