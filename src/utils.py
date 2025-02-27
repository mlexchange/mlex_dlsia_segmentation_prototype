import os
from urllib.parse import urlparse, urlunparse

import numpy as np
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure


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
    job_name,
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
        "job_name": job_name,
        "model": model,
    }

    array_client = last_container.new(
        structure_family="array",
        structure=structure,
        key=array_name,
        metadata=metadata,
    )
    return array_client
