import logging
from urllib.parse import urlparse, urlunparse

import numpy as np
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import DataSource

from mlex_dlsia.dataset import TiledMaskedDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def ensure_parent_containers(tiled_uri, tiled_api_key=None):
    """
    This function ensures that all parent containers exist for a given Tiled uri.
    If any parent container in the URI path does not exist, it is created.

    Input:
        tiled_uri: str, The URI of the Tiled resource.
        tiled_api_key: str (optional), The API key for authentication. Default is None.

    Output:
        last_container: The final container in the uri path after ensuring all parent containers exist.
    """
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

    if tiled_api_key is not None:
        last_container = from_uri(tiled_root, api_key=tiled_api_key)
    else:
        last_container = from_uri(tiled_root)

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
    last_container,
    uid,
    model_name,
    array_name,
):
    """
    This function allocates an array space based on the shape of the tiled_dataset provided
    in order to save the segmentation result.
    Input:
        tiled_dataset: class TiledDataset object
        last_container: parent tiled client container for array saving
        uid: str, uid for saving
        model_name: str, name of the model used for inference
        array_name: str, result name
    Output:
        array_client: tiled client for result saving
    """
    assert (
        uid not in last_container.keys()
    ), f"uid_save: {uid} already existed in Tiled Server"

    last_container = last_container.create_container(key=uid)

    array_shape = tiled_dataset.shape
    # TODO: Check if this case is still valid
    # In case we have 2d array for the single mask case, where the ArrayClient will return a 2d array.
    if len(array_shape) == 2:
        array_shape = (1,) + array_shape
    structure = ArrayStructure.from_array(np.zeros(array_shape, dtype=np.int8))

    # For now, only save image 1 by 1 regardless of the batch_size_inference.
    structure.chunks = ((1,) * array_shape[0], (array_shape[1],), (array_shape[2],))

    mask_uri = None
    if isinstance(tiled_dataset, TiledMaskedDataset):
        mask_uri = tiled_dataset.mask_client.uri

    metadata = {
        "data_uri": tiled_dataset.data_client.uri,
        "mask_uri": mask_uri,
        "mask_idx": tiled_dataset.selected_indices,
        "uid": uid,
        "model": model_name,
    }

    array_client = last_container.new(
        structure_family=StructureFamily.array,
        data_sources=[
            DataSource(
                structure=structure,
                structure_family=StructureFamily.array,
            )
        ],
        metadata=metadata,
        key=array_name,
    )

    logging.info(
        f"Result space allocated in Tiled and segmentation will be saved in {array_client.uri}."
    )

    return array_client


def prepare_tiled_containers(io_parameters, dataset, network_name):
    """
    This function prepares the Tiled containers for saving segmentation results.
    Input:
        tiled_uri: str, The URI of the Tiled resource.
        tiled_api_key: str (optional), The API key for authentication. Default is None.
    Output:
        last_container: The final container in the uri path after ensuring all parent containers exist.
    """
    last_container = ensure_parent_containers(
        io_parameters.seg_tiled_uri, io_parameters.seg_tiled_api_key
    )
    seg_client = allocate_array_space(
        tiled_dataset=dataset,
        last_container=last_container,
        uid=io_parameters.uid_save,
        model_name=network_name,
        array_name="seg_result",
    )
    logging.info("Tiled segmentation result space allocated successfully.")
    return seg_client
