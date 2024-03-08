import os
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure
import numpy as np

# Create directory
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Local directory '{path}' created.")
    else:
        print(f"Local directory '{path}' already exsists.")


# Tiled Saving
def allocate_array_space(
                      tiled_dataset,
                      seg_tiled_uri,
                      seg_tiled_api_key,
                      uid,
                      model,
                      array_name,
                      ):
    
    last_container = from_uri(seg_tiled_uri, api_key=seg_tiled_api_key)
    last_container = last_container.create_container(key=uid)
    array_shape = tiled_dataset.mask_client.shape if tiled_dataset.mask_client else tiled_dataset.data_client.shape
    structure = ArrayStructure.from_array(np.zeros(array_shape,dtype=np.int8))
    # For now, only save image 1 by 1 regardless of the batch_size_inference.
    structure.chunks = ((1,) * array_shape[0], (array_shape[1],), (array_shape[2],))

    metadata={
        'data_tiled_uri': tiled_dataset.data_tiled_uri,
        'mask_uri': tiled_dataset.mask_tiled_uri, 
        'mask_idx': tiled_dataset.mask_idx,
        'uid': uid,
        'model': model, 
        }

    array_client = last_container.new(structure_family='array',
                                      structure=structure,
                                      key=array_name,
                                      metadata=metadata,
                                      )
    return array_client