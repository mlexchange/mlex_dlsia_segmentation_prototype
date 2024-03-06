import os
from tiled.client import from_uri

# Create directory
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Local directory '{path}' created.")
    else:
        print(f"Local directory '{path}' already exsists.")


# Tiled Saving
def save_seg_to_tiled(seg_result,
                      tiled_dataset,
                      seg_tiled_uri,
                      seg_tiled_api_key,
                      uid,
                      model,
                      ):
    
    last_container = from_uri(seg_tiled_uri, api_key=seg_tiled_api_key)
    last_container = last_container.create_container(key=uid)

    metadata={
        'data_tiled_uri': tiled_dataset.data_tiled_uri,
        'mask_uri': tiled_dataset.mask_tiled_uri, 
        'mask_idx': tiled_dataset.mask_idx,
        'uid': uid,
        'model': model, 
        }
    print(f'metadata: {metadata}')
    seg_result = last_container.write_array(key="seg_result", array=seg_result, metadata=metadata)
    print("Segmentaion result array saved in following uri: ", seg_result.uri)
    return seg_result.uri, seg_result.metadata   