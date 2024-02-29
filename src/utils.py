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
                      data_tiled_uri,
                      mask_tiled_uri,
                      seg_tiled_uri,
                      seg_tiled_api_key,
                      container_keys,
                      uid,
                      model,
                      ):
    
    last_container = from_uri(seg_tiled_uri, api_key=seg_tiled_api_key)
    container_keys.append(uid)
    
    for key in container_keys:
        if key not in last_container.keys():
            last_container = last_container.create_container(key=key)
        else:
            last_container = last_container[key]

    metadata={
        'data_tiled_uri': data_tiled_uri,
        'mask_uri': mask_tiled_uri, 
        'uid': uid,
        'model': model, 
        }
    seg_result = last_container.write_array(key="seg_result", array=seg_result, metadata=metadata)
    print("Segmentaion result array saved in following uri: ", seg_result.uri)
    return seg_result.uri, seg_result.metadata   