import os

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
                      container_keys,
                      uuid,
                      model,
                      ):
    last_container = tiled_dataset.seg_client
    for key in container_keys:
        if key not in last_container.keys():
            last_container = last_container.create_container(key=key)
        else:
            last_container = last_container[key]

    metadata={
        'data_tiled_uri': tiled_dataset.data_client.uri,
        'mask_uri': tiled_dataset.mask_client.uri, 
        'uuid': uuid,
        'model': model, 
        }
    seg_result = last_container.write_array(key="seg_result", array=seg_result, metadata=metadata)
    print("Segmentaion result array saved in following uri: ", seg_result.uri)
    return seg_result.uri, seg_result.metadata   