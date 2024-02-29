import  argparse
from    network             import  load_network
from    parameters          import  MSDNetParameters, TUNetParameters, TUNet3PlusParameters
from    seg_utils           import  segment
from    tiled_dataset       import  TiledDataset
import  torch
from    torch.utils.data    import  DataLoader
from    torchvision         import  transforms
from    utils               import  save_seg_to_tiled
import  yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, help='path of yaml file for parameters')
    args = parser.parse_args()

    # Open the YAML file for all parameters
    with open(args.yaml_path, 'r') as file:
        # Load parameters
        parameters = yaml.safe_load(file)

    # Detect which model we have, then load corresponding parameters 
    raw_parameters = parameters['model_parameters']
    network = raw_parameters['network']

    model_parameters = None
    if network == 'MSDNet':
        model_parameters = MSDNetParameters(**raw_parameters)
    elif network == 'TUNet':
        model_parameters = TUNetParameters(**raw_parameters)
    elif network == 'TUNet3+':
        model_parameters = TUNet3PlusParameters(**raw_parameters)
    
    assert model_parameters, f"Received Unsupported Network: {network}"
    
    print('Parameters loaded successfully.')

    
    dataset = TiledDataset(
        data_tiled_uri=parameters['data_tiled_uri'],
        mask_idx=parameters['mask_idx'], # Keeping this for a quick inference for now, in future this will be out with updates from app.
        data_tiled_api_key=parameters['data_tiled_api_key'],
        transform=transforms.ToTensor()
        )
    
    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    inference_loader_params = {'batch_size': model_parameters.batch_size_inference,
                               'shuffle': model_parameters.shuffle_inference}
    # Build Dataloaders
    inference_loader = DataLoader(dataset, **inference_loader_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Network
    model_params_path = f"{parameters['save_path']}/{parameters['uid']}_{network}.pt"
    net = load_network(network, model_params_path)

    # Start segmentation
    seg_result = segment(net, device, inference_loader)
    
    # Save results back to Tiled
    # TODO: Change the hard-coding of container keys
    container_keys = ["mlex_store", 'rec20190524_085542_clay_testZMQ_8bit', 'results']
    
    
    seg_result_uri, seg_result_metadata = save_seg_to_tiled(seg_result, 
                                                            parameters['data_tiled_uri'], 
                                                            parameters['mask_tiled_uri'],
                                                            parameters['seg_tiled_uri'],
                                                            parameters['seg_tiled_api_key'], 
                                                            container_keys, 
                                                            parameters['uid'], 
                                                            network
                                                            )
