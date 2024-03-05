import  argparse
from    network             import  load_network
from    parameters          import  IOParameters, MSDNetParameters, TUNetParameters, TUNet3PlusParameters
from    qlty.qlty2D         import  NCYXQuilt
from    seg_utils           import  custom_collate, segment
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

    # Validate and load I/O related parameters
    io_parameters = parameters['io_parameters']
    io_parameters = IOParameters(**io_parameters)

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
        data_tiled_uri=io_parameters.data_tiled_uri,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        mask_tiled_uri=io_parameters.mask_tiled_uri,
        mask_tiled_api_key=io_parameters.mask_tiled_api_key,
        workflow_type=io_parameters.workflow_type,
        qlty_window=model_parameters.qlty_window,
        qlty_step=model_parameters.qlty_step,
        qlty_border=model_parameters.qlty_border,
        transform=transforms.ToTensor()
        )

    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    inference_loader_params = {'batch_size': model_parameters.batch_size_inference,
                               'shuffle': model_parameters.shuffle_inference}
    # Build Dataloaders
    inference_loader = DataLoader(dataset, **inference_loader_params, collate_fn=custom_collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Network
    model_params_path = f"{io_parameters.uid}/{io_parameters.uid}_{network}.pt"
    net = load_network(network, model_params_path)

    qlty_object = NCYXQuilt(X=dataset.data_client.shape[-1], 
                            Y=dataset.data_client.shape[-2],
                            window = (model_parameters.qlty_window, model_parameters.qlty_window),
                            step = (model_parameters.qlty_step, model_parameters.qlty_step),
                            border = (model_parameters.qlty_border, model_parameters.qlty_border)
                            )
    # Start segmentation
    seg_result = segment(net, device, inference_loader, qlty_object)
    
    seg_result_uri, seg_result_metadata = save_seg_to_tiled(seg_result, 
                                                            io_parameters.data_tiled_uri, 
                                                            io_parameters.mask_tiled_uri,
                                                            io_parameters.seg_tiled_uri,
                                                            io_parameters.seg_tiled_api_key,
                                                            io_parameters.uid, 
                                                            network
                                                            )
