import  argparse
import  glob
from    network                 import  load_network, baggin_smsnet_ensemble
from    parameters              import  IOParameters, MSDNetParameters, TUNetParameters, TUNet3PlusParameters, SMSNetEnsembleParameters
from    seg_utils               import  custom_collate, segment
from    tiled_dataset           import  TiledDataset
import  torch
from    torch.utils.data        import  DataLoader
from    torchvision             import  transforms
from    utils                   import  allocate_array_space
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
    elif network == 'SMSNetEnsemble':
        model_parameters = SMSNetEnsembleParameters(**raw_parameters)
    
    assert model_parameters, f"Received Unsupported Network: {network}"
    
    print('Parameters loaded successfully.')

    dataset = TiledDataset(
        data_tiled_uri=io_parameters.data_tiled_uri,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        mask_tiled_uri=io_parameters.mask_tiled_uri,
        mask_tiled_api_key=io_parameters.mask_tiled_api_key,
        is_training=False,
        qlty_window=model_parameters.qlty_window,
        qlty_step=model_parameters.qlty_step,
        qlty_border=model_parameters.qlty_border,
        transform=transforms.ToTensor()
        )

    # Set Dataloader parameters (Note: we randomly shuffle the training set upon each pass)
    inference_loader_params = {'batch_size': model_parameters.batch_size_inference,
                               'shuffle': False}
    # Build Dataloaders
    inference_loader = DataLoader(dataset, **inference_loader_params, collate_fn=custom_collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Inference will be processed on: {device}')

    # Load Network
    if network == 'SMSNetEnsemble':
        net = baggin_smsnet_ensemble(io_parameters.uid_retrieve)
    else:
        net_files = glob.glob(f"{io_parameters.uid_retrieve}/*.pt")
        net = load_network(network, net_files[0])

    # Allocate Result space in Tiled
    seg_client = allocate_array_space(tiled_dataset=dataset,
                                      seg_tiled_uri=io_parameters.seg_tiled_uri,
                                      seg_tiled_api_key=io_parameters.seg_tiled_api_key,
                                      uid=io_parameters.uid_save,
                                      model=network,
                                      array_name='seg_result',
                                      )
    print(f'Result space allocated in Tiled and segmentation will be saved in {seg_client.uri}.')

    # Start segmentation and save frame by frame
    frame_count = segment(net, device, inference_loader, dataset.qlty_object, seg_client)

    assert frame_count == len(dataset)

    print('Segmentation completed.')
