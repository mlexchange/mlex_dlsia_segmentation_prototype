import  argparse
from    network         import  build_network
from    parameters      import  IOParameters, MSDNetParameters, TUNetParameters, TUNet3PlusParameters
from    seg_utils       import  train_val_split, train_segmentation
from    tiled_dataset   import  TiledDataset
import  torch
import  torch.nn        as      nn
import  torch.optim     as      optim
from    torchvision     import  transforms
from    utils           import  create_directory
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

    # Create Result Directory if not existed
    create_directory(io_parameters.uid)

    dataset = TiledDataset(
        data_tiled_uri=io_parameters.data_tiled_uri,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        mask_tiled_uri=io_parameters.mask_tiled_uri,
        mask_tiled_api_key=io_parameters.mask_tiled_api_key,
        is_training=True,
        qlty_window=model_parameters.qlty_window,
        qlty_step=model_parameters.qlty_step,
        qlty_border=model_parameters.qlty_border,
        transform=transforms.ToTensor()
        )
    train_loader, val_loader = train_val_split(dataset, model_parameters)
      
    # Build network
    net = build_network(
        network=network,
        data_shape=dataset.data_client.shape,
        num_classes=model_parameters.num_classes,
        parameters=model_parameters,
        )

    # Define criterion and optimizer
    criterion = getattr(nn, model_parameters.criterion)
    criterion = criterion(weight=torch.tensor(model_parameters.weights,dtype=torch.float),
                          ignore_index=-1, 
                          size_average=None
                          )    
    optimizer = getattr(optim, model_parameters.optimizer)
    optimizer = optimizer(net.parameters(), lr = model_parameters.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, results = train_segmentation(
        net,
        train_loader,
        val_loader,
        model_parameters.num_epochs,
        criterion,
        optimizer,
        device,
        savepath=io_parameters.uid,
        saveevery=None,
        scheduler=None,
        show=0,
        use_amp=False,
        clip_value=None
        )

    # Save network parameters
    model_params_path = f"{io_parameters.uid}/{io_parameters.uid}_{network}.pt"
    net.save_network_parameters(model_params_path)

    print(f'{network} trained successfully.')
    # Clear out unnecessary variables from device memory
    torch.cuda.empty_cache()

    
