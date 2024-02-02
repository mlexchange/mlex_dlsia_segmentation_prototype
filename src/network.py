import logging
import numpy as np
import torch.nn as nn

from dlsia.core.networks import msdnet, tunet, tunet3plus

def build_msdnet(
        in_channels,
        out_channels,
        num_layers = 10,
        layer_width = 1,
        activation = nn.ReLU(),
        normalization = nn.BatchNorm2d,
        final_layer = nn.Softmax(dim=1),
        convolution = nn.Conv2d,
        custom_dilation = False,
        max_dilation = 10,
        dilation_array = np.array([1,2,4,8]),
        ):
    
    if custom_dilation == False:
        logging.INFO(f'Using maximum dilation: {max_dilation}')
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            layer_width=layer_width,
            max_dilation=max_dilation,
            activation=activation,
            normalization=normalization,
            final_layer=final_layer,
            convolution=convolution
            )
    else:
        dilation_array = np.array(dilation_array)
        logging.INFO(f'Using custom dilation: {dilation_array}')
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            layer_width=layer_width,
            custom_msdnet=dilation_array,
            activation=activation,
            normalization=normalization,
            final_layer=final_layer,
            convolution=convolution
            )
    return network

def build_tunet(
        in_channels,
        out_channels,
        img_size,
        depth = 4,
        base_channels = 16,
        growth_rate = 2,
        hidden_rate = 1,
        activation = nn.ReLU(),
        normalization = nn.BatchNorm2d,
        ):
    image_shape = img_size[2:]
    network = tunet.TUNet(
            image_shape=image_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            base_channels=base_channels,
            growth_rate=growth_rate,
            hidden_rate=hidden_rate,
            activation=activation,
            normalization=normalization,
            )
    return network

def build_tunet3plus(
        in_channels,
        out_channels,
        img_size,
        depth = 4,
        base_channels = 16,
        growth_rate = 2,
        hidden_rate = 1,
        carryover_channels = 16,
        activation = nn.ReLU(),
        normalization = nn.BatchNorm2d,
        ):
    image_shape = img_size[2:]
    network = tunet3plus.TUNet3Plus(
            image_shape=image_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            base_channels=base_channels,
            carryover_channels=carryover_channels,
            growth_rate=growth_rate,
            hidden_rate=hidden_rate,
            activation=activation,
            normalization=normalization,
            )
    return network

def build_network(
        network,
        num_classes,
        img_size,
        layer_width = 1,
        convolution = nn.Conv2d,
        num_layers = 10,
        activation = nn.ReLU(),
        normalization = nn.BatchNorm2d,
        final_layer = nn.Softmax(dim=1),
        custom_dilation = False,
        max_dilation = 10,
        dilation_array = np.array([1,2,4,8]),
        depth = 4,
        base_channels = 16,
        growth_rate = 2,
        hidden_rate = 1,
        carryover_channels = 16,
        ):
    
    in_channels = img_size[1]
    out_channels = num_classes

    if network == 'MSDNet':
        network = build_msdnet(
            in_channels,
            out_channels,
            num_layers = num_layers,
            layer_width = layer_width,
            activation = activation,
            normalization = normalization,
            final_layer = final_layer,
            convolution = convolution,
            custom_dilation = custom_dilation,
            max_dilation = max_dilation,
            dilation_array = dilation_array,
            )
        
    elif network == 'TUNet':
        network = build_tunet(
            in_channels,
            out_channels,
            img_size,
            depth = depth,
            base_channels = base_channels,
            growth_rate = growth_rate,
            hidden_rate = hidden_rate,
            activation = activation,
            normalization = normalization,
            )
        
    elif network == 'TUNet3+':
        network = build_tunet3plus(
            in_channels,
            out_channels,
            img_size,
            depth = depth,
            base_channels = base_channels,
            growth_rate = growth_rate,
            hidden_rate = hidden_rate,
            carryover_channels = carryover_channels,
            activation = activation,
            normalization = normalization,
            )
        
    return network
