import logging
import numpy as np
import torch.nn as nn

from dlsia.core.networks import msdnet, tunet, tunet3plus

def build_msdnet(
        in_channels,
        out_channels,
        msdnet_parameters,
        activation,
        normalization,
        convolution,
        final_layer = nn.Softmax(dim=1),
        ):
    
    if msdnet_parameters.custom_dilation == False:
        logging.info(f'Using maximum dilation: {msdnet_parameters.max_dilation}')
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=msdnet_parameters.num_layers,
            layer_width=msdnet_parameters.layer_width,
            max_dilation=msdnet_parameters.max_dilation,
            activation=activation,
            normalization=normalization,
            final_layer=final_layer,
            convolution=convolution
            )
    else:
        dilation_array = np.array(msdnet_parameters.dilation_array)
        logging.info(f'Using custom dilation: {dilation_array}')
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=msdnet_parameters.num_layers,
            layer_width=msdnet_parameters.layer_width,
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
        image_shape,
        tunet_parameters,
        activation,
        normalization,
        ):

    network = tunet.TUNet(
            image_shape=image_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=tunet_parameters.depth,
            base_channels=tunet_parameters.base_channels,
            growth_rate=tunet_parameters.growth_rate,
            hidden_rate=tunet_parameters.hidden_rate,
            activation=activation,
            normalization=normalization,
            )
    return network

def build_tunet3plus(
        in_channels,
        out_channels,
        image_shape,
        tunet3plus_parameters,
        activation,
        normalization,
        ):

    network = tunet3plus.TUNet3Plus(
            image_shape=image_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            depth=tunet3plus_parameters.depth,
            base_channels=tunet3plus_parameters.base_channels,
            growth_rate=tunet3plus_parameters.growth_rate,
            hidden_rate=tunet3plus_parameters.hidden_rate,
            carryover_channels=tunet3plus_parameters.carryover_channels,
            activation=activation,
            normalization=normalization,
            )
    return network

def build_network(
        network,
        recon_shape,
        num_classes,
        parameters,
        ):
    if len(recon_shape)==3:
        in_channels = 1
        image_shape = recon_shape[1:]
    else:
        in_channels = recon_shape[1]
        image_shape = recon_shape[2:]

    out_channels = num_classes

    if parameters.activation is not None:
        activation = getattr(nn, parameters.activation.value)
        activation = activation()

    if parameters.normalization is not None:
        normalization = getattr(nn, parameters.normalization.value)

    if parameters.convolution is not None:
        convolution = getattr(nn, parameters.convolution.value)   

    if network == 'MSDNet':
        network = build_msdnet(
            in_channels,
            out_channels,
            parameters.msdnet_parameters,
            activation,
            normalization,
            convolution,
            )
        
    elif network == 'TUNet':
        network = build_tunet(
            in_channels,
            out_channels,
            image_shape,
            parameters.tunet_parameters,
            activation,
            normalization,
            )
        
    elif network == 'TUNet3+':
        network = build_tunet3plus(
            in_channels,
            out_channels,
            image_shape,
            parameters.tunet3plus_parameters,
            activation,
            normalization,
            )
        
    return network
