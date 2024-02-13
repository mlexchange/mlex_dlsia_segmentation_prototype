from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
import torch.nn as nn


class DLSIANetwork(str, Enum):
    msdnet = 'MSDNet'
    tunet = 'TUNet'
    tunet3plus = 'TUNet3+'


class Optimizer(str, Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamw = "AdamW"
    sparseadam = "SparseAdam"
    adamax = "Adamax"
    asgd = "ASGD"
    lbfgs = "LBFGS"
    rmsprop = "RMSprop"
    rprop = "Rprop"
    sgd = "SGD"


class Criterion(str, Enum):
    l1loss = "L1Loss" 
    mseloss = "MSELoss" 
    crossentropyloss = "CrossEntropyLoss"
    ctcloss = "CTCLoss"
    poissonnllloss = "PoissonNLLLoss"
    gaussiannllloss = "GaussianNLLLoss"
    kldivloss = "KLDivLoss"
    bceloss = "BCELoss"
    bcewithlogitsloss = "BCEWithLogitsLoss"
    marginrankingloss = "MarginRankingLoss"
    hingeembeddingloss = "HingeEnbeddingLoss"
    multilabelmarginloss = "MultiLabelMarginLoss"
    huberloss = "HuberLoss"
    smoothl1loss = "SmoothL1Loss"
    softmarginloss = "SoftMarginLoss"
    multilabelsoftmarginloss = "MutiLabelSoftMarginLoss"
    cosineembeddingloss = "CosineEmbeddingLoss"
    multimarginloss = "MultiMarginLoss"
    tripletmarginloss = "TripletMarginLoss"
    tripletmarginwithdistanceloss = "TripletMarginWithDistanceLoss"


class Activation(str, Enum):
    relu = "ReLU"
    sigmoid = "Sigmoid"
    tanh = "Tanh"
    softmax = "Softmax"


class Normalization(str, Enum):
    batchnorm2d = "BatchNorm2d"
    batchnorm3d = "BatchNorm3d"


class Convolution(str, Enum):
    conv2d = "Conv2d"
    conv3d = "Conv3d"


class MSDNetParameters(BaseModel):
    layer_width: Optional[int] = Field(1, description="layer width of MSDNet")    
    num_layers: Optional[int] = Field(None, description="number of layers for MSDNet")
    custom_dilation: Optional[bool] = Field(None, description="whether to customize dilation for MSDNet")
    max_dilation: Optional[int] = Field(None, description="maximum dilation for MSDNet")
    dilation_array: Optional[List[int]] = Field(None, description="customized dilation array for MSDNet")
    

class TUNetParameters(BaseModel):    
    depth: Optional[int] = Field(None, description='the depth of the UNet')
    base_channels: Optional[int] = Field(None, description='the number of initial channels for UNet')
    growth_rate: Optional[int] = Field(None, description='multiplicative growth factor of number of '\
                                       'channels per layer of depth for UNet')
    hidden_rate: Optional[int] = Field(None, description='multiplicative growth factor of channels within'\
                                       ' each layer for UNet')

class TUNet3PlusParameters(BaseModel):    
    depth: Optional[int] = Field(None, description='the depth of the UNet3+')
    base_channels: Optional[int] = Field(None, description='the number of initial channels for UNet3+')
    growth_rate: Optional[int] = Field(None, description='multiplicative growth factor of number of '\
                                       'channels per layer of depth for UNet3+')
    hidden_rate: Optional[int] = Field(None, description='multiplicative growth factor of channels within'\
                                       ' each layer for UNet3+')
    carryover_channels: Optional[int] = Field(None, description='the number of channels in each skip '\
                                              'connection for UNet3+')
 

class DataloadersParameters(BaseModel):
    shuffle_train: Optional[bool] = Field(None, description="whether to shuffle data in train set")
    batch_size_train: Optional[int] = Field(None, description="batch size of train set")

    shuffle_val: Optional[bool] = Field(None, description="whether to shuffle data in validation set")
    batch_size_val: Optional[int] = Field(None, description="batch size of validation set")

    shuffle_test: Optional[bool] = Field(None, description="whether to shuffle data during inference")
    batch_size_test: Optional[int] = Field(None, description="batch size for inference")

    num_workers: Optional[int] = Field(None, description="number of workers")
    pin_memory: Optional[bool] = Field(None, description="memory pinning")
    val_pct: Optional[float] = Field(None, description="percentage of data to use for validation")


class TrainingParameters(BaseModel):
    network: DLSIANetwork
    num_classes: int = Field(description="number of classes as output channel")
    num_epochs: int = Field(description="number of epochs")
    optimizer: Optimizer
    criterion: Criterion
    learning_rate: float = Field(description='learning rate')
    
    activation: Optional[Activation] = None
    normalization: Optional[Normalization] = None
    convolution: Optional[Convolution] = None
    msdnet_parameters: Optional[MSDNetParameters] = None
    tunet_parameters: Optional[TUNetParameters] = None 
    tunet3plus_parameters: Optional[TUNet3PlusParameters] = None

    dataloaders: Optional[DataloadersParameters] = None