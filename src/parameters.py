from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List


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


class DataloadersParameters(BaseModel):
    shuffle: Optional[bool] = Field(description="shuffle data")
    batch_size: Optional[int] = Field(description="batch size")
    num_workers: Optional[int] = Field(description="number of workers")
    pin_memory: Optional[bool] = Field(description="memory pinning")
    val_pct: Optional[float] = Field(description="percentage of data to use for validation")


class TrainingParameters(BaseModel):
    network: DLSIANetwork
    num_epochs: int = Field(description="number of epochs")
    optimizer: Optimizer
    criterion: Criterion
    learning_rate: float = Field(description='learning rate')
    num_layers: Optional[int] = Field(description="number of layers for MSDNet")
    custom_dilation: Optional[bool] = Field(description="whether to customize dilation for MSDNet")
    max_dilation: Optional[int] = Field(description="maximum dilation for MSDNet")
    dilation_array: Optional[List[int]] = Field(description="customized dilation array for MSDNet")
    depth: Optional[int] = Field(description='the depth of the UNet')
    base_channels: Optional[int] = Field(description='the number of initial channels for UNet')
    growth_rate: Optional[int] = Field(description='multiplicative growth factor of number of '\
                                       'channels per layer of depth for UNet')
    hidden_rate: Optional[int] = Field(description='multiplicative growth factor of channels within'\
                                       ' each layer for UNet')
    carryover_channels: Optional[int] = Field(description='the number of channels in each skip '\
                                              'connection for UNet3+')
    load: Optional[DataloadersParameters]