from pydantic import BaseModel, Field
from typing import Optional, List

#===========================================General Parameters===========================================#
class TrainingParameters(BaseModel):
    network: str = Field(description="type of dlsia network used")
    num_classes: int = Field(description="number of classes as output channel")
    num_epochs: int = Field(default=10, description="number of epochs")
    optimizer: str = Field(default="Adam", description="optimizer used for training")
    criterion: str = Field(default="CrossEntropyLoss", description="criterion for loss")
    weights: Optional[str] = Field(default=None, description="weights per class for imbalanced labeling")
    learning_rate: float = Field(default=1e-2, description='learning rate')
    
    activation: Optional[str] = Field(default="ReLU", description="activation function used in network")
    normalization: Optional[str] = Field(default="BatchNorm2d", description="normalization used between layers")
    convolution: Optional[str] = Field(default="Conv2d", description="convolution used in network")

    # Below are Data Loader related parameters
    shuffle_train: Optional[bool] = Field(default=True, description="whether to shuffle data in train set")
    batch_size_train: Optional[int] = Field(default=1, description="batch size of train set")

    shuffle_val: Optional[bool] = Field(default=True, description="whether to shuffle data in validation set")
    batch_size_val: Optional[int] = Field(default=1, description="batch size of validation set")

    shuffle_inference: Optional[bool] = Field(default=False, description="whether to shuffle data during inference")
    batch_size_inference: Optional[int] = Field(default=1, description="batch size for inference")

    num_workers: Optional[int] = Field(None, description="number of workers")
    pin_memory: Optional[bool] = Field(None, description="memory pinning")
    val_pct: Optional[float] = Field(default=0.2, description="percentage of data to use for validation")


#===========================================Network-specific Parameters===========================================
class MSDNetParameters(TrainingParameters):
    layer_width: Optional[int] = Field(default=1, description="layer width of MSDNet")    
    num_layers: Optional[int] = Field(default=3, description="number of layers for MSDNet")
    custom_dilation: Optional[bool] = Field(default=False, description="whether to customize dilation for MSDNet")
    max_dilation: Optional[int] = Field(default=5, description="maximum dilation for MSDNet")
    dilation_array: Optional[List[int]] = Field(default=None, description="customized dilation array for MSDNet")

class TUNetParameters(TrainingParameters):    
    depth: Optional[int] = Field(default=4, description='the depth of the UNet')
    base_channels: Optional[int] = Field(default=32, description='the number of initial channels for UNet')
    growth_rate: Optional[int] = Field(default=2, description='multiplicative growth factor of number of '\
                                       'channels per layer of depth for UNet')
    hidden_rate: Optional[int] = Field(default=1, description='multiplicative growth factor of channels within'\
                                       ' each layer for UNet')

class TUNet3PlusParameters(TUNetParameters):    
    carryover_channels: Optional[int] = Field(default=32, description='the number of channels in each skip '\
                                              'connection for UNet3+')
