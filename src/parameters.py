from typing import Optional

from pydantic import BaseModel, Field


# ===========================================I/O Related Parameters===========================================#
class IOParameters(BaseModel):
    data_tiled_uri: str = Field(description="tiled uri for image data")
    data_tiled_api_key: str = Field(description="tiled api key for data client")
    mask_tiled_uri: Optional[str] = Field(
        default=None, description="tiled uri for masks"
    )
    mask_tiled_api_key: Optional[str] = Field(
        default=None, description="tiled api key for masks"
    )
    seg_tiled_uri: Optional[str] = Field(
        default=None, description="tiled uri for segmenation results"
    )
    seg_tiled_api_key: Optional[str] = Field(
        default=None, description="tiled api key for segmentation results"
    )
    uid_save: str = Field(description="uid to save models, metrics and etc")
    job_name: str = Field(description="segmentation job name")
    uid_retrieve: Optional[str] = Field(
        description="optional, uid to retrieve models for inference"
    )
    models_dir: Optional[str] = Field(
        default=None, description="directory to save model results"
    )


# ===========================================General Parameters===========================================#
class TrainingParameters(BaseModel):
    network: str = Field(description="type of dlsia network used")
    num_classes: int = Field(description="number of classes as output channel")
    num_epochs: int = Field(default=10, description="number of epochs")
    optimizer: str = Field(default="Adam", description="optimizer used for training")
    criterion: str = Field(default="CrossEntropyLoss", description="criterion for loss")
    weights: Optional[str] = Field(
        default=None, description="weights per class for imbalanced labeling"
    )
    learning_rate: float = Field(default=1e-2, description="learning rate")

    activation: Optional[str] = Field(
        default="ReLU", description="activation function used in network"
    )
    normalization: Optional[str] = Field(
        default="BatchNorm2d", description="normalization used between layers"
    )
    convolution: Optional[str] = Field(
        default="Conv2d", description="convolution used in network"
    )

    # Below are QLTY related parameters
    qlty_window: Optional[int] = Field(
        default=50, description="window size for stitched patches in qlty"
    )
    qlty_step: Optional[int] = Field(
        default=30, description="shifting size for stitched patches in qlty"
    )
    qlty_border: Optional[int] = Field(
        default=3, description="border parameter for stitched patches in qlty"
    )

    # Below are Data Loader related parameters
    shuffle_train: Optional[bool] = Field(
        default=True, description="whether to shuffle data in train set"
    )
    batch_size_train: Optional[int] = Field(
        default=1, description="batch size of train set"
    )

    batch_size_val: Optional[int] = Field(
        default=1, description="batch size of validation set"
    )

    batch_size_inference: Optional[int] = Field(
        default=1, description="batch size for inference"
    )
    # Commented out in case we need these back in the future
    # num_workers: Optional[int] = Field(None, description="number of workers")
    # pin_memory: Optional[bool] = Field(None, description="memory pinning")
    val_pct: Optional[float] = Field(
        default=0.2, description="percentage of data to use for validation"
    )


# ===========================================Network-specific Parameters===========================================
class MSDNetParameters(TrainingParameters):
    layer_width: Optional[int] = Field(default=1, description="layer width of MSDNet")
    num_layers: Optional[int] = Field(
        default=3, description="number of layers for MSDNet"
    )
    custom_dilation: Optional[bool] = Field(
        default=False, description="whether to customize dilation for MSDNet"
    )
    max_dilation: Optional[int] = Field(
        default=5, description="maximum dilation for MSDNet"
    )
    dilation_array: Optional[str] = Field(
        default=None, description="customized dilation array for MSDNet"
    )


class TUNetParameters(TrainingParameters):
    depth: Optional[int] = Field(default=4, description="the depth of the UNet")
    base_channels: Optional[int] = Field(
        default=32, description="the number of initial channels for UNet"
    )
    growth_rate: Optional[int] = Field(
        default=2,
        description="multiplicative growth factor of number of "
        "channels per layer of depth for UNet",
    )
    hidden_rate: Optional[int] = Field(
        default=1,
        description="multiplicative growth factor of channels within"
        " each layer for UNet",
    )


class TUNet3PlusParameters(TUNetParameters):
    carryover_channels: Optional[int] = Field(
        default=32,
        description="the number of channels in each skip " "connection for UNet3+",
    )


class SMSNetEnsembleParameters(TrainingParameters):
    num_networks: Optional[int] = Field(
        default=3, description="how many networks will be used in the ensemble"
    )
    layers: Optional[int] = Field(
        default=5, description="number of layers, limit from 5 to 20"
    )
    alpha: Optional[float] = Field(default=0.0, description="aplha used in ensemble")
    gamma: Optional[float] = Field(default=0.0, description="gamma used in ensemble")
    hidden_channels: Optional[int] = Field(
        default=None, description="hidden channels, limit from 3 to 20"
    )
    dilation_choices: Optional[str] = Field(
        default=None, description="customized dilation choices"
    )
    max_trial: Optional[int] = Field(
        default=10, description="max trial for the ensemble"
    )
