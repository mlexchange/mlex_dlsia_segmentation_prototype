# mlex_dlsia_segmentation_prototype

This pipeline is developed using DLSIA Package to run segmentation tasks for the High_Res Segmentation Application.

The primary goal is to make this compatible with the updated segmentation application as a paperation for the incoming Diamond Beamtime in March 2024.

## Feature Highlights

- Reading data and mask iput directly from a Tiled Server
- Applying qlty cropping of frame to patches to ensure better training performance
- Using DVC Live to track training loss and metrics
- Saving segmentation results back to Tiled Server on-the-fly during inference
- An expansion of Neural Network choices to sync with the current DLSIA package (MSDNet, TUNet, TUNet3+, SMSNet Ensemble)

## To Test

### 1. Set Up Local Tiled Server

- This step is recommended in order to keep the Public Tiled Server clean for writing-related tests.

### 2. Request Public Tiled URI for Data and Masks

- Please reach out to MLExchange Team for details.

### 3. Installation

- Git Clone the repository.

 - Navigate to the repository, then create a new conda environment (recommended).

- Activate the conda environment.

- Install packages using:

```
pip install -r requirements.txt
```
- Initialize DVC for loss and metric saving, this only needs to be done once with your github account:

First, remove or comment out `.dvc*` and `dvc*` from .gitignore file.

Then run in the following order:

```
git init

dvc init

git config --global [user.name](http://user.name/) "John Doe"

git config --global user.email "johndoe@email.com"
```

Then add back dvc related files in .gitignore file.

Note: this step is only for dvc related repo and directory initialization, it will not enable auto-commits back to the GitHub repo or dvc repo. This feature will be discussed and potentially introduced in the future.

### 4. Parameter Initialization

- Navigate to directory "example_yamls/".
- Open one of the example yaml files for the model of interest you would like to test on.
- Inside the yaml file, change parameters according to you needs.
- `uid_save`: where you want to save the model and metric report.
- `uid_retrieve`: where you want to retrieve your trained model.
- Model parameters: we have pre-filled some values for speed testing, those are not meant for default settings. For more information regarding recommanded values for each neural network, please refer to [this](https://dlsia.readthedocs.io/en/latest/tutorialLinks/segmentation_MSDNet_TUNet_TUNet3plus.html) example in the DLSIA documentation.

### 5. Model Training

- We have prepared some make commands ready to be used in the make file. With the model name in your mind, go back to terminal and type

```
make train_<your model name>
```

For example, if you pre-filled values for a tunet yaml file:
```
make train_tunet
```
For exact commands to run the source code, please refer to the content of the Makefile.

### 6. Running Inference

- Once a trained model has been saved, you can run a quick inference to preview the segmentation result only on frames that have labeling information from masks. To do so:

```
make segment_tunet
```

- The segmented slices will be saved directly into the Tiled Server (`seg_tiled_uri`) you provided in the yaml file. If you are satisfiled with the segmentation result, you can run a full inference of the whole image stack by doing:

- Go back to your example_yaml file.
- Set `mask_tiled_uri` and `mask_tiled_api_key` to `null` (this is `None` in yaml), or simply comment out these two entries.
- Set your `uid_save` to be a different one if you have previously run a quick inference under the same uid.
- Save your yaml file.
- Go back to terminal and use the same make command from above:

```
make segment_tunet
```

Note: depending on the data size, this may take for a while.


# Copyright
MLExchange Copyright (c) 2024, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
