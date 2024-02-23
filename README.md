# mlex_dlsia_segmentation_prototype

This pipeline is built using DLSIA Package to run segmentation tasks for the High_Res Segmentation Application. 

Primary goal is to make this compatible with the updated segmentation application as a paperation for the incoming Diamond Beamtime.

## Feature Highlights

- Reading data and mask iput directly from Tiled Server
- Saving segmentation results back to Tiled Server
- Different Neural Network choices (MSDNet, TUNet, TUNet3+)

## To Test

### Set Up Local Tiled Server

This step is recommended in order to keep the Public Tiled Server clean for writing tests, as right now there is lack of good way or deleting containers rather than database modification.

### Request Public Tiled URI for Data and Masks

Please reach out to MLExchange Team.

### Installation

1. Git Clone the repository. 

2. Navigate to the repository, then activate a new conda environment (recommended).

3. Install packages.

```
pip install -r requirements.txt
```

4. Set environment variables via a `.env` file to configure a connection to the Tiled server.

```
RECON_TILED_URI = https://tiled-seg.als.lbl.gov/api/v1/metadata/reconstruction/rec20190524_085542_clay_testZMQ_8bit/20190524_085542_clay_testZMQ_
RECON_TILED_API_KEY = <key-provided-on-request>
MASK_TILED_URI = https://tiled-seg.als.lbl.gov/api/v1/metadata/reconstruction/seg-partial-rec20190524_085542_clay_testZMQ_8bit/seg-partial-20190524_085542_clay_testZMQ_
MASK_TILED_API_KEY = <key-provided-on-request>
SEG_TILED_URI = <Local Tiled Server URI> (for example: http://0.0.0.0:8888)
SEG_TILED_API_KEY = <Local Tiled API Key>

```

5. Open a Terminal and use pre-build commands from Makefile for testing:

```
make test_tunet
```

# Copyright
MLExchange Copyright (c) 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.