# Makefile for testing main.py

# Load environment variables from .env file
include .env
export $(shell sed 's/=.*//' .env)

# Define variables

MASK_IDX = "[10, 201, 222, 493]"
SHIFT = 2
SAVE_PATH = results/models
UID = uid0001

MSDNET_PARAMETERS_1 = '{"network": "MSDNet", \
			   "num_classes": 3, \
			   "num_epochs": 3, \
			   "optimizer": "Adam", \
			   "criterion": "CrossEntropyLoss", \
			   "learning_rate": 0.1, \
			   "activation": "ReLU", \
			   "normalization": "BatchNorm2d", \
			   "convolution": "Conv2d", \
			   "msdnet_parameters": { \
			   		"layer_width": 1, \
			   		"num_layers": 3, \
			   		"custom_dilation": false, \
			   		"max_dilation": 5 \
			   		}, \
			   "dataloaders": { \
			    	"shuffle_train": true, \
			    	"batch_size_train": 1, \
			    	"shuffle_val": true, \
			   		"batch_size_val": 1, \
			   		"shuffle_inference": false, \
			   		"batch_size_inference": 1, \
			   		"val_pct": 0.2 \
			   		} \
			  }'

MSDNET_PARAMETERS_2 = '{"network": "MSDNet", \
			   "num_classes": 3, \
			   "num_epochs": 3, \
			   "optimizer": "Adam", \
			   "criterion": "CrossEntropyLoss", \
			   "learning_rate": 0.1, \
			   "activation": "ReLU", \
			   "normalization": "BatchNorm2d", \
			   "convolution": "Conv2d", \
			   "msdnet_parameters": { \
			   		"layer_width": 1, \
			   		"num_layers": 3, \
			   		"custom_dilation": true, \
			   		"dilation_array": [1,2,4] \
			   		}, \
			   "dataloaders": { \
			    	"shuffle_train": true, \
			    	"batch_size_train": 1, \
			    	"shuffle_val": true, \
			   		"batch_size_val": 1, \
			   		"shuffle_inference": false, \
			   		"batch_size_inference": 1, \
			   		"val_pct": 0.2 \
			   		} \
			  }'

TUNET_PARAMETERS = '{"network": "TUNet", \
			   "num_classes": 3, \
			   "num_epochs": 3, \
			   "optimizer": "Adam", \
			   "criterion": "CrossEntropyLoss", \
			   "learning_rate": 0.1, \
			   "activation": "ReLU", \
			   "normalization": "BatchNorm2d", \
			   "convolution": "Conv2d", \
			   "tunet_parameters": { \
			   		"depth": 4, \
			   		"base_channels": 8, \
			   		"growth_rate": 2, \
			   		"hidden_rate": 1 \
			   		}, \
			   "dataloaders": { \
			    	"shuffle_train": true, \
			    	"batch_size_train": 1, \
			    	"shuffle_val": true, \
			   		"batch_size_val": 1, \
			   		"shuffle_inference": false, \
			   		"batch_size_inference": 1, \
			   		"val_pct": 0.2 \
			   		} \
			  }'

TUNET3PLUS_PARAMETERS = '{"network": "TUNet3+", \
			   "num_classes": 3, \
			   "num_epochs": 3, \
			   "optimizer": "Adam", \
			   "criterion": "CrossEntropyLoss", \
			   "learning_rate": 0.1, \
			   "activation": "ReLU", \
			   "normalization": "BatchNorm2d", \
			   "convolution": "Conv2d", \
			   "tunet3plus_parameters": { \
			   		"depth": 4, \
			   		"base_channels": 8, \
			   		"growth_rate": 2, \
			   		"hidden_rate": 1, \
					"carryover_channels": 8 \
			   		}, \
			   "dataloaders": { \
			    	"shuffle_train": true, \
			    	"batch_size_train": 1, \
			    	"shuffle_val": true, \
			   		"batch_size_val": 1, \
			   		"shuffle_inference": false, \
			   		"batch_size_inference": 1, \
			   		"val_pct": 0.2 \
			   		} \
			  }'

# Define the default target
#.PHONY: 

# =================Training Commands==================================== #
train_msdnet_maxdil:
	python src/train.py $(DATA_TILED_URI) $(MASK_TILED_URI) \
                   	   	$(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) \
                       	$(MASK_IDX) $(SHIFT) $(SAVE_PATH) $(UID) \
                       	$(MSDNET_PARAMETERS_1)

train_msdnet_customdil:
	python src/train.py $(DATA_TILED_URI) $(MASK_TILED_URI) \
                   	   	$(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) \
                       	$(MASK_IDX) $(SHIFT) $(SAVE_PATH) $(UID) \
                       	$(MSDNET_PARAMETERS_2)

train_tunet:
	python src/train.py $(DATA_TILED_URI) $(MASK_TILED_URI) \
                   	   	$(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) \
                       	$(MASK_IDX) $(SHIFT) $(SAVE_PATH) $(UID) \
                       	$(TUNET_PARAMETERS)

train_tunet3plus:
	python src/train.py $(DATA_TILED_URI) $(MASK_TILED_URI) \
                   	   	$(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) \
                       	$(MASK_IDX) $(SHIFT) $(SAVE_PATH) $(UID) \
                       	$(TUNET3PLUS_PARAMETERS)

# =================Inferening Commands==================================== #
segment_msdnet_maxdil:
	python src/segment.py $(DATA_TILED_URI) $(MASK_TILED_URI) $(SEG_TILED_URI) \
                   	   	  $(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) $(SEG_TILED_API_KEY) \
                       	  $(MASK_IDX) $(SAVE_PATH) $(UID) \
                          $(MSDNET_PARAMETERS_1)

segment_msdnet_customdil:
	python src/segment.py $(DATA_TILED_URI) $(MASK_TILED_URI) $(SEG_TILED_URI) \
                   	   	  $(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) $(SEG_TILED_API_KEY) \
                       	  $(MASK_IDX) $(SAVE_PATH) $(UID) \
                          $(MSDNET_PARAMETERS_2)

segment_tunet:
	python src/segment.py $(DATA_TILED_URI) $(MASK_TILED_URI) $(SEG_TILED_URI) \
                   	   	  $(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) $(SEG_TILED_API_KEY) \
                       	  $(MASK_IDX) $(SAVE_PATH) $(UID) \
                          $(TUNET_PARAMETERS)

segment_tunet3plus:
	python src/segment.py $(DATA_TILED_URI) $(MASK_TILED_URI) $(SEG_TILED_URI) \
                   	   	  $(DATA_TILED_API_KEY) $(MASK_TILED_API_KEY) $(SEG_TILED_API_KEY) \
                       	  $(MASK_IDX) $(SAVE_PATH) $(UID) \
                       	  $(TUNET3PLUS_PARAMETERS)