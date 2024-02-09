# Makefile for testing main.py

# Load environment variables from .env file
include .env
export $(shell sed 's/=.*//' .env)

# Define variables

MASK_IDX = [10, 201, 222, 493]
SHIFT = 2
SAVE_PATH = models
UID = uid0001
PARAMETERS = '{"network": "TUNet", \
			   "num_classes": 3, \
			   "num_epochs": 3, \
			   "optimizer": "Adam", \
			   "criterion": "CrossEntropyLoss", \
			   "learning_rate": 0.01, \
			   "depth": 4, \
			   "base_channels": 16, \
			   "growth_rate": 2, \
			   "hidden_rate": 1, \
			   "load": { \
			   "shuffle_train": true, \
			   "batch_size_train": 1, \
			   "shuffle_val": true, \
			   "batch_size_val": 1, \
			   "shuffle_test": false, \
			   "batch_size_test": 1, \
			   "val_pct": 0.2 \
			   } \
			  }'

# Define the default target
.PHONY: test

test_tunet:
	python src/main.py $(RECON_TILED_URI) $(MASK_TILED_URI) $(SEG_TILED_URI) \
                   $(RECON_TILED_API_KEY) $(MASK_TILED_API_KEY) $(SEG_TILED_API_KEY)\
                   $(MASK_IDX) $(SHIFT) $(SAVE_PATH) \
                   $(UID) $(PARAMETERS)
