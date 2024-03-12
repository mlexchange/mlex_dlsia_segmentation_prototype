# =================Training Commands==================================== #
train_msdnet_maxdil:
	python src/train.py example_yamls/example_msdnet_maxdil.yaml

train_msdnet_customdil:
	python src/train.py example_yamls/example_msdnet_customdil.yaml

train_tunet:
	python src/train.py example_yamls/example_tunet.yaml

train_tunet3plus:
	python src/train.py example_yamls/example_tunet3plus.yaml

train_smsnet_ensemble:
	python src/train.py example_yamls/example_smsnet_ensemble.yaml

# =================Inferening Commands==================================== #
segment_msdnet_maxdil:
	python src/segment.py example_yamls/example_msdnet_maxdil.yaml

segment_msdnet_customdil:
	python src/segment.py example_yamls/example_msdnet_customdil.yaml

segment_tunet:
	python src/segment.py example_yamls/example_tunet.yaml

segment_tunet3plus:
	python src/segment.py example_yamls/example_tunet3plus.yaml

segment_smsnet_ensemble:
	python src/segment.py example_yamls/example_smsnet_ensemble.yaml
