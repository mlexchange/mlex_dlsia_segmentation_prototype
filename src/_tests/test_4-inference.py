import torch

def test_load_network(loaded_network, trained_network):
    assert loaded_network
    trained_network = trained_network[0]
    assert loaded_network.state_dict().keys() == trained_network.state_dict().keys()
    

def test_inference_cropping(inference_patches):
    assert inference_patches.shape == (4, 1, 4, 4)


def test_inference_loader(inference_dataloader, model_parameters, inference_patches):
    assert inference_dataloader
    assert len(inference_dataloader) == 2
    batch_size_inference = model_parameters.batch_size_train
    # Checking each batch
    for image_batch in inference_dataloader:
        assert (
            len(image_batch) == batch_size_inference
            or len(image_batch) == len(inference_patches) % batch_size_inference
        )
        assert image_batch[0].shape[1:] == inference_patches.shape[1:]
        assert image_batch[0].dtype == torch.float64


def test_prediction(prediction):
    assert type(prediction) is torch.Tensor
    assert len(prediction) == 4


def test_result(result):
    assert result.shape == (1, 6, 6)
