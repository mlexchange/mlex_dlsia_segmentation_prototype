import os
import pytest
import torch

from utils import find_device

@pytest.mark.parametrize(
    "cuda_available, expected_device_type",
    [
        (True, "cuda"),
        (False, "cpu"),
    ]
)
def test_find_device(monkeypatch, cuda_available, expected_device_type):
    # Monkey patch torch.cuda.is_available to return the specified value
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    device = find_device()
    assert device.type == expected_device_type, \
        f"Device should be {expected_device_type} when CUDA is {'available' if cuda_available else 'not available'}"

def test_dir_creation(model_directory):
    assert os.path.exists(model_directory)
    assert os.path.isdir(model_directory)
